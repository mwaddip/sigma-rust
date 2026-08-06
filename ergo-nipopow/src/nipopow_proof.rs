use ergo_chain_types::{autolykos_pow_scheme::AutolykosPowSchemeError, BlockId, Header};
use ergo_merkle_tree::{BatchMerkleProof, MerkleNode};
use serde::{Deserialize, Serialize};
use sigma_ser::{
    vlq_encode::{ReadSigmaVlqExt, WriteSigmaVlqExt},
    ScorexParsingError, ScorexSerializable, ScorexSerializeResult,
};
use std::collections::HashSet;

use crate::nipopow_algos::NipopowAlgos;

/// Upper bound for prefix/suffix element counts and proof parameters.
/// Real proofs never exceed a few hundred entries; 20 000 is generous.
const MAX_NIPOPOW_PROOF_ELEMENTS: usize = 20_000;
/// Upper bound for a serialized header within a PoPowHeader (bytes).
const MAX_POPOW_HEADER_BYTES: usize = 10_000;
/// Upper bound for the number of interlinks in a PoPowHeader.
const MAX_POPOW_INTERLINKS: usize = 10_000;
/// Upper bound for the serialized interlinks proof (bytes).
const MAX_POPOW_PROOF_BYTES: usize = 1_000_000;
/// Maximum depth implied by the serialized `u32` Merkle index domain.
const MAX_MERKLE_PROOF_LEVELS: usize = u32::BITS as usize;
/// Upper bound for one serialized `PoPowHeader` element frame.
///
/// This is derived from the existing nested limits, plus room for length
/// fields and other fixed-size structure. It must remain checked before the
/// frame buffer is allocated.
const MAX_POPOW_HEADER_ELEMENT_BYTES: usize =
    MAX_POPOW_HEADER_BYTES + MAX_POPOW_INTERLINKS * 32 + MAX_POPOW_PROOF_BYTES + 64;

/// Read one length-declared element frame.
///
/// The declared size owns the outer stream boundary: exactly that many bytes
/// are consumed before parsing the element. The parser may leave trailing
/// bytes inside its private slice, matching the JVM `parseBytes` behavior.
fn read_framed<T: ScorexSerializable, R: ReadSigmaVlqExt>(
    r: &mut R,
    max_bytes: usize,
    what: &str,
) -> Result<T, ScorexParsingError> {
    let size = r.get_u32()? as usize;
    if size > max_bytes {
        return Err(ScorexParsingError::Io(format!(
            "{what} declared size {size} exceeds sanity limit {max_bytes}"
        )));
    }
    let mut buf = vec![0; size];
    r.read_exact(&mut buf)?;
    T::scorex_parse_bytes(&buf)
}

fn validate_batch_merkle_proof_frame(bytes: &[u8]) -> Result<(), ScorexParsingError> {
    const COUNT_BYTES: usize = 8;
    const INDEX_BYTES: usize = 36;
    const PROOF_BYTES: usize = 33;

    let counts = bytes.get(..COUNT_BYTES).ok_or_else(|| {
        ScorexParsingError::ValueOutOfBounds(
            "BatchMerkleProof counts do not fit declared proof frame".into(),
        )
    })?;
    let indices_len = u32::from_be_bytes([counts[0], counts[1], counts[2], counts[3]]) as usize;
    let proofs_len = u32::from_be_bytes([counts[4], counts[5], counts[6], counts[7]]) as usize;
    let indices_bytes = indices_len.checked_mul(INDEX_BYTES).ok_or_else(|| {
        ScorexParsingError::ValueOutOfBounds(
            "BatchMerkleProof index count does not fit declared proof frame".into(),
        )
    })?;
    let proofs_bytes = proofs_len.checked_mul(PROOF_BYTES).ok_or_else(|| {
        ScorexParsingError::ValueOutOfBounds(
            "BatchMerkleProof proof count does not fit declared proof frame".into(),
        )
    })?;
    let required_bytes = COUNT_BYTES
        .checked_add(indices_bytes)
        .and_then(|size| size.checked_add(proofs_bytes))
        .ok_or_else(|| {
            ScorexParsingError::ValueOutOfBounds(
                "BatchMerkleProof counts do not fit declared proof frame".into(),
            )
        })?;
    if required_bytes != bytes.len() {
        return Err(ScorexParsingError::ValueOutOfBounds(format!(
            "BatchMerkleProof encoded length requires {required_bytes} bytes and does not fit declared proof frame of {} bytes",
            bytes.len()
        )));
    }
    let max_proof_nodes = indices_len
        .checked_mul(MAX_MERKLE_PROOF_LEVELS)
        .ok_or_else(|| {
            ScorexParsingError::ValueOutOfBounds(
                "BatchMerkleProof structural proof-node bound overflow".into(),
            )
        })?;
    if proofs_len > max_proof_nodes {
        return Err(ScorexParsingError::ValueOutOfBounds(format!(
            "BatchMerkleProof proof count {proofs_len} exceeds structural limit {max_proof_nodes} for {indices_len} disclosed indices"
        )));
    }
    Ok(())
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
/// A structure representing NiPoPow proof as a persistent modifier.
pub struct NipopowProof {
    /// Algos
    #[serde(skip_serializing, skip_deserializing)]
    pub popow_algos: NipopowAlgos,
    /// Security parameter (min μ-level superchain length)
    #[serde(rename = "m")]
    pub m: u32,
    /// Security parameter (min suffix length, >= 1)
    #[serde(rename = "k")]
    pub k: u32,
    /// Proof prefix headers
    #[serde(rename = "prefix")]
    pub prefix: Vec<PoPowHeader>,
    /// First header of the suffix
    #[serde(rename = "suffixHead")]
    pub suffix_head: PoPowHeader,
    /// Tail of the proof suffix headers
    #[serde(rename = "suffixTail")]
    pub suffix_tail: Vec<Header>,
}

impl NipopowProof {
    /// Create new proof instance
    pub fn new(
        m: u32,
        k: u32,
        prefix: Vec<PoPowHeader>,
        suffix_head: PoPowHeader,
        suffix_tail: Vec<Header>,
    ) -> Result<NipopowProof, NipopowProofError> {
        Self::validate_parameters(m, k)?;
        Self::validate_suffix_length(k, suffix_tail.len())?;
        Ok(NipopowProof {
            popow_algos: NipopowAlgos::default(),
            m,
            k,
            prefix,
            suffix_head,
            suffix_tail,
        })
    }

    /// Validate proof parameters before any parameter-dependent allocation,
    /// arithmetic, loop, or indexing.
    pub(crate) fn validate_parameters(m: u32, k: u32) -> Result<(), NipopowProofError> {
        if m == 0 || m > MAX_NIPOPOW_PROOF_ELEMENTS as u32 {
            return Err(NipopowProofError::InvalidMParameter(m));
        }
        if k == 0 {
            return Err(NipopowProofError::ZeroKParameter);
        }
        if k > MAX_NIPOPOW_PROOF_ELEMENTS as u32 {
            return Err(NipopowProofError::InvalidKParameter(k));
        }
        Ok(())
    }

    fn validate_suffix_length(k: u32, suffix_tail_len: usize) -> Result<(), NipopowProofError> {
        let actual = u32::try_from(suffix_tail_len)
            .ok()
            .and_then(|len| len.checked_add(1))
            .ok_or(NipopowProofError::SuffixLengthMismatch {
                expected: k,
                actual: suffix_tail_len.saturating_add(1),
            })?;
        if actual != k {
            return Err(NipopowProofError::SuffixLengthMismatch {
                expected: k,
                actual: actual as usize,
            });
        }
        Ok(())
    }

    /// Implementation of the ≥ algorithm from [`KMZ17`], see Algorithm 4
    ///
    /// [`KMZ17`]: https://fc20.ifca.ai/preproceedings/74.pdf
    pub fn is_better_than(&self, that: &NipopowProof) -> Result<bool, NipopowProofError> {
        self.validate()?;
        that.validate()?;
        if (self.m, self.k) != (that.m, that.k) {
            return Err(NipopowProofError::IncompatibleComparisonParameters {
                left_m: self.m,
                left_k: self.k,
                right_m: that.m,
                right_k: that.k,
            });
        }
        if let Some(lca) = self.popow_algos.lowest_common_ancestor(
            &self.headers_chain().collect::<Vec<_>>(),
            &that.headers_chain().collect::<Vec<_>>(),
        ) {
            let self_headers = self
                .headers_chain()
                .filter(|h| h.height > lca.height)
                .collect::<Vec<_>>();
            let that_headers = that
                .headers_chain()
                .filter(|h| h.height > lca.height)
                .collect::<Vec<_>>();
            Ok(self.popow_algos.best_arg(&self_headers, self.m)?
                > self.popow_algos.best_arg(&that_headers, that.m)?)
        } else {
            Ok(false)
        }
    }

    /// Validate parameters, suffix cardinality, chain structure, and interlink proofs.
    pub fn validate(&self) -> Result<(), NipopowProofError> {
        Self::validate_parameters(self.m, self.k)?;
        Self::validate_suffix_length(self.k, self.suffix_tail.len())?;
        if !self.has_valid_connections() {
            return Err(NipopowProofError::InvalidProofStructure("connections"));
        }
        if !self.has_valid_heights() {
            return Err(NipopowProofError::InvalidProofStructure("heights"));
        }
        if !self.has_valid_proofs() {
            return Err(NipopowProofError::InvalidProofStructure("interlink proofs"));
        }
        Ok(())
    }

    /// Returns whether this proof passes [`NipopowProof::validate`].
    pub fn is_valid(&self) -> bool {
        self.validate().is_ok()
    }

    /// Checks the connections of the blocks in the proof.
    ///
    /// Adjacent blocks in the suffix must be linked via parent id (the
    /// suffix is a strict header chain).
    ///
    /// Prefix entries are linked via interlink or parent block id, but the
    /// check is *tolerant*: each entry need not connect to its immediate
    /// predecessor — it may connect to any of the up to
    /// `use_last_epochs + 3` immediately preceding entries. The tolerance
    /// exists because JVM-built proofs include continuous-mode
    /// difficulty-recalculation headers and naturally-skipped entries from
    /// sparse-superlevel walks; these show up as prefix entries that don't
    /// connect to their immediate sorted-by-height neighbour but do connect
    /// to a nearby earlier neighbour.
    ///
    /// Direct port of the JVM
    /// `org.ergoplatform.modifiers.history.popow.NipopowProof.hasValidConnections`
    /// (see `ergo-core/.../popow/NipopowProof.scala`, lines ~128-148):
    ///
    /// ```scala
    /// val maxDiffHeaders = popowAlgos.chainSettings.useLastEpochs + 1
    /// val prefixToCheck = prefix :+ suffixHead
    /// val prefixConnections = (1 until prefixToCheck.length).forall { checkIdx =>
    ///   val next = prefixToCheck(checkIdx)
    ///   (checkIdx - 1).to(Math.max(0, checkIdx - maxDiffHeaders - 1 - 1), -1).exists { prevIdx =>
    ///     val prev = prefixToCheck(prevIdx)
    ///     next.interlinks.contains(prev.id) || next.header.parentId == prev.id
    ///   }
    /// }
    /// ```
    ///
    /// The Scala `(checkIdx - 1).to(max(0, checkIdx - maxDiffHeaders - 2), -1)`
    /// range is **inclusive on both ends**, so the lookback covers indices
    /// `[max(0, checkIdx - maxDiffHeaders - 2), checkIdx - 1]` — a window of
    /// up to `maxDiffHeaders + 2 = use_last_epochs + 3` predecessors.
    pub fn has_valid_connections(&self) -> bool {
        let use_last_epochs = self.popow_algos.use_last_epochs as usize;
        // `maxDiffHeaders = useLastEpochs + 1` in JVM. The full lookback span
        // is `maxDiffHeaders + 2 = use_last_epochs + 3` predecessors.
        let lookback_span = use_last_epochs + 3;

        let prefix_len = self.prefix.len();
        // `prefixToCheck = prefix :+ suffixHead` — virtual concatenation,
        // accessed via the closure below to avoid an unnecessary clone.
        let prefix_to_check_len = prefix_len + 1;
        let get = |idx: usize| -> &PoPowHeader {
            if idx == prefix_len {
                &self.suffix_head
            } else {
                &self.prefix[idx]
            }
        };

        let prefix_connections = (1..prefix_to_check_len).all(|check_idx| {
            let next = get(check_idx);
            // JVM walks `(checkIdx - 1).to(max(0, checkIdx - maxDiffHeaders - 2), -1)`,
            // i.e. indices `[lookback_start, check_idx - 1]` inclusive,
            // descending. The descending order is observable only as a
            // micro-optimization (closer predecessors are likeliest to match);
            // any iteration order is semantically equivalent for `exists`.
            let lookback_start = check_idx.saturating_sub(lookback_span);
            (lookback_start..check_idx).rev().any(|prev_idx| {
                let prev = get(prev_idx);
                // Note that blocks with level 0 do not appear at all within
                // interlinks, which is why we need to check the parent
                // block id as well.
                next.interlinks.contains(&prev.header.id) || next.header.parent_id == prev.header.id
            })
        });

        let suffix_connections = std::iter::once(&self.suffix_head.header)
            .chain(self.suffix_tail.iter())
            .zip(self.suffix_tail.iter())
            .all(|(prev, next)| next.parent_id == prev.id);

        prefix_connections && suffix_connections
    }

    /// Checks if the heights of the header-chain provided are consistent, meaning that for any two
    /// blocks b1 and b2, if b1 precedes b2 then b1's height should be smaller. Return true if the
    /// heights of the header-chain are consistent
    fn has_valid_heights(&self) -> bool {
        self.headers_chain()
            .zip(self.headers_chain().skip(1))
            .all(|(prev, next)| prev.height < next.height)
    }
    /// Checks interlink proofs for each block using `PoPowHeader::check_interlinks_proof`
    fn has_valid_proofs(&self) -> bool {
        std::iter::once(&self.suffix_head)
            .chain(self.prefix.iter())
            .all(PoPowHeader::check_interlinks_proof)
    }

    /// Returns an iterator representing a chain of `Headers` from `self.prefix`, to
    /// `self.suffix_head` and `self.suffix_tail`.
    pub(crate) fn headers_chain(&self) -> impl Iterator<Item = &Header> {
        self.prefix
            .iter()
            .map(|p| &p.header)
            .chain(std::iter::once(&self.suffix_head.header).chain(self.suffix_tail.iter()))
    }
}

impl ScorexSerializable for NipopowProof {
    fn scorex_serialize<W: WriteSigmaVlqExt>(&self, w: &mut W) -> ScorexSerializeResult {
        w.put_u32(self.m)?;
        w.put_u32(self.k)?;
        w.put_u32(self.prefix.len() as u32)?;
        for p in &self.prefix {
            let prefix_num_bytes = p.scorex_serialize_bytes()?.len();
            w.put_u32(prefix_num_bytes as u32)?;
            p.scorex_serialize(w)?;
        }
        let suffix_head_num_bytes = self.suffix_head.scorex_serialize_bytes()?.len();
        w.put_u32(suffix_head_num_bytes as u32)?;
        self.suffix_head.scorex_serialize(w)?;
        w.put_u32(self.suffix_tail.len() as u32)?;
        for h in &self.suffix_tail {
            let header_num_bytes = h.scorex_serialize_bytes()?.len();
            w.put_u32(header_num_bytes as u32)?;
            h.scorex_serialize(w)?;
        }
        Ok(())
    }

    fn scorex_parse<R: ReadSigmaVlqExt>(r: &mut R) -> Result<Self, ScorexParsingError> {
        let m = r.get_u32()?;
        let k = r.get_u32()?;
        Self::validate_parameters(m, k)
            .map_err(|err| ScorexParsingError::ValueOutOfBounds(err.to_string()))?;
        let num_prefixes = r.get_u32()? as usize;
        if num_prefixes > MAX_NIPOPOW_PROOF_ELEMENTS {
            return Err(ScorexParsingError::Io(
                "num_prefixes exceeds sanity limit".into(),
            ));
        }
        let mut prefix = Vec::with_capacity(num_prefixes);
        for _ in 0..num_prefixes {
            prefix.push(read_framed(
                r,
                MAX_POPOW_HEADER_ELEMENT_BYTES,
                "prefix element",
            )?);
        }
        let suffix_head = read_framed(r, MAX_POPOW_HEADER_ELEMENT_BYTES, "suffix head")?;
        let num_suffix_tail = r.get_u32()? as usize;
        if num_suffix_tail > MAX_NIPOPOW_PROOF_ELEMENTS {
            return Err(ScorexParsingError::Io(
                "num_suffix_tail exceeds sanity limit".into(),
            ));
        }
        Self::validate_suffix_length(k, num_suffix_tail)
            .map_err(|err| ScorexParsingError::ValueOutOfBounds(err.to_string()))?;
        let mut suffix_tail = Vec::with_capacity(num_suffix_tail);
        for _ in 0..num_suffix_tail {
            suffix_tail.push(read_framed(
                r,
                MAX_POPOW_HEADER_BYTES,
                "suffix-tail header",
            )?);
        }
        Self::new(m, k, prefix, suffix_head, suffix_tail)
            .map_err(|err| ScorexParsingError::ValueOutOfBounds(err.to_string()))
    }
}

/// `NipopowProof` errors
#[derive(PartialEq, Eq, Debug, Clone, thiserror::Error)]
pub enum NipopowProofError {
    /// Errors from `AutolykosPowScheme`
    #[error("{0:?}")]
    AutolykosPowSchemeError(#[from] AutolykosPowSchemeError),
    /// `m` is outside the supported proof-resource range.
    #[error("m parameter {0} must be in 1..=20000")]
    InvalidMParameter(u32),
    /// `k` parameter == 0. Must be >= 1.
    #[error("k parameter == 0. Must be >= 1")]
    ZeroKParameter,
    /// `k` exceeds the supported proof-resource range.
    #[error("k parameter {0} must be in 1..=20000")]
    InvalidKParameter(u32),
    /// Declared `k` does not match the number of suffix headers.
    #[error("suffix length {actual} does not match k parameter {expected}")]
    SuffixLengthMismatch {
        /// Declared suffix length.
        expected: u32,
        /// Actual suffix length, including the suffix head.
        actual: usize,
    },
    /// Proofs with different security parameters cannot be compared.
    #[error(
        "cannot compare proofs with parameters ({left_m}, {left_k}) and ({right_m}, {right_k})"
    )]
    IncompatibleComparisonParameters {
        /// Left proof `m`.
        left_m: u32,
        /// Left proof `k`.
        left_k: u32,
        /// Right proof `m`.
        right_m: u32,
        /// Right proof `k`.
        right_k: u32,
    },
    /// A structural or cryptographic proof predicate failed.
    #[error("invalid NiPoPoW proof structure: {0}")]
    InvalidProofStructure(&'static str),
    /// Can not prove non-anchored (first block is non-Genesis) chain
    #[error("Can not prove non-anchored (first block is non-Genesis) chain")]
    NonAnchoredChain,
    /// Chain must be of length `>= k + m`
    #[error("Chain must be of length `>= k + m`")]
    ChainTooShort,
    /// A `PopowHeaderReader` lookup returned `None` for a header that the
    /// proof construction algorithm expected to be present (genesis, an
    /// interlink target, the suffix head, or a header in the suffix tail).
    /// Indicates the reader is inconsistent with the chain it claims to
    /// expose.
    #[error("Popow header reader returned None for an expected header")]
    MissingPopowHeader,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
/// PoPowHeader structure. Represents the block header and unpacked interlinks
pub struct PoPowHeader {
    /// The block header
    pub header: Header,
    /// Interlinks are stored in reverse order: first element is always genesis header, then level
    /// of lowest target met etc
    pub interlinks: Vec<BlockId>,
    /// BatchMerkleProof for interlinks in extension field
    #[serde(rename = "interlinksProof")]
    pub interlinks_proof: BatchMerkleProof,
}

impl PoPowHeader {
    fn has_canonical_interlink_runs(&self) -> bool {
        let Some(mut current) = self.interlinks.first() else {
            return false;
        };
        let max_run_length = usize::from(u8::MAX);
        let max_key_position = usize::from(u8::MAX);
        let mut run_length = 1usize;
        let mut closed_runs = HashSet::new();

        for (position, interlink) in self.interlinks.iter().enumerate().skip(1) {
            if interlink == current {
                if run_length == max_run_length {
                    return false;
                }
                run_length += 1;
            } else {
                if position > max_key_position {
                    return false;
                }
                closed_runs.insert(*current);
                if closed_runs.contains(interlink) {
                    return false;
                }
                current = interlink;
                run_length = 1;
            }
        }
        true
    }

    /// Validates the exact packed interlink leaves against the full extension root
    pub fn check_interlinks_proof(&self) -> bool {
        let proof_is_empty = self.interlinks_proof.get_indices().is_empty()
            && self.interlinks_proof.get_proofs().is_empty();
        if self.header.height == 1 {
            return self.interlinks.is_empty() && proof_is_empty;
        }
        if self.interlinks.is_empty() {
            return false;
        }
        if !self.has_canonical_interlink_runs() {
            return false;
        }

        let expected_leaf_hashes: Vec<_> = NipopowAlgos::pack_interlinks(self.interlinks.clone())
            .into_iter()
            .map(|(key, value)| -> Vec<u8> {
                std::iter::once(2u8)
                    .chain(key.iter().copied())
                    .chain(value)
                    .collect()
            })
            .map(MerkleNode::from_bytes)
            .filter_map(|node| node.get_hash().copied())
            .collect();
        let proven_leaves = self.interlinks_proof.get_indices();

        expected_leaf_hashes.len() == proven_leaves.len()
            && expected_leaf_hashes
                .iter()
                .zip(proven_leaves)
                .all(|(expected, proven)| expected == &proven.hash)
            && self
                .interlinks_proof
                .valid(self.header.extension_root.as_ref())
    }
}

impl ScorexSerializable for PoPowHeader {
    fn scorex_serialize<W: WriteSigmaVlqExt>(&self, w: &mut W) -> ScorexSerializeResult {
        let bytes = self.header.scorex_serialize_bytes()?;
        w.put_u32(bytes.len() as u32)?;
        w.write_all(&bytes)?;
        w.put_u32(self.interlinks.len() as u32)?;
        for interlink in self.interlinks.iter() {
            w.write_all(&interlink.0 .0)?;
        }
        let proof_bytes = self.interlinks_proof.scorex_serialize_bytes()?;
        w.put_u32(proof_bytes.len() as u32)?;
        w.write_all(&proof_bytes)?;

        Ok(())
    }

    fn scorex_parse<R: ReadSigmaVlqExt>(r: &mut R) -> Result<Self, ScorexParsingError> {
        let header_size = r.get_u32()? as usize;
        if header_size > MAX_POPOW_HEADER_BYTES {
            return Err(ScorexParsingError::Io(
                "header_size exceeds sanity limit".into(),
            ));
        }
        let mut buf = vec![0; header_size];
        r.read_exact(&mut buf)?;
        let header = Header::scorex_parse(&mut std::io::Cursor::new(buf))?;

        let interlinks_size = r.get_u32()? as usize;
        if interlinks_size > MAX_POPOW_INTERLINKS {
            return Err(ScorexParsingError::Io(
                "interlinks_size exceeds sanity limit".into(),
            ));
        }

        let interlinks: Result<Vec<BlockId>, ScorexParsingError> = (0..interlinks_size)
            .map(|_| {
                let mut buf = [0; 32];
                r.read_exact(&mut buf)?;
                Ok(BlockId(buf.into()))
            })
            .collect();

        let proof_bytes = r.get_u32()? as usize;
        if proof_bytes > MAX_POPOW_PROOF_BYTES {
            return Err(ScorexParsingError::Io(
                "proof_bytes exceeds sanity limit".into(),
            ));
        }
        let mut proof_buf = vec![0u8; proof_bytes];
        r.read_exact(&mut proof_buf)?;
        validate_batch_merkle_proof_frame(&proof_buf)?;
        let interlinks_proof = BatchMerkleProof::scorex_parse_bytes(&proof_buf);

        Ok(Self {
            header,
            interlinks: interlinks?,
            interlinks_proof: interlinks_proof?,
        })
    }
}

#[cfg(feature = "arbitrary")]
#[allow(clippy::unwrap_used)]
mod arbitrary {
    use super::*;
    use ergo_chain_types::Digest32;
    use ergo_chain_types::ExtensionCandidate;
    use proptest::prelude::*;
    use proptest::{arbitrary::Arbitrary, collection::vec};

    impl Arbitrary for PoPowHeader {
        type Parameters = ();
        type Strategy = BoxedStrategy<PoPowHeader>;

        fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
            (any::<Box<Header>>(), vec(any::<Digest32>(), 1..10))
                .prop_map(|(header, digests)| PoPowHeader {
                    header: *header,
                    interlinks: digests.iter().cloned().map(BlockId).collect(),
                    interlinks_proof: NipopowAlgos::proof_for_interlink_vector(
                        &ExtensionCandidate::new(NipopowAlgos::pack_interlinks(
                            digests.into_iter().map(BlockId).collect(),
                        ))
                        .unwrap(),
                    )
                    .unwrap(),
                })
                .boxed()
        }
    }

    impl Arbitrary for NipopowProof {
        type Parameters = ();
        type Strategy = BoxedStrategy<NipopowProof>;

        fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
            (
                1_u32..=MAX_NIPOPOW_PROOF_ELEMENTS as u32,
                vec(any::<PoPowHeader>(), 1..10),
                any::<PoPowHeader>(),
                vec(any::<Header>(), 0..10),
            )
                .prop_map(|(m, prefix, suffix_head, suffix_tail)| {
                    let k = suffix_tail.len() as u32 + 1;
                    NipopowProof {
                        popow_algos: NipopowAlgos::default(),
                        m,
                        k,
                        prefix,
                        suffix_head,
                        suffix_tail,
                    }
                })
                .boxed()
        }
    }
}

#[cfg(test)]
#[cfg(feature = "arbitrary")]
#[allow(clippy::unwrap_used, clippy::panic)]
pub mod tests {
    use super::*;
    use ergo_chain_types::{Digest32, ExtensionCandidate};
    use ergo_merkle_tree::{BatchMerkleProof, MerkleNode, MerkleTree, NodeSide};
    use proptest::prelude::*;
    use proptest::strategy::ValueTree;
    use proptest::test_runner::TestRunner;
    use sigma_ser::scorex_serialize_roundtrip;
    proptest! {

        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn nipopowproof_roundtrip(v in any::<NipopowProof>()) {
            prop_assert_eq![scorex_serialize_roundtrip(&v), v];
        }


    }

    /// Build a `BlockId` filled with the given byte. Used by the
    /// `has_valid_connections` regression tests below to mint distinct,
    /// human-traceable block ids without paying the cost of real hashing.
    fn id_from_byte(byte: u8) -> BlockId {
        BlockId(Digest32::from([byte; 32]))
    }

    /// Generate one valid base `Header` via the proptest `Arbitrary` impl,
    /// then return a customizing closure. The closure rewrites the
    /// `id`, `parent_id`, and `height` fields without rebuilding the rest
    /// of the (irrelevant-to-this-test) header content. We only need the
    /// three fields above because `has_valid_connections` only inspects
    /// `header.id`, `header.parent_id`, and `PoPowHeader::interlinks`.
    fn header_factory() -> impl Fn(BlockId, BlockId, u32) -> Header {
        let mut runner = TestRunner::default();
        let base = any::<Box<Header>>()
            .new_tree(&mut runner)
            .unwrap()
            .current();
        move |id, parent_id, height| {
            let mut h = (*base).clone();
            h.id = id;
            h.parent_id = parent_id;
            h.height = height;
            h
        }
    }

    /// Build a `PoPowHeader` from a header plus an interlink vector. The
    /// `interlinks_proof` is a no-op empty proof — `has_valid_connections`
    /// never invokes `check_interlinks_proof`, so this is sufficient for
    /// connection-check tests.
    fn pop_header(header: Header, interlinks: Vec<BlockId>) -> PoPowHeader {
        PoPowHeader {
            header,
            interlinks,
            interlinks_proof: BatchMerkleProof::new(vec![], vec![]),
        }
    }

    fn extension_tree(fields: &[([u8; 2], Vec<u8>)]) -> MerkleTree {
        MerkleTree::new(
            fields
                .iter()
                .map(|(key, value)| {
                    let leaf: Vec<u8> = std::iter::once(2u8)
                        .chain(key.iter().copied())
                        .chain(value.iter().copied())
                        .collect();
                    MerkleNode::from_bytes(leaf)
                })
                .collect::<Vec<_>>(),
        )
    }

    /// Build a structurally valid proof whose suffix contains exactly `k`
    /// headers. A genesis suffix head needs neither interlinks nor a Merkle
    /// proof, so this fixture isolates parameter validation from cryptography.
    fn valid_proof(m: u32, k: u32) -> NipopowProof {
        assert!(k >= 1);
        let mk_header = header_factory();
        let suffix_head_id = id_from_byte(0x40);
        let suffix_head = pop_header(mk_header(suffix_head_id, id_from_byte(0), 1), vec![]);
        let mut suffix_tail = Vec::new();
        let mut parent_id = suffix_head_id;
        for index in 1..k {
            let id = id_from_byte(0x40_u8.wrapping_add(index as u8));
            suffix_tail.push(mk_header(id, parent_id, index + 1));
            parent_id = id;
        }
        NipopowProof {
            popow_algos: NipopowAlgos::default(),
            m,
            k,
            prefix: vec![],
            suffix_head,
            suffix_tail,
        }
    }

    #[test]
    fn new_rejects_zero_m() {
        let proof = valid_proof(1, 1);
        assert_eq!(
            NipopowProof::new(
                0,
                proof.k,
                proof.prefix,
                proof.suffix_head,
                proof.suffix_tail,
            )
            .unwrap_err(),
            NipopowProofError::InvalidMParameter(0)
        );
    }

    #[test]
    fn parameter_bounds_accept_sanity_limit() {
        let max = MAX_NIPOPOW_PROOF_ELEMENTS as u32;
        assert_eq!(NipopowProof::validate_parameters(max, max), Ok(()));
    }

    #[test]
    fn new_rejects_zero_k() {
        let proof = valid_proof(1, 1);
        assert_eq!(
            NipopowProof::new(
                proof.m,
                0,
                proof.prefix,
                proof.suffix_head,
                proof.suffix_tail,
            )
            .unwrap_err(),
            NipopowProofError::ZeroKParameter
        );
    }

    #[test]
    fn new_rejects_m_above_sanity_limit() {
        let proof = valid_proof(1, 1);
        assert_eq!(
            NipopowProof::new(
                20_001,
                proof.k,
                proof.prefix,
                proof.suffix_head,
                proof.suffix_tail,
            )
            .unwrap_err(),
            NipopowProofError::InvalidMParameter(20_001)
        );
    }

    #[test]
    fn new_rejects_k_above_sanity_limit() {
        let proof = valid_proof(1, 1);
        assert_eq!(
            NipopowProof::new(
                proof.m,
                20_001,
                proof.prefix,
                proof.suffix_head,
                proof.suffix_tail,
            )
            .unwrap_err(),
            NipopowProofError::InvalidKParameter(20_001)
        );
    }

    #[test]
    fn new_rejects_suffix_length_not_equal_to_k() {
        let proof = valid_proof(6, 1);
        assert_eq!(
            NipopowProof::new(
                proof.m,
                2,
                proof.prefix,
                proof.suffix_head,
                proof.suffix_tail,
            )
            .unwrap_err(),
            NipopowProofError::SuffixLengthMismatch {
                expected: 2,
                actual: 1,
            }
        );
    }

    #[test]
    fn parser_rejects_zero_m() {
        let bytes = valid_proof(0, 1).scorex_serialize_bytes().unwrap();
        assert!(NipopowProof::scorex_parse_bytes(&bytes).is_err());
    }

    #[test]
    fn parser_rejects_suffix_length_not_equal_to_k() {
        let mut proof = valid_proof(6, 2);
        proof.k = 1;
        let bytes = proof.scorex_serialize_bytes().unwrap();
        let mut cursor = std::io::Cursor::new(bytes.as_slice());
        assert!(NipopowProof::scorex_parse(&mut cursor).is_err());
        assert!(
            cursor.position() < bytes.len() as u64,
            "suffix mismatch must be rejected before parsing tail elements"
        );
    }

    #[test]
    fn comparison_rejects_different_m() {
        assert_eq!(
            valid_proof(6, 1)
                .is_better_than(&valid_proof(7, 1))
                .unwrap_err(),
            NipopowProofError::IncompatibleComparisonParameters {
                left_m: 6,
                left_k: 1,
                right_m: 7,
                right_k: 1,
            }
        );
    }

    #[test]
    fn comparison_rejects_different_k() {
        assert_eq!(
            valid_proof(6, 1)
                .is_better_than(&valid_proof(6, 2))
                .unwrap_err(),
            NipopowProofError::IncompatibleComparisonParameters {
                left_m: 6,
                left_k: 1,
                right_m: 6,
                right_k: 2,
            }
        );
    }

    #[test]
    fn validate_accepts_consistent_proof() {
        let proof = valid_proof(6, 2);
        assert_eq!(proof.validate(), Ok(()));
        assert!(proof.is_valid());
    }

    #[test]
    fn validate_rejects_zero_m() {
        let proof = valid_proof(0, 1);
        assert_eq!(
            proof.validate(),
            Err(NipopowProofError::InvalidMParameter(0))
        );
    }

    #[test]
    fn validate_rejects_zero_k() {
        let mut proof = valid_proof(6, 1);
        proof.k = 0;
        assert_eq!(proof.validate(), Err(NipopowProofError::ZeroKParameter));
    }

    #[test]
    fn validate_rejects_suffix_length_not_equal_to_k() {
        let mut proof = valid_proof(6, 1);
        proof.k = 2;
        assert_eq!(
            proof.validate(),
            Err(NipopowProofError::SuffixLengthMismatch {
                expected: 2,
                actual: 1,
            })
        );
    }

    #[test]
    fn validate_rejects_connection_failure() {
        let mut proof = valid_proof(6, 2);
        proof.suffix_tail[0].parent_id = id_from_byte(0xff);
        assert_eq!(
            proof.validate(),
            Err(NipopowProofError::InvalidProofStructure("connections"))
        );
        assert!(!proof.is_valid());
    }

    #[test]
    fn validate_rejects_height_failure() {
        let mut proof = valid_proof(6, 2);
        proof.suffix_tail[0].height = 1;
        assert!(proof.has_valid_connections());
        assert_eq!(
            proof.validate(),
            Err(NipopowProofError::InvalidProofStructure("heights"))
        );
    }

    #[test]
    fn validate_rejects_interlink_proof_failure() {
        let mut proof = valid_proof(6, 1);
        proof.suffix_head.header.height = 2;
        assert!(proof.has_valid_connections());
        assert_eq!(
            proof.validate(),
            Err(NipopowProofError::InvalidProofStructure("interlink proofs"))
        );
    }

    fn interlinks_only_popow_header(height: u32, interlinks: Vec<BlockId>) -> PoPowHeader {
        let fields = NipopowAlgos::pack_interlinks(interlinks.clone());
        let extension = ExtensionCandidate::new(fields.clone()).unwrap();
        let proof = NipopowAlgos::proof_for_interlink_vector(&extension).unwrap();
        let make_header = header_factory();
        let mut header = make_header(id_from_byte(0x31), id_from_byte(0x30), height);
        header.extension_root = extension_tree(&fields).root_hash();
        PoPowHeader {
            header,
            interlinks,
            interlinks_proof: proof,
        }
    }

    /// Constructs a deliberately-skipped prefix and asserts the JVM-tolerant
    /// `has_valid_connections` accepts it.
    ///
    /// Layout: `h0 -> h1 -> h2 -> h3 -> suffix_head`. `h2.parent_id` is set
    /// to an unrelated id (NOT `h1.id`) and `h1.id` is **not** in
    /// `h2.interlinks`, so the strict (pre-fix) verifier would have rejected
    /// at index 2. The tolerant verifier MUST accept because `h0.id` is in
    /// `h2.interlinks` and `h0` is within the lookback window
    /// (`use_last_epochs + 3 = 11` predecessors by default).
    #[test]
    fn has_valid_connections_accepts_skipped_prefix_entry() {
        let mk_header = header_factory();

        let h0_id = id_from_byte(1);
        let h1_id = id_from_byte(2);
        let h2_id = id_from_byte(3);
        let h3_id = id_from_byte(4);
        let suffix_head_id = id_from_byte(5);
        let suffix_tail_id = id_from_byte(6);
        let unrelated_parent = id_from_byte(0xff);

        // h0: genesis. parent_id is irrelevant for index 0.
        let h0 = pop_header(mk_header(h0_id, id_from_byte(0), 1), vec![h0_id]);
        // h1: connects via parent_id == h0.id.
        let h1 = pop_header(mk_header(h1_id, h0_id, 10), vec![h0_id]);
        // h2: parent_id is UNRELATED (not h1) and h1.id is NOT in interlinks,
        // but h0.id IS in interlinks → tolerant verifier connects via h0
        // through the lookback window.
        let h2 = pop_header(mk_header(h2_id, unrelated_parent, 20), vec![h0_id]);
        // h3: connects via parent_id == h2.id.
        let h3 = pop_header(mk_header(h3_id, h2_id, 30), vec![h0_id, h2_id]);
        // suffix_head: connects via parent_id == h3.id.
        let suffix_head = pop_header(
            mk_header(suffix_head_id, h3_id, 40),
            vec![h0_id, h2_id, h3_id],
        );
        // suffix_tail must be a strict parent_id chain.
        let suffix_tail = vec![mk_header(suffix_tail_id, suffix_head_id, 41)];

        let proof =
            NipopowProof::new(6, 2, vec![h0, h1, h2, h3], suffix_head, suffix_tail).unwrap();

        assert!(
            proof.has_valid_connections(),
            "tolerant verifier must accept a prefix where an entry skips its \
             immediate predecessor but still connects to a header within the \
             lookback window"
        );
    }

    /// Constructs a prefix with a gap LARGER than the lookback window and
    /// asserts the verifier still rejects. This proves the fix is not a
    /// blanket accept-all.
    ///
    /// We squeeze the lookback by setting `use_last_epochs = 0`, which gives
    /// `lookback_span = 3`. The chain is then designed so that the bad entry
    /// (suffix_head, at index 4) only connects backward to index 0, which is
    /// outside the `[1, 3]` lookback range.
    #[test]
    fn has_valid_connections_rejects_too_far_skip() {
        let mk_header = header_factory();

        let h0_id = id_from_byte(1);
        let h1_id = id_from_byte(2);
        let h2_id = id_from_byte(3);
        let h3_id = id_from_byte(4);
        let suffix_head_id = id_from_byte(5);
        let suffix_tail_id = id_from_byte(6);
        let unrelated_parent = id_from_byte(0xff);

        let h0 = pop_header(mk_header(h0_id, id_from_byte(0), 1), vec![h0_id]);
        let h1 = pop_header(mk_header(h1_id, h0_id, 10), vec![h0_id]);
        let h2 = pop_header(mk_header(h2_id, h1_id, 20), vec![h0_id, h1_id]);
        let h3 = pop_header(mk_header(h3_id, h2_id, 30), vec![h0_id, h2_id]);
        // suffix_head's parent is unrelated, h0 is its only interlink, and
        // h1/h2/h3 ids are NOT among its interlinks. With lookback span 3,
        // the lookback window for index 4 covers indices [1, 3] only —
        // h0 (index 0) is excluded → no valid predecessor → REJECT.
        let suffix_head = pop_header(mk_header(suffix_head_id, unrelated_parent, 40), vec![h0_id]);
        let suffix_tail = vec![mk_header(suffix_tail_id, suffix_head_id, 41)];

        let mut proof =
            NipopowProof::new(6, 2, vec![h0, h1, h2, h3], suffix_head, suffix_tail).unwrap();

        // Squeeze the lookback window to size 3 (= use_last_epochs + 3)
        // so a small synthetic chain can demonstrate the boundary.
        proof.popow_algos.use_last_epochs = 0;

        assert!(
            !proof.has_valid_connections(),
            "verifier must reject when the only valid backward connection is \
             outside the lookback window — the fix is tolerant, not blanket"
        );
    }

    /// Sanity check: a proof whose suffix tail is broken (parent_id chain
    /// violated) must still be rejected. Ensures we didn't accidentally
    /// loosen the suffix-side check while loosening the prefix-side check.
    #[test]
    fn has_valid_connections_rejects_broken_suffix_tail() {
        let mk_header = header_factory();

        let h0_id = id_from_byte(1);
        let suffix_head_id = id_from_byte(2);
        let bad_parent = id_from_byte(0xee);
        let suffix_tail_id = id_from_byte(3);

        let h0 = pop_header(mk_header(h0_id, id_from_byte(0), 1), vec![h0_id]);
        let suffix_head = pop_header(mk_header(suffix_head_id, h0_id, 10), vec![h0_id]);
        // suffix_tail header's parent_id is unrelated to suffix_head.id
        // → suffix-side check must fail.
        let suffix_tail = vec![mk_header(suffix_tail_id, bad_parent, 11)];

        let proof = NipopowProof::new(6, 2, vec![h0], suffix_head, suffix_tail).unwrap();

        assert!(
            !proof.has_valid_connections(),
            "broken suffix tail (parent_id chain violation) must still be \
             rejected after the prefix-tolerance fix"
        );
    }

    #[test]
    fn interlink_proof_accepts_empty_genesis() {
        let make_header = header_factory();
        let header = make_header(id_from_byte(0x31), id_from_byte(0x30), 1);
        let popow_header = pop_header(header, Vec::new());

        assert!(popow_header.check_interlinks_proof());
    }

    #[test]
    fn interlink_proof_rejects_non_empty_genesis() {
        let popow_header = interlinks_only_popow_header(1, vec![id_from_byte(0x11)]);

        assert!(!popow_header.check_interlinks_proof());
    }

    #[test]
    fn interlink_proof_rejects_empty_non_genesis() {
        let make_header = header_factory();
        let header = make_header(id_from_byte(0x31), id_from_byte(0x30), 2);
        let popow_header = pop_header(header, Vec::new());

        assert!(!popow_header.check_interlinks_proof());
    }

    #[test]
    fn interlink_proof_accepts_a_255_element_run() {
        let popow_header =
            interlinks_only_popow_header(2, vec![id_from_byte(0x11); usize::from(u8::MAX)]);

        assert!(popow_header.check_interlinks_proof());
    }

    #[test]
    fn interlink_proof_rejects_a_256_element_run() {
        let make_header = header_factory();
        let header = make_header(id_from_byte(0x31), id_from_byte(0x30), 2);
        let popow_header = pop_header(header, vec![id_from_byte(0x11); usize::from(u8::MAX) + 1]);

        assert!(!popow_header.check_interlinks_proof());
    }

    #[test]
    fn interlink_proof_rejects_a_run_starting_at_position_256() {
        let mut interlinks = vec![id_from_byte(0x11); usize::from(u8::MAX)];
        interlinks.push(id_from_byte(0x22));
        interlinks.push(id_from_byte(0x33));
        let make_header = header_factory();
        let header = make_header(id_from_byte(0x31), id_from_byte(0x30), 2);
        let popow_header = pop_header(header, interlinks);

        assert!(!popow_header.check_interlinks_proof());
    }

    #[test]
    fn interlink_proof_rejects_a_reopened_id() {
        let first = id_from_byte(0x11);
        let popow_header = interlinks_only_popow_header(2, vec![first, id_from_byte(0x22), first]);

        assert!(!popow_header.check_interlinks_proof());
    }

    #[test]
    fn full_root_accepts_a_complete_mixed_extension_proof() {
        let interlinks = vec![id_from_byte(0x11), id_from_byte(0x22)];
        let mut fields = NipopowAlgos::pack_interlinks(interlinks.clone());
        fields.push(([0x02, 0x00], vec![0x01]));
        let extension = ExtensionCandidate::new(fields.clone()).unwrap();
        let proof = NipopowAlgos::proof_for_interlink_vector(&extension).unwrap();
        let full_root = extension_tree(&fields).root_hash();
        assert!(proof.valid(full_root.as_ref()));

        let make_header = header_factory();
        let mut header = make_header(id_from_byte(0x31), id_from_byte(0x30), 2);
        header.extension_root = full_root;
        let popow_header = PoPowHeader {
            header,
            interlinks,
            interlinks_proof: proof,
        };

        assert!(popow_header.check_interlinks_proof());
    }

    #[test]
    fn full_root_rejects_a_one_byte_header_root_mutation() {
        let interlinks = vec![id_from_byte(0x11), id_from_byte(0x22)];
        let mut fields = NipopowAlgos::pack_interlinks(interlinks.clone());
        fields.push(([0x02, 0x00], vec![0x01]));
        let extension = ExtensionCandidate::new(fields.clone()).unwrap();
        let proof = NipopowAlgos::proof_for_interlink_vector(&extension).unwrap();
        let full_root = extension_tree(&fields).root_hash();
        let mut wrong_root = full_root;
        wrong_root.0[0] ^= 1;

        let make_header = header_factory();
        let mut header = make_header(id_from_byte(0x31), id_from_byte(0x30), 2);
        header.extension_root = wrong_root;
        let popow_header = PoPowHeader {
            header,
            interlinks,
            interlinks_proof: proof,
        };

        assert!(!popow_header.check_interlinks_proof());
    }

    #[test]
    fn full_root_rejects_an_interlink_mutation_with_the_original_proof() {
        let interlinks = vec![id_from_byte(0x11), id_from_byte(0x22)];
        let mut fields = NipopowAlgos::pack_interlinks(interlinks);
        fields.push(([0x02, 0x00], vec![0x01]));
        let extension = ExtensionCandidate::new(fields.clone()).unwrap();
        let proof = NipopowAlgos::proof_for_interlink_vector(&extension).unwrap();
        let full_root = extension_tree(&fields).root_hash();

        let make_header = header_factory();
        let mut header = make_header(id_from_byte(0x31), id_from_byte(0x30), 2);
        header.extension_root = full_root;
        let popow_header = PoPowHeader {
            header,
            interlinks: vec![id_from_byte(0x11), id_from_byte(0x33)],
            interlinks_proof: proof,
        };

        assert!(!popow_header.check_interlinks_proof());
    }

    #[test]
    fn full_root_rejects_an_incomplete_interlink_disclosure() {
        let interlinks = vec![id_from_byte(0x11), id_from_byte(0x22)];
        let mut fields = NipopowAlgos::pack_interlinks(interlinks.clone());
        let first_interlink_key = fields[0].0;
        fields.push(([0x02, 0x00], vec![0x01]));
        let extension = ExtensionCandidate::new(fields.clone()).unwrap();
        let incomplete_proof =
            NipopowAlgos::extension_batch_proof_for(&extension, &[first_interlink_key]).unwrap();
        let full_root = extension_tree(&fields).root_hash();
        assert!(incomplete_proof.valid(full_root.as_ref()));

        let make_header = header_factory();
        let mut header = make_header(id_from_byte(0x31), id_from_byte(0x30), 2);
        header.extension_root = full_root;
        let popow_header = PoPowHeader {
            header,
            interlinks,
            interlinks_proof: incomplete_proof,
        };

        assert!(!popow_header.check_interlinks_proof());
    }

    #[test]
    fn full_root_rejects_an_extra_disclosed_extension_leaf() {
        let interlinks = vec![id_from_byte(0x11), id_from_byte(0x22)];
        let mut fields = NipopowAlgos::pack_interlinks(interlinks.clone());
        fields.push(([0x02, 0x00], vec![0x01]));
        let extension = ExtensionCandidate::new(fields.clone()).unwrap();
        let all_keys: Vec<_> = fields.iter().map(|(key, _)| *key).collect();
        let overcomplete_proof =
            NipopowAlgos::extension_batch_proof_for(&extension, &all_keys).unwrap();
        let full_root = extension_tree(&fields).root_hash();
        assert!(overcomplete_proof.valid(full_root.as_ref()));

        let make_header = header_factory();
        let mut header = make_header(id_from_byte(0x31), id_from_byte(0x30), 2);
        header.extension_root = full_root;
        let popow_header = PoPowHeader {
            header,
            interlinks,
            interlinks_proof: overcomplete_proof,
        };

        assert!(!popow_header.check_interlinks_proof());
    }

    #[test]
    fn full_root_rejects_an_interlinks_only_proof_under_a_mixed_root() {
        let interlinks = vec![id_from_byte(0x11), id_from_byte(0x22)];
        let interlink_fields = NipopowAlgos::pack_interlinks(interlinks.clone());
        let interlinks_only = ExtensionCandidate::new(interlink_fields.clone()).unwrap();
        let legacy_proof = NipopowAlgos::proof_for_interlink_vector(&interlinks_only).unwrap();
        let mut mixed_fields = interlink_fields.clone();
        mixed_fields.push(([0x02, 0x00], vec![0x01]));
        let legacy_root = extension_tree(&interlink_fields).root_hash();
        let mixed_root = extension_tree(&mixed_fields).root_hash();
        assert!(legacy_proof.valid(legacy_root.as_ref()));
        assert!(!legacy_proof.valid(mixed_root.as_ref()));

        let make_header = header_factory();
        let mut header = make_header(id_from_byte(0x31), id_from_byte(0x30), 2);
        header.extension_root = mixed_root;
        let popow_header = PoPowHeader {
            header,
            interlinks,
            interlinks_proof: legacy_proof,
        };

        assert!(!popow_header.check_interlinks_proof());
    }

    #[test]
    fn full_root_keeps_interlinks_only_extensions_compatible() {
        let popow_header =
            interlinks_only_popow_header(2, vec![id_from_byte(0x11), id_from_byte(0x22)]);

        assert!(popow_header.check_interlinks_proof());
    }

    /// Helper: VLQ-encode a u32 into bytes.
    fn vlq_encode_u32(v: u32) -> Vec<u8> {
        let mut buf = Vec::new();
        sigma_ser::vlq_encode::WriteSigmaVlqExt::put_u32(&mut buf, v).unwrap();
        buf
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum FrameSite {
        Prefix(usize),
        SuffixHead,
        SuffixTail(usize),
    }

    #[derive(Clone, Copy, Debug)]
    enum DeclaredSizeMutation {
        Delta(i64),
        Override(u32),
        Raw(&'static [u8]),
    }

    #[derive(Clone, Copy, Debug)]
    struct FrameMutation {
        site: FrameSite,
        declared_size: DeclaredSizeMutation,
        filler_len: usize,
    }

    fn sample_framing_proof() -> NipopowProof {
        let mut runner = TestRunner::default();
        let mut proof = any::<NipopowProof>()
            .new_tree(&mut runner)
            .unwrap()
            .current();
        if proof.prefix.len() == 1 {
            proof.prefix.push(proof.prefix[0].clone());
        }
        if proof.suffix_tail.is_empty() {
            proof.suffix_tail.push(proof.suffix_head.header.clone());
        }
        if proof.suffix_tail.len() == 1 {
            proof.suffix_tail.push(proof.suffix_tail[0].clone());
        }
        proof.k = u32::try_from(proof.suffix_tail.len() + 1).unwrap();
        proof
    }

    fn frame_sites(proof: &NipopowProof) -> Vec<FrameSite> {
        let mut sites = (0..proof.prefix.len())
            .map(FrameSite::Prefix)
            .collect::<Vec<_>>();
        sites.push(FrameSite::SuffixHead);
        sites.extend((0..proof.suffix_tail.len()).map(FrameSite::SuffixTail));
        sites
    }

    fn write_test_frame<T: ScorexSerializable>(
        w: &mut Vec<u8>,
        value: &T,
        site: FrameSite,
        mutation: FrameMutation,
    ) {
        let bytes = value.scorex_serialize_bytes().unwrap();
        if site == mutation.site {
            match mutation.declared_size {
                DeclaredSizeMutation::Delta(delta) => {
                    let declared = i64::try_from(bytes.len()).unwrap() + delta;
                    w.put_u32(u32::try_from(declared).unwrap()).unwrap();
                }
                DeclaredSizeMutation::Override(declared) => {
                    w.put_u32(declared).unwrap();
                }
                DeclaredSizeMutation::Raw(raw) => w.extend_from_slice(raw),
            }
        } else {
            w.put_u32(u32::try_from(bytes.len()).unwrap()).unwrap();
        }
        w.extend_from_slice(&bytes);
        if site == mutation.site {
            w.resize(w.len() + mutation.filler_len, 0x7f);
        }
    }

    fn serialize_with_frame_mutation(proof: &NipopowProof, mutation: FrameMutation) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.put_u32(proof.m).unwrap();
        bytes.put_u32(proof.k).unwrap();
        bytes
            .put_u32(u32::try_from(proof.prefix.len()).unwrap())
            .unwrap();
        for (index, prefix) in proof.prefix.iter().enumerate() {
            write_test_frame(&mut bytes, prefix, FrameSite::Prefix(index), mutation);
        }
        write_test_frame(
            &mut bytes,
            &proof.suffix_head,
            FrameSite::SuffixHead,
            mutation,
        );
        bytes
            .put_u32(u32::try_from(proof.suffix_tail.len()).unwrap())
            .unwrap();
        for (index, header) in proof.suffix_tail.iter().enumerate() {
            write_test_frame(&mut bytes, header, FrameSite::SuffixTail(index), mutation);
        }
        bytes
    }

    #[test]
    fn declared_element_frames_roundtrip() {
        let proof = sample_framing_proof();
        let canonical = proof.scorex_serialize_bytes().unwrap();
        for site in frame_sites(&proof) {
            let reemitted = serialize_with_frame_mutation(
                &proof,
                FrameMutation {
                    site,
                    declared_size: DeclaredSizeMutation::Delta(0),
                    filler_len: 0,
                },
            );
            assert_eq!(reemitted, canonical, "honest re-emission changed {site:?}");
            assert_eq!(
                NipopowProof::scorex_parse_bytes(&reemitted).unwrap(),
                proof,
                "honest frame did not round-trip at {site:?}"
            );
        }
    }

    #[test]
    fn declared_element_frames_reject_overstatement_without_filler() {
        let proof = sample_framing_proof();
        for site in frame_sites(&proof) {
            let mutated = serialize_with_frame_mutation(
                &proof,
                FrameMutation {
                    site,
                    declared_size: DeclaredSizeMutation::Delta(1),
                    filler_len: 0,
                },
            );
            assert!(
                NipopowProof::scorex_parse_bytes(&mutated).is_err(),
                "accepted over-declared {site:?} without filler"
            );
        }
    }

    #[test]
    fn declared_element_frames_reject_understatement() {
        let proof = sample_framing_proof();
        for site in frame_sites(&proof) {
            let mutated = serialize_with_frame_mutation(
                &proof,
                FrameMutation {
                    site,
                    declared_size: DeclaredSizeMutation::Delta(-1),
                    filler_len: 0,
                },
            );
            assert!(
                NipopowProof::scorex_parse_bytes(&mutated).is_err(),
                "accepted under-declared {site:?}"
            );
        }
    }

    #[test]
    fn declared_element_frames_accept_matching_filler() {
        let proof = sample_framing_proof();
        for site in frame_sites(&proof) {
            let mutated = serialize_with_frame_mutation(
                &proof,
                FrameMutation {
                    site,
                    declared_size: DeclaredSizeMutation::Delta(1),
                    filler_len: 1,
                },
            );
            let parsed = NipopowProof::scorex_parse_bytes(&mutated)
                .unwrap_or_else(|err| panic!("rejected padded {site:?}: {err}"));
            assert_eq!(parsed, proof, "padded {site:?} changed the proof");
        }
    }

    fn serialize_popow_header_with_nested_frames(
        value: &PoPowHeader,
        header_frame: &[u8],
        proof_frame: &[u8],
    ) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes
            .put_u32(u32::try_from(header_frame.len()).unwrap())
            .unwrap();
        bytes.extend_from_slice(header_frame);
        bytes
            .put_u32(u32::try_from(value.interlinks.len()).unwrap())
            .unwrap();
        for interlink in &value.interlinks {
            bytes.extend_from_slice(&interlink.0 .0);
        }
        bytes
            .put_u32(u32::try_from(proof_frame.len()).unwrap())
            .unwrap();
        bytes.extend_from_slice(proof_frame);
        bytes
    }

    #[test]
    fn nested_header_frame_accepts_trailing_padding() {
        let value = sample_framing_proof().prefix.remove(0);
        let mut header_frame = value.header.scorex_serialize_bytes().unwrap();
        header_frame.push(0x7f);
        let proof_frame = value.interlinks_proof.scorex_serialize_bytes().unwrap();
        let bytes = serialize_popow_header_with_nested_frames(&value, &header_frame, &proof_frame);

        assert_eq!(PoPowHeader::scorex_parse_bytes(&bytes).unwrap(), value);
    }

    #[test]
    fn nested_merkle_proof_frame_rejects_trailing_padding() {
        let value = sample_framing_proof().prefix.remove(0);
        let header_frame = value.header.scorex_serialize_bytes().unwrap();
        let mut proof_frame = value.interlinks_proof.scorex_serialize_bytes().unwrap();
        proof_frame.push(0x7f);
        let bytes = serialize_popow_header_with_nested_frames(&value, &header_frame, &proof_frame);

        let error = PoPowHeader::scorex_parse_bytes(&bytes).unwrap_err();
        let ScorexParsingError::ValueOutOfBounds(message) = error else {
            panic!("padding reached the nested parser before preflight: {error:?}");
        };
        assert!(
            message.contains("does not fit declared proof frame"),
            "unexpected padding preflight message: {message}"
        );
    }

    fn assert_merkle_count_preflight(indices_len: u32, proofs_len: u32, label: &str) {
        let mut value = sample_framing_proof().prefix.remove(0);
        value.interlinks_proof = BatchMerkleProof::new(vec![], vec![]);
        let header_frame = value.header.scorex_serialize_bytes().unwrap();
        let mut proof_frame = value.interlinks_proof.scorex_serialize_bytes().unwrap();
        assert_eq!(proof_frame.len(), 8);
        proof_frame[..4].copy_from_slice(&indices_len.to_be_bytes());
        proof_frame[4..8].copy_from_slice(&proofs_len.to_be_bytes());
        let bytes = serialize_popow_header_with_nested_frames(&value, &header_frame, &proof_frame);

        let error = PoPowHeader::scorex_parse_bytes(&bytes).unwrap_err();
        let ScorexParsingError::ValueOutOfBounds(message) = error else {
            panic!("{label} count reached the nested parser before preflight: {error:?}");
        };
        assert!(
            message.contains("does not fit declared proof frame"),
            "unexpected {label} preflight message: {message}"
        );
    }

    #[test]
    fn merkle_index_count_outside_declared_frame_is_rejected_before_parse() {
        assert_merkle_count_preflight(1, 0, "index");
    }

    #[test]
    fn merkle_proof_count_outside_declared_frame_is_rejected_before_parse() {
        assert_merkle_count_preflight(0, 1, "proof");
    }

    #[test]
    fn impossible_singleton_merkle_depth_is_rejected_before_parse() {
        let value = sample_framing_proof().prefix.remove(0);
        let header_frame = value.header.scorex_serialize_bytes().unwrap();
        let proof_count = u32::BITS + 1;
        let mut proof_frame = Vec::new();
        proof_frame.extend_from_slice(&1_u32.to_be_bytes());
        proof_frame.extend_from_slice(&proof_count.to_be_bytes());
        proof_frame.extend_from_slice(&0_u32.to_be_bytes());
        proof_frame.extend_from_slice(&[0x11; 32]);
        for _ in 0..proof_count {
            proof_frame.extend_from_slice(&[0x22; 32]);
            proof_frame.push(NodeSide::Left as u8);
        }
        let bytes = serialize_popow_header_with_nested_frames(&value, &header_frame, &proof_frame);

        assert!(PoPowHeader::scorex_parse_bytes(&bytes).is_err());
    }

    #[test]
    fn declared_element_frames_reject_sizes_above_caps() {
        let proof = sample_framing_proof();
        let popow_header_cap =
            MAX_POPOW_HEADER_BYTES + MAX_POPOW_INTERLINKS * 32 + MAX_POPOW_PROOF_BYTES + 64;
        let cases = [
            (FrameSite::Prefix(0), popow_header_cap + 1),
            (FrameSite::SuffixHead, popow_header_cap + 1),
            (FrameSite::SuffixTail(0), MAX_POPOW_HEADER_BYTES + 1),
        ];
        for (site, declared) in cases {
            let mutated = serialize_with_frame_mutation(
                &proof,
                FrameMutation {
                    site,
                    declared_size: DeclaredSizeMutation::Override(u32::try_from(declared).unwrap()),
                    filler_len: 0,
                },
            );
            assert!(
                NipopowProof::scorex_parse_bytes(&mutated).is_err(),
                "accepted {site:?} above its declared-size cap"
            );
        }
    }

    #[test]
    fn malformed_suffix_tail_frame_size_is_rejected() {
        const MALFORMED_VLQ: &[u8] = &[0x80; 10];

        let proof = sample_framing_proof();
        let mutated = serialize_with_frame_mutation(
            &proof,
            FrameMutation {
                site: FrameSite::SuffixTail(0),
                declared_size: DeclaredSizeMutation::Raw(MALFORMED_VLQ),
                filler_len: 0,
            },
        );
        assert!(
            NipopowProof::scorex_parse_bytes(&mutated).is_err(),
            "accepted malformed suffix-tail size VLQ"
        );
    }

    fn jvm_nipopow_fixture() -> (serde_json::Value, Vec<u8>) {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../tests/fixtures/nipopow-full-root-mixed-nipopow-proof.json"
        ))
        .unwrap();
        let bytes = base16::decode(fixture["bytes_hex"].as_str().unwrap()).unwrap();
        (fixture, bytes)
    }

    fn fixture_usize(fixture: &serde_json::Value, field: &str) -> usize {
        usize::try_from(fixture[field].as_u64().unwrap()).unwrap()
    }

    fn fixture_u8(fixture: &serde_json::Value, field: &str) -> u8 {
        u8::try_from(fixture[field].as_u64().unwrap()).unwrap()
    }

    /// Parse the Rust-owned core and validate the one-byte JVM envelope.
    ///
    /// `continuous` remains an adapter concern: it is checked here, but is
    /// intentionally not represented by the Rust `NipopowProof` core type.
    fn parse_jvm_nipopow_fixture(
        bytes: &[u8],
        core_length: usize,
        terminal_mode: u8,
    ) -> Result<NipopowProof, String> {
        if terminal_mode > 1 {
            return Err(format!("invalid JVM terminal mode {terminal_mode}"));
        }
        if bytes.len()
            != core_length
                .checked_add(1)
                .ok_or("fixture length overflow")?
        {
            return Err(format!(
                "expected one terminal byte after {core_length} core bytes, got {} total bytes",
                bytes.len()
            ));
        }

        let mut cursor = std::io::Cursor::new(bytes);
        let proof = NipopowProof::scorex_parse(&mut cursor).map_err(|err| err.to_string())?;
        let parsed_core_length =
            usize::try_from(cursor.position()).map_err(|err| err.to_string())?;
        if parsed_core_length != core_length {
            return Err(format!(
                "Rust parser stopped at {parsed_core_length}, expected {core_length}"
            ));
        }
        let terminal = bytes
            .get(parsed_core_length)
            .copied()
            .ok_or("missing JVM terminal byte")?;
        if terminal != terminal_mode {
            return Err(format!(
                "JVM terminal mode {terminal} does not match expected {terminal_mode}"
            ));
        }

        proof.validate().map_err(|err| err.to_string())?;
        Ok(proof)
    }

    fn read_fixture_vlq(bytes: &[u8], offset: &mut usize) -> Result<u32, String> {
        let mut value = 0_u32;
        for shift in (0..35).step_by(7) {
            let byte = bytes.get(*offset).copied().ok_or("truncated fixture VLQ")?;
            *offset = offset.checked_add(1).ok_or("fixture offset overflow")?;
            value |= u32::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                return Ok(value);
            }
        }
        Err("fixture VLQ exceeds u32".to_owned())
    }

    fn fixture_suffix_head_range(bytes: &[u8]) -> std::ops::Range<usize> {
        let mut offset = 0;
        read_fixture_vlq(bytes, &mut offset).unwrap();
        read_fixture_vlq(bytes, &mut offset).unwrap();
        let prefix_count = read_fixture_vlq(bytes, &mut offset).unwrap();
        for _ in 0..prefix_count {
            let frame_length =
                usize::try_from(read_fixture_vlq(bytes, &mut offset).unwrap()).unwrap();
            offset = offset.checked_add(frame_length).unwrap();
        }
        let suffix_head_length =
            usize::try_from(read_fixture_vlq(bytes, &mut offset).unwrap()).unwrap();
        let end = offset.checked_add(suffix_head_length).unwrap();
        assert!(end <= bytes.len());
        offset..end
    }

    fn single_subslice_offset(bytes: &[u8], range: std::ops::Range<usize>, needle: &[u8]) -> usize {
        assert!(!needle.is_empty());
        let matches = bytes[range.clone()]
            .windows(needle.len())
            .enumerate()
            .filter_map(|(index, window)| (window == needle).then_some(range.start + index))
            .collect::<Vec<_>>();
        assert_eq!(matches.len(), 1, "fixture mutation target must be unique");
        matches[0]
    }

    #[test]
    fn jvm_nipopow_fixture_roundtrips_at_the_explicit_core_boundary() {
        let (fixture, bytes) = jvm_nipopow_fixture();
        let core_length = fixture_usize(&fixture, "rust_core_length");
        let terminal_mode = fixture_u8(&fixture, "terminal_continuous_byte");

        assert_eq!(
            fixture["format"].as_str().unwrap(),
            "scorex-nipopow-proof-with-jvm-mode-v1"
        );
        assert_eq!(bytes.len(), core_length + 1);
        assert_eq!(
            base16::encode_lower(sigma_util::hash::sha256_hash(&bytes).as_slice()),
            fixture["sha256"].as_str().unwrap()
        );

        let mut cursor = std::io::Cursor::new(bytes.as_slice());
        let parsed = NipopowProof::scorex_parse(&mut cursor).unwrap();
        assert_eq!(cursor.position(), core_length as u64);
        assert_eq!(&bytes[core_length..], &[terminal_mode]);
        assert_eq!(parsed.m, fixture["m"].as_u64().unwrap() as u32);
        assert_eq!(parsed.k, fixture["k"].as_u64().unwrap() as u32);
        assert_eq!(parsed.prefix.len(), fixture_usize(&fixture, "prefix_count"));
        assert_eq!(
            parsed.suffix_tail.len() + 1,
            fixture_usize(&fixture, "suffix_count")
        );
        assert_eq!(
            parsed.suffix_tail.len(),
            fixture_usize(&fixture, "suffix_tail_count")
        );
        assert_eq!(
            parsed.suffix_head.header.extension_root.to_string(),
            fixture["extension_root"].as_str().unwrap()
        );
        assert_eq!(
            parsed.suffix_head.interlinks,
            vec![id_from_byte(0x11), id_from_byte(0x22)]
        );
        assert!(parsed.prefix[0].check_interlinks_proof());
        assert!(parsed.suffix_head.check_interlinks_proof());
        assert_eq!(parsed.validate(), Ok(()));
        assert_eq!(
            parsed.scorex_serialize_bytes().unwrap(),
            bytes[..core_length]
        );
        assert!(parse_jvm_nipopow_fixture(&bytes, core_length, terminal_mode).is_ok());
    }

    #[test]
    fn jvm_nipopow_fixture_rejects_extension_root_mutation() {
        let (fixture, mut bytes) = jvm_nipopow_fixture();
        let core_length = fixture_usize(&fixture, "rust_core_length");
        let terminal_mode = fixture_u8(&fixture, "terminal_continuous_byte");
        let extension_root = base16::decode(fixture["extension_root"].as_str().unwrap()).unwrap();
        let root_offset =
            single_subslice_offset(&bytes, fixture_suffix_head_range(&bytes), &extension_root);
        bytes[root_offset] ^= 1;

        assert!(parse_jvm_nipopow_fixture(&bytes, core_length, terminal_mode).is_err());
    }

    #[test]
    fn jvm_nipopow_fixture_rejects_disclosed_interlink_mutation() {
        let (fixture, mut bytes) = jvm_nipopow_fixture();
        let core_length = fixture_usize(&fixture, "rust_core_length");
        let terminal_mode = fixture_u8(&fixture, "terminal_continuous_byte");
        let interlink = [0x22; 32];
        let interlink_offset =
            single_subslice_offset(&bytes, fixture_suffix_head_range(&bytes), &interlink);
        bytes[interlink_offset] ^= 1;

        assert!(parse_jvm_nipopow_fixture(&bytes, core_length, terminal_mode).is_err());
    }

    #[test]
    fn jvm_nipopow_fixture_rejects_m_mutation() {
        let (fixture, mut bytes) = jvm_nipopow_fixture();
        let core_length = fixture_usize(&fixture, "rust_core_length");
        let terminal_mode = fixture_u8(&fixture, "terminal_continuous_byte");
        assert_eq!(bytes[0], 1);
        bytes[0] = 0;

        assert!(parse_jvm_nipopow_fixture(&bytes, core_length, terminal_mode).is_err());
    }

    #[test]
    fn jvm_nipopow_fixture_rejects_k_mutation() {
        let (fixture, mut bytes) = jvm_nipopow_fixture();
        let core_length = fixture_usize(&fixture, "rust_core_length");
        let terminal_mode = fixture_u8(&fixture, "terminal_continuous_byte");
        assert_eq!(&bytes[..2], &[1, 2]);
        bytes[1] = 1;

        assert!(parse_jvm_nipopow_fixture(&bytes, core_length, terminal_mode).is_err());
    }

    #[test]
    fn jvm_nipopow_fixture_rejects_nested_header_frame_mutation() {
        let (fixture, mut bytes) = jvm_nipopow_fixture();
        let core_length = fixture_usize(&fixture, "rust_core_length");
        let terminal_mode = fixture_u8(&fixture, "terminal_continuous_byte");
        let suffix_head_range = fixture_suffix_head_range(&bytes);
        let nested_size_offset = suffix_head_range.start;
        let mut after_size = nested_size_offset;
        assert_eq!(read_fixture_vlq(&bytes, &mut after_size).unwrap(), 218);
        assert_eq!(bytes[nested_size_offset], 0xda);
        bytes[nested_size_offset] = 0xd9;

        assert!(parse_jvm_nipopow_fixture(&bytes, core_length, terminal_mode).is_err());
    }

    #[test]
    fn jvm_nipopow_fixture_rejects_missing_terminal_byte() {
        let (fixture, bytes) = jvm_nipopow_fixture();
        let core_length = fixture_usize(&fixture, "rust_core_length");
        let terminal_mode = fixture_u8(&fixture, "terminal_continuous_byte");

        assert!(
            parse_jvm_nipopow_fixture(&bytes[..bytes.len() - 1], core_length, terminal_mode)
                .is_err()
        );
    }

    #[test]
    fn jvm_nipopow_fixture_rejects_extra_terminal_byte() {
        let (fixture, mut bytes) = jvm_nipopow_fixture();
        let core_length = fixture_usize(&fixture, "rust_core_length");
        let terminal_mode = fixture_u8(&fixture, "terminal_continuous_byte");
        bytes.push(terminal_mode);

        assert!(parse_jvm_nipopow_fixture(&bytes, core_length, terminal_mode).is_err());
    }

    #[test]
    fn jvm_nipopow_fixture_rejects_terminal_mode_mutation() {
        let (fixture, mut bytes) = jvm_nipopow_fixture();
        let core_length = fixture_usize(&fixture, "rust_core_length");
        let terminal_mode = fixture_u8(&fixture, "terminal_continuous_byte");
        let last = bytes.len() - 1;
        bytes[last] = 1;
        assert!(parse_jvm_nipopow_fixture(&bytes, core_length, terminal_mode).is_err());

        bytes[last] = 2;
        assert!(parse_jvm_nipopow_fixture(&bytes, core_length, 2).is_err());
    }

    #[test]
    fn crafted_huge_prefix_count_returns_err() {
        let mut payload = Vec::new();
        payload.extend(vlq_encode_u32(1)); // m
        payload.extend(vlq_encode_u32(1)); // k
        payload.extend(vlq_encode_u32(0x7FFF_FFFF)); // num_prefixes
        payload.extend_from_slice(&[0u8; 16]); // padding

        let result = NipopowProof::scorex_parse(&mut std::io::Cursor::new(payload));
        assert!(
            result.is_err(),
            "Expected Err for huge num_prefixes, got Ok"
        );
    }

    #[test]
    fn crafted_huge_header_size_returns_err() {
        let mut payload = Vec::new();
        payload.extend(vlq_encode_u32(0x7FFF_FFFF)); // header_size
        payload.extend_from_slice(&[0u8; 16]); // padding

        let result = PoPowHeader::scorex_parse(&mut std::io::Cursor::new(payload));
        assert!(result.is_err(), "Expected Err for huge header_size, got Ok");
    }

    #[test]
    fn crafted_header_size_just_over_limit_returns_err() {
        let mut payload = Vec::new();
        let over_limit = (MAX_POPOW_HEADER_BYTES as u32) + 1;
        payload.extend(vlq_encode_u32(over_limit)); // header_size
        payload.extend_from_slice(&[0u8; 16]); // padding

        let result = PoPowHeader::scorex_parse(&mut std::io::Cursor::new(payload));
        assert!(
            result.is_err(),
            "Expected Err for header_size > limit, got Ok"
        );
    }
}
