//! Wire-tier round-trip: parse a `kind`'s canonical bytes with sigma-rust's serializer and
//! reserialize for byte-identity comparison downstream. The wire analog of [`crate::eval::run_entry`]
//! — a single round-trip outcome, no value/cost (docs/specs/wire-tier.md). A parse/serialize failure
//! is `errored` (sigma-rust rejected bytes the reference accepted — a real divergence); a `kind` with no
//! sigma serializer wired here is `not-implemented`.

use ergo_lib::chain::transaction::Transaction;
use ergotree_ir::chain::ergo_box::ErgoBox;
use ergotree_ir::ergo_tree::ErgoTree;
use ergotree_ir::ergo_tree::ErgoTreeVersion;
use ergotree_ir::mir::constant::Constant;
use ergotree_ir::serialization::sigma_byte_reader::{self, SigmaByteRead};
use ergotree_ir::serialization::sigma_byte_writer::{SigmaByteWrite, SigmaByteWriter};
use ergotree_ir::serialization::SigmaSerializable;
use ergotree_ir::sigma_protocol::sigma_boolean::SigmaBoolean;

/// One wire entry's outcome — the round-trip analog of [`crate::eval::Outcome`]: `bytes_hex`
/// replaces value+cost (the wire tier has no cost dimension).
pub enum WireOutcome {
    RoundTrip { bytes_hex: String },
    Errored,
    NotImplemented,
    Panicked { note: String },
}

impl WireOutcome {
    pub fn to_json(&self) -> serde_json::Value {
        match self {
            WireOutcome::RoundTrip { bytes_hex } => {
                serde_json::json!({"bytes_hex": bytes_hex, "error": serde_json::Value::Null})
            }
            WireOutcome::Errored => serde_json::json!({"bytes_hex": null, "error": "errored"}),
            WireOutcome::NotImplemented => {
                serde_json::json!({"bytes_hex": null, "error": "not-implemented"})
            }
            WireOutcome::Panicked { note } => {
                serde_json::json!({"bytes_hex": null, "error": "panicked", "note": note})
            }
        }
    }
}

/// Round-trip `bytes` through `T`'s sigma serializer: parse then reserialize. A parse or serialize
/// failure is `errored` — sigma-rust couldn't reproduce bytes the reference accepted (a real divergence).
fn roundtrip<T: SigmaSerializable>(bytes: &[u8]) -> WireOutcome {
    match T::sigma_parse_bytes(bytes) {
        Ok(obj) => match obj.sigma_serialize_bytes() {
            Ok(out) => WireOutcome::RoundTrip {
                bytes_hex: bytes_to_hex(&out),
            },
            Err(_) => WireOutcome::Errored,
        },
        Err(_) => WireOutcome::Errored,
    }
}

/// Box round-trip under the entry's activated script version: SANTA runs a wire entry under
/// its version context, and from activated 2 the JVM refuses a tree whose version is above the
/// activated one (`ErgoTreeSerializer.scala:150-154`, `VersionContext.scala:17-21`).
/// sigma-rust's reader does the same once it is given the activated version. The tree
/// version stays the one `sigma_parse_bytes` reads at.
fn roundtrip_box(bytes: &[u8], activated: Option<ErgoTreeVersion>) -> WireOutcome {
    let Some(activated) = activated else {
        return roundtrip::<ErgoBox>(bytes);
    };
    match sigma_byte_reader::from_bytes(bytes).with_versions(
        activated,
        ErgoTreeVersion::MAX_SCRIPT_VERSION,
        ErgoBox::sigma_parse,
    ) {
        Ok(b) => match b.sigma_serialize_bytes() {
            Ok(out) => WireOutcome::RoundTrip {
                bytes_hex: bytes_to_hex(&out),
            },
            Err(_) => WireOutcome::Errored,
        },
        Err(_) => WireOutcome::Errored,
    }
}

/// Transaction round-trip at the entry's ErgoTree version: SANTA runs a wire entry under its
/// version context, and sigma-rust reads and writes a transaction at its reader's and writer's
/// version (a v6 block's at 3, an earlier one's below). The reader also takes the entry's
/// activated script version, when it has one.
fn roundtrip_transaction(
    bytes: &[u8],
    version: ErgoTreeVersion,
    activated: Option<ErgoTreeVersion>,
) -> WireOutcome {
    let mut r = sigma_byte_reader::from_bytes(bytes);
    let parsed = match activated {
        Some(activated) => r.with_versions(activated, version, Transaction::sigma_parse),
        None => r.with_tree_version(version, Transaction::sigma_parse),
    };
    match parsed {
        Ok(tx) => {
            let mut out = Vec::new();
            let mut w = SigmaByteWriter::new(&mut out, None);
            match w.with_tree_version(version, |w| tx.sigma_serialize(w)) {
                Ok(()) => WireOutcome::RoundTrip {
                    bytes_hex: bytes_to_hex(&out),
                },
                Err(_) => WireOutcome::Errored,
            }
        }
        Err(_) => WireOutcome::Errored,
    }
}

/// ErgoTree round-trip through `sigma_parse_bytes_lenient`, the arbitrary-root parse (the JVM's `checkType = false`
/// reader). The wire vectors carry trees whose root is not a SigmaProp (an Int, a type-variable witness), which the
/// strict `sigma_parse_bytes` keeps as unparsed bytes (an echo of the input) instead of reading; the lenient parse
/// reads them to structure, so reserializing re-encodes them: the structural round-trip the ErgoTree kind asks for
/// (runner-contract-wire §5).
fn roundtrip_ergotree(bytes: &[u8]) -> WireOutcome {
    match ErgoTree::sigma_parse_bytes_lenient(bytes) {
        Ok(tree) => match tree.sigma_serialize_bytes() {
            Ok(out) => WireOutcome::RoundTrip {
                bytes_hex: bytes_to_hex(&out),
            },
            Err(_) => WireOutcome::Errored,
        },
        Err(_) => WireOutcome::Errored,
    }
}

/// Round-trip one wire entry. `kind` selects the serializer. Header isn't wired here — it parses
/// via ScorexSerializable, not SigmaSerializable → not-implemented.
pub fn run_entry(
    kind: &str,
    bytes_hex: &str,
    tree_version: u8,
    activated_version: Option<u8>,
) -> WireOutcome {
    let bytes = match crate::hex_to_bytes(bytes_hex) {
        Ok(b) => b,
        Err(e) => {
            return WireOutcome::Panicked {
                note: format!("bad bytes_hex: {e}"),
            }
        }
    };
    let activated = activated_version.map(ErgoTreeVersion::from);
    match kind {
        "Box" => roundtrip_box(&bytes, activated),
        "Constant" => roundtrip::<Constant>(&bytes),
        "SigmaBoolean" => roundtrip::<SigmaBoolean>(&bytes),
        "Transaction" => {
            roundtrip_transaction(&bytes, ErgoTreeVersion::from(tree_version), activated)
        }
        "BlockTransactions" => roundtrip_block_transactions(&bytes),
        "ErgoTree" => roundtrip_ergotree(&bytes),
        _ => WireOutcome::NotImplemented,
    }
}

/// A block's transactions section, parsed the way ergo-node-rust parses one (`validation/src/sections.rs`
/// `parse_block_transactions`). The section is the 32-byte header id, then a VLQ that is either the tx count or
/// `10_000_000 + block version` (followed by the count), then the transactions. Every transaction is read by
/// `Transaction::sigma_parse` from ONE reader over the rest, so sigma-rust decides what reader state a transaction
/// starts with; the JVM starts each on a fresh reader (`ErgoTransactionSerializer.parse`). Reserialized with the same
/// framing.
fn roundtrip_block_transactions(bytes: &[u8]) -> WireOutcome {
    match block_transactions_reserialized(bytes) {
        Some(out) => WireOutcome::RoundTrip {
            bytes_hex: bytes_to_hex(&out),
        },
        None => WireOutcome::Errored,
    }
}

fn block_transactions_reserialized(bytes: &[u8]) -> Option<Vec<u8>> {
    const BLOCK_VERSION_SENTINEL: u32 = 10_000_000;
    let header_id = bytes.get(..32)?;
    let mut pos = 32;
    let ver_or_count = read_vlq_u32(bytes, &mut pos)?;
    let (version_marker, count) = if ver_or_count > BLOCK_VERSION_SENTINEL {
        (Some(ver_or_count), read_vlq_u32(bytes, &mut pos)?)
    } else {
        (None, ver_or_count)
    };
    // ergo reads a v6 block's transactions at version context (blockVersion - 1,
    // blockVersion - 1), an earlier block's at the default (1, 1), and writes them at
    // (blockVersion, blockVersion) from v3 blocks (ergo v6.0.6 `BlockTransactions.scala:150-160`,
    // `:184-202`); the reader's and writer's versions are sigma-rust's context
    let block_version = version_marker.map_or(1, |m| m - BLOCK_VERSION_SENTINEL);
    let read_version = if block_version >= 4 {
        ErgoTreeVersion::V3
    } else {
        ErgoTreeVersion::V0
    };
    let write_version = if block_version >= 3 {
        ErgoTreeVersion::V3
    } else {
        ErgoTreeVersion::V0
    };
    let mut r = sigma_byte_reader::from_bytes(&bytes[pos..]);
    let mut out = header_id.to_vec();
    if let Some(marker) = version_marker {
        write_vlq_u32(marker, &mut out);
    }
    write_vlq_u32(count, &mut out);
    for _ in 0..count {
        // from block version 4 the section is read under activated (blockVersion - 1); an
        // earlier block's has no context, and a tree of any version parses
        let tx = if block_version >= 4 {
            let activated = ErgoTreeVersion::from((block_version - 1) as u8);
            r.with_versions(activated, read_version, Transaction::sigma_parse)
        } else {
            r.with_tree_version(read_version, Transaction::sigma_parse)
        }
        .ok()?;
        let mut w = SigmaByteWriter::new(&mut out, None);
        w.with_tree_version(write_version, |w| tx.sigma_serialize(w))
            .ok()?;
    }
    Some(out)
}

/// Unsigned LEB128: the JVM's `getUInt` / `putUInt` section framing.
fn read_vlq_u32(bytes: &[u8], pos: &mut usize) -> Option<u32> {
    let mut value: u64 = 0;
    for shift in (0..35).step_by(7) {
        let b = *bytes.get(*pos)?;
        *pos += 1;
        value |= u64::from(b & 0x7f) << shift;
        if b & 0x80 == 0 {
            return u32::try_from(value).ok();
        }
    }
    None
}

fn write_vlq_u32(mut v: u32, out: &mut Vec<u8>) {
    loop {
        let b = (v & 0x7f) as u8;
        v >>= 7;
        if v == 0 {
            out.push(b);
            return;
        }
        out.push(b | 0x80);
    }
}

fn bytes_to_hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::run_entry;
    use serde_json::Value as J;

    /// SANTA `wire/v6/authored/BlockTransactions.reader_scope` #3: a block section (header id, the version-4
    /// marker `84 ad e2 04`, count 2) holding two txs whose output trees each define ValDef(1) and use it. Shallow on
    /// purpose: a debug build takes tens of KB of stack per expression level, so #1's 109-deep tree would overflow
    /// the 2 MB test thread (the release runner parses it fine).
    const TWO_TX_SECTION: &str = "bfbc8c5cb61b59dd3cd2918450c7272600312b3b8c929808b09e76253d1bfecf84ade2040201456971d1f52b5bafce6e5c94848a0892d936cfe52a65d6a75d48d1fcc586f9f60000000001c0843d00d801d60108d37201010000015dea0e5b89dbbc08eb741989221dda7428575bb695d2a75bb15e434e379af43b0000000001c0843d00d801d60108d37201010000";

    #[test]
    fn block_transactions_round_trips_the_section_framing() {
        let j = run_entry("BlockTransactions", TWO_TX_SECTION, 3, None).to_json();
        assert_eq!(j["error"], J::Null);
        assert_eq!(j["bytes_hex"], TWO_TX_SECTION);
    }

    #[test]
    fn block_transactions_rejects_a_truncated_section() {
        // the header id alone: no version marker, no count
        let j = run_entry("BlockTransactions", &TWO_TX_SECTION[..64], 3, None).to_json();
        assert_eq!(j["error"], "errored");
    }

    #[test]
    fn box_round_trips_to_its_own_bytes() {
        // sbox_minimal from vectors/wire/v5/authored/Box.json
        let hex = "c0843d09020101000000000000000000000000000000000000000000000000000000000000000000000000";
        let j = run_entry("Box", hex, 3, None).to_json();
        assert_eq!(j["error"], J::Null);
        assert_eq!(j["bytes_hex"], hex);
    }

    #[test]
    fn sigma_boolean_round_trips_to_its_own_bytes() {
        let hex = "d3"; // TrivialProp(true) from vectors/wire/v5/authored/SigmaBoolean.json
        let j = run_entry("SigmaBoolean", hex, 3, None).to_json();
        assert_eq!(j["error"], J::Null);
        assert_eq!(j["bytes_hex"], hex);
    }

    #[test]
    fn constant_round_trips_to_its_own_bytes() {
        // bool_0 from vectors/wire/v5/vendored/Constant.json (Fleet)
        let hex = "0101";
        let j = run_entry("Constant", hex, 3, None).to_json();
        assert_eq!(j["error"], J::Null);
        assert_eq!(j["bytes_hex"], hex);
    }

    #[test]
    fn transaction_round_trips_to_its_own_bytes() {
        // tx_466f1aef from vectors/wire/v5/vendored/Transaction.json (Fleet signed tx)
        let hex = "010c7a0f145994fa15b02eca8189b454eeff9eec5ca33fa135b4474d4ee8ed35c70000000220fa2bf23962cdf51b07722d6237c0c7b8a44f78856c0f7ec308dc1ef1a92a51d9a2cc8a09abfaed87afacfbb7daee79a6b26f10c6613fc13d3f3953e5521d1a0280cc9497e9d1870f101004020e36100204a00b08cd0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798ea02d192a39a8cc7a7017300730110010204020404040004c0fd4f05808c82f5f6030580b8c9e5ae040580f882ad16040204c0944004c0f407040004000580f882ad16d19683030191a38cc7a7019683020193c2b2a57300007473017302830108cdeeac93a38cc7b2a573030001978302019683040193b1a5730493c2a7c2b2a573050093958fa3730673079973089c73097e9a730a9d99a3730b730c0599c1a7c1b2a5730d00938cc7b2a5730e0001a390c1a7730fb3825c0200010180b0abe9c1c7fa1300809ccdca64100204a00b08cd0274e729bb6615cbda94d9d176a2f1525068f12b330e38bbbf387232797dfd891fea02d192a39a8cc7a70173007301b3825c010180f085da2c00";
        let j = run_entry("Transaction", hex, 3, None).to_json();
        assert_eq!(j["error"], J::Null);
        assert_eq!(j["bytes_hex"], hex);
    }

    /// An ErgoTree whose root is not a SigmaProp (an Int constant) round-trips through the lenient parse: the
    /// arbitrary-root parse the JVM's `checkType = false` reader is, which the wire ErgoTree kind needs (the strict
    /// parse refuses or keeps such a tree as bytes). Header v0, no size: `00`, then the root `04 02` (Int 1).
    #[test]
    fn ergotree_with_a_non_sigmaprop_root_round_trips() {
        let j = run_entry("ErgoTree", "000402", 3, None).to_json();
        assert_eq!(j["error"], J::Null);
        assert_eq!(j["bytes_hex"], "000402");
    }

    #[test]
    fn ergotree_that_does_not_parse_is_errored() {
        // header v0, then an opcode with no node behind it
        let j = run_entry("ErgoTree", "00ff", 3, None).to_json();
        assert_eq!(j["error"], "errored");
    }

    #[test]
    fn unwired_kind_is_not_implemented() {
        // Header still parses via ScorexSerializable, not SigmaSerializable -> not wired here.
        let j = run_entry("Header", "00", 3, None).to_json();
        assert_eq!(j["error"], "not-implemented");
        assert_eq!(j["bytes_hex"], J::Null);
    }

    /// A Box's registers are parsed at the reader's TREE version (set from the wire runner's
    /// `MAX_SCRIPT_VERSION = V3`), not from the explicit `activated` version. A bare box whose
    /// R4 is `Coll[SFunc(SInt => SInt)]()` — a function type, parseable only at tree_version
    /// `>= V3` — round-trips at explicit `activated = 0`: reader `with_versions(V0, V3, parse)`.
    /// A mutant that threaded the activated through as the tree version would read R4 at V0
    /// and refuse the SFunc type code.
    #[test]
    fn a_box_reader_uses_the_max_tree_version_not_activated() {
        // Box layout: value(VLQ u64), ergo_tree, creation_height(VLQ), tokens(count u8),
        // registers(count u8 + values), tx_id(32), index(u16 VLQ). Here:
        //   value=1_000_000 (c0843d), ergo_tree=0008d3 (v0 header, body SigmaProp true),
        //   creation_height=0, tokens=0, regs=1 with R4=Coll[SFunc(Int=>Int)]() `83007001040400`,
        //   tx_id=zero, index=0.
        let hex = "c0843d0008d300000183007001040400000000000000000000000000000000000000000000000000000000000000000000";
        let j = run_entry("Box", hex, 3, Some(0)).to_json();
        assert_eq!(
            j["error"],
            J::Null,
            "box with SFunc R4 at activated=0 must round-trip (reader tree_version = V3): {j}"
        );
        assert_eq!(j["bytes_hex"], hex);
    }

    /// Without an explicit activated version, a Transaction wire parse uses the ENTRY's
    /// tree_version alone — `with_tree_version(version, parse)`. A tx carrying `Coll[SFunc]`
    /// in an input's context extension parses at tree_version=V3 but not below. A mutant that
    /// pins the tree_version to V0 regardless would refuse the SFunc type code.
    #[test]
    fn a_tx_without_activated_reads_at_the_entry_tree_version() {
        let hex = "0100a19de1b5fa998df5a48630a611180690abad5270c33f23a79baba2f8840d710001008300700204040400000001c0843d0008d3010000";
        let j = run_entry("Transaction", hex, 3, None).to_json();
        assert_eq!(
            j["error"],
            J::Null,
            "tx with SFunc ctx extension at tree_version=V3 must round-trip: {j}"
        );
        assert_eq!(j["bytes_hex"], hex);
    }

    /// With an explicit activated version, a Transaction wire parse uses the ENTRY's
    /// tree_version — `with_versions(activated, version, parse)`. A `Tuple(MethodCall(... no
    /// args))` in a context extension parses only at tree_version < V3 (a 0-arg MethodCall is
    /// refused from tree version 3). At tree_version=V0 the tx
    /// round-trips, with the 0-arg MethodCall written back as a PropertyCall (opcode 0xdb
    /// replaces 0xdc, dropping its args-count byte). A mutant that forced tree_version=V3
    /// regardless would refuse the MethodCall at parse.
    #[test]
    fn a_tx_with_activated_reads_at_the_entry_tree_version() {
        // Tx bytes: 1 input with box_id and ProverResult carrying `{0: Tuple(Box.id of SELF)}`.
        // Box.id = SMethod(type_id=0x63, method_id=0x01). SELF opcode = 0xa7. MethodCall args
        // count = 0 (u32 VLQ `00`). Tuple opcode `86`, count `01`.
        let input_hex = "0100a19de1b5fa998df5a48630a611180690abad5270c33f23a79baba2f8840d710001008601dc6301a700000001c0843d0008d3010000";
        // Round-trip at V0 rewrites the inner MethodCall as a PropertyCall (`dc 63 01 a7 00` →
        // `db 63 01 a7`), so the Tuple value re-serializes shorter by one byte.
        let expected_out = "0100a19de1b5fa998df5a48630a611180690abad5270c33f23a79baba2f8840d710001008601db6301a7000001c0843d0008d3010000";
        let j = run_entry("Transaction", input_hex, 0, Some(1)).to_json();
        assert_eq!(
            j["error"],
            J::Null,
            "tx with 0-arg MethodCall at tree_version=V0 must parse: {j}"
        );
        assert_eq!(j["bytes_hex"], expected_out);
    }

    /// At block_version 4 a BlockTransactions section reads its transactions at
    /// `(activated = V3, tree_version = V3)`. A section whose one transaction carries
    /// `Coll[SFunc]` in a context extension round-trips. A mutant that lifted the threshold to
    /// `>= 5` (so a v4 block reads at V0) would refuse the SFunc inside.
    #[test]
    fn block_v4_section_reads_transactions_at_tree_version_v3() {
        // header_id (32 zero bytes) + `84 ad e2 04` (VLQ 10_000_004 = marker + block v4) + `01`
        // (one tx) + the ext-c2 transaction (SFunc in the single input's context extension).
        let hex = format!(
            "{}{}{}{}",
            "0".repeat(64),
            "84ade204",
            "01",
            "0100a19de1b5fa998df5a48630a611180690abad5270c33f23a79baba2f8840d710001008300700204040400000001c0843d0008d3010000"
        );
        let j = run_entry("BlockTransactions", &hex, 3, None).to_json();
        assert_eq!(
            j["error"],
            J::Null,
            "block v4 section with SFunc tx must round-trip: {j}"
        );
        assert_eq!(j["bytes_hex"], hex);
    }

    /// At block_version 3 a BlockTransactions section reads its transactions at
    /// `tree_version = V0` (no activated context). A section whose one transaction carries
    /// `Coll[SFunc]` in a context extension is refused. A mutant that lowered the threshold
    /// to `>= 3` (so a v3 block reads at V3) would accept the SFunc inside.
    #[test]
    fn block_v3_section_reads_transactions_at_tree_version_v0() {
        let hex = format!(
            "{}{}{}{}",
            "0".repeat(64),
            "83ade204",
            "01",
            "0100a19de1b5fa998df5a48630a611180690abad5270c33f23a79baba2f8840d710001008300700204040400000001c0843d0008d3010000"
        );
        let j = run_entry("BlockTransactions", &hex, 3, None).to_json();
        assert_eq!(
            j["error"], "errored",
            "block v3 section with SFunc tx must refuse (read at V0): {j}"
        );
    }

    /// At block_version 3 a BlockTransactions section WRITES its transactions at
    /// `tree_version = V3`: an `Upcast(Const)` in a context extension is written preserved.
    /// A mutant that lifted the threshold to `>= 4` would write at V0 and inline the Upcast.
    /// The witness is a tx whose extension holds `Tuple(Int 1, Upcast(1, Long))` (ext-x15).
    #[test]
    fn block_v3_section_writes_transactions_at_tree_version_v3() {
        // ext-x15: `... 86 02 04 02 7e 04 02 05 ...`. Read at V0, written at V3 preserves Upcast.
        let hex = format!(
            "{}{}{}{}",
            "0".repeat(64),
            "83ade204",
            "01",
            "0100a19de1b5fa998df5a48630a611180690abad5270c33f23a79baba2f8840d71000100860204027e040205000001c0843d0008d3010000"
        );
        let j = run_entry("BlockTransactions", &hex, 3, None).to_json();
        assert_eq!(j["error"], J::Null, "block v3 section must round-trip: {j}");
        // At write_version=V3 the Upcast opcode stays; the output == input.
        assert_eq!(
            j["bytes_hex"], hex,
            "Upcast in ctx ext must be preserved at write V3"
        );
    }

    /// At block_version 2 a BlockTransactions section WRITES its transactions at
    /// `tree_version = V0`: an `Upcast(Const)` in a context extension is inlined to the
    /// constant. A mutant that lowered the write threshold to `>= 2` would write at V3 and
    /// preserve the Upcast (producing input-identity).
    #[test]
    fn block_v2_section_writes_transactions_at_tree_version_v0() {
        let input_hex = format!(
            "{}{}{}{}",
            "0".repeat(64),
            "82ade204",
            "01",
            "0100a19de1b5fa998df5a48630a611180690abad5270c33f23a79baba2f8840d71000100860204027e040205000001c0843d0008d3010000"
        );
        // Written at V0, `86 02 04 02 7e 04 02 05` becomes `86 02 04 02 04 02` (Upcast inlined
        // into the Long constant), so the section is written back shorter than the input.
        let expected_out = format!(
            "{}{}{}{}",
            "0".repeat(64),
            "82ade204",
            "01",
            "0100a19de1b5fa998df5a48630a611180690abad5270c33f23a79baba2f8840d71000100860204020402000001c0843d0008d3010000"
        );
        let j = run_entry("BlockTransactions", &input_hex, 3, None).to_json();
        assert_eq!(j["error"], J::Null, "block v2 section must round-trip: {j}");
        assert_eq!(
            j["bytes_hex"], expected_out,
            "Upcast must be inlined at write V0"
        );
    }

    #[test]
    fn ergotree_lenient_roundtrip_matches_jvm_canonical() {
        // eda080: a UTF-16-surrogate STypeVar name. sigma-rust matches the JVM's 1-FFFD collapse, so the
        // lenient (structural) round-trip yields the JVM-canonical efbfbd form — NOT the input
        // (echo) and NOT 3-FFFD. From vectors/wire/v6/authored/STypeVar.name_utf8_roundtrip.json.
        let input = "1b1901040ad801d701016703eda080d901026703eda08072027300";
        let expected = "1b1901040ad801d701016703efbfbdd901026703efbfbd72027300";
        let j = run_entry("ErgoTree", input, 3, None).to_json();
        assert_eq!(j["error"], J::Null);
        assert_eq!(j["bytes_hex"], expected);
    }
}
