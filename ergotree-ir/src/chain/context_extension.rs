//! ContextExtension type
use crate::chain::evaluated_value::EvaluatedValue;
use crate::mir::constant::Constant;
use crate::mir::constant::TryExtractFromError;
use crate::serialization::sigma_byte_reader::SigmaByteRead;
use crate::serialization::sigma_byte_writer::SigmaByteWrite;
use crate::serialization::SigmaParsingError;
use crate::serialization::SigmaSerializable;
use crate::serialization::SigmaSerializationError;
use crate::serialization::SigmaSerializeResult;
use alloc::string::String;
use core::convert::TryFrom;
use core::fmt;
use core::hash::BuildHasher;
use thiserror::Error;

use super::IndexMap;

/// User-defined variables to be put into context
#[derive(Debug, PartialEq, Eq, Clone)]
#[cfg_attr(
    feature = "json",
    derive(serde::Deserialize),
    serde(try_from = "IndexMap<String, String>")
)]
pub struct ContextExtension {
    /// key-value pairs of variable id and its value: any value sigmastate's cast to
    /// `EvaluatedValue` accepts (v6.0.6 `ContextExtension.scala:61`)
    pub values: IndexMap<u8, EvaluatedValue>,
}

impl ContextExtension {
    /// Returns an empty ContextExtension
    pub fn empty() -> Self {
        Self {
            values: IndexMap::with_hasher(Default::default()),
        }
    }

    /// The value of variable `id` as a constant (see [`EvaluatedValue::to_constant`]), or `None`
    /// when there is no such variable
    pub fn get_constant(&self, id: u8) -> Result<Option<Constant>, TryExtractFromError> {
        self.values
            .get(&id)
            .map(EvaluatedValue::to_constant)
            .transpose()
    }
}

impl fmt::Display for ContextExtension {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.values.iter()).finish()
    }
}

/// Scala 2.12 immutable.HashMap hash improvement function.
/// Used to predict the HAMT (Hash Array Mapped Trie) iteration order
/// that the Ergo node (Scala 2.12) uses for ContextExtension serialization.
///
/// The Ergo node's ContextExtension uses `scala.collection.immutable.Map` which,
/// for 5+ entries, becomes a HashMap with hash-based iteration order. This order
/// differs from sigma-rust's BTreeMap/IndexMap sorted order, causing bytes_to_sign
/// divergence and transaction rejection.
///
/// See: <https://github.com/scala/scala/blob/v2.12.20/src/library/scala/collection/immutable/HashMap.scala>
/// See: <https://github.com/ergoplatform/sigma-rust/issues/763>
fn scala_212_improve(hc: i32) -> i32 {
    let mut h: i32 = hc.wrapping_add(!(hc.wrapping_shl(9)));
    h = h ^ (((h as u32) >> 14) as i32);
    h = h.wrapping_add(h.wrapping_shl(4));
    h ^ (((h as u32) >> 10) as i32)
}

/// Compute a sort key that matches Scala 2.12 HashMap's HAMT iteration order.
/// The HAMT uses 5 bits per level from the improved hash, iterating slots 0-31
/// at each level. The sort key encodes levels from outermost (most significant)
/// to innermost (least significant).
fn scala_212_hamt_sort_key(key: u8) -> u64 {
    let hash = scala_212_improve(key as i32) as u32;
    let mut sort_key: u64 = 0;
    for level in 0..7 {
        sort_key <<= 5;
        sort_key |= ((hash >> (level * 5)) & 0x1f) as u64;
    }
    sort_key
}

impl SigmaSerializable for ContextExtension {
    fn sigma_serialize<W: SigmaByteWrite>(&self, w: &mut W) -> SigmaSerializeResult {
        // sigmastate v6.0.6 `ContextExtension.serializer.serialize` (`:44-50`):
        // more than `Byte.MaxValue` entries is an error, never a truncated count.
        let len = self.values.len();
        if len > i8::MAX as usize {
            return Err(SigmaSerializationError::NotSupported(format!(
                "Number of ContextExtension values {len} exceeds {}",
                i8::MAX
            )));
        }
        w.put_u8(len as u8)?;
        if self.values.len() >= 5 {
            // For 5+ entries, Scala 2.12 uses HashMap which iterates in HAMT order
            // (based on hash of keys). We must match this order for bytes_to_sign
            // compatibility with the Ergo node.
            // See: https://github.com/ergoplatform/sigma-rust/issues/763
            let mut entries: alloc::vec::Vec<_> = self.values.iter().collect();
            entries.sort_by_key(|(&idx, _)| scala_212_hamt_sort_key(idx));
            for (&idx, c) in entries {
                w.put_u8(idx)?;
                c.sigma_serialize(w)?;
            }
        } else {
            // For 1-4 entries, Scala uses Map1-Map4 which preserves insertion order.
            // IndexMap also preserves insertion order, so they match.
            self.values.iter().try_for_each(|(idx, c)| {
                w.put_u8(*idx)?;
                c.sigma_serialize(w)
            })?;
        }
        Ok(())
    }

    /// Port of sigmastate v6.0.6 `ContextExtension.serializer.parse`
    /// (`data/shared/src/main/scala/sigma/interpreter/ContextExtension.scala:52-66`).
    /// The count and every variable id are signed bytes, and a negative one is
    /// rejected at parse: count ≥ 128 (`:53-55`), id ≥ 0x80 (`:58-60`, checked
    /// before the value is read). Not version-gated, as in the reference.
    fn sigma_parse<R: SigmaByteRead>(r: &mut R) -> Result<Self, SigmaParsingError> {
        let values_count = r.get_i8()?;
        if values_count < 0 {
            return Err(SigmaParsingError::ValueOutOfBounds(format!(
                "Negative amount of context extension values: {values_count}"
            )));
        }
        let mut values: IndexMap<u8, EvaluatedValue> =
            IndexMap::with_capacity_and_hasher(values_count as usize, Default::default());
        for _ in 0..values_count {
            let idx = r.get_i8()?;
            if idx < 0 {
                return Err(SigmaParsingError::ValueOutOfBounds(format!(
                    "Negative id of context extension variable: {idx}"
                )));
            }
            // `getValue`, the cast to `EvaluatedValue`, then `CheckV6Type` (sigmastate v6.0.6
            // `ContextExtension.scala:61-62`). `getValue` takes one value level
            // (`ValueSerializer.scala:396-409`) on top of the value's own, released only on
            // success.
            let value = EvaluatedValue::sigma_parse(r)?;
            value.check_v6_type()?;
            values.insert(idx as u8, value);
        }
        Ok(ContextExtension { values })
    }
}

/// Error parsing Constant from base16-encoded string
#[derive(Error, Eq, PartialEq, Debug, Clone)]
#[error("Error parsing constant: {0}")]
pub struct ConstantParsingError(pub String);

// for JSON encoding in ergo-lib
impl<H: BuildHasher> TryFrom<indexmap::IndexMap<String, String, H>> for ContextExtension {
    type Error = ConstantParsingError;
    fn try_from(values_str: indexmap::IndexMap<String, String, H>) -> Result<Self, Self::Error> {
        let values = values_str.iter().try_fold(
            IndexMap::with_capacity_and_hasher(values_str.len(), Default::default()),
            |mut acc, pair| {
                let idx: u8 = pair.0.parse().map_err(|_| {
                    ConstantParsingError(format!("cannot parse index from {0:?}", pair.0))
                })?;
                // The JVM decodes a key as a `Byte`: 128..=255 cannot be represented.
                if idx > i8::MAX as u8 {
                    return Err(ConstantParsingError(format!(
                        "context extension variable id {idx} is outside 0..=127"
                    )));
                }
                let value_bytes = base16::decode(pair.1).map_err(|_| {
                    ConstantParsingError(format!(
                        "cannot decode base16 value bytes from {0:?}",
                        pair.1
                    ))
                })?;
                // sigmastate's decoder is `getValue` and the cast, without `CheckV6Type`
                // (v6.0.6 `JsonCodecs.scala:192-196`)
                acc.insert(
                    idx,
                    EvaluatedValue::sigma_parse_bytes(&value_bytes).map_err(|_| {
                        ConstantParsingError(format!(
                            "cannot deserialize value bytes from {0:?}",
                            pair.1
                        ))
                    })?,
                );
                Ok(acc)
            },
        )?;
        Ok(ContextExtension { values })
    }
}

#[cfg(feature = "json")]
impl serde::Serialize for ContextExtension {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::Error;
        use serde::ser::SerializeMap;
        let mut map = serializer.serialize_map(Some(self.values.len()))?;
        for (k, v) in &self.values {
            map.serialize_entry(
                &format!("{}", k),
                &base16::encode_lower(&v.sigma_serialize_bytes().map_err(Error::custom)?),
            )?;
        }
        map.end()
    }
}

#[cfg(feature = "arbitrary")]
mod arbitrary {
    use super::*;
    use proptest::{arbitrary::Arbitrary, collection::vec, prelude::*};

    impl Arbitrary for ContextExtension {
        type Parameters = ();
        type Strategy = BoxedStrategy<Self>;

        fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
            vec(
                any::<Constant>().prop_filter(
                    "Filter out types that can't be serialized in ContextExtension",
                    |c| c.tpe.check_v6_type().is_ok(),
                ),
                0..10,
            )
            .prop_map(|constants| {
                let pairs = constants
                    .into_iter()
                    .enumerate()
                    .map(|(idx, c)| (idx as u8, c.into()))
                    .collect();
                Self { values: pairs }
            })
            .boxed()
        }
    }
}

#[cfg(test)]
#[cfg(feature = "arbitrary")]
#[allow(clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::{
        serialization::{sigma_serialize_roundtrip, SigmaSerializable},
        unsignedbigint256::UnsignedBigInt,
    };
    use proptest::prelude::*;

    const SCALA_212_HAMT_ORDER_8: [u8; 8] = [0, 5, 1, 6, 2, 7, 3, 4];

    const SCALA_212_HAMT_ORDER_32: [u8; 32] = [
        0, 5, 10, 24, 25, 14, 20, 29, 1, 6, 28, 21, 9, 13, 2, 17, 22, 27, 12, 7, 3, 18, 16, 31, 11,
        26, 23, 8, 30, 19, 4, 15,
    ];

    // Ordering only: the JVM ContextExtension serializer allows at most 127 entries.
    const SCALA_212_HAMT_ORDER_128: [u8; 128] = [
        69, 101, 0, 88, 115, 5, 120, 10, 56, 42, 24, 37, 25, 52, 14, 110, 125, 20, 46, 93, 57, 78,
        29, 106, 121, 84, 61, 89, 116, 1, 74, 6, 60, 117, 85, 102, 28, 38, 70, 21, 33, 92, 65, 97,
        9, 53, 109, 124, 77, 96, 13, 41, 73, 105, 2, 32, 34, 45, 64, 17, 22, 44, 59, 118, 27, 71,
        12, 54, 49, 86, 113, 81, 76, 7, 39, 98, 103, 91, 66, 108, 3, 80, 35, 112, 123, 48, 63, 18,
        95, 50, 67, 16, 127, 31, 11, 72, 43, 99, 87, 104, 40, 26, 55, 114, 23, 8, 75, 119, 58, 82,
        36, 30, 51, 19, 107, 4, 126, 79, 94, 47, 15, 68, 62, 90, 111, 122, 83, 100,
    ];

    const SCALA_212_CONTEXT_EXTENSION_8_BYTES: [u8; 25] = [
        8, 0, 4, 0, 5, 4, 10, 1, 4, 2, 6, 4, 12, 2, 4, 4, 7, 4, 14, 3, 4, 6, 4, 4, 8,
    ];

    fn context_extension_with_int_constants(
        keys: impl IntoIterator<Item = u8>,
    ) -> ContextExtension {
        let mut ext = ContextExtension::empty();
        for key in keys {
            ext.values.insert(key, (key as i32).into());
        }
        ext
    }

    fn serialized_keys(ext: &ContextExtension) -> Vec<u8> {
        let bytes = ext.sigma_serialize_bytes().unwrap();
        let mut keys = Vec::new();
        let mut pos = 1;
        while pos < bytes.len() {
            keys.push(bytes[pos]);
            let constant = Constant::sigma_parse_bytes(&bytes[pos + 1..]).unwrap();
            pos += 1 + constant.sigma_serialize_bytes().unwrap().len();
        }
        keys
    }

    #[test]
    #[should_panic]
    fn test_v6_type_reject() {
        let mut extension = ContextExtension::empty();
        extension
            .values
            .insert(0, UnsignedBigInt::from(1u32).into());
        sigma_serialize_roundtrip(&extension);
    }

    proptest! {
        #[test]
        fn ser_roundtrip(v in any::<ContextExtension>()) {
            prop_assert_eq![sigma_serialize_roundtrip(&v), v];
        }
    }
    #[test]
    fn test_scala_212_improve() {
        // Verify that the improve function produces distinct hashes and that
        // the lowest 5 bits (HAMT level-0 slot) match the empirically observed
        // Ergo node iteration order for keys 0-5: [0, 5, 1, 2, 3, 4].
        // Level-0 slots: key->slot: 0->0, 1->7, 2->14, 3->20, 4->29, 5->1
        assert_eq!((scala_212_improve(0) as u32) & 0x1f, 0);
        assert_eq!((scala_212_improve(1) as u32) & 0x1f, 7);
        assert_eq!((scala_212_improve(2) as u32) & 0x1f, 14);
        assert_eq!((scala_212_improve(3) as u32) & 0x1f, 20);
        assert_eq!((scala_212_improve(4) as u32) & 0x1f, 29);
        assert_eq!((scala_212_improve(5) as u32) & 0x1f, 1);
    }

    #[test]
    fn test_hamt_sort_order_6_entries() {
        // For keys {0,1,2,3,4,5}, the Scala 2.12 HashMap HAMT iterates in order
        // [0, 5, 1, 2, 3, 4] due to the improve hash function's slot assignments.
        // This was verified empirically against the Ergo node.
        let mut keys: Vec<u8> = vec![0, 1, 2, 3, 4, 5];
        keys.sort_by_key(|&k| scala_212_hamt_sort_key(k));
        assert_eq!(keys, vec![0, 5, 1, 2, 3, 4]);
    }

    #[test]
    fn test_hamt_sort_order_matches_scala_212_golden_vectors() {
        let mut keys_8: Vec<u8> = (0..8).collect();
        keys_8.sort_by_key(|&k| scala_212_hamt_sort_key(k));
        assert_eq!(keys_8.as_slice(), &SCALA_212_HAMT_ORDER_8);

        let mut keys_32: Vec<u8> = (0..32).collect();
        keys_32.sort_by_key(|&k| scala_212_hamt_sort_key(k));
        assert_eq!(keys_32.as_slice(), &SCALA_212_HAMT_ORDER_32);

        let mut keys_128: Vec<u8> = (0..128).collect();
        keys_128.sort_by_key(|&k| scala_212_hamt_sort_key(k));
        assert_eq!(keys_128.as_slice(), &SCALA_212_HAMT_ORDER_128);
    }

    #[test]
    fn test_serialize_order_matches_scala_212_golden_vectors() {
        let ext_8 = context_extension_with_int_constants(0..8);
        let bytes_8 = ext_8.sigma_serialize_bytes().unwrap();
        assert_eq!(bytes_8, SCALA_212_CONTEXT_EXTENSION_8_BYTES);
        assert_eq!(serialized_keys(&ext_8).as_slice(), &SCALA_212_HAMT_ORDER_8);

        let ext_32 = context_extension_with_int_constants(0..32);
        assert_eq!(
            serialized_keys(&ext_32).as_slice(),
            &SCALA_212_HAMT_ORDER_32
        );
    }

    #[test]
    fn test_serialize_all_five_entry_insertion_orders() {
        // Scala 2.12 switches from Map4 insertion order to HashMap traversal at five entries.
        // For these keys the JVM order is [0, 5, 1, 2, 3], distinct from numeric order.
        const EXPECTED_BYTES: [u8; 16] = [5, 0, 4, 0, 5, 4, 10, 1, 4, 2, 2, 4, 4, 3, 4, 6];

        fn check_permutations(keys: &mut [u8; 5], start: usize) -> usize {
            if start == keys.len() {
                let ext = context_extension_with_int_constants(keys.iter().copied());
                assert_eq!(ext.sigma_serialize_bytes().unwrap(), EXPECTED_BYTES);
                return 1;
            }
            let mut checked = 0;
            for next in start..keys.len() {
                keys.swap(start, next);
                checked += check_permutations(keys, start + 1);
                keys.swap(start, next);
            }
            checked
        }

        assert_eq!(check_permutations(&mut [0, 1, 2, 3, 5], 0), 120);
    }

    #[test]
    fn test_serialize_order_5plus_entries() {
        // Verify that serialization of 6-entry ContextExtension produces entries
        // in Scala 2.12 HAMT iteration order, not insertion/sorted order.
        let mut ext = ContextExtension::empty();
        for i in 0..6u8 {
            ext.values.insert(i, (i as i32).into());
        }
        let bytes = ext.sigma_serialize_bytes().unwrap();
        // bytes[0] = count (6)
        assert_eq!(bytes[0], 6);
        // After count, each entry is: key_byte, serialized_constant
        // Extract just the key bytes (every entry is key + 2 bytes for SInt constant)
        let mut keys = Vec::new();
        let mut pos = 1;
        while pos < bytes.len() {
            keys.push(bytes[pos]);
            // SInt constants serialize as type_byte + vlq_value (2 bytes for small ints)
            let c = Constant::sigma_parse_bytes(&bytes[pos + 1..]).unwrap();
            pos += 1 + c.sigma_serialize_bytes().unwrap().len();
        }
        assert_eq!(keys, vec![0, 5, 1, 2, 3, 4]);
    }

    #[test]
    fn test_serialize_order_4_entries_unchanged() {
        // Verify that serialization of 4-entry ContextExtension preserves
        // insertion order (Scala Map1-Map4 behavior).
        let mut ext = ContextExtension::empty();
        for i in 0..4u8 {
            ext.values.insert(i, (i as i32).into());
        }
        let bytes = ext.sigma_serialize_bytes().unwrap();
        assert_eq!(bytes[0], 4);
        let mut keys = Vec::new();
        let mut pos = 1;
        while pos < bytes.len() {
            keys.push(bytes[pos]);
            let c = Constant::sigma_parse_bytes(&bytes[pos + 1..]).unwrap();
            pos += 1 + c.sigma_serialize_bytes().unwrap().len();
        }
        // 4 entries: insertion order preserved
        assert_eq!(keys, vec![0, 1, 2, 3]);
    }

    #[cfg(feature = "json")]
    mod json {
        use super::*;
        #[test]
        fn parse_empty_context_extension() {
            let c: ContextExtension = serde_json::from_str("{}").unwrap();
            assert_eq!(c, ContextExtension::empty());
        }

        #[test]
        fn parse_context_extension() {
            let json = r#"
            {"1" :"05b0b5cad8e6dbaef44a", "3":"048ce5d4e505"}
            "#;
            let c: ContextExtension = serde_json::from_str(json).unwrap();
            assert_eq!(c.values.len(), 2);
            assert!(c.values.get(&1u8).is_some());
            assert!(c.values.get(&3u8).is_some());
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod parse_bounds_tests {
    //! JVM parity for `ContextExtension` bounds: sigmastate v6.0.6
    //! `ContextExtension.serializer` (`ContextExtension.scala:44-66`).
    use super::*;

    /// Wire bytes: the count byte, then `(id, IntConstant(1))` for each id.
    fn wire(count: u8, ids: &[u8]) -> Vec<u8> {
        let value = Constant::from(1i32).sigma_serialize_bytes().unwrap();
        let mut bytes = vec![count];
        for &id in ids {
            bytes.push(id);
            bytes.extend_from_slice(&value);
        }
        bytes
    }

    fn out_of_bounds(r: Result<ContextExtension, SigmaParsingError>) -> bool {
        matches!(r, Err(SigmaParsingError::ValueOutOfBounds(_)))
    }

    #[test]
    fn parse_accepts_count_127_and_id_0x7f() {
        let ids: Vec<u8> = (0..127).collect();
        let ext = ContextExtension::sigma_parse_bytes(&wire(127, &ids)).unwrap();
        assert_eq!(ext.values.len(), 127);
        let ext = ContextExtension::sigma_parse_bytes(&wire(1, &[0x7f])).unwrap();
        assert!(ext.values.contains_key(&0x7f));
    }

    #[test]
    fn parse_rejects_count_128_even_with_valid_ids() {
        let ids: Vec<u8> = (0..128).collect(); // 0..=127, every id valid
        assert!(out_of_bounds(ContextExtension::sigma_parse_bytes(&wire(
            128, &ids
        ))));
    }

    #[test]
    fn parse_rejects_count_255() {
        let ids: Vec<u8> = (0..255u16).map(|i| (i % 128) as u8).collect();
        assert!(out_of_bounds(ContextExtension::sigma_parse_bytes(&wire(
            255, &ids
        ))));
    }

    #[test]
    fn parse_rejects_id_0x80_and_0xff() {
        for id in [0x80u8, 0xff] {
            assert!(
                out_of_bounds(ContextExtension::sigma_parse_bytes(&wire(1, &[id]))),
                "id {id:#04x}"
            );
        }
    }

    #[test]
    fn parse_rejects_bad_id_before_reading_its_value() {
        // count 1, id 0x80, no value bytes: the id check (Scala :58-60) precedes the value read (:61)
        assert!(out_of_bounds(ContextExtension::sigma_parse_bytes(&[
            1, 0x80
        ])));
    }

    #[test]
    fn parse_keeps_duplicate_id_semantics() {
        // Unchanged by this fix: last value wins, first position kept (Scala `toMap`).
        let one = Constant::from(1i32).sigma_serialize_bytes().unwrap();
        let two = Constant::from(2i32).sigma_serialize_bytes().unwrap();
        let mut bytes = vec![3, 5];
        bytes.extend_from_slice(&one);
        bytes.push(7);
        bytes.extend_from_slice(&one);
        bytes.push(5);
        bytes.extend_from_slice(&two);
        let ext = ContextExtension::sigma_parse_bytes(&bytes).unwrap();
        assert_eq!(ext.values.keys().copied().collect::<Vec<_>>(), vec![5, 7]);
        assert_eq!(ext.values[&5], EvaluatedValue::from(2i32));
    }

    #[test]
    fn serialize_rejects_more_than_127_entries() {
        let mut ext = ContextExtension::empty();
        for id in 0..=127u8 {
            ext.values.insert(id, 1i32.into());
        }
        assert!(ext.sigma_serialize_bytes().is_err());
    }

    #[test]
    fn serialize_accepts_127_entries() {
        let mut ext = ContextExtension::empty();
        for id in 0..127u8 {
            ext.values.insert(id, 1i32.into());
        }
        assert_eq!(ext.sigma_serialize_bytes().unwrap()[0], 127);
    }

    #[cfg(feature = "json")]
    #[test]
    fn json_rejects_keys_above_127() {
        let hex = base16::encode_lower(&Constant::from(1i32).sigma_serialize_bytes().unwrap());
        let ok = format!(r#"{{"127": "{hex}"}}"#);
        assert!(serde_json::from_str::<ContextExtension>(&ok).is_ok());
        for key in ["128", "200", "255"] {
            let json = format!(r#"{{"{key}": "{hex}"}}"#);
            assert!(
                serde_json::from_str::<ContextExtension>(&json).is_err(),
                "key {key}"
            );
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod depth_limit_tests {
    //! JVM parity: the extension reads each value with `getValue` (sigmastate v6.0.6
    //! `ContextExtension.scala:61`), so a value takes one value level
    //! (`ValueSerializer.deserialize`, `ValueSerializer.scala:396-409`) on top of its
    //! data levels, up to `MaxTreeDepth` = 110 per reader.
    use super::*;
    use crate::chain::ergo_box::box_value::BoxValue;
    use crate::chain::ergo_box::{ErgoBox, NonMandatoryRegisters};
    use crate::chain::tx_id::TxId;
    use crate::ergo_tree::ErgoTree;
    use crate::has_opcode::HasStaticOpCode;
    use crate::serialization::op_code::OpCode;
    use crate::serialization::sigma_byte_writer::SigmaByteWriter;
    use crate::sigma_protocol::sigma_boolean::cand::Cand;
    use alloc::vec::Vec;
    use sigma_ser::vlq_encode::WriteSigmaVlqExt;

    /// `Coll^n[Byte]` constant (n ≥ 2): type `0c`×(n−2) `1a`, data `01`×(n−1) `00`.
    fn coll_n_byte(n: usize) -> Vec<u8> {
        let mut bytes = vec![0x0c; n - 2];
        bytes.push(0x1a);
        bytes.extend(vec![0x01; n - 1]);
        bytes.push(0x00);
        bytes
    }

    /// A `SigmaProp` constant of `k` nested CANDs around `TrueProp`,
    /// `CAND(CAND(… CAND(true, true) …, true), true)`.
    fn cand_chain(k: usize) -> Vec<u8> {
        let and = Cand::OP_CODE.value();
        let t = OpCode::TRIVIAL_PROP_TRUE.value();
        let mut bytes = vec![0x08]; // SSigmaProp type code
        for _ in 0..k {
            bytes.extend([and, 2]);
        }
        bytes.extend([t, t]);
        bytes.extend(vec![t; k - 1]);
        bytes
    }

    /// A `Box` constant whose tree is size-flagged (header `0x18`) and holds one
    /// segregated constant `Coll^n[Byte]`, with a `SigmaProp(true)` root.
    fn box_with_deep_tree_constant(n: usize) -> Constant {
        let mut body = vec![1];
        body.extend(coll_n_byte(n));
        body.extend([0x08, OpCode::TRIVIAL_PROP_TRUE.value()]);
        let mut tree_bytes = Vec::new();
        {
            let mut w = SigmaByteWriter::new(&mut tree_bytes, None);
            w.put_u8(0x18).unwrap();
            w.put_u32(body.len() as u32).unwrap();
            body.iter().for_each(|b| w.put_u8(*b).unwrap());
        }
        // On its own reader the tree takes n levels, within the limit here.
        let tree = ErgoTree::sigma_parse_bytes(&tree_bytes).unwrap();
        assert!(matches!(tree, ErgoTree::Parsed(_)));
        ErgoBox::new(
            BoxValue::SAFE_USER_MIN,
            tree,
            None,
            NonMandatoryRegisters::empty(),
            0,
            TxId::zero(),
            0,
        )
        .unwrap()
        .into()
    }

    /// Wire bytes of the one-entry extension `{1: value}`.
    fn extension(value: Vec<u8>) -> Vec<u8> {
        let mut bytes = vec![1, 1];
        bytes.extend(value);
        bytes
    }

    fn depth_exceeded(r: Result<ContextExtension, SigmaParsingError>) -> bool {
        matches!(r, Err(SigmaParsingError::DeserializeCallDepthExceeded(111)))
    }

    #[test]
    fn extension_value_takes_a_value_level() {
        // Coll^n[Byte]: a value level and n data levels.
        assert!(ContextExtension::sigma_parse_bytes(&extension(coll_n_byte(109))).is_ok());
        assert!(depth_exceeded(ContextExtension::sigma_parse_bytes(
            &extension(coll_n_byte(110))
        )));
    }

    #[test]
    fn extension_sigma_prop_takes_a_value_level() {
        // A value level, a data level, k CAND levels and the innermost leaf: k + 3.
        assert!(ContextExtension::sigma_parse_bytes(&extension(cand_chain(107))).is_ok());
        assert!(depth_exceeded(ContextExtension::sigma_parse_bytes(
            &extension(cand_chain(108))
        )));
    }

    #[test]
    fn extension_box_tree_continues_from_the_box_level() {
        // A value level and the Box data level (`DataSerializer.scala:31-49`), then the
        // box's size-flagged tree on the same counter: its constant reaches 2 + n.
        let ext_bytes = |n| {
            let value = box_with_deep_tree_constant(n);
            extension(value.sigma_serialize_bytes().unwrap())
        };
        assert!(ContextExtension::sigma_parse_bytes(&ext_bytes(108)).is_ok());
        assert!(depth_exceeded(ContextExtension::sigma_parse_bytes(
            &ext_bytes(109)
        )));
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod evaluated_value_tests {
    //! JVM parity: sigmastate reads each context-extension value with `getValue` and casts it to
    //! `EvaluatedValue` (v6.0.6 `ContextExtension.scala:61`), so a variable may hold a tuple, a
    //! concrete collection or the group generator. The tx id hashes each value as `putValue`
    //! writes it back (`:49`).
    use super::*;
    use crate::ergo_tree::ErgoTreeVersion;
    use crate::serialization::sigma_byte_writer::SigmaByteWriter;
    use alloc::format;
    use alloc::string::ToString;
    use alloc::vec::Vec;

    /// One variable, id 0, holding `value`
    fn extension_hex(value: &str) -> String {
        format!("0100{value}")
    }

    fn parse(value: &str) -> Result<ContextExtension, SigmaParsingError> {
        ContextExtension::sigma_parse_bytes(&base16::decode(&extension_hex(value)).unwrap())
    }

    #[test]
    fn values_the_jvm_accepts_parse_and_are_written_back_as_it_writes_them() {
        // SANTA `extension_evaluated_values` X1-X14, as in `chain::evaluated_value`'s tests
        for (value, written_back) in [
            ("7f", "0101"),
            ("80", "0100"),
            ("82", "82"),
            ("83020404020404", "83020404020404"),
            ("83020101010100", "850201"),
            ("8302017f80", "850201"),
            ("830001", "8500"),
            ("850201", "850201"),
            ("860204020404", "860204020404"),
            ("8603040204040406", "8603040204040406"),
            ("86010402", "86010402"),
            ("8600", "8600"),
            ("86020402a3", "86020402a3"),
            ("830104a3", "830104a3"),
        ] {
            let ext = parse(value).unwrap();
            assert_eq!(
                base16::encode_lower(&ext.sigma_serialize_bytes().unwrap()),
                extension_hex(written_back),
                "{value}"
            );
        }
    }

    #[test]
    fn an_upcast_of_a_constant_is_written_as_the_writer_s_version_writes_it() {
        // SANTA X15, `Tuple(1, Upcast(1, Long))`: a v6 block's transaction, parsed at tree
        // version 3, keeps the `Upcast`; below version 3 the constant is written in its place
        // (`ValueSerializer.scala:157-169`)
        let ext = parse("860204027e040205").unwrap();
        let written_at = |version: ErgoTreeVersion| {
            let mut data = Vec::new();
            let mut w = SigmaByteWriter::new(&mut data, None);
            w.with_tree_version(version, |w| ext.sigma_serialize(w))
                .unwrap();
            base16::encode_lower(&data)
        };
        assert_eq!(
            written_at(ErgoTreeVersion::V0),
            extension_hex("860204020402")
        );
        assert_eq!(
            written_at(ErgoTreeVersion::V3),
            extension_hex("860204027e040205")
        );
    }

    #[test]
    fn values_the_jvm_rejects_do_not_parse() {
        // SANTA N1-N6: a placeholder, `HEIGHT`, `Plus(1, 2)`, a tuple holding a placeholder, a
        // tuple holding `GetVar[Int](0)` (rule 1019), and a tuple size of 0x80
        let size_0x80 = format!("8680{}", "0402".repeat(128));
        for value in [
            "7300",
            "a3",
            "9a04020404",
            "860204027300",
            "86020402e30004",
            &size_0x80,
        ] {
            assert!(parse(value).is_err(), "{value}");
        }
    }

    #[test]
    fn get_constant_derives_a_value_s_constant_form() {
        // `Coll[Int](1, 2)`, against its constant encoding; `Tuple(1, HEIGHT)` has none
        let coll = parse("83020404020404").unwrap();
        assert_eq!(
            coll.get_constant(0).unwrap(),
            Some(Constant::sigma_parse_bytes(&base16::decode("10020204").unwrap()).unwrap())
        );
        assert!(parse("86020402a3").unwrap().get_constant(0).is_err());
    }

    #[test]
    fn json_values_parse_as_the_jvm_decoder_reads_them() {
        // sigmastate's JSON decoder is `getValue` and the cast, without rule 1019
        // (`JsonCodecs.scala:192-196`): `TrueLeaf` and a tuple holding an `Option` pass,
        // `HEIGHT` does not
        let json = |value: &str| {
            let mut map: IndexMap<String, String> = IndexMap::with_hasher(Default::default());
            map.insert("0".to_string(), value.to_string());
            ContextExtension::try_from(map)
        };
        assert_eq!(
            json("7f").unwrap().get_constant(0).unwrap(),
            Some(true.into())
        );
        assert!(json("86020402e30004").is_ok());
        assert!(json("a3").is_err());
    }
}
