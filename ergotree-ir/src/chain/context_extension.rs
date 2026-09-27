//! ContextExtension type
use crate::mir::constant::Constant;
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
    /// key-value pairs of variable id and it's value
    pub values: IndexMap<u8, Constant>,
}

impl ContextExtension {
    /// Returns an empty ContextExtension
    pub fn empty() -> Self {
        Self {
            values: IndexMap::with_hasher(Default::default()),
        }
    }
}

impl fmt::Display for ContextExtension {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.values.iter()).finish()
    }
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
        self.values.iter().try_for_each(|(idx, c)| {
            w.put_u8(*idx)?;
            c.sigma_serialize(w)
        })?;
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
        let mut values: IndexMap<u8, Constant> =
            IndexMap::with_capacity_and_hasher(values_count as usize, Default::default());
        for _ in 0..values_count {
            let idx = r.get_i8()?;
            if idx < 0 {
                return Err(SigmaParsingError::ValueOutOfBounds(format!(
                    "Negative id of context extension variable: {idx}"
                )));
            }
            // The JVM reads the value with `getValue` (sigmastate v6.0.6
            // `ContextExtension.scala:61`), so it takes one value level
            // (`ValueSerializer.deserialize`, `ValueSerializer.scala:396-409`) on top of the
            // constant's data levels, released only on success.
            let depth = r.level();
            r.set_level(depth + 1)?;
            let value = Constant::sigma_parse(r)?;
            r.set_level(r.level().saturating_sub(1))?;
            value.tpe.check_v6_type()?;
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
                let constant_bytes = base16::decode(pair.1).map_err(|_| {
                    ConstantParsingError(format!(
                        "cannot decode base16 constant bytes from {0:?}",
                        pair.1
                    ))
                })?;
                acc.insert(
                    idx,
                    Constant::sigma_parse_bytes(&constant_bytes).map_err(|_| {
                        ConstantParsingError(format!(
                            "cannot deserialize constant bytes from {0:?}",
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
                    .map(|(idx, c)| (idx as u8, c))
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
    use crate::{serialization::sigma_serialize_roundtrip, unsignedbigint256::UnsignedBigInt};
    use proptest::prelude::*;

    #[test]
    #[should_panic]
    fn test_v6_type_reject() {
        let mut extension = ContextExtension::empty();
        extension
            .values
            .insert(0, Constant::from(UnsignedBigInt::from(1u32)));
        sigma_serialize_roundtrip(&extension);
    }

    proptest! {
        #[test]
        fn ser_roundtrip(v in any::<ContextExtension>()) {
            prop_assert_eq![sigma_serialize_roundtrip(&v), v];
        }
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
        assert_eq!(ext.values[&5], Constant::from(2i32));
    }

    #[test]
    fn serialize_rejects_more_than_127_entries() {
        let mut ext = ContextExtension::empty();
        for id in 0..=127u8 {
            ext.values.insert(id, Constant::from(1i32));
        }
        assert!(ext.sigma_serialize_bytes().is_err());
    }

    #[test]
    fn serialize_accepts_127_entries() {
        let mut ext = ContextExtension::empty();
        for id in 0..127u8 {
            ext.values.insert(id, Constant::from(1i32));
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
