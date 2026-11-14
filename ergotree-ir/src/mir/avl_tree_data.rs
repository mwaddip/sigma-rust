use super::coll_value::CollValue;
use super::constant::TryExtractFromError;
use super::constant::TryExtractInto;
use super::value::Value;
use alloc::boxed::Box;
use alloc::vec::Vec;
use ergo_chain_types::ADDigest;
use sigma_ser::ScorexSerializable;

use crate::serialization::{
    sigma_byte_reader::SigmaByteRead, sigma_byte_writer::SigmaByteWrite, SigmaParsingError,
    SigmaSerializable, SigmaSerializeResult,
};

#[derive(PartialEq, Eq, Debug, Clone)]
/// AVL tree flags
pub struct AvlTreeFlags(u8);

impl AvlTreeFlags {
    /// Create tree-flags
    pub fn new(insert_allowed: bool, update_allowed: bool, remove_allowed: bool) -> Self {
        let read_only = 0;
        let i = if insert_allowed {
            read_only | 0x01
        } else {
            read_only
        };
        let u = if update_allowed { i | 0x02 } else { i };
        AvlTreeFlags(if remove_allowed { u | 0x04 } else { u })
    }

    /// Get byte-representation of the tree-flags
    pub fn serialize(&self) -> u8 {
        self.0
    }

    /// Parse tree-flags from byte
    pub fn parse(serialized_flags: u8) -> Self {
        let insert_allowed = serialized_flags & 0x01 != 0;
        let update_allowed = serialized_flags & 0x02 != 0;
        let remove_allowed = serialized_flags & 0x04 != 0;
        Self::new(insert_allowed, update_allowed, remove_allowed)
    }

    /// Returns true if inserting is allowed
    pub fn insert_allowed(&self) -> bool {
        self.0 & 0x01 != 0
    }

    /// Returns true if updating is allowed
    pub fn update_allowed(&self) -> bool {
        self.0 & 0x02 != 0
    }

    /// Returns true if removal is allowed
    pub fn remove_allowed(&self) -> bool {
        self.0 & 0x04 != 0
    }
}

/// Type of data which efficiently authenticates potentially huge dataset having key-value
/// dictionary interface. Only root hash of dynamic AVL+ tree, tree height, key length, optional
/// value length, and access flags are stored in an instance of the datatype.
#[derive(PartialEq, Eq, Debug, Clone)]
pub struct AvlTreeData {
    /// Authenticated tree digest: root hash along with tree height. The parser
    /// always reads exactly 33 bytes (`ADDigest` = 32-byte root hash + height
    /// byte), matching the JVM `AvlTreeData.serializer` (`getBytes(DigestSize)`).
    /// Bytes of another length come only from a runtime tree ([`AvlTreeValue`])
    /// taken as data: `updateDigest` stores its argument without a length check.
    pub digest: Vec<u8>,
    /// Allowed modifications
    pub tree_flags: AvlTreeFlags,
    /// All the elements under the tree have the same length
    pub key_length: u32,
    /// If non-empty, all the values under the tree are of the same length
    pub value_length_opt: Option<Box<u32>>,
}

/// A runtime `AvlTree` (sigmastate's `CAvlTree`). Its digest is a collection value, as
/// sigmastate's `AvlTreeData.digest` is a `Coll` (`AvlTreeData.scala:51`)
#[derive(PartialEq, Eq, Debug, Clone)]
pub struct AvlTreeValue<'ctx> {
    /// The digest as the collection it is
    pub digest: CollValue<'ctx>,
    /// Allowed modifications
    pub tree_flags: AvlTreeFlags,
    /// All the elements under the tree have the same length
    pub key_length: u32,
    /// If non-empty, all the values under the tree are of the same length
    pub value_length_opt: Option<Box<u32>>,
}

/// Data as a runtime tree: the digest is the specialised collection of bytes
impl From<AvlTreeData> for AvlTreeValue<'_> {
    fn from(d: AvlTreeData) -> Self {
        AvlTreeValue {
            digest: CollValue::Bytes(d.digest.iter().map(|&b| b as i8).collect()),
            tree_flags: d.tree_flags,
            key_length: d.key_length,
            value_length_opt: d.value_length_opt,
        }
    }
}

/// A runtime tree as data. This is sigmastate's cast of `digest.toArray` to bytes
/// (`AvlTreeData.scala:75`, `CAvlTreeVerifier.scala:59`): an error for a digest whose array
/// is not of bytes
impl TryFrom<AvlTreeValue<'_>> for AvlTreeData {
    type Error = TryExtractFromError;
    fn try_from(t: AvlTreeValue<'_>) -> Result<Self, Self::Error> {
        Ok(AvlTreeData {
            digest: Value::Coll(t.digest).try_extract_into::<Vec<u8>>()?,
            tree_flags: t.tree_flags,
            key_length: t.key_length,
            value_length_opt: t.value_length_opt,
        })
    }
}

impl SigmaSerializable for AvlTreeData {
    fn sigma_serialize<W: SigmaByteWrite>(&self, w: &mut W) -> SigmaSerializeResult {
        // Mirror the JVM `AvlTreeData.serializer`: write the digest bytes raw
        // (`putBytes`), whatever their length. The parser always reads exactly
        // 33 bytes, so a non-33 digest (only producible in memory via
        // `updateDigest`) does not round-trip — same as the reference impl.
        // Metered as PutChunkCost(33) under `Global.serialize`: treeFlags =
        // PutByteCost via the virtual put; keyLength & valueLength use the no-info
        // `putUInt` (unmetered); the option tag byte is PutByteCost.
        w.write_all(&self.digest)?;
        w.add_put_chunk_cost(33);
        w.put_u8(self.tree_flags.0)?;
        w.add_put_byte_cost();
        w.put_u32(self.key_length)?;
        self.value_length_opt.sigma_serialize(w)?;
        w.add_put_byte_cost();
        Ok(())
    }
    fn sigma_parse<R: SigmaByteRead>(r: &mut R) -> Result<Self, SigmaParsingError> {
        // Always read exactly `DigestSize` (33) bytes, matching the JVM parser.
        let digest = ADDigest::scorex_parse(r)?.0.to_vec();
        let tree_flags = AvlTreeFlags::parse(r.get_u8()?);
        let key_length = r.get_u32()?;
        let value_length_opt = <Option<Box<u32>> as SigmaSerializable>::sigma_parse(r)?;
        Ok(AvlTreeData {
            digest,
            tree_flags,
            key_length,
            value_length_opt,
        })
    }
}

#[cfg(feature = "arbitrary")]
#[allow(clippy::unwrap_used)]
mod arbitrary {

    use super::*;
    use proptest::prelude::*;

    type OptBox = Option<Box<u32>>;
    impl Arbitrary for AvlTreeData {
        type Strategy = BoxedStrategy<Self>;
        type Parameters = ();

        fn arbitrary_with(_: Self::Parameters) -> Self::Strategy {
            (
                // 33-byte digests only: a non-33 digest is producible solely in
                // memory via `updateDigest` and does not survive serialization.
                any::<ADDigest>().prop_map(|d| d.0.to_vec()),
                any::<AvlTreeFlags>(),
                any::<u32>(),
                any::<OptBox>(),
            )
                .prop_map(
                    |(digest, tree_flags, key_length, value_length_opt)| AvlTreeData {
                        digest,
                        tree_flags,
                        key_length,
                        value_length_opt,
                    },
                )
                .boxed()
        }
    }
    impl Arbitrary for AvlTreeFlags {
        type Strategy = BoxedStrategy<Self>;
        type Parameters = ();

        fn arbitrary_with(_: Self::Parameters) -> Self::Strategy {
            (any::<bool>(), any::<bool>(), any::<bool>())
                .prop_map(|(insert_allowed, update_allowed, remove_allowed)| {
                    AvlTreeFlags::new(insert_allowed, update_allowed, remove_allowed)
                })
                .boxed()
        }
    }
}

#[cfg(test)]
#[cfg(feature = "arbitrary")]
#[allow(clippy::unwrap_used)]
#[allow(clippy::panic)]
mod tests {
    use super::*;
    use crate::chain::ergo_box::ErgoBox;
    use crate::mir::constant::Literal;
    use crate::mir::expr::Expr;
    use crate::reference::Ref;
    use crate::serialization::sigma_serialize_roundtrip;
    use crate::types::stype::SType;
    use alloc::vec;

    use proptest::prelude::*;
    use sigma_test_util::force_any_val;

    proptest! {

        #[test]
        fn ser_roundtrip(v in any::<AvlTreeData>()) {
            let expr = Expr::Const(v.into());
            prop_assert_eq![sigma_serialize_roundtrip(&expr), expr];
        }

    }

    fn tree_with(digest: CollValue<'_>) -> AvlTreeValue<'_> {
        AvlTreeValue {
            digest,
            tree_flags: AvlTreeFlags::new(true, true, true),
            key_length: 1,
            value_length_opt: None,
        }
    }

    proptest! {

        // data becomes a runtime tree whose digest is the specialised collection of bytes, and
        // comes back as it was
        #[test]
        fn a_byte_digest_goes_to_a_runtime_tree_and_back(v in any::<AvlTreeData>()) {
            let tree = AvlTreeValue::from(v.clone());
            prop_assert!(matches!(tree.digest, CollValue::Bytes(_)));
            prop_assert_eq!(AvlTreeData::try_from(tree), Ok(v.clone()));
            prop_assert_eq!(Value::from(v.clone()).try_extract_into::<AvlTreeData>(), Ok(v));
        }

    }

    // sigmastate casts `digest.toArray` to bytes (`AvlTreeData.scala:75`,
    // `CAvlTreeVerifier.scala:59`): an array of bytes passes whatever the collection's class,
    // any other array does not, a tuple of bytes among them (its array is of objects)
    #[test]
    fn only_a_digest_of_bytes_is_data() {
        let generic_bytes = CollValue::generic(SType::SByte, vec![Value::Byte(1), Value::Byte(2)]);
        assert_eq!(
            AvlTreeData::try_from(tree_with(generic_bytes))
                .unwrap()
                .digest,
            vec![1u8, 2]
        );
        for digest in [
            CollValue::Ints(vec![1, 2].into()),
            CollValue::Booleans(vec![true].into()),
            CollValue::generic(SType::SLong, vec![Value::Long(1), Value::Long(2)]),
            CollValue::generic(SType::SAny, vec![Value::Byte(1), Value::Byte(2)]),
            CollValue::pairs(
                CollValue::Ints(vec![1].into()),
                CollValue::Ints(vec![2].into()),
            ),
        ] {
            assert!(
                AvlTreeData::try_from(tree_with(digest.clone())).is_err(),
                "{digest:?}"
            );
            let v = Value::AvlTree(Box::new(tree_with(digest.clone())));
            // the narrowing extraction refuses it, the runtime one hands it back
            assert!(v.clone().try_extract_into::<AvlTreeData>().is_err());
            assert_eq!(
                v.clone().try_extract_into::<AvlTreeValue>().unwrap().digest,
                digest
            );
            // and such a tree is not data
            assert!(Literal::try_from(v).is_err(), "{digest:?}");
        }
    }

    // a digest's items are converted with the collection's own conversion: a borrowed box
    // becomes an owned one
    #[test]
    fn to_static_converts_the_digests_items() {
        let b = force_any_val::<ErgoBox>();
        let digest = CollValue::generic(SType::SBox, vec![Value::CBox(Ref::Borrowed(&b))]);
        let v = Value::AvlTree(Box::new(tree_with(digest)));
        let owned: Value<'static> = v.to_static();
        assert_eq!(owned, v);
        match owned {
            Value::AvlTree(t) => match &t.digest {
                CollValue::Generic { items, .. } => {
                    assert!(matches!(items[0], Value::CBox(Ref::Arc(_))))
                }
                other => panic!("{other:?}"),
            },
            other => panic!("{other:?}"),
        }
    }
}
