//! AND conjunction for sigma propositions

use alloc::vec::Vec;

use crate::serialization::op_code::OpCode;
use crate::serialization::sigma_byte_reader::SigmaByteRead;
use crate::serialization::sigma_byte_writer::SigmaByteWrite;
use crate::serialization::SigmaParsingError;
use crate::serialization::SigmaSerializable;
use crate::serialization::SigmaSerializeResult;
use crate::traversable::impl_traversable_expr;
use crate::types::stype::SType;

use super::expr::Expr;
use super::expr::InvalidArgumentError;
use crate::has_opcode::HasStaticOpCode;

/// AND conjunction for sigma propositions
#[derive(PartialEq, Eq, Debug, Clone)]
pub struct SigmaAnd {
    /// Collection of SSigmaProp
    pub items: Vec<Expr>,
}

impl SigmaAnd {
    /// Create new object, returns an error if any of the requirements failed
    pub fn new(items: Vec<Expr>) -> Result<Self, InvalidArgumentError> {
        let item_types: Vec<SType> = items
            .clone()
            .into_iter()
            .map(|it| it.post_eval_tpe())
            .collect();
        if item_types
            .iter()
            .all(|tpe| matches!(tpe, SType::SSigmaProp))
        {
            Ok(Self { items })
        } else {
            Err(InvalidArgumentError(format!(
                "Sigma conjecture: expected all items be of type SSigmaProp, got {:?},\n items: {:?}",
                item_types, items
            )))
        }
    }

    /// Type
    pub fn tpe(&self) -> SType {
        SType::SSigmaProp
    }
}

impl HasStaticOpCode for SigmaAnd {
    const OP_CODE: OpCode = OpCode::SIGMA_AND;
}

impl SigmaSerializable for SigmaAnd {
    fn sigma_serialize<W: SigmaByteWrite>(&self, w: &mut W) -> SigmaSerializeResult {
        self.items.sigma_serialize(w)
    }

    fn sigma_parse<R: SigmaByteRead>(r: &mut R) -> Result<Self, SigmaParsingError> {
        Ok(Self::new(Vec::<Expr>::sigma_parse(r)?)?)
    }
}

impl_traversable_expr!(SigmaAnd, arr items);

/// Arbitrary impl
#[cfg(feature = "arbitrary")]
#[allow(clippy::unwrap_used)]
mod arbitrary {
    use super::*;
    use crate::mir::constant::Constant;
    use proptest::collection::vec;
    use proptest::prelude::*;

    impl Arbitrary for SigmaAnd {
        type Strategy = BoxedStrategy<Self>;
        type Parameters = ();

        fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
            vec(any_with::<Constant>(SType::SSigmaProp.into()), 2..5)
                .prop_map(|constants| Self {
                    items: constants
                        .into_iter()
                        .map(|c| c.into())
                        .collect::<Vec<Expr>>(),
                })
                .boxed()
        }
    }
}

#[cfg(test)]
#[cfg(feature = "arbitrary")]
#[allow(clippy::panic)]
mod tests {
    use super::*;
    use crate::mir::expr::Expr;
    use crate::serialization::sigma_serialize_roundtrip;
    use proptest::prelude::*;

    proptest! {

        #[test]
        fn ser_roundtrip(v in any::<SigmaAnd>()) {
            let expr: Expr = v.into();
            prop_assert_eq![sigma_serialize_roundtrip(&expr), expr];
        }

    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod bounds_tests {
    //! JVM parity: `SigmaTransformerSerializer` reads a `getUIntExact` count into `safeNewArray`
    //! and builds the node unchecked (`SigmaTransformerSerializer.scala:20-30`), so a node of
    //! 0 to 100000 items parses.
    use crate::ergo_tree::ErgoTree;
    use crate::serialization::SigmaParsingError;
    use crate::serialization::SigmaSerializable;
    use alloc::vec::Vec;

    /// An unsized tree whose root is `SigmaAnd(n × sigmaProp(true))`, the count as written
    fn tree(count: &[u8], items: usize) -> Vec<u8> {
        let mut bytes = [&[0x00, 0xea][..], count].concat();
        for _ in 0..items {
            bytes.extend_from_slice(&[0x08, 0xd3]);
        }
        bytes
    }

    #[test]
    fn no_items_and_256_items_parse() {
        // SANTA `tree_sigmaboolean_bounds` #5 and #7
        for bytes in [tree(&[0x00], 0), tree(&[0x80, 0x02], 256)] {
            let parsed = ErgoTree::sigma_parse_bytes(&bytes).unwrap();
            assert!(matches!(parsed, ErgoTree::Parsed(_)));
            assert_eq!(parsed.sigma_serialize_bytes().unwrap(), bytes);
        }
    }

    #[test]
    fn more_than_100000_items_do_not_parse() {
        // `safeNewArray` refuses the count, 100001, before an item is read. This passes before
        // the change too: the cap is `Vec<T>`'s, and it is the one bound the node keeps. The
        // error is the cap's: no item follows the count, so any reader fails here.
        assert_eq!(
            ErgoTree::sigma_parse_bytes(&tree(&[0xa1, 0x8d, 0x06], 0)),
            Err(SigmaParsingError::ArrayLengthExceeded(100001))
        );
    }
}
