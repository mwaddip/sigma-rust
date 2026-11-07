//! Block of statements ending with an expression

use alloc::boxed::Box;
use alloc::format;
use alloc::vec::Vec;

use super::expr::Expr;
use crate::has_opcode::HasStaticOpCode;
use crate::serialization::op_code::OpCode;
use crate::serialization::sigma_byte_reader::{SigmaByteRead, MAX_ARRAY_LENGTH};
use crate::serialization::sigma_byte_writer::SigmaByteWrite;
use crate::serialization::SigmaParsingError;
use crate::serialization::SigmaSerializable;
use crate::serialization::SigmaSerializeResult;
use crate::traversable::impl_traversable_expr;
use crate::types::stype::SType;

/** Block of statements ending with an expression
 * The order of ValDefs in the block is used to assign ids to ValUse(id) nodes
 * For all i: items(i).id == {number of ValDefs preceded in a graph} with respect to topological order.
 * Specific topological order doesn't really matter, what is important is to preserve semantic linkage
 * between ValUse(id) and ValDef with the corresponding id.
 * This convention allow to valid serializing ids because we always serializing and deserializing
 * in a fixed well defined order.
 */
#[derive(PartialEq, Eq, Debug, Clone)]
pub struct BlockValue {
    /// Statements
    pub items: Vec<Expr>,
    /// Resulting expr
    pub result: Box<Expr>,
}

impl BlockValue {
    /// Type
    pub fn tpe(&self) -> SType {
        self.result.tpe()
    }
}

impl HasStaticOpCode for BlockValue {
    const OP_CODE: OpCode = OpCode::BLOCK_VALUE;
}

impl SigmaSerializable for BlockValue {
    fn sigma_serialize<W: SigmaByteWrite>(&self, w: &mut W) -> SigmaSerializeResult {
        self.items.sigma_serialize(w)?;
        self.result.sigma_serialize(w)
    }

    fn sigma_parse<R: SigmaByteRead>(r: &mut R) -> Result<Self, SigmaParsingError> {
        // sigmastate reads the count with `getUIntExact`, allocates with `safeNewArray`, and
        // casts each item to `BlockItem` (a `ValDef`) as it reads it, before the next (v6.0.7
        // `BlockValueSerializer.scala:27-43`)
        let items_count = r.get_u32()?;
        if items_count as usize > MAX_ARRAY_LENGTH {
            return Err(SigmaParsingError::ArrayLengthExceeded(items_count as usize));
        }
        let mut items = Vec::new();
        for _ in 0..items_count {
            let item = Expr::sigma_parse(r)?;
            if !matches!(item, Expr::ValDef(_)) {
                return Err(SigmaParsingError::BlockItemNotValDef(format!("{item:?}")));
            }
            items.push(item);
        }
        let result = Expr::sigma_parse(r)?;
        Ok(BlockValue {
            items,
            result: Box::new(result),
        })
    }
}

impl_traversable_expr!(BlockValue, arr items, boxed result);

/// Arbitrary impl
#[cfg(feature = "arbitrary")]
mod arbitrary {
    use super::*;
    use crate::mir::val_def::{ValDef, ValId};
    use proptest::collection::vec;
    use proptest::prelude::*;

    impl Arbitrary for BlockValue {
        type Strategy = BoxedStrategy<Self>;
        type Parameters = ();

        fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
            // A block's items are `ValDef`s: sigmastate casts each to `BlockItem` as it reads
            // it (`BlockValueSerializer.scala:39`)
            let val_def = (any::<ValId>(), any::<Expr>()).prop_map(|(id, rhs)| {
                Expr::ValDef(
                    ValDef {
                        id,
                        tpe_args: Vec::new(),
                        rhs: Box::new(rhs),
                    }
                    .into(),
                )
            });
            (any::<Expr>(), vec(val_def, 0..10))
                .prop_map(|(result, items)| Self {
                    items,
                    result: Box::new(result),
                })
                .boxed()
        }
    }
}

#[cfg(test)]
#[cfg(feature = "arbitrary")]
#[allow(clippy::panic)]
mod tests {
    use crate::mir::block::BlockValue;
    use crate::mir::expr::Expr;
    use crate::serialization::sigma_serialize_roundtrip;
    use proptest::prelude::*;

    proptest! {

        #[test]
        fn ser_roundtrip(block in any::<BlockValue>()) {
            let e = Expr::BlockValue(block.into());
            prop_assert_eq![sigma_serialize_roundtrip(&e), e];
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod block_item_tests {
    use crate::ergo_tree::ErgoTree;
    use crate::serialization::{SigmaParsingError, SigmaSerializable};

    fn tree(hex: &str) -> Result<ErgoTree, SigmaParsingError> {
        ErgoTree::sigma_parse_bytes(&base16::decode(hex.as_bytes()).unwrap())
    }

    #[test]
    fn a_block_item_that_is_not_a_val_def_rejects() {
        // SANTA `tree_parse_acceptance` #35, #36: v0 BlockValue([Int 1], SigmaProp(true)),
        // unsized and size-flagged; #37: BlockValue([ValDef(1, Int 1)], SigmaProp(true))
        for hex in ["00d801040208d3", "0806d801040208d3"] {
            assert!(matches!(
                tree(hex),
                Err(SigmaParsingError::BlockItemNotValDef(_))
            ));
        }
        let bytes = base16::decode(b"00d801d601040208d3").unwrap();
        let parsed = ErgoTree::sigma_parse_bytes(&bytes).unwrap();
        assert_eq!(parsed.sigma_serialize_bytes().unwrap(), bytes);
    }
}
