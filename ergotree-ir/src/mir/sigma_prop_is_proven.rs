//! `SigmaPropIsProven`: a node sigmastate parses and never evaluates

use alloc::boxed::Box;

use super::expr::Expr;
use super::expr::InvalidArgumentError;
use super::unary_op::OneArgOp;
use super::unary_op::OneArgOpTryBuild;
use crate::has_opcode::HasStaticOpCode;
use crate::serialization::op_code::OpCode;
use crate::types::stype::SType;

/// Whether a `SigmaProp` is proven (v6.0.7 `transformers.scala:321-329`). sigmastate reads it
/// (`SigmaPropIsProvenSerializer`) and defines no `eval` for it
#[derive(PartialEq, Eq, Debug, Clone)]
pub struct SigmaPropIsProven {
    /// `SigmaProp` input
    pub input: Box<Expr>,
}

impl SigmaPropIsProven {
    /// Type
    pub fn tpe(&self) -> SType {
        SType::SBoolean
    }
}

impl HasStaticOpCode for SigmaPropIsProven {
    const OP_CODE: OpCode = OpCode::SIGMA_PROP_IS_PROVEN;
}

impl OneArgOp for SigmaPropIsProven {
    fn input(&self) -> &Expr {
        &self.input
    }
    fn input_mut(&mut self) -> &mut Expr {
        &mut self.input
    }
}

impl OneArgOpTryBuild for SigmaPropIsProven {
    fn try_build(input: Expr) -> Result<Self, InvalidArgumentError> {
        input.check_post_eval_tpe(&SType::SSigmaProp)?;
        Ok(Self {
            input: input.into(),
        })
    }
}

#[cfg(feature = "arbitrary")]
#[allow(clippy::unwrap_used)]
mod arbitrary {
    use super::*;
    use crate::mir::expr::arbitrary::ArbExprParams;
    use proptest::prelude::*;

    impl Arbitrary for SigmaPropIsProven {
        type Strategy = BoxedStrategy<Self>;
        type Parameters = ();

        fn arbitrary_with(_: Self::Parameters) -> Self::Strategy {
            any_with::<Expr>(ArbExprParams {
                tpe: SType::SSigmaProp,
                depth: 0,
            })
            .prop_map(|input| Self::try_build(input).unwrap())
            .boxed()
        }
    }
}

#[cfg(test)]
#[cfg(feature = "arbitrary")]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::ergo_tree::ErgoTree;
    use crate::serialization::sigma_serialize_roundtrip;
    use crate::serialization::SigmaParsingError;
    use crate::serialization::SigmaSerializable;
    use proptest::prelude::*;

    /// `sigmaProp(SigmaPropIsProven(sigmaProp(true)))`, an unsized v0 tree as sigmastate writes it
    /// (SANTA `Unevaluated.node_kinds`; `true` is a constant, as `TrueLeaf` is a `ConstantNode`,
    /// `values.scala:771`): sigmastate reads it (v6.0.7 `SigmaPropIsProvenSerializer`)
    #[test]
    fn is_proven_parses_and_round_trips() {
        let bytes = base16::decode("00d1cfd10101").unwrap();
        let tree = ErgoTree::sigma_parse_bytes(&bytes).unwrap();
        assert!(matches!(tree, ErgoTree::Parsed(_)), "{tree:?}");
        assert_eq!(tree.sigma_serialize_bytes().unwrap(), bytes);
    }

    /// `SigmaPropIsProven`'s `val opType` reads its input's type when the node is built
    /// (`transformers.scala:321-325`): over `Filter(1, (x: Int) => true)`, whose type casts an
    /// `Int` to a collection, the build throws a `ClassCastException`, which rejects a tree
    /// whether or not it is size-flagged
    #[test]
    fn is_proven_reads_its_input_type() {
        let body = base16::decode("d1cfb50402d90101040101").unwrap();
        let unsized_tree = [&[0x00][..], &body].concat();
        let sized_tree = [&[0x0b, body.len() as u8][..], &body].concat();
        for bytes in [unsized_tree, sized_tree] {
            assert!(
                matches!(
                    ErgoTree::sigma_parse_bytes_no_fallback(&bytes),
                    Err(SigmaParsingError::TypeReadFailed(_))
                ),
                "{bytes:02x?}"
            );
        }
    }

    proptest! {

        #[test]
        fn ser_roundtrip(v in any::<SigmaPropIsProven>()) {
            let expr: Expr = v.into();
            prop_assert_eq![sigma_serialize_roundtrip(&expr), expr];
        }
    }
}
