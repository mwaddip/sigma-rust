use alloc::boxed::Box;

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

/// Exponentiate op for GroupElement
#[derive(PartialEq, Eq, Debug, Clone)]
pub struct Exponentiate {
    /// GroupElement
    pub left: Box<Expr>,
    /// Expr of type BigInt
    pub right: Box<Expr>,
}

impl Exponentiate {
    /// Create new object, returns an error if any of the requirements failed
    pub fn new(left: Expr, right: Expr) -> Result<Self, InvalidArgumentError> {
        match (left.post_eval_tpe(), right.post_eval_tpe()) {
            (SType::SGroupElement, SType::SBigInt) => Ok(Exponentiate {
                left: left.into(),
                right: right.into(),
            }),
            (_, _) => Err(InvalidArgumentError(format!(
                "Exponentiate Expected: (SGroupElement, SBigInt), Actual: {0:?}",
                (left.tpe(), right.tpe())
            ))),
        }
    }

    /// Type: `GroupElement`, whatever the operands' types are. sigmastate's is that constant
    /// (`NotReadyValueGroupElement`, v6.0.6 `values.scala:755-757`, mixed in at `trees.scala:1028-1031`),
    /// and its parser builds the node over operands of any type, so the operands' types are not read here.
    pub fn tpe(&self) -> SType {
        SType::SGroupElement
    }
}

impl HasStaticOpCode for Exponentiate {
    const OP_CODE: OpCode = OpCode::EXPONENTIATE;
}

impl SigmaSerializable for Exponentiate {
    fn sigma_serialize<W: SigmaByteWrite>(&self, w: &mut W) -> SigmaSerializeResult {
        self.left.sigma_serialize(w)?;
        self.right.sigma_serialize(w)
    }

    fn sigma_parse<R: SigmaByteRead>(r: &mut R) -> Result<Self, SigmaParsingError> {
        let left = Expr::sigma_parse(r)?.into();
        let right = Expr::sigma_parse(r)?.into();
        Ok(Exponentiate { left, right })
    }
}

impl_traversable_expr!(Exponentiate, boxed left, boxed right);

#[cfg(feature = "arbitrary")]
#[allow(clippy::unwrap_used)]
mod arbitrary {
    use super::*;
    use crate::mir::expr::arbitrary::ArbExprParams;
    use proptest::prelude::*;

    impl Arbitrary for Exponentiate {
        type Strategy = BoxedStrategy<Self>;
        type Parameters = usize;

        fn arbitrary_with(args: Self::Parameters) -> Self::Strategy {
            (
                any_with::<Expr>(ArbExprParams {
                    tpe: SType::SGroupElement,
                    depth: args,
                }),
                any_with::<Expr>(ArbExprParams {
                    tpe: SType::SBigInt,
                    depth: args,
                }),
            )
                .prop_map(|(left, right)| Exponentiate::new(left, right).unwrap())
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

        #![proptest_config(ProptestConfig::with_cases(16))]

        #[test]
        fn ser_roundtrip(v in any::<Exponentiate>()) {
            let expr: Expr = v.into();
            prop_assert_eq![sigma_serialize_roundtrip(&expr), expr];
        }
    }
}
