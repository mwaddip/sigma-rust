//! `ModQ`, `PlusModQ` and `MinusModQ`: nodes sigmastate parses and never evaluates. Its readers
//! stay "because their removal may break deserialization of transaction, when for example ModQ
//! operation happen to be in one of the outputs" (v6.0.7 `ValueSerializer.scala:141-147`).

use alloc::boxed::Box;

use super::expr::Expr;
use super::expr::InvalidArgumentError;
use super::unary_op::OneArgOp;
use super::unary_op::OneArgOpTryBuild;
use crate::has_opcode::HasOpCode;
use crate::has_opcode::HasStaticOpCode;
use crate::serialization::op_code::OpCode;
use crate::serialization::sigma_byte_reader::SigmaByteRead;
use crate::serialization::SigmaParsingError;
use crate::serialization::SigmaSerializable;
use crate::traversable::impl_traversable_expr;
use crate::types::stype::SType;

/// `ModQ`: a `BigInt` modulo the order of the group (`trees.scala:954-963`), never evaluated
#[derive(PartialEq, Eq, Debug, Clone)]
pub struct ModQ {
    /// `BigInt` input
    pub input: Box<Expr>,
}

impl ModQ {
    /// Type
    pub fn tpe(&self) -> SType {
        SType::SBigInt
    }
}

impl HasStaticOpCode for ModQ {
    const OP_CODE: OpCode = OpCode::MOD_Q;
}

impl OneArgOp for ModQ {
    fn input(&self) -> &Expr {
        &self.input
    }
    fn input_mut(&mut self) -> &mut Expr {
        &mut self.input
    }
}

impl OneArgOpTryBuild for ModQ {
    fn try_build(input: Expr) -> Result<Self, InvalidArgumentError> {
        input.check_post_eval_tpe(&SType::SBigInt)?;
        Ok(Self {
            input: input.into(),
        })
    }
}

/// Which of the two modular operations a [`ModQArithOp`] is
#[derive(PartialEq, Eq, Debug, Clone, Copy)]
#[cfg_attr(feature = "arbitrary", derive(proptest_derive::Arbitrary))]
pub enum ModQArithOpKind {
    /// `PlusModQ`
    PlusModQ,
    /// `MinusModQ`
    MinusModQ,
}

/// `PlusModQ` / `MinusModQ` over two `BigInt`s (`trees.scala:965-975`), never evaluated
#[derive(PartialEq, Eq, Debug, Clone)]
pub struct ModQArithOp {
    /// `BigInt` left operand
    pub left: Box<Expr>,
    /// `BigInt` right operand
    pub right: Box<Expr>,
    /// The operation
    pub kind: ModQArithOpKind,
}

impl ModQArithOp {
    /// Creates the node, checking that both operands are `BigInt`
    pub fn new(
        left: Expr,
        right: Expr,
        kind: ModQArithOpKind,
    ) -> Result<Self, InvalidArgumentError> {
        left.check_post_eval_tpe(&SType::SBigInt)?;
        right.check_post_eval_tpe(&SType::SBigInt)?;
        Ok(Self {
            left: left.into(),
            right: right.into(),
            kind,
        })
    }

    /// Type
    pub fn tpe(&self) -> SType {
        SType::SBigInt
    }

    /// sigmastate reads two values and builds the node with no check of them
    /// (`ModQArithOpSerializer.scala`; its casts are to an erased `Value[T]`)
    pub(crate) fn sigma_parse_operands<R: SigmaByteRead>(
        kind: ModQArithOpKind,
        r: &mut R,
    ) -> Result<Self, SigmaParsingError> {
        let left = Expr::sigma_parse(r)?;
        let right = Expr::sigma_parse(r)?;
        Ok(Self {
            left: left.into(),
            right: right.into(),
            kind,
        })
    }
}

impl HasOpCode for ModQArithOp {
    fn op_code(&self) -> OpCode {
        match self.kind {
            ModQArithOpKind::PlusModQ => OpCode::PLUS_MOD_Q,
            ModQArithOpKind::MinusModQ => OpCode::MINUS_MOD_Q,
        }
    }
}

impl_traversable_expr!(ModQArithOp, boxed left, boxed right);

#[cfg(feature = "arbitrary")]
#[allow(clippy::unwrap_used)]
mod arbitrary {
    use super::*;
    use crate::mir::expr::arbitrary::ArbExprParams;
    use proptest::prelude::*;

    fn big_int() -> BoxedStrategy<Expr> {
        any_with::<Expr>(ArbExprParams {
            tpe: SType::SBigInt,
            depth: 0,
        })
    }

    impl Arbitrary for ModQ {
        type Strategy = BoxedStrategy<Self>;
        type Parameters = ();

        fn arbitrary_with(_: Self::Parameters) -> Self::Strategy {
            big_int()
                .prop_map(|input| Self::try_build(input).unwrap())
                .boxed()
        }
    }

    impl Arbitrary for ModQArithOp {
        type Strategy = BoxedStrategy<Self>;
        type Parameters = ();

        fn arbitrary_with(_: Self::Parameters) -> Self::Strategy {
            (big_int(), big_int(), any::<ModQArithOpKind>())
                .prop_map(|(left, right, kind)| Self::new(left, right, kind).unwrap())
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
    use crate::serialization::SigmaSerializable;
    use proptest::prelude::*;

    /// `sigmaProp(ModQ(1) == 1)`, `sigmaProp(PlusModQ(1, 1) == 1)` and
    /// `sigmaProp(MinusModQ(1, 1) == 1)` over `BigInt` constants, unsized v0 trees: sigmastate
    /// reads them (v6.0.7 `ValueSerializer.scala:141-147`)
    #[test]
    fn mod_q_trees_parse_and_round_trip() {
        for hex in [
            "00d193e7060101060101",
            "00d193e8060101060101060101",
            "00d193e9060101060101060101",
        ] {
            let bytes = base16::decode(hex).unwrap();
            let tree = ErgoTree::sigma_parse_bytes(&bytes).unwrap();
            assert!(matches!(tree, ErgoTree::Parsed(_)), "{hex}: {tree:?}");
            assert_eq!(tree.sigma_serialize_bytes().unwrap(), bytes, "{hex}");
        }
    }

    proptest! {

        #[test]
        fn mod_q_ser_roundtrip(v in any::<ModQ>()) {
            let expr: Expr = v.into();
            prop_assert_eq![sigma_serialize_roundtrip(&expr), expr];
        }

        #[test]
        fn mod_q_arith_op_ser_roundtrip(v in any::<ModQArithOp>()) {
            let expr: Expr = v.into();
            prop_assert_eq![sigma_serialize_roundtrip(&expr), expr];
        }
    }
}
