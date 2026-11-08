//! The nodes sigmastate parses but never evaluates. None of them overrides `Value.eval`, whose
//! default throws (v6.0.7 `values.scala:101-102`), so a spend that reaches one fails; the readers
//! stay because outputs are parsed without being executed (`ValueSerializer.scala:141-147`).

use alloc::format;

use crate::eval::EvalError;

/// The error of evaluating `node`, which sigmastate parses and never evaluates. It is returned
/// before any child of the node is evaluated, as sigmastate throws at the node itself.
pub(crate) fn not_evaluated(node: &str) -> EvalError {
    EvalError::UnexpectedExpr(format!("{node} is not supported for evaluation"))
}

#[allow(clippy::unwrap_used, clippy::panic)]
#[cfg(test)]
#[cfg(feature = "arbitrary")]
mod tests {
    use crate::eval::test_util::try_eval_out_wo_ctx;
    use alloc::string::ToString;
    use alloc::vec;
    use ergotree_ir::bigint256::BigInt256;
    use ergotree_ir::mir::avl_tree_data::AvlTreeData;
    use ergotree_ir::mir::bin_op::{BinOp, BinOpKind, BitOp, LogicalOp, RelationOp};
    use ergotree_ir::mir::bit_inversion::BitInversion;
    use ergotree_ir::mir::constant::Constant;
    use ergotree_ir::mir::create_avl_tree::CreateAvlTree;
    use ergotree_ir::mir::expr::Expr;
    use ergotree_ir::mir::mod_q::{ModQ, ModQArithOp, ModQArithOpKind};
    use ergotree_ir::mir::option_get::OptionGet;
    use ergotree_ir::mir::unary_op::OneArgOpTryBuild;

    fn bin_op(kind: BinOpKind, left: Expr, right: Expr) -> Expr {
        BinOp {
            kind,
            left: left.into(),
            right: right.into(),
        }
        .into()
    }

    /// The error a node sigmastate does not evaluate fails with
    fn assert_not_evaluated<T: core::fmt::Debug>(
        res: Result<T, crate::eval::EvalError>,
        node: &str,
    ) {
        let err = res.unwrap_err().to_string();
        assert!(
            err.contains(&alloc::format!("{node} is not supported for evaluation")),
            "{err}"
        );
    }

    /// The node fails before its children are evaluated: a child that would fail on its own
    /// (`OptionGet` of `None`) does not decide the error
    #[test]
    fn create_avl_tree_does_not_evaluate() {
        let failing_flags: Expr = OptionGet::try_build(Constant::from(None::<i8>).into())
            .unwrap()
            .into();
        let expr: Expr = CreateAvlTree::new(
            failing_flags,
            Constant::from(vec![0u8; 33]).into(),
            Constant::from(32i32).into(),
            Constant::from(None::<i32>).into(),
        )
        .unwrap()
        .into();
        assert_not_evaluated(try_eval_out_wo_ctx::<AvlTreeData>(&expr), "CreateAvlTree");
    }

    /// sigmastate defines no `eval` for `BitOp` and `BitInversion` (`trees.scala:899-916`)
    #[test]
    fn bitwise_operations_do_not_evaluate() {
        for op in [
            BitOp::BitOr,
            BitOp::BitAnd,
            BitOp::BitXor,
            BitOp::BitShiftRight,
            BitOp::BitShiftLeft,
            BitOp::BitShiftRightZeroed,
        ] {
            let expr = bin_op(op.into(), 6i32.into(), 1i32.into());
            assert_not_evaluated(try_eval_out_wo_ctx::<i32>(&expr), "BitOp");
        }
        let expr: Expr = BitInversion {
            input: Expr::from(6i32).into(),
        }
        .into();
        assert_not_evaluated(try_eval_out_wo_ctx::<i32>(&expr), "BitInversion");
    }

    /// `&&` does not evaluate its right side when the left is false, so a bitwise node there
    /// never fails the spend, as in sigmastate (`BinAnd.eval`)
    #[test]
    fn an_untaken_bitwise_operation_does_not_fail() {
        let bit_or = bin_op(BitOp::BitOr.into(), 1i32.into(), 2i32.into());
        let eq = bin_op(RelationOp::Eq.into(), bit_or, 3i32.into());
        let expr = bin_op(LogicalOp::And.into(), false.into(), eq);
        assert!(!try_eval_out_wo_ctx::<bool>(&expr).unwrap());
    }

    /// sigmastate defines no `eval` for `ModQ` and `ModQArithOp` (`trees.scala:954-975`)
    #[test]
    fn mod_q_family_does_not_evaluate() {
        let one = || Expr::from(BigInt256::from(1i64));
        let expr: Expr = ModQ {
            input: one().into(),
        }
        .into();
        assert_not_evaluated(try_eval_out_wo_ctx::<BigInt256>(&expr), "ModQ");
        for kind in [ModQArithOpKind::PlusModQ, ModQArithOpKind::MinusModQ] {
            let expr: Expr = ModQArithOp {
                left: one().into(),
                right: one().into(),
                kind,
            }
            .into();
            assert_not_evaluated(try_eval_out_wo_ctx::<BigInt256>(&expr), "ModQArithOp");
        }
    }
}
