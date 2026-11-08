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
    use ergotree_ir::mir::avl_tree_data::AvlTreeData;
    use ergotree_ir::mir::constant::Constant;
    use ergotree_ir::mir::create_avl_tree::CreateAvlTree;
    use ergotree_ir::mir::expr::Expr;
    use ergotree_ir::mir::option_get::OptionGet;
    use ergotree_ir::mir::unary_op::OneArgOpTryBuild;

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
}
