use ergotree_ir::mir::select_field::SelectField;
use ergotree_ir::mir::value::Value;

use crate::eval::env::Env;
use crate::eval::Context;
use crate::eval::EvalError;
use crate::eval::Evaluable;

impl Evaluable for SelectField {
    fn eval<'ctx>(
        &self,
        env: &mut Env<'ctx>,
        ctx: &Context<'ctx>,
    ) -> Result<Value<'ctx>, EvalError> {
        // sigmastate evaluates the input, then adds the cost (`SelectField` = Fixed(10)), then
        // selects (v6.0.6 `transformers.scala:297-307`): an input that throws leaves the node
        // uncharged. It selects only from a pair, its `Tuple2`; it holds every other tuple as a
        // `Coll[Any]` and raises a type error there
        let input_v = self.input.eval(env, ctx)?;
        ctx.add_jit_cost(10)?;
        match input_v {
            Value::Pair(pair) => match (self.field_index.zero_based_index(), *pair) {
                (0, (first, _)) => Ok(first),
                (1, (_, second)) => Ok(second),
                (_, pair) => Err(EvalError::NotFound(format!(
                    "SelectField field index is out of bounds. Index: {0:?}, pair: {1:?}",
                    self.field_index, pair
                ))),
            },
            _ => Err(EvalError::UnexpectedValue(format!(
                "expected SelectField input to be a pair, got: {0:?}",
                input_v
            ))),
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use crate::eval::test_util::{assert_trees, eval_tree_err, innermost, try_eval_out_wo_ctx};
    use crate::eval::EvalError;
    use ergotree_ir::ergo_tree::ErgoTree;
    use ergotree_ir::mir::expr::Expr;
    use ergotree_ir::mir::value::Value;
    use ergotree_ir::serialization::SigmaSerializable;

    /// `c._1 == 1` under `sigmaProp`, from SANTA's bytes: the comparison's verdict
    fn eval(hex: &str) -> Result<bool, EvalError> {
        let tree = ErgoTree::sigma_parse_bytes(&base16::decode(hex).unwrap()).unwrap();
        let Expr::BoolToSigmaProp(p) = tree.proposition().unwrap() else {
            panic!("{hex}: not a sigmaProp root")
        };
        try_eval_out_wo_ctx::<bool>(&p.input)
    }

    #[test]
    fn selecting_from_a_one_item_tuple_fails() {
        // sigmastate holds a `(Int)` constant as a `Coll[Any]`, and `SelectField.eval` takes only a
        // `Tuple2` (v6.0.7 `transformers.scala:297-307`): "Invalid type returned by evaluator"
        // (SANTA case 1)
        let err = eval("00d1938c60010402010402").unwrap_err();
        assert!(
            matches!(innermost(&err), EvalError::UnexpectedValue(_)),
            "{err:?}"
        );
    }

    #[test]
    fn selecting_from_a_three_item_tuple_fails() {
        // the same for any tuple but a pair (SANTA 7c, `(1, 2, 3)._1 == 1`)
        let err = eval("00d1938c48040404020406010402").unwrap_err();
        assert!(
            matches!(innermost(&err), EvalError::UnexpectedValue(_)),
            "{err:?}"
        );
    }

    /// sigmastate evaluates the input, THEN adds the cost, then selects from a `Tuple2` or
    /// raises its type error (v6.0.6 `transformers.scala:297-307`). Measured on the sigma-state
    /// 6.0.7 jars (a profiling evaluator on the expression's bytes, its accumulator read also at
    /// a throw)
    #[test]
    fn select_field_adds_its_cost_after_its_input() {
        // S1 `(1 / 0, 2)._1`: the input throws (an `ArithmeticException` at 25 on the JVM),
        // and neither the `Tuple` node nor the selection is charged
        let (e, stopped_at) = eval_tree_err("0b0b8c86029d04020400040401");
        assert!(matches!(e, EvalError::ArithmeticException(_)), "{e:?}");
        assert_eq!(stopped_at, 25);
        // S2 `(1, 2, 3)._1`, the constant: not a pair, refused AFTER the cost (the JVM's type
        // error at 15)
        let (e, stopped_at) = eval_tree_err("0b098c4804040402040601");
        assert!(matches!(e, EvalError::UnexpectedValue(_)), "{e:?}");
        assert_eq!(stopped_at, 15);
        // S3 `(1, 2)._1`
        assert_trees(&[("0b088c86020402040401", Some((Value::Int(1), 35)))]);
    }

    #[test]
    fn selecting_from_a_pair_still_works() {
        // `(1, 2)._1 == 1`
        assert!(eval("00d1938c580204010402").unwrap());
    }
}
