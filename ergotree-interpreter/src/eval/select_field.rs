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
        ctx.add_jit_cost(10)?; // SelectField = Fixed(10)
        let input_v = self.input.eval(env, ctx)?;
        // sigmastate selects only from a pair, its `Tuple2`; it holds every other tuple as a
        // `Coll[Any]` and raises a type error there (v6.0.7 `transformers.scala:297-307`)
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
    use crate::eval::test_util::try_eval_out_wo_ctx;
    use crate::eval::EvalError;
    use ergotree_ir::ergo_tree::ErgoTree;
    use ergotree_ir::mir::expr::Expr;
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
        assert!(eval("00d1938c60010402010402").is_err());
    }

    #[test]
    fn selecting_from_a_three_item_tuple_fails() {
        // the same for any tuple but a pair (SANTA 7c, `(1, 2, 3)._1 == 1`)
        assert!(eval("00d1938c48040404020406010402").is_err());
    }

    #[test]
    fn selecting_from_a_pair_still_works() {
        // `(1, 2)._1 == 1`
        assert!(eval("00d1938c580204010402").unwrap());
    }
}
