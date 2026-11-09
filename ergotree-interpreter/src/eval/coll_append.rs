use ergotree_ir::mir::coll_append::Append;
use ergotree_ir::mir::value::Value;

use crate::eval::env::Env;
use crate::eval::Context;
use crate::eval::EvalError;
use crate::eval::Evaluable;

impl Evaluable for Append {
    fn eval<'ctx>(
        &self,
        env: &mut Env<'ctx>,
        ctx: &Context<'ctx>,
    ) -> Result<Value<'ctx>, EvalError> {
        let input_v = self.input.eval(env, ctx)?;
        let col2_v = self.col_2.eval(env, ctx)?;
        let (Value::Coll(left), Value::Coll(right)) = (&input_v, &col2_v) else {
            return Err(EvalError::UnexpectedValue(format!(
                "Append: expected two collections, got: {input_v:?} and {col2_v:?}"
            )));
        };
        // charged before the append, as `Append.eval`'s `addSeqCost` does (v6.0.7
        // `transformers.scala:67-68`)
        ctx.add_per_item_jit_cost(20, 2, 100, (left.len() + right.len()) as u32)?;
        Ok(Value::Coll(super::coll_shape::append(left, right)?))
    }
}

#[allow(clippy::unwrap_used)]
#[allow(clippy::panic)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::test_util::eval_out_wo_ctx;
    use ergotree_ir::mir::collection::Collection;
    use ergotree_ir::mir::constant::Constant;
    use ergotree_ir::mir::expr::Expr;
    use ergotree_ir::types::stype::SType;

    #[test]
    fn test_append_basic_numbers() {
        for (test_inp, test_col2, expected_output) in [
            (
                vec![1, 2, 3, 4],
                vec![5, 6, 7, 8],
                vec![1, 2, 3, 4, 5, 6, 7, 8],
            ),
            (vec![], vec![], vec![]),
            (vec![], vec![1], vec![1]),
            (vec![], vec![2, 2], vec![2, 2]),
            (vec![1], vec![], vec![1]),
            (vec![2, 2], vec![], vec![2, 2]),
            (vec![1], vec![2, 2], vec![1, 2, 2]),
        ]
        .iter()
        {
            let in_expr = Expr::from(test_inp.clone());
            let col2_expr = Expr::from(test_col2.clone());
            let append_expr = Expr::from(Append::new(in_expr, col2_expr).unwrap());
            let append_eval: Vec<i32> = eval_out_wo_ctx(&append_expr);
            assert_eq!(append_eval, *expected_output);
            match eval_out_wo_ctx::<Value>(&append_expr) {
                Value::Coll(coll) => assert_eq!(coll.elem_tpe(), SType::SInt),
                _ => panic!("fail"),
            }
        }
    }

    #[test]
    fn append_byte_array_and_byte() {
        let byte_coll: Constant = vec![1i8, 2i8].into();
        let byte: Expr = Expr::Collection(Collection::new(SType::SByte, vec![3i8.into()]).unwrap());
        let expr: Expr = Expr::Append(Append::new(byte_coll.into(), byte).unwrap().into());
        assert_eq!(eval_out_wo_ctx::<Vec<i8>>(&expr), vec![1i8, 2, 3]);
    }
}
