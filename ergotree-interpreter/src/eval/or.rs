use ergotree_ir::mir::or::Or;
use ergotree_ir::mir::value::Value;

use crate::eval::and::booleans_read_until;
use crate::eval::env::Env;
use crate::eval::Context;
use crate::eval::EvalError;
use crate::eval::Evaluable;

impl Evaluable for Or {
    fn eval<'ctx>(
        &self,
        env: &mut Env<'ctx>,
        ctx: &Context<'ctx>,
    ) -> Result<Value<'ctx>, EvalError> {
        let input_v = self.input.eval(env, ctx)?;
        // the loop stops at the first `true`, and the cost is of the items it read
        // (`trees.scala:199-213`)
        let (decided, read) = booleans_read_until(&input_v, true)?;
        ctx.add_per_item_jit_cost(5, 5, 64, read)?;
        Ok(decided.into())
    }
}

#[allow(clippy::panic)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::test_util::eval_out;
    use ergotree_ir::chain::context::Context;
    use ergotree_ir::mir::expr::Expr;
    use proptest::collection;
    use proptest::prelude::*;
    use sigma_test_util::force_any_val;

    proptest! {

        #[test]
        fn eval(bools in collection::vec(any::<bool>(), 0..10)) {
            let expr: Expr = Or {input: Expr::Const(bools.clone().into()).into()}.into();
            let ctx = force_any_val::<Context>();
            let res = eval_out::<bool>(&expr, &ctx);
            prop_assert_eq!(res, bools.iter().any(|b| *b));
        }
    }
}
