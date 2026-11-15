use ergotree_ir::mir::constant::TryExtractInto;
use ergotree_ir::mir::logical_not::LogicalNot;
use ergotree_ir::mir::value::Value;

use crate::eval::env::Env;
use crate::eval::operand::{eval_to, OperandClass};
use crate::eval::Context;
use crate::eval::EvalError;
use crate::eval::Evaluable;

impl Evaluable for LogicalNot {
    fn eval<'ctx>(
        &self,
        env: &mut Env<'ctx>,
        ctx: &Context<'ctx>,
    ) -> Result<Value<'ctx>, EvalError> {
        let input_v = eval_to(&self.input, OperandClass::Boolean, env, ctx)?;
        let input_v_bool = input_v.try_extract_into::<bool>()?;
        ctx.add_jit_cost(15)?; // LogicalNot = Fixed(15)
        Ok((!input_v_bool).into())
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::eval::test_util::eval_out_wo_ctx;
    use ergotree_ir::mir::expr::Expr;

    fn check(input: bool) -> bool {
        let expr: Expr = LogicalNot {
            input: Expr::Const(input.into()).into(),
        }
        .into();
        eval_out_wo_ctx::<bool>(&expr)
    }

    #[test]
    fn eval() {
        assert!(!check(true));
        assert!(check(false));
    }
}
