use ergotree_ir::mir::create_provedlog::CreateProveDlog;
use ergotree_ir::mir::value::Value;
use ergotree_ir::sigma_protocol::sigma_boolean::ProveDlog;

use crate::eval::env::Env;
use crate::eval::operand::{eval_to, OperandClass};
use crate::eval::Context;
use crate::eval::EvalError;
use crate::eval::Evaluable;

impl Evaluable for CreateProveDlog {
    fn eval<'ctx>(
        &self,
        env: &mut Env<'ctx>,
        ctx: &Context<'ctx>,
    ) -> Result<Value<'ctx>, EvalError> {
        let value_v = eval_to(&self.input, OperandClass::GroupElement, env, ctx)?;
        ctx.add_jit_cost(10)?; // CreateProveDlog = Fixed(10)
        match value_v {
            Value::GroupElement(ecpoint) => {
                let prove_dlog = ProveDlog::new(*ecpoint);
                Ok(prove_dlog.into())
            }
            _ => Err(EvalError::UnexpectedValue(format!(
                "Expected CreateProveDlog input to be Value::GroupElement, got {0:?}",
                value_v
            ))),
        }
    }
}
