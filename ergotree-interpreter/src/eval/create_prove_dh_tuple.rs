use ergo_chain_types::EcPoint;
use ergotree_ir::mir::constant::TryExtractInto;
use ergotree_ir::mir::create_prove_dh_tuple::CreateProveDhTuple;
use ergotree_ir::mir::value::Value;
use ergotree_ir::sigma_protocol::sigma_boolean::ProveDhTuple;

use crate::eval::env::Env;
use crate::eval::operand::{eval_to, OperandClass};
use crate::eval::Context;
use crate::eval::EvalError;
use crate::eval::Evaluable;

impl Evaluable for CreateProveDhTuple {
    fn eval<'ctx>(
        &self,
        env: &mut Env<'ctx>,
        ctx: &Context<'ctx>,
    ) -> Result<Value<'ctx>, EvalError> {
        let g = eval_to(&self.g, OperandClass::GroupElement, env, ctx)?;
        let h = eval_to(&self.h, OperandClass::GroupElement, env, ctx)?;
        let u = eval_to(&self.u, OperandClass::GroupElement, env, ctx)?;
        let v = eval_to(&self.v, OperandClass::GroupElement, env, ctx)?;
        ctx.add_jit_cost(20)?; // CreateProveDHTuple = Fixed(20)
        Ok(ProveDhTuple::new(
            g.try_extract_into::<EcPoint>()?,
            h.try_extract_into::<EcPoint>()?,
            u.try_extract_into::<EcPoint>()?,
            v.try_extract_into::<EcPoint>()?,
        )
        .into())
    }
}
