use alloc::vec::Vec;
use ergotree_ir::chain::ergo_box::ErgoBox;
use ergotree_ir::mir::coll_value::CollValue;
use ergotree_ir::mir::extract_creation_info::ExtractCreationInfo;
use ergotree_ir::mir::value::Value;
use ergotree_ir::types::stype::SType;

use crate::eval::env::Env;
use crate::eval::Context;
use crate::eval::EvalError;
use crate::eval::Evaluable;

/// `Box.creationInfo` as sigmastate builds it: the creation height, and the creating transaction's id
/// and the output's index as a GENERIC `Coll[Byte]` (SANTA `Coll.producer_specialisation_witness`
/// H2), unlike the box's own byte fields
pub(crate) fn creation_info_value<'ctx>(b: &ErgoBox) -> Value<'ctx> {
    let (height, bytes) = b.creation_info();
    let bytes: Vec<Value> = bytes.into_iter().map(Value::Byte).collect();
    Value::pair(
        Value::Int(height),
        Value::Coll(CollValue::generic(SType::SByte, bytes)),
    )
}

impl Evaluable for ExtractCreationInfo {
    fn eval<'ctx>(
        &self,
        env: &mut Env<'ctx>,
        ctx: &Context<'ctx>,
    ) -> Result<Value<'ctx>, EvalError> {
        // the box (`evalTo[Box]`: an input of another class fails here), the node's cost,
        // then `creationInfo`, which is `getReg(3)` and so converts the box's registers
        // (`transformers.scala:519-523`, `CBox.scala:46-54`)
        let input_v = self.input.eval(env, ctx)?;
        let Value::CBox(b) = input_v else {
            return Err(EvalError::UnexpectedValue(format!(
                "Expected ExtractCreationInfo input to be Value::CBox, got {0:?}",
                input_v
            )));
        };
        ctx.add_jit_cost(16)?; // ExtractCreationInfo = Fixed(16)
        super::sbox::convert_registers(&b)?;
        Ok(creation_info_value(&b))
    }
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
#[cfg(feature = "arbitrary")]
mod tests {
    use crate::eval::test_util::eval_out;
    use crate::eval::Context;

    use ergotree_ir::mir::expr::Expr;
    use ergotree_ir::mir::global_vars::GlobalVars;
    use ergotree_ir::mir::unary_op::OneArgOpTryBuild;
    use sigma_test_util::force_any_val;

    use super::*;

    #[test]
    fn eval() {
        let expr: Expr = ExtractCreationInfo::try_build(GlobalVars::SelfBox.into())
            .unwrap()
            .into();
        let ctx = force_any_val::<Context>();
        let v = eval_out::<(i32, Vec<i8>)>(&expr, &ctx);
        assert_eq!(v, ctx.self_box.creation_info());
    }
}
