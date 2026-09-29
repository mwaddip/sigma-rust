use core::convert::TryInto;

use alloc::boxed::Box;
use ergotree_ir::chain::ergo_box::ErgoBox;
use ergotree_ir::mir::constant::TryExtractInto;
use ergotree_ir::mir::extract_reg_as::ExtractRegisterAs;
use ergotree_ir::mir::value::Value;
use ergotree_ir::reference::Ref;

use crate::eval::env::Env;
use crate::eval::sbox::get_register_script_value;
use crate::eval::Context;
use crate::eval::EvalError;
use crate::eval::Evaluable;

impl Evaluable for ExtractRegisterAs {
    fn eval<'ctx>(
        &self,
        env: &mut Env<'ctx>,
        ctx: &Context<'ctx>,
    ) -> Result<Value<'ctx>, EvalError> {
        ctx.add_jit_cost(50)?; // ExtractRegisterAs = Fixed(50)
        let ir_box = self
            .input
            .eval(env, ctx)?
            .try_extract_into::<Ref<'_, ErgoBox>>()?;
        let id = self.register_id.try_into().map_err(|e| {
            EvalError::RegisterIdOutOfBounds(format!(
                "register index {} is out of bounds: {:?} ",
                self.register_id, e
            ))
        })?;
        match get_register_script_value(&ir_box, id)? {
            Some(value) if value.tpe.as_ref() == Some(&*self.elem_tpe) => {
                Ok(Value::Opt(Some(Box::new(value.v.into()))))
            }
            Some(value) => Err(EvalError::UnexpectedValue(format!(
                "Expected register {id} to be of type {}, got {:?}",
                self.elem_tpe, value.tpe
            ))),
            None => Ok(Value::Opt(None)),
        }
    }
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
#[cfg(feature = "arbitrary")]
mod tests {
    use super::*;
    use crate::eval::test_util::{eval_out, try_eval_out};
    use ergotree_ir::chain::context::Context;
    use ergotree_ir::mir::expr::Expr;
    use ergotree_ir::mir::global_vars::GlobalVars;
    use ergotree_ir::mir::option_get::OptionGet;
    use ergotree_ir::mir::unary_op::OneArgOpTryBuild;
    use ergotree_ir::types::stype::SType;
    use sigma_test_util::force_any_val;

    #[test]
    fn eval_box_get_reg_r0() {
        let get_reg_expr: Expr = ExtractRegisterAs::new(
            GlobalVars::SelfBox.into(),
            0,
            SType::SOption(SType::SLong.into()),
        )
        .unwrap()
        .into();
        let option_get_expr: Expr = OptionGet::try_build(get_reg_expr).unwrap().into();
        let ctx = force_any_val::<Context>();
        let v = eval_out::<i64>(&option_get_expr, &ctx);
        assert_eq!(v, ctx.self_box.value.as_i64());
    }

    #[test]
    fn eval_box_get_reg_r0_wrong_type() {
        let get_reg_expr: Expr = ExtractRegisterAs::new(
            GlobalVars::SelfBox.into(),
            0,
            SType::SOption(SType::SInt.into()), // R0 (value) is long, but we're expecting int
        )
        .unwrap()
        .into();
        let option_get_expr: Expr = OptionGet::try_build(get_reg_expr).unwrap().into();
        let ctx = force_any_val::<Context>();
        assert!(try_eval_out::<Value>(&option_get_expr, &ctx).is_err());
    }
}

#[cfg(test)]
#[cfg(feature = "arbitrary")]
#[allow(clippy::unwrap_used)]
pub(crate) mod register_conversion_tests {
    //! JVM parity: sigmastate reads a register through `CBox.registers`, a lazy val that
    //! converts every register of the box on the first read (v6.0.6 `CBox.scala:28`,
    //! `:77-92`), so a register that does not convert fails a read of any register of the box.
    use super::*;
    use crate::eval::test_util::try_eval_out;
    use ergotree_ir::chain::context::Context;
    use ergotree_ir::chain::ergo_box::{NonMandatoryRegisters, RegisterValue};
    use ergotree_ir::mir::constant::Constant;
    use ergotree_ir::mir::expr::Expr;
    use ergotree_ir::mir::global_vars::GlobalVars;
    use ergotree_ir::types::stype::SType;
    use sigma_test_util::force_any_val;

    /// SELF with R4 = `r4` and R5 = Int 1
    pub(crate) fn ctx_with_r4(r4: &str) -> Context<'static> {
        let regs = NonMandatoryRegisters::try_from(vec![
            RegisterValue::sigma_parse_bytes(&base16::decode(r4).unwrap()),
            Constant::from(1i32).into(),
        ])
        .unwrap();
        let b = force_any_val::<ErgoBox>().with_additional_registers(regs);
        Context {
            self_box: Box::leak(Box::new(b)),
            ..force_any_val::<Context>()
        }
    }

    #[test]
    fn a_register_read_converts_every_register_of_the_box() {
        // SANTA V10: R4 = `Tuple(1, HEIGHT)` fails the read of R5; its twin, R4 = `Tuple(1, 2)`,
        // does not; and C1 as R4, a `Coll[(Int, Int)]` holding a tuple expression, fails it
        let r5: Expr = ExtractRegisterAs::new(
            GlobalVars::SelfBox.into(),
            5,
            SType::SOption(SType::SInt.into()),
        )
        .unwrap()
        .into();
        for (r4, reads) in [
            ("86020402a3", false),
            ("860204020404", true),
            ("830158860204020404", false),
        ] {
            let res = try_eval_out::<Option<i32>>(&r5, &ctx_with_r4(r4));
            assert_eq!(res.is_ok(), reads, "{r4}");
        }
    }
}
