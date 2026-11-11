use alloc::boxed::Box;
use ergotree_ir::mir::constant::TryExtractFromError;
use ergotree_ir::mir::get_var::GetVar;
use ergotree_ir::mir::value::Value;

use crate::eval::env::Env;
use crate::eval::Context;
use crate::eval::EvalError;
use crate::eval::Evaluable;

impl Evaluable for GetVar {
    fn eval<'ctx>(&self, _env: &mut Env, ctx: &Context<'ctx>) -> Result<Value<'ctx>, EvalError> {
        ctx.add_jit_cost(10)?; // GetVar = Fixed(10)

        // `CContext.getVar` (v6.0.6 `CContext.scala:60-74`): the variable as the script context
        // holds it, when its type is the one asked for, and an error otherwise
        let value = match ctx.extension.values.get(&self.var_id) {
            None => return Ok(Value::Opt(None)),
            Some(value) => value,
        };
        let script_value = value.to_script_value()?;
        if script_value.tpe == self.var_tpe {
            Ok(Value::Opt(Some(Box::new(script_value.v))))
        } else {
            Err(TryExtractFromError(format!(
                "GetVar: expected extension value id {} to have type {:?}, found {:?} in context extension map {}",
                self.var_id, self.var_tpe, value, ctx.extension
            ))
            .into())
        }
    }
}

#[cfg(test)]
#[cfg(feature = "arbitrary")]
pub mod tests {
    use super::*;
    use crate::eval::test_util::{eval_out, try_eval_out};
    use ergotree_ir::chain::context::arbitrary::DummyContextExtensionProvider;
    use ergotree_ir::chain::context::Context;
    use ergotree_ir::chain::context_extension::ContextExtension;
    use ergotree_ir::mir::expr::Expr;
    use ergotree_ir::types::stype::SType;
    use sigma_test_util::force_any_val;

    const VAR_IDX: u8 = 3;
    const VAR_VAL: i32 = 123;

    /// Prepare context with single extension variable
    pub fn prepare_context() -> Context<'static> {
        let mut ctx = force_any_val::<Context>();
        ctx.extension = Box::leak(
            ContextExtension {
                values: [(VAR_IDX, VAR_VAL.into())].into_iter().collect(),
            }
            .into(),
        );
        ctx.extension_provider =
            Box::leak(DummyContextExtensionProvider(vec![ctx.extension.clone()]).into());
        ctx
    }

    #[test]
    #[allow(clippy::unwrap_used)]
    fn a_tuple_of_three_reads_as_a_collection() {
        // SANTA's tuple-values reply, G1: SizeOf(getVar[(Int, Int, Int)](1).get) is 3 @39 for a
        // variable holding the constant `(1, 2, 3)`
        use crate::eval::test_util::eval_v3_body_in;
        use ergotree_ir::chain::evaluated_value::EvaluatedValue;
        use ergotree_ir::mir::value::Value;
        use ergotree_ir::serialization::SigmaSerializable;
        let triple =
            EvaluatedValue::sigma_parse_bytes(&base16::decode("48040404020406").unwrap()).unwrap();
        let mut ctx = force_any_val::<Context>();
        ctx.extension = Box::leak(
            ContextExtension {
                values: [(1u8, triple)].into_iter().collect(),
            }
            .into(),
        );
        ctx.extension_provider =
            Box::leak(DummyContextExtensionProvider(vec![ctx.extension.clone()]).into());
        let (_, res, cost) = eval_v3_body_in(ctx, "0b08b1e4e30148040404");
        assert_eq!(res.unwrap(), Value::Int(3));
        assert_eq!(cost, 39);
    }

    /// Normal evaluation
    #[test]
    fn eval_success() {
        let ctx = prepare_context();
        let expr: Expr = GetVar {
            var_id: VAR_IDX,
            var_tpe: SType::SInt,
        }
        .into();
        let res = eval_out::<Option<i32>>(&expr, &ctx);
        assert_eq!(res, Some(VAR_VAL));
    }

    /// Variable index out of range
    #[test]
    fn eval_fail() {
        let ctx = prepare_context();
        let expr: Expr = GetVar {
            var_id: VAR_IDX + 1,
            var_tpe: SType::SInt,
        }
        .into();
        let res = eval_out::<Option<i32>>(&expr, &ctx);
        assert_eq!(res, None);
    }

    /// Variable has wrong type
    #[test]
    fn eval_wrong_type() {
        let ctx = prepare_context();
        let expr: Expr = GetVar {
            var_id: VAR_IDX,
            var_tpe: SType::SBoolean,
        }
        .into();
        let res = try_eval_out::<Value>(&expr, &ctx);
        assert!(res.is_err());
    }
}
