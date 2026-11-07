use alloc::vec::Vec;
use ergotree_ir::mir::method_call::MethodCall;
use ergotree_ir::mir::value::Value;

use super::smethod_eval_fn;
use super::Context;
use super::Env;
use super::EvalError;
use super::Evaluable;
use ergotree_ir::types::smethod::SMethod;
use ergotree_ir::types::stype_companion::STypeCompanion;

impl Evaluable for MethodCall {
    fn eval<'ctx>(
        &self,
        env: &mut Env<'ctx>,
        ectx: &Context<'ctx>,
    ) -> Result<Value<'ctx>, EvalError> {
        ectx.add_jit_cost(4)?; // MethodCall = Fixed(4)
        let ov = self.obj.eval(env, ectx)?;
        let argsv: Result<Vec<Value>, EvalError> =
            self.args.iter().map(|arg| arg.eval(env, ectx)).collect();
        let argsv = argsv?;
        check_arity(&self.method, argsv.len())?;
        smethod_eval_fn(&self.method)?(&self.method, env, ectx, ov, argsv)
    }
}
/// sigmastate parses a call of any arity, and its evaluator invokes the method with the
/// arguments it has, after evaluating them all (v6.0.7 `values.scala:1361-1393`): a reflective
/// invocation (`javaMethod.invoke`, `evalMethod.invoke`) fails on any other count than the
/// method's, and a `userDefinedInvoke` handler on a missing argument only. The parser builds a
/// call of any arity too, so the evaluator makes the same checks.
pub(crate) fn check_arity(method: &SMethod, n_args: usize) -> Result<(), EvalError> {
    let expected = method.tpe().t_dom.len().saturating_sub(1);
    if n_args != expected && !(n_args > expected && ignores_extra_arguments(method)) {
        return Err(EvalError::Misc(format!(
            "method {} takes {expected} arguments, got {n_args}",
            method.name()
        )));
    }
    Ok(())
}

/// The methods sigmastate implements with a `userDefinedInvoke` handler, which reads the
/// arguments it needs (`other.head`, or none) and ignores any more: the numeric `toBytes`,
/// `toBits`, `bitwiseInverse`, `bitwiseOr`, `bitwiseAnd`, `bitwiseXor`, `shiftLeft` and
/// `shiftRight` (v6.0.7 `methods.scala:311-458`, the only `withUserDefinedInvoke` calls)
fn ignores_extra_arguments(method: &SMethod) -> bool {
    matches!(
        method.obj_type,
        STypeCompanion::SByte
            | STypeCompanion::SShort
            | STypeCompanion::SInt
            | STypeCompanion::SLong
            | STypeCompanion::SBigInt
            | STypeCompanion::SUnsignedBigInt
    ) && matches!(
        method.name(),
        "toBytes"
            | "toBits"
            | "bitwiseInverse"
            | "bitwiseOr"
            | "bitwiseAnd"
            | "bitwiseXor"
            | "shiftLeft"
            | "shiftRight"
    )
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod arity_tests {
    use crate::eval::test_util::try_eval_out_wo_ctx;
    use alloc::boxed::Box;
    use alloc::vec;
    use ergotree_ir::mir::expr::Expr;
    use ergotree_ir::mir::method_call::MethodCall;
    use ergotree_ir::mir::property_call::PropertyCall;
    use ergotree_ir::mir::value::Value;
    use ergotree_ir::types::scontext::GET_VAR_FROM_INPUT_METHOD;
    use hashbrown::HashMap;

    // The parser builds a call of any arity, as sigmastate's does; evaluating one whose
    // count is not the method's errors and does not panic.

    #[test]
    fn a_property_call_of_a_method_with_arguments_errors() {
        let pc: Expr = PropertyCall {
            obj: Box::new(Expr::Context),
            method: GET_VAR_FROM_INPUT_METHOD.clone(),
        }
        .into();
        // the arity check's error, not the method's own read of a missing argument
        let err = try_eval_out_wo_ctx::<Value>(&pc).unwrap_err();
        assert!(
            alloc::format!("{err:?}").contains("takes 2 arguments, got 0"),
            "{err:?}"
        );
    }

    #[test]
    fn a_user_defined_invoke_method_ignores_extra_arguments() {
        // `Int.bitwiseOr` reads `other.head` and ignores the rest: 1 | 2 with an extra 4 is 3.
        // `Int.toBytes` reads no argument at all. Each with too few still errors.
        use ergotree_ir::mir::constant::Constant;
        use ergotree_ir::types::snumeric::sint;
        let method = |name: &str| {
            sint::METHODS
                .iter()
                .find(|m| m.name() == name)
                .unwrap()
                .clone()
        };
        let call = |name: &str, args: alloc::vec::Vec<Expr>| -> Expr {
            MethodCall {
                obj: Box::new(Constant::from(1i32).into()),
                method: method(name),
                args,
                explicit_type_args: HashMap::new(),
            }
            .into()
        };
        assert_eq!(
            try_eval_out_wo_ctx::<i32>(&call("bitwiseOr", vec![2i32.into(), 4i32.into()])).unwrap(),
            3
        );
        assert!(try_eval_out_wo_ctx::<i32>(&call("bitwiseOr", vec![])).is_err());
        assert!(try_eval_out_wo_ctx::<Value>(&call("toBytes", vec![5i32.into()])).is_ok());
    }

    #[test]
    fn a_method_call_with_too_many_or_too_few_arguments_errors() {
        for args in [
            vec![],
            vec![0i16.into()],
            vec![0i16.into(), 1i8.into(), 2i8.into()],
        ] {
            let mc: Expr = MethodCall {
                obj: Box::new(Expr::Context),
                method: GET_VAR_FROM_INPUT_METHOD.clone(),
                args,
                explicit_type_args: HashMap::new(),
            }
            .into();
            assert!(try_eval_out_wo_ctx::<Value>(&mc).is_err());
        }
    }
}
