use ergotree_ir::mir::func_value::FuncValue;
use ergotree_ir::mir::value::Value;

use crate::eval::env::Env;
use crate::eval::Context;
use crate::eval::EvalError;
use crate::eval::Evaluable;

impl Evaluable for FuncValue {
    fn eval<'ctx>(
        &self,
        env: &mut Env<'ctx>,
        ctx: &Context<'ctx>,
    ) -> Result<Value<'ctx>, EvalError> {
        ctx.add_jit_cost(5)?; // FuncValue = Fixed(5)

        // The JVM supports only unary functions: FuncValue.eval
        // (values.scala:1042-1056) charges its fixed cost, then builds the
        // closure iff args.length == 1 and otherwise throws "Function must
        // have 1 argument". The reject fires at closure CREATION (BlockValue
        // evaluates ValDefs eagerly), so a bound non-unary lambda rejects even
        // when never applied — but NOT at parse: both serializers are
        // arity-agnostic, and a non-unary lambda on the dead branch of a lazy
        // `if` must still accept.
        if self.args().len() != 1 {
            return Err(EvalError::Misc(format!(
                "Function must have 1 argument, but was: {self:?}"
            )));
        }
        // The JVM's `FuncValue.eval` returns a closure over the defining env —
        // capture it so a lambda that escapes its creation site (returned from
        // another lambda, bound to a `val`) still sees these bindings when
        // applied. Both the env and the body are shared, not copied
        Ok(Value::Lambda(self.closure(env.shared())))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use crate::eval::test_util::try_eval_out_with_version;
    use ergotree_ir::chain::context::Context;
    use ergotree_ir::ergo_tree::ErgoTree;
    use sigma_test_util::force_any_val;

    fn hx(s: &str) -> alloc::vec::Vec<u8> {
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
            .collect()
    }

    // SANTA FuncValue.non_unary_arity (eval/v5/authored, JVM-blessed): the
    // closure-creation arity gate, end-to-end from the vector bytes (v0 trees,
    // no size bit — the unsized parse path takes the Int roots as-is).
    #[test]
    fn non_unary_lambda_rejects_at_closure_creation() {
        let ctx = force_any_val::<Context>();
        let run = |hex: &str| -> Result<i32, alloc::string::String> {
            let tree = ErgoTree::sigma_parse_bytes_lenient(&hx(hex))
                .map_err(|e| alloc::format!("parse: {e:?}"))?;
            let expr = tree
                .proposition()
                .map_err(|e| alloc::format!("proposition: {e:?}"))?;
            try_eval_out_with_version::<i32>(&expr, &ctx, 0, 2)
                .map_err(|e| alloc::format!("eval: {e:?}"))
        };
        // { val add = (x:Int,y:Int) => x+y; add(3,4) }  — applied
        // { val add = (x:Int,y:Int) => x+y; 5 }         — bound, never applied
        //                                                  (ValDefs eval eagerly)
        // { val f = () => 5; f() }                      — the zero-arg side
        for hex in [
            "100204060408d801d601d902020403049a72027203da72010273007301",
            "1001040ad801d601d902020403049a720272037300",
            "1001040ad801d601d9007300da720100",
        ] {
            assert!(
                run(hex).is_err(),
                "{hex}: non-unary lambda must reject at closure creation"
            );
        }
        // if (true) 5 else { val add = (x:Int,y:Int) => x+y; add(3,4) } — the
        // 2-arg lambda sits on the dead branch of a lazy If and never evaluates:
        // the tree ACCEPTS (the gate is eval-time, not parse-time).
        assert_eq!(
            run("10040101040a040604089573007301d801d601d902020403049a72027203da72010273027303")
                .unwrap(),
            5,
            "lazy-if twin: a never-evaluated non-unary lambda must not reject the tree"
        );
    }

    mod sharing {
        use alloc::boxed::Box;
        use alloc::sync::Arc;
        use alloc::vec;
        use ergotree_ir::chain::context::Context;
        use ergotree_ir::mir::apply::Apply;
        use ergotree_ir::mir::block::BlockValue;
        use ergotree_ir::mir::expr::Expr;
        use ergotree_ir::mir::func_value::{FuncArg, FuncValue};
        use ergotree_ir::mir::val_def::{ValDef, ValId};
        use ergotree_ir::mir::val_use::ValUse;
        use ergotree_ir::mir::value::{Lambda, Value};
        use ergotree_ir::types::sfunc::SFunc;
        use ergotree_ir::types::stype::SType;
        use sigma_test_util::force_any_val;

        use crate::eval::env::Env;
        use crate::eval::Evaluable;

        fn int_to_int() -> SType {
            SType::SFunc(SFunc::new(vec![SType::SInt], SType::SInt))
        }

        fn int(id: u32) -> Expr {
            ValUse {
                val_id: id.into(),
                tpe: SType::SInt,
            }
            .into()
        }

        /// `(x: Int) => body`, `x` bound to `arg`
        fn lambda(arg: u32, body: Expr) -> Expr {
            FuncValue::new(
                vec![FuncArg {
                    idx: arg.into(),
                    tpe: SType::SInt,
                }],
                body,
            )
            .into()
        }

        /// `f(x)`, `f` a `ValUse` of an `Int => Int`
        fn call(f: u32, arg: u32) -> Expr {
            Apply {
                func: Box::new(
                    ValUse {
                        val_id: f.into(),
                        tpe: int_to_int(),
                    }
                    .into(),
                ),
                args: vec![int(arg)],
            }
            .into()
        }

        fn val(id: u32, rhs: Expr) -> Expr {
            ValDef {
                id: id.into(),
                tpe_args: vec![],
                rhs: Box::new(rhs),
            }
            .into()
        }

        /// `{ val f1 = (x: Int) => x; val f2 = (x: Int) => f1(x); val f3 = (x: Int) => f2(x);
        /// val v = 1; true }`, evaluated into an environment the test reads afterwards: a top-level
        /// block keeps its `ValDef`s
        fn eval_into_env<'ctx>(ctx: &Context<'ctx>) -> Env<'ctx> {
            let block: Expr = BlockValue {
                items: vec![
                    val(1, lambda(10, int(10))),
                    val(2, lambda(11, call(1, 11))),
                    val(3, lambda(12, call(2, 12))),
                    val(4, 1i32.into()),
                ],
                result: Box::new(true.into()),
            }
            .into();
            let mut env = Env::empty();
            assert_eq!(block.eval(&mut env, ctx).unwrap(), Value::Boolean(true));
            env
        }

        fn id(n: u32) -> ValId {
            n.into()
        }

        fn lambda_of<'a, 'ctx>(v: Option<&'a Value<'ctx>>) -> Option<&'a Lambda<'ctx>> {
            match v {
                Some(Value::Lambda(l)) => Some(l),
                _ => None,
            }
        }

        #[test]
        fn nested_lambdas_share_their_captured_environments() {
            // the JVM's closure holds its defining `DataEnv` (an immutable Map) and its body
            // (v6.0.7 `values.scala:1069-1086`); copying them made each nested definition double
            let ctx = force_any_val::<Context>();
            let env = eval_into_env(&ctx);
            let f1 = lambda_of(env.get(1.into())).unwrap();
            let f2 = lambda_of(env.get(2.into())).unwrap();
            let f3 = lambda_of(env.get(3.into())).unwrap();
            let f1_in_f2 = lambda_of(f2.captured.get(&id(1))).unwrap();
            let f2_in_f3 = lambda_of(f3.captured.get(&id(2))).unwrap();
            assert!(Arc::ptr_eq(&f1_in_f2.captured, &f1.captured));
            assert!(Arc::ptr_eq(&f2_in_f3.captured, &f2.captured));
            assert!(Arc::ptr_eq(&f2_in_f3.body, &f2.body));
        }

        #[test]
        fn a_later_binding_does_not_reach_an_earlier_lambda() {
            let ctx = force_any_val::<Context>();
            let env = eval_into_env(&ctx);
            let f1 = lambda_of(env.get(1.into())).unwrap();
            let f3 = lambda_of(env.get(3.into())).unwrap();
            assert!(f1.captured.is_empty());
            assert!(f3.captured.contains_key(&id(1)));
            assert!(f3.captured.contains_key(&id(2)));
            assert!(!f3.captured.contains_key(&id(4)));
        }

        #[test]
        fn to_static_keeps_shared_environments_shared() {
            let ctx = force_any_val::<Context>();
            let env = eval_into_env(&ctx);
            let f3 = env.get(3.into()).unwrap().to_static();
            let f3 = lambda_of(Some(&f3)).unwrap();
            let f2 = lambda_of(f3.captured.get(&id(2))).unwrap();
            let f1_via_f2 = lambda_of(f2.captured.get(&id(1))).unwrap();
            let f1_direct = lambda_of(f3.captured.get(&id(1))).unwrap();
            assert!(Arc::ptr_eq(&f1_via_f2.captured, &f1_direct.captured));
        }

        #[test]
        fn a_lambda_prints_the_ids_it_captured() {
            // printing a captured value printed the lambdas it holds, and theirs: an error or
            // the diagnostics of such a script printed exponentially much
            let ctx = force_any_val::<Context>();
            let env = eval_into_env(&ctx);
            let printed = alloc::format!("{:?}", env.get(3.into()).unwrap());
            // one lambda value, `Value::Lambda(Lambda { .. })`
            assert_eq!(printed.matches("Lambda(Lambda").count(), 1, "{printed}");
        }
    }
}
