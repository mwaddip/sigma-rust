use alloc::vec::Vec;
use ergotree_ir::mir::apply::Apply;
use ergotree_ir::mir::value::Value;
use ergotree_ir::types::stype::SType;

use crate::eval::env::Env;
use crate::eval::Context;
use crate::eval::EvalError;
use crate::eval::Evaluable;
use crate::eval::LambdaInvoker;

impl Evaluable for Apply {
    fn eval<'ctx>(
        &self,
        env: &mut Env<'ctx>,
        ctx: &Context<'ctx>,
    ) -> Result<Value<'ctx>, EvalError> {
        ctx.add_jit_cost(30)?; // Apply = Fixed(30)
        let func_v: Value<'ctx> = self.func.eval(env, ctx)?;
        let args_v: Vec<Value> = self
            .args
            .iter()
            .map(|arg| arg.eval(env, ctx))
            .collect::<Result<_, EvalError>>()?;
        match func_v {
            Value::Lambda(fv) => {
                // The JVM rejects applying a lambda whose argument type is
                // still a type variable (a `FunDef`-bound polymorphic lambda
                // applied without instantiation): `Value.checkType(argTpe, v)`
                // inside the closure reaches `isValueOfType`'s catch-all →
                // `Unknown type T`. The error fires only on application —
                // binding such a lambda and never applying it is accepted,
                // since the closure body never runs.
                if let Some(arg) = fv.args.iter().find(|a| matches!(a.tpe, SType::STypeVar(_))) {
                    return Err(EvalError::Misc(format!(
                        "Apply: Unknown type {:?} of lambda argument {}",
                        arg.tpe, arg.idx
                    )));
                }
                // The body evaluates in the lambda's CAPTURED environment
                // (extended with the argument bindings) — not in the caller's
                // env. Mirrors the JVM, where `FuncValue.eval` returns a
                // closure over the defining env; the previous bind-into-the-
                // caller's-env dance lost the outer binding of a curried
                // lambda (`add(3)(1)`) as soon as the outer `Apply` returned.
                // ADD_TO_ENV_COST per arg binding is charged by the invoker.
                LambdaInvoker::new(&fv).invoke(ctx, args_v)
            }
            _ => Err(EvalError::UnexpectedValue(format!(
                "expected func_v to be Value::FuncValue got: {0:?}",
                func_v
            ))),
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use alloc::boxed::Box;
    use ergotree_ir::mir::bin_op::BinOp;
    use ergotree_ir::mir::bin_op::RelationOp;
    use ergotree_ir::mir::block::BlockValue;
    use ergotree_ir::mir::expr::Expr;
    use ergotree_ir::mir::func_value::FuncArg;
    use ergotree_ir::mir::func_value::FuncValue;
    use ergotree_ir::mir::val_def::ValDef;
    use ergotree_ir::mir::val_use::ValUse;
    use ergotree_ir::types::stype::SType;

    use crate::eval::test_util::eval_out_wo_ctx;

    use super::*;

    // SANTA HOF regression (`eval/v6/authored` vectors): `FunDef` (0xd7)
    // trees deserialize, the polymorphic *construct* evaluates, and applying
    // a lambda whose argument type is still a type variable is rejected —
    // mirroring the JVM boundary (`Unknown type T` fires at application via
    // `Value.checkType` inside the closure, never at binding).
    #[test]
    fn fundef_polymorphic_apply_boundary() {
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
        // The vector trees carry arbitrary-typed roots (eval-tier corpus); the
        // sized parse path rejects non-SigmaProp roots, so use the lenient entry
        // (skips only the root-type check) — the same leniency the conformance
        // runner applies; the header (size bit set) keeps Rule-1012 satisfied.
        // End-to-end: parse → substitute constants → eval. The reject trees
        // may fail at either layer (Rust's `Apply::new` arg-type check fires
        // at parse, where the JVM defers to eval — both reject the spend).
        let run = |hex: &str| -> Result<i32, alloc::string::String> {
            let tree = ErgoTree::sigma_parse_bytes_lenient(&hx(hex))
                .map_err(|e| alloc::format!("parse: {e:?}"))?;
            let expr = tree
                .proposition()
                .map_err(|e| alloc::format!("proposition: {e:?}"))?;
            let ctx = force_any_val::<Context>();
            try_eval_out_with_version::<i32>(&expr, &ctx, 3, 3)
                .map_err(|e| alloc::format!("eval: {e:?}"))
        };

        // { val id[T] = {(x: Int) => x}; id(7) } — tpeArgs=[T], concrete arg type
        // { val id[T] = {(x: T) => x}; 5 }      — type-var arg, bound but never applied
        let accepts: &[(&str, i32)] = &[
            ("1b1701040ed801d70101670154d90102047202da7201017300", 7),
            ("1b1501040ad801d70101670154d9010267015472027300", 5),
        ];
        // Applying through the type var — all reject:
        // (x: T) => x  /  (x: T) => 5  /  (x: T) => x + x, each applied as id(7)
        let rejects: &[&str] = &[
            "1b1901040ed801d70101670154d901026701547202da7201017300",
            "1b1b02040a040ed801d70101670154d901026701547300da7201017301",
            "1b1c01040ed801d70101670154d901026701549a72027202da7201017300",
        ];
        for (hex, expected) in accepts {
            assert_eq!(run(hex).unwrap(), *expected, "{hex}: value mismatch");
        }
        for hex in rejects {
            assert!(
                run(hex).is_err(),
                "{hex}: applying a type-var-typed lambda argument must reject"
            );
        }
    }

    // SANTA HOF regression (`HOF_currying_Apply_of_Apply` vector): a lambda
    // returned from another lambda must capture its defining environment —
    // JVM `FuncValue.eval` returns a closure over the env it was created in.
    // Pre-fix, `Apply` bound arguments into the CALLER's env and removed them
    // after the body ran, so the inner lambda of
    // `add = (a: Int) => (b: Int) => a + b` lost `a` the moment the outer
    // Apply returned, and `add(3)(1)` errored instead of evaluating to 4.
    #[test]
    fn curried_lambda_captures_environment() {
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
        // { val add = {(a: Int) => {(b: Int) => a + b}}; add(3)(1) } — the vector
        // tree (v3, segregated). The eval-tier corpus carries an Int-typed root, so
        // use the lenient parse (skips only the root-type check); the real header
        // keeps Rule-1012 satisfied.
        let tree = ErgoTree::sigma_parse_bytes_lenient(&hx(
            "1b200204060402d801d601d9010204d90103049a72027203dada7201017300017301",
        ))
        .unwrap();
        let expr = tree.proposition().unwrap();
        let ctx = force_any_val::<Context>();
        assert_eq!(
            try_eval_out_with_version::<i32>(&expr, &ctx, 3, 3).unwrap(),
            4,
            "add(3)(1) must evaluate to 4 — the returned lambda must keep its captured `a`"
        );
    }

    // The capture must hold uniformly across invocation sites: a lambda that
    // escapes its creation site and is then fed to a HOF must still see its
    // captured bindings. `{ val mk = (a: Int) => (b: Int) => a + b;
    // val f = mk(3); Coll(1, 2).map(f) }` → [4, 5]. Pre-fix, `Coll.map`
    // bound the argument into the caller's env, where `a` no longer existed.
    #[test]
    fn escaped_lambda_into_hof_keeps_captured_env() {
        use ergotree_ir::mir::apply::Apply;
        use ergotree_ir::mir::bin_op::ArithOp;
        use ergotree_ir::mir::coll_map::Map;
        use ergotree_ir::mir::collection::Collection;

        let inner_lambda: Expr = FuncValue::new(
            vec![FuncArg {
                idx: 3.into(),
                tpe: SType::SInt,
            }],
            Expr::BinOp(
                BinOp {
                    kind: ArithOp::Plus.into(),
                    left: Box::new(
                        ValUse {
                            val_id: 2.into(),
                            tpe: SType::SInt,
                        }
                        .into(),
                    ),
                    right: Box::new(
                        ValUse {
                            val_id: 3.into(),
                            tpe: SType::SInt,
                        }
                        .into(),
                    ),
                }
                .into(),
            ),
        )
        .into();
        let mk: Expr = FuncValue::new(
            vec![FuncArg {
                idx: 2.into(),
                tpe: SType::SInt,
            }],
            inner_lambda,
        )
        .into();
        let f: Expr = Apply::new(
            ValUse {
                val_id: 1.into(),
                tpe: mk.tpe(),
            }
            .into(),
            vec![Expr::Const(3i32.into())],
        )
        .unwrap()
        .into();
        let block: Expr = BlockValue {
            items: vec![
                ValDef {
                    id: 1.into(),
                    tpe_args: vec![],
                    rhs: Box::new(mk),
                }
                .into(),
                ValDef {
                    id: 4.into(),
                    tpe_args: vec![],
                    rhs: Box::new(f.clone()),
                }
                .into(),
            ],
            result: Box::new(
                Map::new(
                    Collection::new(
                        SType::SInt,
                        vec![Expr::Const(1i32.into()), Expr::Const(2i32.into())],
                    )
                    .unwrap()
                    .into(),
                    ValUse {
                        val_id: 4.into(),
                        tpe: f.tpe(),
                    }
                    .into(),
                )
                .unwrap()
                .into(),
            ),
        }
        .into();
        assert_eq!(
            eval_out_wo_ctx::<alloc::vec::Vec<i32>>(&block),
            alloc::vec![4, 5],
            "the escaped lambda must keep `a = 3` when invoked by Coll.map"
        );
    }

    #[test]
    fn eval_user_defined_func_call() {
        let arg = Expr::Const(1i32.into());
        let bin_op = Expr::BinOp(
            BinOp {
                kind: RelationOp::Eq.into(),
                left: Box::new(
                    ValUse {
                        val_id: 1.into(),
                        tpe: SType::SInt,
                    }
                    .into(),
                ),
                right: Box::new(
                    ValUse {
                        val_id: 2.into(),
                        tpe: SType::SInt,
                    }
                    .into(),
                ),
            }
            .into(),
        );
        let body = Expr::BlockValue(
            BlockValue {
                items: vec![ValDef {
                    id: 2.into(),
                    tpe_args: vec![],
                    rhs: Box::new(Expr::Const(1i32.into())),
                }
                .into()],
                result: Box::new(bin_op),
            }
            .into(),
        );
        let apply: Expr = Apply::new(
            FuncValue::new(
                vec![FuncArg {
                    idx: 1.into(),
                    tpe: SType::SInt,
                }],
                body,
            )
            .into(),
            vec![arg],
        )
        .unwrap()
        .into();
        assert!(eval_out_wo_ctx::<bool>(&apply));
    }

    // Parity gap: Scala charges ADD_TO_ENV_COST (5 JIT) on every env
    // insertion. Apply binds each lambda arg into the interpreter env, so an
    // N-arg application must pay 5×N ADD_TO_ENV on top of Apply=Fixed(30),
    // FuncValue=Fixed(5), and the per-arg value eval. Pre-fix the binding
    // loop charged 0, undercharging every user-lambda application by 5×N.
    #[test]
    fn apply_charges_add_to_env_per_arg_binding() {
        use crate::eval::test_util::try_eval_out;
        use ergotree_ir::chain::context::Context;
        use sigma_test_util::force_any_val;

        // Apply a FuncValue of `n_args` Int params (body ignores them and
        // returns a Bool const) to `n_args` Int constants; return JIT cost.
        let run = |n_args: u32| -> u64 {
            let ctx = force_any_val::<Context>();
            let before = ctx.jit_cost_value();
            let args: alloc::vec::Vec<FuncArg> = (1..=n_args)
                .map(|i| FuncArg {
                    idx: i.into(),
                    tpe: SType::SInt,
                })
                .collect();
            let arg_exprs: alloc::vec::Vec<Expr> =
                (1..=n_args).map(|_| Expr::Const(1i32.into())).collect();
            let apply: Expr = Apply::new(
                FuncValue::new(args, Expr::Const(true.into())).into(),
                arg_exprs,
            )
            .unwrap()
            .into();
            let _: bool = try_eval_out(&apply, &ctx).unwrap();
            ctx.jit_cost_value() - before
        };

        // Each extra arg adds: arg Const eval (5) + ADD_TO_ENV_COST (5) = 10.
        // Apply (30), FuncValue (5) and the body Const (5) are identical
        // across both runs, so they cancel in the delta.
        let delta_1 = run(1);
        let delta_2 = run(2);
        assert_eq!(
            delta_2 - delta_1,
            10,
            "Apply must charge ADD_TO_ENV_COST (5 JIT) per lambda-arg binding \
             on top of the per-arg value eval (5); got {} JIT delta between a \
             2-arg and 1-arg application, expected 10 (pre-fix would be 5).",
            delta_2 - delta_1,
        );
    }
}
