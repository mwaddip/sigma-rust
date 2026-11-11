use alloc::vec::Vec;
use ergotree_ir::mir::method_call::MethodCall;
use ergotree_ir::mir::value::Value;

use super::smethod_eval_fn;
use super::Context;
use super::Env;
use super::EvalError;
use super::Evaluable;
use super::MethodCallSite;
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
        let site = MethodCallSite {
            method: &self.method,
            args: &self.args,
        };
        smethod_eval_fn(&self.method)?(&site, env, ectx, ov, argsv)
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
            explicit_type_args: Default::default(),
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

/// SANTA's eval-tier rows for the method ids sigmastate resolves (`Coll.methodcall_v5_ids.json`,
/// `MethodCall.v5_methods_missing_in_rust.json`, `SigmaProp.methodcall_ids.json`), with the
/// JVM's values and costs
#[cfg(test)]
#[cfg(feature = "arbitrary")]
#[allow(clippy::unwrap_used, clippy::unreachable)]
mod jvm_method_rows {
    use alloc::vec::Vec;
    use ergotree_ir::mir::constant::TryExtractInto;
    use ergotree_ir::mir::expr::Expr;
    use ergotree_ir::mir::value::Value;
    use ergotree_ir::serialization::SigmaSerializable;
    use ergotree_ir::types::stype::SType;

    use crate::eval::test_util::{assert_accepts, assert_evaluator_refuses, eval_v3_body};

    #[test]
    fn coll_v5_accept_rows() {
        assert_accepts("0b0adb0c0183020404020404", Value::Int(2), 48);
        assert_accepts("0b0fdc0c028302040402040402040a0412", Value::Int(9), 74);
        assert_accepts("0b0fdc0c02830204040204040204000412", Value::Int(1), 74);
        assert_accepts("0b0ddc0c0a83020404020404010402", Value::Int(2), 69);
        let map = "0b14dc0c038302040402040401d90101049a72010402";
        let (_, res, c) = eval_v3_body(map);
        assert_eq!(res.unwrap().try_extract_into::<Vec<i32>>().unwrap(), [2, 3]);
        assert_eq!(c, 120);
    }

    #[test]
    fn coll_v5_errored_rows() {
        // exists, forall, slice, filter, append, fold (2-arg op), fold (pair op): the JVM
        // parses them and has no evaluator for any
        for tree_hex in [
            "0b14dc0c048302040402040401d90101049172010402",
            "0b14dc0c068302040402040401d90101049172010400",
            "0b11dc0c078303040402040404060204020406",
            "0b14dc0c088302040402040401d90101049172010402",
            "0b0edc0c098301040402018301040404",
            "0b18dc0c0583020404020404020400d902010402049a72017202",
            "0b1adc0c0583020404020404020400d90101589a8c7201018c720102",
        ] {
            assert!(eval_v3_body(tree_hex).1.is_err(), "{tree_hex}");
        }
    }

    #[test]
    fn coll_get_or_else_generic_default() {
        // Coll(1, 2).getOrElse(i, true): the default's type is not the items', so the call's
        // types don't unify and the method stays generic. sigmastate checks the default against
        // its own type (`getOrElse_eval`, `mc.args.last.tpe`) and returns it out of range.
        for (tree_hex, expected) in [
            ("0b0fdc0c028302040402040402040a0101", Value::Boolean(true)),
            ("0b0fdc0c02830204040204040204000101", Value::Int(1)),
        ] {
            let (expr, res, _) = eval_v3_body(tree_hex);
            let Expr::MethodCall(mc) = &expr else {
                unreachable!("{expr:?}")
            };
            assert!(
                matches!(
                    mc.expr().method.tpe().t_dom.last(),
                    Some(SType::STypeVar(_))
                ),
                "{tree_hex}: {:?}",
                mc.expr().method
            );
            assert_eq!(res.unwrap(), expected, "{tree_hex}");
        }
    }

    #[test]
    fn coll_map_generic_range_fails_on_empty() {
        // Coll[Int]().map((x: Long) => x): the mapper doesn't unify with the items, so the
        // result's item type is still `OV`, which `stypeToRType` can't convert: sigmastate
        // throws before mapping, so no element (and no lambda argument check) is reached
        let tree_hex = "0b0cdc0c03100001d90101057201";
        let (_, res, _) = eval_v3_body(tree_hex);
        let err = res.unwrap_err();
        assert!(
            alloc::format!("{err:?}").contains("type variable"),
            "{err:?}"
        );
    }

    #[test]
    fn coll_negative_index() {
        assert_evaluator_refuses("0b0ddc0c0a83020404020404010401"); // apply(-1)
        assert_accepts("0b0fdc0c02830204040204040204010412", Value::Int(9), 74);
        // getOrElse(-1, 9)
    }

    #[test]
    fn coll_new_evaluators_check_their_receiver() {
        // size, apply(0), getOrElse(0, 9), map((x: Int) => x), each on the Int 1
        for tree_hex in [
            "0b05db0c010402",
            "0b08dc0c0a0402010400",
            "0b0adc0c0204020204000412",
            "0b0cdc0c03040201d90101047201",
        ] {
            assert_evaluator_refuses(tree_hex);
        }
    }

    #[test]
    fn v5_missing_accept_rows() {
        use ergo_chain_types::EcPoint;
        // groupGenerator.exp(BigInt 5) @928, groupGenerator.multiply(groupGenerator) @82
        for (tree_hex, point_hex, cost) in [
            (
                "0b0bdc0703db6a01dd01060105",
                "022f8bde4d1a07209355b4a7250a5c5128e88b84bddc619ab7cba8d569b240efe4",
                928,
            ),
            (
                "0b0cdc0704db6a01dd01db6a01dd",
                "02c6047f9441ed7d6d3045406e95c07cd85c778e4b8cef3ca7abac09b95c709ee5",
                82,
            ),
        ] {
            let (_, res, c) = eval_v3_body(tree_hex);
            let point = res.unwrap().try_extract_into::<EcPoint>().unwrap();
            assert_eq!(
                base16::encode_lower(&point.sigma_serialize_bytes().unwrap()),
                point_hex
            );
            assert_eq!(c, cost, "{tree_hex}");
        }
        // Global.some[Int](5).isDefined, Global.none[Int].isDefined, Global.some[Int](5).get
        assert_accepts("0b0bdb2402dc6a09dd01040a04", Value::Boolean(true), 33);
        assert_accepts("0b08db2402db6a0add04", Value::Boolean(false), 28);
        assert_accepts("0b0bdb2403dc6a09dd01040a04", Value::Int(5), 38);
    }

    #[test]
    fn v5_missing_and_sigmaprop_errored_rows() {
        // SELF.R0.isDefined, SELF.R4.isDefined (no `R0()` to invoke), none.get (throws
        // inside), some(5).getOrElse(9), none.getOrElse(9) (no `getOrElse(int)`), propBytes,
        // isProven (no `_eval`), 1.toBigInt (no cost kind)
        for tree_hex in [
            "0b05e6db6309a7",
            "0b05e6db630da7",
            "0b08db2403db6a0add04",
            "0b0edc2404dc6a09dd01040a04010412",
            "0b0bdc2404db6a0add04010412",
            "0b0bd191b1db0801d101010400",
            "0b07d1db0802d10101",
            "0b05db04050402",
        ] {
            assert!(eval_v3_body(tree_hex).1.is_err(), "{tree_hex}");
        }
    }

    #[test]
    fn global_serialize_writes_a_tuple_by_its_type() {
        // SANTA's tuple-values reply, W1 and W2: Global.serialize((1, 2, 3)) writes the items
        // with no length, Global.serialize(Coll((1, 2, 3))) the collection's length and the items
        use ergotree_ir::mir::coll_value::CollValue;
        let bytes = |b: &[i8]| Value::Coll(CollValue::Bytes(b.into()));
        assert_accepts("0b0cdc6a03dd0148040404020406", bytes(&[2, 4, 6]), 33);
        assert_accepts("0b0edc6a03dd010c4804040401020406", bytes(&[1, 2, 4, 6]), 36);
    }

    #[test]
    fn option_new_evaluators_check_their_receiver() {
        // isDefined and get on Coll(1, 2)
        assert_evaluator_refuses("0b0adb240283020404020404");
        assert_evaluator_refuses("0b0adb240383020404020404");
    }

    #[test]
    fn map_result_without_runtime_type_fails_on_empty() {
        // sigmastate's `MapCollection.eval` and `map_eval` both convert the result's item type
        // with `stypeToRType` before charging or mapping (v6.0.7 `transformers.scala:40-45`,
        // `methods.scala:862-867`); it has no arm for a type variable or a function of other than
        // one argument (`Evaluation.scala:51-55`), so an empty input fails too
        for tree_hex in [
            // Coll[Int]().map((x: Int) => (a: Int, b: Int) => a), the method
            "0b12dc0c03100001d9010104d902020403047202",
            // Map(Coll[Int](), (x: Int) => SELF.R4), the node: `R4`'s `Option[T]`
            "0b0bad1000d9010104db630da7",
            // Map(Coll[Int](), (x: Int) => SELF.getRegV5(4)), the node
            "0b0ead1000d9010104dc6307a7010408",
        ] {
            let (_, res, _) = eval_v3_body(tree_hex);
            let err = res.unwrap_err();
            assert!(
                alloc::format!("{err:?}").contains("no runtime type"),
                "{tree_hex}: {err:?}"
            );
        }
        // a function of one argument converts: Coll[Int]().map((x: Int) => (y: Int) => y), as
        // the method and as the node
        for tree_hex in [
            "0b0fdc0c03100001d9010104d90102047202",
            "0b0dad1000d9010104d90102047202",
        ] {
            let (_, res, _) = eval_v3_body(tree_hex);
            assert!(res.is_ok(), "{tree_hex}: {res:?}");
        }
    }

    #[test]
    fn coll_runtime_shapes_rows() {
        // SANTA `Coll.runtime_shapes_methodcall.json` (bar the two `Coll[Byte]`-constant rows)
        let pair = |a: i32, b: i32| Value::pair(Value::Int(a), Value::Int(b));
        // a collection of pairs is a `PairOfCols` on the JVM: the `Coll.getOrElse` bridge
        // casts the default to `Tuple2` whatever the index (`CollsOverArrays.scala:309-315`)
        assert_evaluator_refuses("0b0edc0c0283015858020402040a0101"); // getOrElse(5, true)
        assert_evaluator_refuses("0b0edc0c028301585802040204000101"); // getOrElse(0, true)
        assert_accepts(
            "0b15dc0c0283015886020402040402040a8602040e0410",
            pair(7, 8),
            109,
        ); // getOrElse(5, (7, 8))
        assert_accepts(
            "0b15dc0c028301588602040204040204008602040e0410",
            pair(1, 2),
            109,
        ); // getOrElse(0, (7, 8))
           // a ConcreteCollection of bytes is a generic `CollOverArray`: the default is returned
        assert_accepts(
            "0b0fdc0c028302020201020202040a0101",
            Value::Boolean(true),
            74,
        );
        assert_accepts("0b0fdc0c02830202020102020204000101", Value::Byte(1), 74);
        // a tuple of other than two items is a `Coll[Any]` on the JVM (`values.scala:818-822`)
        assert_accepts("0b0adb0c0148040404020406", Value::Int(3), 23); // (1, 2, 3).size
        assert_accepts("0b0ddc0c0a48040404020406010400", Value::Int(1), 44); // .apply(0)
        assert_accepts("0b0fdc0c024804040402040602040a0412", Value::Int(9), 49);
        // .getOrElse(5, 9)
    }
}
