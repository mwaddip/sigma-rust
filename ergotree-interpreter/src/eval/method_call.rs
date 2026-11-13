use alloc::vec::Vec;
use ergotree_ir::mir::expr::Expr;
use ergotree_ir::mir::method_call::MethodCall;
use ergotree_ir::mir::value::Value;

use super::method_cost::fixed_method_cost;
use super::smethod_eval_fn;
use super::Context;
use super::Env;
use super::EvalError;
use super::Evaluable;
use super::MethodCallSite;
use ergotree_ir::types::smethod::SMethod;
use ergotree_ir::types::stype::SType;
use ergotree_ir::types::stype_companion::STypeCompanion;

impl Evaluable for MethodCall {
    fn eval<'ctx>(
        &self,
        env: &mut Env<'ctx>,
        ectx: &Context<'ctx>,
    ) -> Result<Value<'ctx>, EvalError> {
        call_method(&self.method, &self.obj, &self.args, env, ectx)
    }
}

/// A method call, with or without arguments: sigmastate's `MethodCall.eval`
/// (`values.scala:1361-1391`; a `PropertyCall` is the same node with no arguments, `:1351`).
/// The receiver is evaluated first and the call's own cost is added after it (`:1362-1363`).
/// Then the arguments, the method's cost if it is a fixed one, the refusals of a call the JVM's
/// invoke refuses (the argument count, the receiver's class, each argument's class), and the
/// evaluator.
pub(crate) fn call_method<'ctx>(
    method: &SMethod,
    obj: &Expr,
    args: &[Expr],
    env: &mut Env<'ctx>,
    ectx: &Context<'ctx>,
) -> Result<Value<'ctx>, EvalError> {
    let ov = obj.eval(env, ectx)?;
    ectx.add_jit_cost(4)?; // MethodCall = Fixed(4)
    let argsv: Result<Vec<Value>, EvalError> = args.iter().map(|arg| arg.eval(env, ectx)).collect();
    let argsv = argsv?;
    // a fixed-cost method is paid for before it is invoked, so before every refusal below
    // (`:1364-1380`); any other method adds its costs in its body (`:1381-1391`)
    if let Some(cost) = fixed_method_cost(method) {
        ectx.add_jit_cost(cost)?;
    }
    check_arity(method, argsv.len())?;
    check_receiver(method, &ov)?;
    let argsv = widen_arguments(method, argsv);
    check_arguments(method, &argsv)?;
    let site = MethodCallSite { method, args };
    smethod_eval_fn(method)?(&site, env, ectx, ov, argsv)
}
/// sigmastate's parser checks no receiver against the method: the owner is the type code in
/// the bytes. Its evaluator then gives the receiver's value to the method (v6.0.6
/// `values.scala:1361-1393`), and every way it does that refuses a value that is not of the
/// owner's class: the reflective call of a fixed-cost method, looked up on the owner's class
/// (`SMethod.scala:92-101`, `:119-126`); the `_eval` method of any other, whose receiver
/// parameter has that class (`SMethod.scala:138-160`); and the numeric handlers' casts
/// (`methods.scala:314-458`). The exception's class varies, so this is an error and no more.
/// Measured by SANTA: `MethodCall.receiver_kind` and `MethodCall.numeric_receiver_width`.
pub(crate) fn check_receiver(method: &SMethod, obj: &Value<'_>) -> Result<(), EvalError> {
    if receiver_is_of(method.obj_type, obj) {
        Ok(())
    } else {
        Err(EvalError::UnexpectedValue(format!(
            "method {} of {:?} called on a receiver of another kind",
            method.name(),
            method.obj_type
        )))
    }
}

/// Whether `obj` is a value of the owner's class (`SType.scala`, each type's `reprClass`).
/// The kind only: no range, no item type. A `BigInt` of any magnitude is a `BigInt`, a
/// collection of any form a `Coll`; a pair is a `Tuple2` and not a collection.
fn receiver_is_of(owner: STypeCompanion, obj: &Value<'_>) -> bool {
    match owner {
        STypeCompanion::SByte => matches!(obj, Value::Byte(_)),
        STypeCompanion::SShort => matches!(obj, Value::Short(_)),
        STypeCompanion::SInt => matches!(obj, Value::Int(_)),
        STypeCompanion::SLong => matches!(obj, Value::Long(_)),
        STypeCompanion::SBigInt => matches!(obj, Value::BigInt(_)),
        STypeCompanion::SUnsignedBigInt => matches!(obj, Value::UnsignedBigInt(_)),
        STypeCompanion::Context => matches!(obj, Value::Context),
        STypeCompanion::Box => matches!(obj, Value::CBox(_)),
        STypeCompanion::Coll => matches!(obj, Value::Coll(_)),
        STypeCompanion::GroupElem => matches!(obj, Value::GroupElement(_)),
        STypeCompanion::SigmaProp => matches!(obj, Value::SigmaProp(_)),
        STypeCompanion::Global => matches!(obj, Value::Global),
        STypeCompanion::Header => matches!(obj, Value::Header(_)),
        STypeCompanion::PreHeader => matches!(obj, Value::PreHeader(_)),
        STypeCompanion::Option => matches!(obj, Value::Opt(_)),
        STypeCompanion::AvlTree => matches!(obj, Value::AvlTree(_)),
    }
}

/// sigmastate's parser checks no argument against the method either, and its evaluator hands
/// the argument values to the method by Java reflection (v6.0.6 `values.scala:1361-1393`: the
/// Java method of a fixed-cost method, `SMethod.scala:92-101`, `:119-126`; the `_eval` method
/// of any other, `:138-160`; `JavaImpl.scala:137`). Reflection applies a widening primitive
/// conversion to a boxed argument of a primitive parameter: a `Byte` passes for a `short`, a
/// `Byte` or a `Short` for an `int`, and those and an `Int` for a `long`. So
/// `Coll[Int](0,1,2).apply(1.toByte)` is 1 on the JVM, at the cost of `apply(1)`.
///
/// A type variable is never widened, also where the call specialised it to `Long`, so the
/// type that decides is the one the method DECLARES. The `_eval` method is looked up with
/// the declared types (`genericMethod`'s `stype.tDom`, `values.scala:1390`,
/// `SMethod.scala:139-151`), where a type variable is `Object` and takes the value as it is.
/// The Java method of a fixed-cost method is looked up with the call's own types unless it
/// is given explicitly (`SMethod.scala:92-101`); there a type variable the call specialised
/// finds no Java method at all (`Option.getOrElse`: `NoSuchMethodException`), so nothing is
/// invoked.
///
/// A method a handler serves is not invoked by reflection and is left alone
/// ([`served_by_a_handler`]). Nothing else is converted. A value that is still not of its
/// parameter's class (a narrowing, any other kind) is refused by [`check_arguments`], which
/// runs next; the items of a collection stay for the evaluator to refuse. The call's
/// static type is not touched. Measured by SANTA (`sigma-rust-argument-kind-reply.md`, oracle
/// sigma-state 6.0.7).
pub(crate) fn widen_arguments<'ctx>(
    method: &SMethod,
    mut args: Vec<Value<'ctx>>,
) -> Vec<Value<'ctx>> {
    if served_by_a_handler(method) {
        return args;
    }
    // the method as its owner's table declares it, fetched only for a call that has a
    // narrower integer where an integer is expected: no well-typed call does
    let mut declared: Option<Option<SMethod>> = None;
    for (i, arg) in args.iter_mut().enumerate() {
        let Some(param) = method.tpe().t_dom.get(i + 1) else {
            break;
        };
        let widened = match (param, &*arg) {
            (SType::SShort, Value::Byte(b)) => Value::Short(i16::from(*b)),
            (SType::SInt, Value::Byte(b)) => Value::Int(i32::from(*b)),
            (SType::SInt, Value::Short(s)) => Value::Int(i32::from(*s)),
            (SType::SLong, Value::Byte(b)) => Value::Long(i64::from(*b)),
            (SType::SLong, Value::Short(s)) => Value::Long(i64::from(*s)),
            (SType::SLong, Value::Int(n)) => Value::Long(i64::from(*n)),
            _ => continue,
        };
        let declared =
            declared.get_or_insert_with(|| method.obj_type.method_by_id(&method.method_id()));
        let declared_param = declared.as_ref().and_then(|m| m.tpe().t_dom.get(i + 1));
        if declared_param == Some(param) {
            *arg = widened;
        }
    }
    args
}

/// sigmastate invokes a method by Java reflection, which refuses an argument that is not of
/// the class of its parameter (v6.0.6 `values.scala:1361-1391`). The parameter's class comes
/// from the type the method DECLARES in its owner's table, for the Java method of a
/// fixed-cost method (`SMethod.scala:92-101`) and for the `_eval` method of any other
/// (`:138-160`): an integer type is its primitive (a narrower integer has been widened by
/// then, [`widen_arguments`]), `BigInt`, `UnsignedBigInt` and `GroupElement` their class, any
/// `Coll[..]` is `sigma.Coll` whatever its items, a function is `scala.Function1`, and a type
/// variable is `java.lang.Object`, which takes any value. So a method that adds its own costs
/// refuses such an argument before anything of its own is charged, and a fixed-cost method
/// after its cost.
///
/// A method a handler serves is not invoked by reflection and is left alone
/// ([`served_by_a_handler`]): its evaluator casts what it reads to the exact class. What a
/// class does not say stays for the evaluator: a collection's item type, a value's range.
/// Read from the sigma-state 6.0.7 jars for every method (each Java method's parameter
/// classes) and measured against the JVM on every parameter with eighteen kinds of value.
pub(crate) fn check_arguments(method: &SMethod, args: &[Value<'_>]) -> Result<(), EvalError> {
    if served_by_a_handler(method) {
        return Ok(());
    }
    let Some(declared) = method.obj_type.method_by_id(&method.method_id()) else {
        return Ok(());
    };
    for (i, arg) in args.iter().enumerate() {
        let Some(param) = declared.tpe().t_dom.get(i + 1) else {
            break;
        };
        if !is_of_class(param, arg) {
            return Err(EvalError::UnexpectedValue(format!(
                "method {} of {:?}: argument {i} is not of the class of {param:?}",
                method.name(),
                method.obj_type
            )));
        }
    }
    Ok(())
}

/// Whether `v` is a value of the JVM class of the type `tpe`: the class reflection checks an
/// argument against (`Evaluation.stypeToRType(tpe).classTag.runtimeClass`). The class only: no
/// item type, no range. The rows of the types a method declares for a parameter today (the
/// four integers, `BigInt`, `UnsignedBigInt`, `GroupElement`, a collection, a function, a
/// type variable) are measured; the others are by reading.
fn is_of_class(tpe: &SType, v: &Value<'_>) -> bool {
    match tpe {
        // `java.lang.Object`
        SType::STypeVar(_) => true,
        // by reading: `AnyType`, also `Object`
        SType::SAny => true,
        SType::SByte => matches!(v, Value::Byte(_)),
        SType::SShort => matches!(v, Value::Short(_)),
        SType::SInt => matches!(v, Value::Int(_)),
        SType::SLong => matches!(v, Value::Long(_)),
        SType::SBigInt => matches!(v, Value::BigInt(_)),
        SType::SUnsignedBigInt => matches!(v, Value::UnsignedBigInt(_)),
        SType::SGroupElement => matches!(v, Value::GroupElement(_)),
        SType::SColl(_) => matches!(v, Value::Coll(_)),
        SType::SFunc(_) => matches!(v, Value::Lambda(_)),
        // by reading, from here on: no method declares a parameter of these types
        SType::SUnit => matches!(v, Value::Unit),
        SType::SBoolean => matches!(v, Value::Boolean(_)),
        SType::SSigmaProp => matches!(v, Value::SigmaProp(_)),
        SType::SBox => matches!(v, Value::CBox(_)),
        SType::SAvlTree => matches!(v, Value::AvlTree(_)),
        SType::SOption(_) => matches!(v, Value::Opt(_)),
        // a pair is a `Tuple2`; a tuple of any other length is a `Coll[Any]`
        SType::STuple(t) => {
            if t.items.len() == 2 {
                matches!(v, Value::Pair(_))
            } else {
                matches!(v, Value::Coll(_))
            }
        }
        SType::SContext => matches!(v, Value::Context),
        SType::SString => matches!(v, Value::String(_)),
        SType::SHeader => matches!(v, Value::Header(_)),
        SType::SPreHeader => matches!(v, Value::PreHeader(_)),
        SType::SGlobal => matches!(v, Value::Global),
        // `NoType` has no runtime type (`stypeToRType` throws, `Evaluation.scala:55`), so a
        // method declaring it could not be looked up at all
        SType::SNoType => false,
    }
}

/// sigmastate parses a call of any arity, and its evaluator invokes the method with the
/// arguments it has, after evaluating them all (v6.0.7 `values.scala:1361-1393`): a reflective
/// invocation (`javaMethod.invoke`, `evalMethod.invoke`) fails on any other count than the
/// method's, and a `userDefinedInvoke` handler on a missing argument only. The parser builds a
/// call of any arity too, so the evaluator makes the same checks.
pub(crate) fn check_arity(method: &SMethod, n_args: usize) -> Result<(), EvalError> {
    let expected = method.tpe().t_dom.len().saturating_sub(1);
    if n_args != expected && !(n_args > expected && served_by_a_handler(method)) {
        return Err(EvalError::Misc(format!(
            "method {} takes {expected} arguments, got {n_args}",
            method.name()
        )));
    }
    Ok(())
}

/// The methods sigmastate implements with a `userDefinedInvoke` handler: the numeric `toBytes`,
/// `toBits`, `bitwiseInverse`, `bitwiseOr`, `bitwiseAnd`, `bitwiseXor`, `shiftLeft` and
/// `shiftRight` (v6.0.7 `methods.scala:311-458`, the only `withUserDefinedInvoke` calls). A
/// handler is not invoked by reflection. It reads the arguments it needs (`other.head`, or
/// none) and ignores any more, and it casts what it reads to the exact class
/// (`other.head.asInstanceOf[Int]`).
fn served_by_a_handler(method: &SMethod) -> bool {
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

/// The owners of sigmastate's methods, for the tests of this file that run over all of them
#[cfg(test)]
#[cfg(feature = "arbitrary")]
mod test_owners {
    use ergotree_ir::types::stype_companion::STypeCompanion;

    /// The names are written once and expand to the list AND to a `match` with no wildcard:
    /// a new owner does not compile until it is named here, a name left out is a pattern not
    /// covered, and a name written twice is an unreachable pattern
    macro_rules! owners {
        ($($owner:ident),* $(,)?) => {{
            match STypeCompanion::SByte {
                $(STypeCompanion::$owner)|* => {}
            }
            alloc::vec![$(STypeCompanion::$owner),*]
        }};
    }

    pub(super) fn all() -> alloc::vec::Vec<STypeCompanion> {
        owners![
            SByte,
            SShort,
            SInt,
            SLong,
            SBigInt,
            SUnsignedBigInt,
            Context,
            Box,
            Coll,
            GroupElem,
            SigmaProp,
            Global,
            Header,
            PreHeader,
            Option,
            AvlTree,
        ]
    }
}

/// The call site refuses a receiver that is not of the kind of the method's owner
/// ([`check_receiver`]): SANTA's rows, the owner table, and the receivers that must still pass
#[cfg(test)]
#[cfg(feature = "arbitrary")]
#[allow(clippy::unwrap_used)]
mod receiver_kind_tests {
    use alloc::boxed::Box;
    use alloc::sync::Arc;
    use alloc::vec;
    use alloc::vec::Vec;
    use ergo_chain_types::ec_point::generator;
    use ergo_chain_types::{Header, PreHeader};
    use ergotree_ir::bigint256::BigInt256;
    use ergotree_ir::chain::ergo_box::ErgoBox;
    use ergotree_ir::mir::avl_tree_data::AvlTreeData;
    use ergotree_ir::mir::coll_value::CollValue;
    use ergotree_ir::mir::expr::Expr;
    use ergotree_ir::mir::value::{Lambda, Value};
    use ergotree_ir::reference::Ref;
    use ergotree_ir::sigma_protocol::sigma_boolean::SigmaProp;
    use ergotree_ir::types::stype::SType;
    use ergotree_ir::types::stype_companion::STypeCompanion;
    use ergotree_ir::unsignedbigint256::UnsignedBigInt;
    use num_traits::Num;
    use sigma_test_util::force_any_val;

    use super::{is_of_class, receiver_is_of};
    use crate::eval::test_util::{assert_trees, eval_v3_body};

    #[test]
    fn santa_receiver_kind_rows() {
        // SANTA `eval/v6/authored/MethodCall.receiver_kind.json` (sigma-state 6.0.7), by row:
        // the JVM parses each tree and refuses the wrong receiver when it invokes the method
        let pow_hit = UnsignedBigInt::from_str_radix(
            "326674862673836209462483453386286740270338859283019276168539876024851191344",
            10,
        )
        .unwrap();
        assert_trees(&[
            // 1 `Global.encodeNbits(BigInt 1)`, 2 on `Int 1`, 3 on `CONTEXT`
            ("0b08dc6a06dd01060101", Some((Value::Long(16842752), 39))),
            ("0b09dc6a06040201060101", None),
            ("0b08dc6a06fe01060101", None),
            // 4 `Global.decodeNbits(16842752L)`, 5 on `Int 1`
            (
                "0b0adc6a07dd010580808810",
                Some((BigInt256::from(1i64).into(), 64)),
            ),
            ("0b0bdc6a070402010580808810", None),
            // 6 `Global.powHit(32, msg, nonce, h, 1048576)`, 7 on `Int 1`
            (
                "0b25dc6a08dd0504400e070a101b8c6a4f2e0e08000000000000002c0e04000000000480808001",
                Some((Value::UnsignedBigInt(pow_hit), 765)),
            ),
            (
                "0b26dc6a0804020504400e070a101b8c6a4f2e0e08000000000000002c0e04000000000480808001",
                None,
            ),
            // 9 `getVarFromInput[Int](0, 1)` on `Global`, 10 on `Int 1`
            ("0b0adc650cdd020300020104", None),
            ("0b0bdc650c0402020300020104", None),
        ]);
        // 8 the same call on `CONTEXT`: `None` @25 in SANTA's context, which binds no variable.
        // This context is random, so the row pins an option and the cost
        let (_, res, cost) = eval_v3_body("0b0adc650cfe020300020104");
        assert!(matches!(res, Ok(Value::Opt(_))), "{res:?}");
        assert_eq!(cost, 25);
    }

    #[test]
    fn santa_numeric_receiver_width_rows() {
        // SANTA `eval/v6/authored/MethodCall.numeric_receiver_width.json` (sigma-state 6.0.7):
        // the numeric handlers cast the receiver to the OWNER's class, the type code in the
        // bytes (`methods.scala:314-458`), so a value of another width throws
        let bytes = |b: &[i8]| Value::Coll(CollValue::Bytes(b.into()));
        assert_trees(&[
            // N0 `SLong.toBytes` on `Long 1`, N1 on `Int 1`
            (
                "0b05db05060502",
                Some((bytes(&[0, 0, 0, 0, 0, 0, 0, 1]), 14)),
            ),
            ("0b05db05060402", None),
            // N2 `SInt.toBytes`, N3 `SInt.toBits`, N4 `SInt.bitwiseInverse`, each on `Long 1`
            ("0b05db04060502", None),
            ("0b05db04070502", None),
            ("0b05db04080502", None),
            // N5 `SLong.shiftLeft(1)` on `Long 1`, N6 on `Int 1`; N7 `SLong.shiftRight(1)` on `Int 1`
            ("0b08dc050c0502010402", Some((Value::Long(2), 19))),
            ("0b08dc050c0402010402", None),
            ("0b08dc050d0402010402", None),
            // N8 `SBigInt.toBytes`, N9 `SByte.toBytes`, N10 `SLong.bitwiseOr(1)`, each on `Int 1`
            ("0b05db06060402", None),
            ("0b05db02060402", None),
            ("0b08dc05090402010402", None),
            // A1 `SLong.bitwiseOr(Int 1)` on `Long 1`: the ARGUMENT is of another width
            ("0b08dc05090502010402", None),
        ]);
    }

    /// The variant of a value, written out so that a new variant fails to compile here
    fn variant(v: &Value) -> &'static str {
        match v {
            Value::Boolean(_) => "Boolean",
            Value::Byte(_) => "Byte",
            Value::Short(_) => "Short",
            Value::Int(_) => "Int",
            Value::Long(_) => "Long",
            Value::Unit => "Unit",
            Value::BigInt(_) => "BigInt",
            Value::GroupElement(_) => "GroupElement",
            Value::SigmaProp(_) => "SigmaProp",
            Value::UnsignedBigInt(_) => "UnsignedBigInt",
            Value::CBox(_) => "CBox",
            Value::AvlTree(_) => "AvlTree",
            Value::Coll(_) => "Coll",
            Value::Pair(_) => "Pair",
            Value::Context => "Context",
            Value::String(_) => "String",
            Value::Header(_) => "Header",
            Value::PreHeader(_) => "PreHeader",
            Value::Global => "Global",
            Value::Opt(_) => "Opt",
            Value::Lambda(_) => "Lambda",
        }
    }

    /// One value of every variant, and more of those that have forms
    fn one_of_each() -> Vec<Value<'static>> {
        let ints = || CollValue::Ints(vec![1, 2].into());
        vec![
            Value::Boolean(true),
            Value::Byte(1),
            Value::Short(1),
            Value::Int(1),
            Value::Long(1),
            Value::Unit,
            Value::BigInt(num_bigint::BigInt::from(1)),
            // wider than 256 bits (below tree version 3): the check is of the kind only
            Value::BigInt(num_bigint::BigInt::from(1) << 255u32),
            Value::from(generator()),
            Value::SigmaProp(Box::new(force_any_val::<SigmaProp>())),
            Value::UnsignedBigInt(UnsignedBigInt::from(1u32)),
            Value::CBox(Ref::from(force_any_val::<ErgoBox>())),
            Value::AvlTree(Box::new(force_any_val::<AvlTreeData>())),
            // a collection in each of its forms
            Value::Coll(CollValue::Bytes(vec![1i8].into())),
            Value::Coll(CollValue::Booleans(vec![true].into())),
            Value::Coll(ints()),
            Value::Coll(CollValue::generic(SType::SLong, vec![Value::Long(1)])),
            Value::Coll(CollValue::Pairs {
                ls: Arc::new(ints()),
                rs: Arc::new(ints()),
            }),
            Value::pair(Value::Int(1), Value::Int(2)),
            Value::Context,
            Value::String("a".into()),
            Value::Header(Box::new(force_any_val::<Header>())),
            Value::PreHeader(Box::new(force_any_val::<PreHeader>())),
            Value::Global,
            Value::Opt(Some(Box::new(Value::Int(1)))),
            Value::Opt(None),
            Value::Lambda(Lambda {
                args: vec![],
                body: Arc::new(Expr::Const(1i32.into())),
                captured: Default::default(),
            }),
        ]
    }

    #[test]
    fn the_owner_table_accepts_exactly_its_own_kind() {
        let values = one_of_each();
        // every variant is offered
        let mut offered: Vec<&str> = values.iter().map(variant).collect();
        offered.sort_unstable();
        offered.dedup();
        assert_eq!(offered.len(), 21);
        // sigmastate's class of each owner (`SType.scala`, `reprClass`), as the variant holding
        // it; no wildcard, so a new owner does not compile until it has its row
        fn takes(owner: STypeCompanion) -> &'static str {
            match owner {
                STypeCompanion::SByte => "Byte",
                STypeCompanion::SShort => "Short",
                STypeCompanion::SInt => "Int",
                STypeCompanion::SLong => "Long",
                STypeCompanion::SBigInt => "BigInt",
                STypeCompanion::SUnsignedBigInt => "UnsignedBigInt",
                STypeCompanion::Context => "Context",
                STypeCompanion::Box => "CBox",
                STypeCompanion::Coll => "Coll",
                STypeCompanion::GroupElem => "GroupElement",
                STypeCompanion::SigmaProp => "SigmaProp",
                STypeCompanion::Global => "Global",
                STypeCompanion::Header => "Header",
                STypeCompanion::PreHeader => "PreHeader",
                STypeCompanion::Option => "Opt",
                STypeCompanion::AvlTree => "AvlTree",
            }
        }
        for owner in super::test_owners::all() {
            let takes = takes(owner);
            for v in &values {
                assert_eq!(
                    receiver_is_of(owner, v),
                    variant(v) == takes,
                    "{owner:?} on a {}",
                    variant(v)
                );
            }
        }
    }

    #[test]
    fn the_class_table_accepts_exactly_its_own_class() {
        use ergotree_ir::types::sfunc::SFunc;
        use ergotree_ir::types::stuple::STuple;
        use ergotree_ir::types::stype_param::STypeVar;
        // The JVM class of each type (`Evaluation.stypeToRType`), as the variant holding it;
        // "any" for `Object` and "none" for a type with no class. Each row is written once
        // and expands to its sample type AND to an arm of a `match` with no wildcard: a new
        // type does not compile until it has a row, and no row can be left out of the samples
        macro_rules! class_rows {
            ($($sample:expr => $pat:pat $(if $guard:expr)? => $takes:literal),* $(,)?) => {
                fn takes(tpe: &SType) -> &'static str {
                    match tpe {
                        $($pat $(if $guard)? => $takes),*
                    }
                }
                fn samples() -> Vec<SType> {
                    vec![$($sample),*]
                }
            };
        }
        class_rows! {
            SType::STypeVar(STypeVar::t()) => SType::STypeVar(_) => "any",
            SType::SAny => SType::SAny => "any",
            SType::SNoType => SType::SNoType => "none",
            SType::SUnit => SType::SUnit => "Unit",
            SType::SBoolean => SType::SBoolean => "Boolean",
            SType::SByte => SType::SByte => "Byte",
            SType::SShort => SType::SShort => "Short",
            SType::SInt => SType::SInt => "Int",
            SType::SLong => SType::SLong => "Long",
            SType::SBigInt => SType::SBigInt => "BigInt",
            SType::SGroupElement => SType::SGroupElement => "GroupElement",
            SType::SSigmaProp => SType::SSigmaProp => "SigmaProp",
            SType::SUnsignedBigInt => SType::SUnsignedBigInt => "UnsignedBigInt",
            SType::SBox => SType::SBox => "CBox",
            SType::SAvlTree => SType::SAvlTree => "AvlTree",
            SType::SOption(Arc::new(SType::SInt)) => SType::SOption(_) => "Opt",
            // the item type is no part of the class
            SType::SColl(Arc::new(SType::SColl(Arc::new(SType::SByte)))) => SType::SColl(_) => "Coll",
            SType::STuple(STuple::pair(SType::SInt, SType::SInt))
                => SType::STuple(t) if t.items.len() == 2 => "Pair",
            SType::STuple(STuple::triple(SType::SInt, SType::SInt, SType::SInt))
                => SType::STuple(_) => "Coll",
            SType::SFunc(SFunc::new(vec![SType::SInt], SType::SInt)) => SType::SFunc(_) => "Lambda",
            SType::SContext => SType::SContext => "Context",
            SType::SString => SType::SString => "String",
            SType::SHeader => SType::SHeader => "Header",
            SType::SPreHeader => SType::SPreHeader => "PreHeader",
            SType::SGlobal => SType::SGlobal => "Global",
        }
        let types = samples();
        assert_eq!(types.len(), 25);
        let values = one_of_each();
        for tpe in &types {
            let takes = takes(tpe);
            for v in &values {
                assert_eq!(
                    is_of_class(tpe, v),
                    takes == "any" || variant(v) == takes,
                    "{tpe:?} {v:?}"
                );
            }
        }
    }

    #[test]
    fn a_right_receiver_evaluates_for_every_owner() {
        // Rust's own costs on fork `master` `f3c2cbc7`: a guard against a wrong row in the
        // owner table. Header, PreHeader and AvlTree receivers are covered by the suites of
        // `eval/sheader.rs`, `eval/spreheader.rs` and `eval/savltree.rs`
        for (tree_hex, cost) in [
            ("0b05db02060201", 14),           // SByte.toBytes on Byte 1
            ("0b05db03060302", 14),           // SShort.toBytes on Short 1
            ("0b05db04060402", 14),           // SInt.toBytes on Int 1
            ("0b05db05060502", 14),           // SLong.toBytes on Long 1
            ("0b06db0606060101", 14),         // SBigInt.toBytes on BigInt 1
            ("0b06db0906090101", 14),         // SUnsignedBigInt.toBytes on UnsignedBigInt 1
            ("0b0adb0c0183020404020404", 48), // Coll.size on Coll(1, 2)
            ("0b08db2402db6a0add04", 28),     // Option.isDefined on Global.none[Int]
            ("0b08dc6a06dd01060101", 39),     // Global.encodeNbits(BigInt 1)
            ("0b0adc650cfe020300020104", 25), // CONTEXT.getVarFromInput[Int](0, 1)
            ("0b04db6301a7", 22),             // Box.value on SELF
            ("0b07db0702db6a01dd", 273),      // GroupElement.getEncoded on groupGenerator
        ] {
            let (_, res, c) = eval_v3_body(tree_hex);
            assert!(res.is_ok(), "{tree_hex}: {res:?}");
            assert_eq!(c, cost, "{tree_hex}");
        }
    }

    #[test]
    fn a_pair_is_not_a_collection_receiver() {
        // a `Tuple` node of two items evaluates to a `Tuple2`, which is not a `Coll`
        // (`values.scala:824-837`); a tuple CONSTANT of three items is the `Coll[Any]` its data
        // holds (`Evaluation.scala:99-101`). `(1, 2).size`, the node, and `(1, 2, 3).size`, the
        // constant, through `Coll.size`. Measured on the sigma-state 6.0.7 jars (a profiling
        // evaluator on the bodies `db0c01860204020404` and `db0c0148040404020406`): the first
        // throws `IllegalArgumentException`, "object is not an instance of declaring class",
        // the second is 3 @23.
        // The first row cannot tell `check_receiver` from `Coll.size`'s own refusal: without
        // the check the evaluator fails with the same variant. On the JVM the two are one
        // event, the invoke
        assert_trees(&[
            ("0b09db0c01860204020404", None),
            ("0b0adb0c0148040404020406", Some((Value::Int(3), 23))),
        ]);
    }

    #[test]
    fn an_any_typed_receiver_is_judged_by_its_value() {
        // by reading: the receiver is `(1, 2, 3)(0)`, an `Int` typed `Any`; the handler casts
        // the VALUE to the owner's class (`methods.scala:314-321`), so `SLong.toBytes` throws
        assert_trees(&[("0b0edb0506b248040404020406040000", None)]);
        // and `SInt.toBytes` returns (Rust's own cost)
        let (_, res, cost) = eval_v3_body("0b0edb0406b248040404020406040000");
        assert!(res.is_ok(), "{res:?}");
        assert_eq!(cost, 49);
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

#[cfg(test)]
#[cfg(feature = "arbitrary")]
#[allow(clippy::unwrap_used)]
mod argument_widening_tests {
    use alloc::boxed::Box;
    use alloc::string::String;
    use alloc::vec;
    use alloc::vec::Vec;
    use ergotree_ir::chain::context::Context;
    use ergotree_ir::mir::value::Value;
    use ergotree_ir::serialization::sigma_byte_reader::{from_bytes, SigmaByteRead};
    use ergotree_ir::serialization::SigmaSerializable;
    use ergotree_ir::types::smethod::SMethod;
    use ergotree_ir::types::stype::SType;
    use ergotree_ir::types::stype_param::STypeVar;
    use ergotree_ir::types::{scoll, scontext, sglobal};
    use sigma_test_util::force_any_val;

    use super::widen_arguments;
    use crate::eval::test_util::{assert_trees, eval_tree, eval_v3_body, eval_v3_body_in};

    /// Each row evaluates to its control's value, at the control's cost plus the row's extra
    /// (15 for a `lie`: the `If`, its condition and the taken branch), in one context
    fn assert_as_control(control: &str, control_cost: Option<u64>, rows: &[(&str, u64)]) {
        let ctx = force_any_val::<Context>();
        let (_, expected, cost) = eval_v3_body_in(ctx.clone(), control);
        let expected = expected.unwrap();
        if let Some(c) = control_cost {
            assert_eq!(cost, c, "{control}");
        }
        for (tree, extra) in rows {
            let (_, res, c) = eval_v3_body_in(ctx.clone(), tree);
            assert_eq!(res.as_ref().ok(), Some(&expected), "{tree}: {res:?}");
            assert_eq!(c, cost + extra, "{tree}");
        }
    }

    // SANTA's rows (`sigma-rust-argument-kind-reply.md`, oracle sigma-state 6.0.7): Java
    // reflection widens a `Byte` or `Short` for an `Int` parameter, and those and an `Int`
    // for a `Long` one. `c` is the constant `Coll[Int](0,1,2)`; `lie(a, b)` is
    // `If(false, a, b)`, typed as `a` and valued as `b`.
    #[test]
    fn santa_widened_argument_rows() {
        // W1: c.apply(1) @44; lie(1, 1.toByte), 1.toByte, lie(1, 1.toShort)
        assert_as_control(
            "0b0bdc0c0a1003000204010402",
            Some(44),
            &[
                ("0b10dc0c0a10030002040195010004020201", 15),
                ("0b0bdc0c0a1003000204010201", 0),
                ("0b10dc0c0a10030002040195010004020302", 15),
            ],
        );
        // W2: c.getOrElse(1, 7) @49
        assert_as_control(
            "0b0ddc0c021003000204020402040e",
            Some(49),
            &[
                ("0b12dc0c0210030002040295010004020201040e", 15),
                ("0b0ddc0c021003000204020302040e", 0),
            ],
        );
        // W3: c.get(1) @44
        assert_as_control(
            "0b0bdc0c211003000204010402",
            Some(44),
            &[("0b10dc0c2110030002040195010004020201", 15)],
        );
        // W4: c.indexOf(1, 0) @55
        assert_as_control(
            "0b0ddc0c1a10030002040204020400",
            Some(55),
            &[("0b12dc0c1a100300020402040295010004000200", 15)],
        );
        // W5: c.patch(0, Coll(7), 1) @56: a Byte `from`, a Short `replaced`
        assert_as_control(
            "0b10dc0c13100300020403040010010e0402",
            Some(56),
            &[
                ("0b15dc0c131003000204039501000400020010010e0402", 15),
                ("0b15dc0c13100300020403040010010e95010004020302", 15),
            ],
        );
        // W6: c.updated(0, 7) @40
        assert_as_control(
            "0b0ddc0c141003000204020400040e",
            Some(40),
            &[("0b12dc0c1410030002040295010004000300040e", 15)],
        );
        // W7: Global.decodeNbits(1L) @64: an Int, a Byte; a Short (the JVM by this round's
        // harness on the sigma-state 6.0.7 jars, not SANTA's oracle)
        assert_as_control(
            "0b07dc6a07dd010502",
            Some(64),
            &[
                ("0b0cdc6a07dd0195010005020402", 15),
                ("0b07dc6a07dd010201", 0),
                ("0b07dc6a07dd010302", 0),
            ],
        );
        // W8: Global.powHit(32, msg, nonce, h, 1024) @765: a Byte k, a Short N
        assert_as_control(
            "0b1ddc6a08dd0504400e01010e0800000000000000010e0400000001048010",
            Some(765),
            &[
                (
                    "0b22dc6a08dd05950100044002200e01010e0800000000000000010e0400000001048010",
                    15,
                ),
                (
                    "0b23dc6a08dd0504400e01010e0800000000000000010e0400000001950100048010038010",
                    15,
                ),
            ],
        );
        // W9: SELF.getReg[Long](0) (@69 in SANTA's context)
        assert_as_control(
            "0b08dc6313a701040005",
            None,
            &[
                ("0b0ddc6313a7019501000400020005", 15),
                ("0b08dc6313a701030005", 0),
            ],
        );
        // W10: CONTEXT.getVarFromInput[Int](0.toShort, 1.toByte): a Byte input index
        assert_as_control(
            "0b0adc650cfe020300020104",
            None,
            &[("0b0fdc650cfe0295010003000200020104", 15)],
        );
        // V1, V2: tree version 0
        assert_trees(&[
            (
                "00dc0c0a10030002040195010004020201",
                Some((Value::Int(1), 59)),
            ),
            ("00dc0c021003000204020302040e", Some((Value::Int(1), 49))),
        ]);
    }

    // A negative argument is widened with its sign. The JVM by this round's harness on the
    // sigma-state 6.0.7 jars, not SANTA's oracle.
    #[test]
    fn a_negative_argument_keeps_its_sign() {
        // c.indexOf(1, -1) @55 searches from the start: 1. With lie(-1, (-1).toByte) and with
        // (-1).toShort; a start of 255 or 65535 would find nothing
        assert_as_control(
            "0b0ddc0c1a10030002040204020401",
            Some(55),
            &[
                ("0b12dc0c1a1003000204020402950100040102ff", 15),
                ("0b0ddc0c1a10030002040204020301", 0),
            ],
        );
        // Global.decodeNbits(-1L) throws (out of the 256-bit range), and so do (-1).toByte and
        // (-1).toShort; decodeNbits(255L) and (65535L) are 0
        assert_trees(&[
            ("0b07dc6a07dd010501", None),
            ("0b07dc6a07dd0102ff", None),
            ("0b07dc6a07dd010301", None),
            (
                "0b09dc6a07dd0105fe03",
                Some((Value::BigInt(num_bigint::BigInt::from(0)), 64)),
            ),
            (
                "0b0adc6a07dd0105feff07",
                Some((Value::BigInt(num_bigint::BigInt::from(0)), 64)),
            ),
        ]);
    }

    // SANTA's receivers: a `Coll[Byte]`, a `Coll[Long]`, a pair collection, a three-item
    // tuple constant, a `Coll[Box]`, a `ConcreteCollection`
    #[test]
    fn santa_widened_argument_receivers() {
        assert_trees(&[
            ("0b0bdc0c0a0e03000102010201", Some((Value::Byte(1), 44))),
            (
                "0b10dc0c0a11030002040195010004020201",
                Some((Value::Long(1), 59)),
            ),
            (
                "0b18dc0c0adc0c1d830204040204040183020404060408010201",
                Some((Value::Pair(Box::new((Value::Int(2), Value::Int(4)))), 114)),
            ),
            ("0b0ddc0c0a48040404020406010201", Some((Value::Int(2), 44))),
            (
                "0b11dc0c02830304040004020404020201040e",
                Some((Value::Int(1), 79)),
            ),
            (
                "0b12dc0c020e0300010202950100040203020207",
                Some((Value::Byte(1), 64)),
            ),
        ]);
        // Coll[Box](SELF).apply(0) and (0.toByte)
        assert_as_control(
            "0b0adc0c0a830163a7010400",
            None,
            &[("0b0adc0c0a830163a7010200", 0)],
        );
    }

    // What the JVM refuses stays refused: a narrowing, a handler's strict cast, a `byte`
    // parameter, a collection's items, a type variable, another kind
    #[test]
    fn santa_unwidened_argument_rows() {
        assert_trees(&[
            // W1n: c.apply(lie(1, 1L))
            ("0b10dc0c0a10030002040195010004020502", None),
            // X1-X3: 1.shiftLeft(lie(1, 1.toByte)), 1L.shiftRight(1.toShort), BigInt(1).shiftLeft(..)
            ("0b0ddc040c04020195010004020201", None),
            ("0b08dc050d0502010302", None),
            ("0b0edc060c0601010195010004020201", None),
            // X4: tree.updateOperations(lie(1.toByte, 1.toShort)).enabledOperations
            ("0b33db6402dc6408640000000000000000000000000000000000000000000000000000000000000000000720000195010002010302", None),
            // X5: c.updateMany(lie(Coll[Int](0), Coll[Byte](0)), Coll(7))
            ("0b15dc0c151003000204029501001001000e010010010e", None),
            // T1: Coll[Long](0,1,2).getOrElse(9, lie(7L, 7.toByte)): the default is a type
            // variable, which the specialised call types as Long
            ("0b12dc0c021103000204020412950100050e0207", None),
            // T1's control: getOrElse(9, 7L)
            ("0b0ddc0c021103000204020412050e", Some((Value::Long(7), 49))),
            // K1, K2, K3, K6: a SigmaProp, SELF, a PreHeader, a String for an Int
            ("0b11dc0c0a1003000204019501000402d10101", None),
            ("0b0fdc0c0a1003000204019501000402a7", None),
            ("0b12dc0c0a1003000204019501000402db6503fe", None),
            ("0b0cdc0c0a100300020401660161", None),
        ]);
    }

    // A Byte CONSTANT as the argument of a generic method leaves the call typed as the
    // method's type variable (the types do not unify), and the widening does not change that
    #[test]
    fn a_widened_call_keeps_its_static_type() {
        // W1d: c.apply(1.toByte)
        let (expr, res, _) = eval_v3_body("0b0bdc0c0a1003000204010201");
        assert_eq!(res.unwrap(), Value::Int(1));
        assert_eq!(expr.tpe(), SType::STypeVar(STypeVar::iv()));
        // Q1: c.apply(1.toByte) == 1 is refused when read (the JVM: `SameType2`)
        let bytes = base16::decode("93dc0c0a10030002040102010402").unwrap();
        assert!(from_bytes(&bytes)
            .with_tree_version(
                ergotree_ir::ergo_tree::ErgoTreeVersion::V3,
                ergotree_ir::mir::expr::Expr::sigma_parse
            )
            .is_err());
        // Q1e: sigmaProp(c.apply(1.toByte) == c.apply(1.toByte)) reads, and throws when evaluated
        let (res, _) = eval_tree("0b18d193dc0c0a1003000204010201dc0c0a1003000204010201");
        assert!(res.is_err());
        // Q2: sigmaProp(Coll(true,true).apply(1.toByte)) is its control, sigmaProp(.apply(1)) @59
        assert_as_control(
            "0b0ad1dc0c0a0d0203010402",
            Some(59),
            &[("0b0ad1dc0c0a0d0203010201", 0)],
        );
    }

    fn widened(method: &SMethod, args: Vec<Value<'static>>) -> Vec<Value<'static>> {
        widen_arguments(method, args)
    }

    #[test]
    fn the_six_widening_pairs_and_no_other() {
        let short = scontext::GET_VAR_FROM_INPUT_METHOD.clone(); // (Short, Byte)
        let int = scoll::APPLY_METHOD.clone(); // (Int)
        let long = sglobal::DECODE_NBITS_METHOD.clone(); // (Long)
        let ints = || {
            vec![
                Value::Byte(1),
                Value::Short(1),
                Value::Int(1),
                Value::Long(1),
            ]
        };
        let run = |m: &SMethod| -> Vec<Value<'static>> {
            ints()
                .into_iter()
                .map(|v| widened(m, vec![v, Value::Byte(1)]).remove(0))
                .collect()
        };
        assert_eq!(
            run(&short),
            vec![
                Value::Short(1),
                Value::Short(1),
                Value::Int(1),
                Value::Long(1)
            ]
        );
        assert_eq!(
            run(&int),
            vec![Value::Int(1), Value::Int(1), Value::Int(1), Value::Long(1)]
        );
        assert_eq!(
            run(&long),
            vec![
                Value::Long(1),
                Value::Long(1),
                Value::Long(1),
                Value::Long(1)
            ]
        );
        // the sign is kept, by each of the six pairs
        let neg = |m: &SMethod, v: Value<'static>| widened(m, vec![v, Value::Byte(1)]).remove(0);
        assert_eq!(neg(&short, Value::Byte(-1)), Value::Short(-1));
        assert_eq!(neg(&int, Value::Byte(-1)), Value::Int(-1));
        assert_eq!(neg(&int, Value::Short(-1)), Value::Int(-1));
        assert_eq!(neg(&long, Value::Byte(-1)), Value::Long(-1));
        assert_eq!(neg(&long, Value::Short(-1)), Value::Long(-1));
        assert_eq!(neg(&long, Value::Int(-1)), Value::Long(-1));
        // nothing but an integer is touched
        for v in [
            Value::Boolean(true),
            Value::Unit,
            Value::BigInt(num_bigint::BigInt::from(1)),
        ] {
            assert_eq!(widened(&int, vec![v.clone()]), vec![v]);
        }
        // getVarFromInput's second parameter is a Byte: nothing widens into it
        assert_eq!(
            widened(&short, vec![Value::Short(0), Value::Short(1)])[1],
            Value::Short(1)
        );
    }

    #[test]
    fn a_type_variable_is_not_widened_when_the_call_specialised_it() {
        // Coll[Long].getOrElse(Int, Long): the default is `IV` in the method's own table
        let method = scoll::GET_OR_ELSE_METHOD
            .clone()
            .with_concrete_types(&core::iter::once((STypeVar::iv(), SType::SLong)).collect());
        assert_eq!(method.tpe().t_dom[2], SType::SLong);
        assert_eq!(
            widened(&method, vec![Value::Byte(9), Value::Byte(7)]),
            vec![Value::Int(9), Value::Byte(7)]
        );
    }

    /// Every (owner, method, argument index, declared type) where a `Byte` argument comes
    /// back as another kind: the parameters the rule reaches
    fn reached() -> Vec<(&'static str, &'static str, usize, String)> {
        let mut rows = vec![];
        for owner in super::test_owners::all() {
            for method in owner.methods() {
                let n = method.tpe().t_dom.len() - 1;
                for i in 0..n {
                    let mut args = vec![Value::Unit; n];
                    args[i] = Value::Byte(1);
                    if widened(&method, args)[i] != Value::Byte(1) {
                        let tpe = alloc::format!("{:?}", method.tpe().t_dom[i + 1]);
                        rows.push((owner.type_name(), method.name(), i, tpe));
                    }
                }
            }
        }
        rows.sort();
        rows
    }

    // The JVM's list: the parameters declared `Short`, `Int` or `Long` of the methods it
    // invokes by reflection. `slice` and `getRegV5` are in it and cannot be invoked at all.
    #[test]
    fn the_rule_reaches_exactly_these_parameters() {
        let expected: Vec<(&str, &str, usize, &str)> = vec![
            ("Box", "getReg", 0, "SInt"),
            ("Box", "getRegV5", 0, "SInt"),
            ("Coll", "apply", 0, "SInt"),
            ("Coll", "get", 0, "SInt"),
            ("Coll", "getOrElse", 0, "SInt"),
            ("Coll", "indexOf", 1, "SInt"),
            ("Coll", "patch", 0, "SInt"),
            ("Coll", "patch", 2, "SInt"),
            ("Coll", "slice", 0, "SInt"),
            ("Coll", "slice", 1, "SInt"),
            ("Coll", "updated", 0, "SInt"),
            ("Context", "getVarFromInput", 0, "SShort"),
            ("Global", "decodeNBits", 0, "SLong"),
            ("Global", "powHit", 0, "SInt"),
            ("Global", "powHit", 4, "SInt"),
        ];
        let got = reached();
        let got: Vec<(&str, &str, usize, &str)> = got
            .iter()
            .map(|(o, m, i, t)| (*o, *m, *i, t.as_str()))
            .collect();
        assert_eq!(got, expected);
    }
}

/// Where a call's charges sit against its refusals. Every figure is the sigma-state 6.0.7
/// evaluator's accumulator at the throw, measured on the jars.
#[cfg(test)]
#[cfg(feature = "arbitrary")]
#[allow(clippy::unwrap_used)]
mod charge_order {
    use crate::eval::test_util::{eval_body, eval_body_err};
    use crate::eval::EvalError;
    use ergotree_ir::ergo_tree::ErgoTreeVersion;
    use ergotree_ir::mir::value::Value;

    // sigmastate evaluates the receiver and then adds the call's own cost
    // (`values.scala:1361-1363`): a receiver that throws has not paid for the call
    #[test]
    fn the_calls_own_cost_follows_the_receiver() {
        for body in [
            "b21000040000",               // the receiver alone: Coll[Int]().apply(0)
            "db0c01b21000040000",         // .size       (a fixed cost)
            "dc0c1db210000400000110010e", // .zip(..)    (per item)
            "db0401b21000040000",         // .toByte     (no cost kind)
            "dc040cb21000040000010402",   // .shiftLeft  (a handler)
        ] {
            let (e, c) = eval_body_err(body);
            assert!(matches!(e, EvalError::Misc(_)), "{body}: {e:?}");
            assert_eq!(c, 40, "{body}");
        }
    }

    // sigmastate invokes a method by reflection, which refuses an argument that is not of the
    // class of the parameter the method declares (`SMethod.scala:92-101`, `:138-160`): before
    // the body of a method that adds its own costs, after the cost of a fixed-cost method
    #[test]
    fn a_wrong_class_is_refused_before_the_method_charges() {
        const T: &str =
            "64000000000000000000000000000000000000000000000000000000000000000000072000";
        let tree = |id: &str, args: &str| alloc::format!("dc64{id}{T}{args}");
        let rows: alloc::vec::Vec<(alloc::string::String, u64)> = alloc::vec![
            ("dc0c021003000204020101040e".into(), 19), // G1 getOrElse(true, 7)
            ("dc0c141003000204020101040e".into(), 19), // G2 updated(true, 7)
            ("dc0c15100300020402010110010e".into(), 19), // G3 updateMany(true, ..)
            ("dc0c151003000204021001000101".into(), 19), // G4 updateMany(.., true)
            ("dc0c1d1003000204010101".into(), 14),     // G5 zip(true)
            ("dc0c1d10030002040195010010010e62".into(), 29), // G6 zip(If(false, coll, ()))
            ("dc0c1f1003000204010101".into(), 14),     // G7 startsWith(true)
            ("dc0c201003000204010101".into(), 14),     // G8 endsWith(true)
            (tree("0c", "0201010e0102"), 19),          // G9 tree.insert(true, proof)
            (tree("0c", "020c3c0e0e000101"), 19),      // G10 tree.insert(entries, true)
            (tree("0d", "020c3c0e0e000101"), 19),      // G11 tree.update(entries, true)
            (tree("0e", "0201010e0102"), 19),          // G12 tree.remove(true, proof)
            (tree("10", "020c3c0e0e000101"), 19),      // G13 tree.insertOrUpdate(entries, true)
            ("dc0c1d100300020401580204".into(), 14),   // K2 zip((1, 2)): a pair is no collection
            ("dc0c0a1003000204010101".into(), 44),     // D1 apply(true): after its fixed 30
            // D2 g.exp(()): after its fixed 900
            (
                "dc0703070279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f817980162"
                    .into(),
                914,
            ),
        ];
        for (body, cost) in &rows {
            let (e, c) = eval_body_err(body);
            assert!(matches!(e, EvalError::UnexpectedValue(_)), "{body}: {e:?}");
            assert_eq!(c, *cost, "{body}");
        }
    }

    // an argument of the declared class reaches the method: a non-pair tuple is a collection, a
    // type variable takes anything, a collection's item type is not a class, and a narrower
    // integer is widened first. A handler-served method is not invoked by reflection: its
    // evaluator makes the exact check
    #[test]
    fn a_passing_class_reaches_the_body() {
        for (body, value, cost) in [
            ("dc0c1d10030002040148040404020406", None, 25), // K1 zip((1, 2, 3))
            ("dc0c0210030002040204020101", Some(Value::Int(1)), 49), // K3 getOrElse(1, true)
            ("dc0c1d10030002040111020204", None, 25),       // K4 zip(Coll[Long])
            ("dc0c0a1003000204010201", Some(Value::Int(1)), 44), // W1 apply(1.toByte)
        ] {
            let (res, c) = eval_body(body, ErgoTreeVersion::V3);
            let v = res.unwrap();
            if let Some(value) = value {
                assert_eq!(v, value, "{body}");
            }
            assert_eq!(c, cost, "{body}");
        }
        // W2 1.shiftLeft(1.toByte)
        let (e, c) = eval_body_err("dc040c0402010201");
        assert!(!matches!(e, EvalError::UnexpectedValue(_)), "{e:?}");
        assert_eq!(c, 19);
    }

    // The class check is the call's, so it is made also where the method's body would never
    // look at the argument: a modifying `AvlTree` method on a tree whose flag is OFF answers
    // `None` without reading anything, and the JVM still refuses a call whose argument is of
    // another class, before the body runs (each tree below has no flag set)
    #[test]
    fn a_wrong_class_is_refused_where_the_body_would_not_read_it() {
        #[rustfmt::skip]
        let rows = [
            // flagoff-insert-true-entries
            "dc640c6415cc1ba1054bf74431a2ccfc57e6051b9d0e7561f70586e5512c0ae3f3bac4fd020001000201010e0102",
            // flagoff-update-entries-true
            "dc640d6415cc1ba1054bf74431a2ccfc57e6051b9d0e7561f70586e5512c0ae3f3bac4fd02000100020c3c0e0e000101",
            // flagoff-remove-true-keys
            "dc640e6415cc1ba1054bf74431a2ccfc57e6051b9d0e7561f70586e5512c0ae3f3bac4fd020001000201010e0102",
            // flagoff-iou-entries-true
            "dc64106415cc1ba1054bf74431a2ccfc57e6051b9d0e7561f70586e5512c0ae3f3bac4fd02000100020c3c0e0e000101",
            // flagoff-zero-insert-true
            "dc640c640000000000000000000000000000000000000000000000000000000000000000000020000201010e0102",
        ];
        for body in rows {
            let (e, c) = eval_body_err(body);
            assert!(matches!(e, EvalError::UnexpectedValue(_)), "{body}: {e:?}");
            assert_eq!(c, 19, "{body}");
        }
    }
}
