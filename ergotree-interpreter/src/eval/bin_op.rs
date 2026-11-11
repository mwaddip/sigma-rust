//! Operators in ErgoTree

use ergotree_ir::bigint256::BigInt256;
use ergotree_ir::mir::bin_op::BinOp;
use ergotree_ir::mir::bin_op::BinOpKind;
use ergotree_ir::mir::bin_op::RelationOp;
use ergotree_ir::mir::bin_op::{ArithOp, LogicalOp};
use ergotree_ir::mir::constant::TryExtractFrom;
use ergotree_ir::mir::constant::TryExtractInto;
use ergotree_ir::mir::value::Value;
use ergotree_ir::types::stype::SType;
use ergotree_ir::unsignedbigint256::UnsignedBigInt;
use num_traits::CheckedAdd;
use num_traits::CheckedDiv;
use num_traits::CheckedMul;
use num_traits::CheckedRem;
use num_traits::CheckedSub;
use num_traits::Num;

use crate::eval::env::Env;
use crate::eval::unevaluated::not_evaluated;
use crate::eval::Context;
use crate::eval::EvalError;
use crate::eval::Evaluable;

fn arithmetic_err<T: core::fmt::Display>(
    op: &str,
    lv_raw: T,
    rv_raw: T,
    err_str: &str,
) -> EvalError {
    EvalError::ArithmeticException(format!(
        "({0}) {1} ({2}) resulted in {3}",
        lv_raw, op, rv_raw, err_str
    ))
}

fn eval_plus<'ctx, T>(lv_raw: T, rv_raw: T) -> Result<Value<'ctx>, EvalError>
where
    T: Num + CheckedAdd + Into<Value<'ctx>> + core::fmt::Display,
{
    lv_raw
        .checked_add(&rv_raw)
        .ok_or_else(|| arithmetic_err("+", lv_raw, rv_raw, "overflow"))
        .map(|t| t.into()) // convert T to Value
}

fn eval_minus<'ctx, T>(lv_raw: T, rv_raw: T) -> Result<Value<'ctx>, EvalError>
where
    T: Num + CheckedSub + Into<Value<'ctx>> + core::fmt::Display,
{
    lv_raw
        .checked_sub(&rv_raw)
        .ok_or_else(|| arithmetic_err("-", lv_raw, rv_raw, "overflow"))
        .map(|t| t.into()) // convert T to Value
}

fn eval_mul<'ctx, T>(lv_raw: T, rv_raw: T) -> Result<Value<'ctx>, EvalError>
where
    T: Num + CheckedMul + Into<Value<'ctx>> + core::fmt::Display,
{
    lv_raw
        .checked_mul(&rv_raw)
        .ok_or_else(|| arithmetic_err("*", lv_raw, rv_raw, "overflow"))
        .map(|t| t.into()) // convert T to Value
}

fn eval_div<'ctx, T>(lv_raw: T, rv_raw: T) -> Result<Value<'ctx>, EvalError>
where
    T: Num + CheckedDiv + Into<Value<'ctx>> + core::fmt::Display,
{
    lv_raw
        .checked_div(&rv_raw)
        .ok_or_else(|| arithmetic_err("/", lv_raw, rv_raw, "exception"))
        .map(|t| t.into()) // convert T to Value
}

fn eval_mod<'ctx, T>(lv_raw: T, rv_raw: T) -> Result<Value<'ctx>, EvalError>
where
    T: Num + CheckedRem + Into<Value<'ctx>> + core::fmt::Display,
{
    lv_raw
        .checked_rem(&rv_raw)
        .ok_or_else(|| arithmetic_err("%", lv_raw, rv_raw, "exception"))
        .map(|t| t.into()) // convert T to Value
}

fn eval_ge<'ctx>(lv: Value<'ctx>, rv: Value<'ctx>) -> Result<Value<'ctx>, EvalError> {
    match lv {
        Value::Byte(lv_raw) => Ok((lv_raw >= rv.try_extract_into::<i8>()?).into()),
        Value::Short(lv_raw) => Ok((lv_raw >= rv.try_extract_into::<i16>()?).into()),
        Value::Int(lv_raw) => Ok((lv_raw >= rv.try_extract_into::<i32>()?).into()),
        Value::Long(lv_raw) => Ok((lv_raw >= rv.try_extract_into::<i64>()?).into()),
        Value::BigInt(lv_raw) => Ok((lv_raw >= rv.try_extract_into::<BigInt256>()?).into()),
        Value::UnsignedBigInt(lv_raw) => {
            Ok((lv_raw >= rv.try_extract_into::<UnsignedBigInt>()?).into())
        }
        _ => Err(EvalError::UnexpectedValue(format!(
            "expected BinOp::left to be numeric value, got {0:?}",
            lv
        ))),
    }
}

fn eval_gt<'ctx>(lv: Value<'ctx>, rv: Value<'ctx>) -> Result<Value<'ctx>, EvalError> {
    match lv {
        Value::Byte(lv_raw) => Ok((lv_raw > rv.try_extract_into::<i8>()?).into()),
        Value::Short(lv_raw) => Ok((lv_raw > rv.try_extract_into::<i16>()?).into()),
        Value::Int(lv_raw) => Ok((lv_raw > rv.try_extract_into::<i32>()?).into()),
        Value::Long(lv_raw) => Ok((lv_raw > rv.try_extract_into::<i64>()?).into()),
        Value::BigInt(lv_raw) => Ok((lv_raw > rv.try_extract_into::<BigInt256>()?).into()),
        Value::UnsignedBigInt(lv_raw) => {
            Ok((lv_raw > rv.try_extract_into::<UnsignedBigInt>()?).into())
        }
        _ => Err(EvalError::UnexpectedValue(format!(
            "expected BinOp::left to be numeric value, got {0:?}",
            lv
        ))),
    }
}

fn eval_lt<'ctx>(lv: Value<'ctx>, rv: Value<'ctx>) -> Result<Value<'ctx>, EvalError> {
    match lv {
        Value::Byte(lv_raw) => Ok((lv_raw < rv.try_extract_into::<i8>()?).into()),
        Value::Short(lv_raw) => Ok((lv_raw < rv.try_extract_into::<i16>()?).into()),
        Value::Int(lv_raw) => Ok((lv_raw < rv.try_extract_into::<i32>()?).into()),
        Value::Long(lv_raw) => Ok((lv_raw < rv.try_extract_into::<i64>()?).into()),
        Value::BigInt(lv_raw) => Ok((lv_raw < rv.try_extract_into::<BigInt256>()?).into()),
        Value::UnsignedBigInt(lv_raw) => {
            Ok((lv_raw < rv.try_extract_into::<UnsignedBigInt>()?).into())
        }
        _ => Err(EvalError::UnexpectedValue(format!(
            "expected BinOp::left to be numeric value, got {0:?}",
            lv
        ))),
    }
}

fn eval_le<'ctx>(lv: Value<'ctx>, rv: Value<'ctx>) -> Result<Value<'ctx>, EvalError> {
    match lv {
        Value::Byte(lv_raw) => Ok((lv_raw <= rv.try_extract_into::<i8>()?).into()),
        Value::Short(lv_raw) => Ok((lv_raw <= rv.try_extract_into::<i16>()?).into()),
        Value::Int(lv_raw) => Ok((lv_raw <= rv.try_extract_into::<i32>()?).into()),
        Value::Long(lv_raw) => Ok((lv_raw <= rv.try_extract_into::<i64>()?).into()),
        Value::BigInt(lv_raw) => Ok((lv_raw <= rv.try_extract_into::<BigInt256>()?).into()),
        Value::UnsignedBigInt(lv_raw) => {
            Ok((lv_raw <= rv.try_extract_into::<UnsignedBigInt>()?).into())
        }
        _ => Err(EvalError::UnexpectedValue(format!(
            "expected BinOp::left to be numeric value, got {0:?}",
            lv
        ))),
    }
}

/// An arithmetic operation at `T`, the Rust type of the left operand's static type. sigmastate
/// charges the operation's cost for that type, then casts BOTH operand values to it (v6.0.6
/// `trees.scala:729-739`): a value of another type, numeric or not, is an error. `bigint`: only
/// `SBigInt` has costs of its own (every `ArithOpCompanion.costKind`, `:746-848`)
fn eval_arith<'ctx, T>(
    op: ArithOp,
    lv: Value<'ctx>,
    rv: Value<'ctx>,
    bigint: bool,
    ctx: &Context<'ctx>,
) -> Result<Value<'ctx>, EvalError>
where
    T: Num
        + CheckedAdd
        + CheckedSub
        + CheckedMul
        + CheckedDiv
        + CheckedRem
        + Ord
        + TryExtractFrom<Value<'ctx>>
        + Into<Value<'ctx>>
        + core::fmt::Display,
{
    ctx.add_jit_cost(match (op, bigint) {
        (ArithOp::Plus | ArithOp::Minus, false) => 15,
        (ArithOp::Plus | ArithOp::Minus, true) => 20,
        (ArithOp::Multiply | ArithOp::Divide | ArithOp::Modulo, false) => 15,
        (ArithOp::Multiply | ArithOp::Divide | ArithOp::Modulo, true) => 25,
        (ArithOp::Max | ArithOp::Min, false) => 5,
        (ArithOp::Max | ArithOp::Min, true) => 10,
    })?;
    let lv_raw = lv.try_extract_into::<T>()?;
    let rv_raw = rv.try_extract_into::<T>()?;
    match op {
        ArithOp::Plus => eval_plus(lv_raw, rv_raw),
        ArithOp::Minus => eval_minus(lv_raw, rv_raw),
        ArithOp::Multiply => eval_mul(lv_raw, rv_raw),
        // MIN / -1 can overflow
        ArithOp::Divide => eval_div(lv_raw, rv_raw),
        ArithOp::Modulo => eval_mod(lv_raw, rv_raw),
        ArithOp::Max => Ok(lv_raw.max(rv_raw).into()),
        ArithOp::Min => Ok(lv_raw.min(rv_raw).into()),
    }
}

impl Evaluable for BinOp {
    fn eval<'ctx>(
        &self,
        env: &mut Env<'ctx>,
        ctx: &Context<'ctx>,
    ) -> Result<Value<'ctx>, EvalError> {
        // sigmastate defines no `eval` for `BitOp` (v6.0.7 `trees.scala:911-916`): it throws at
        // the node, before its operands are evaluated
        if let BinOpKind::Bit(_) = self.kind {
            return Err(not_evaluated("BitOp"));
        }
        let lv = self.left.eval(env, ctx)?;
        match self.kind {
            // The arithmetic cost is charged in `eval_arith`, by the left operand's static type
            BinOpKind::Arith(_) => {}
            BinOpKind::Relation(op) => match op {
                // Eq/NEq cost is charged by eq_with_cost during the eval
                // dispatch below (per-type + per-coll-element), since the cost
                // depends on the runtime value type.
                RelationOp::Eq | RelationOp::NEq => {}
                _ => {
                    ctx.add_jit_cost(20)?;
                } // LT, LE, GT, GE = Fixed(20)
            },
            BinOpKind::Logical(_) => {
                ctx.add_jit_cost(20)?; // BinOr, BinAnd, BinXor = Fixed(20)
            }
            BinOpKind::Bit(_) => return Err(not_evaluated("BitOp")),
        }
        // using closure to keep right value from evaluation (for lazy AND, OR, XOR)
        let mut rv = || self.right.eval(env, ctx);
        match self.kind {
            BinOpKind::Logical(op) => match op {
                LogicalOp::And => Ok(Value::Boolean(if lv.try_extract_into::<bool>()? {
                    rv()?.try_extract_into::<bool>()?
                } else {
                    false
                })),
                LogicalOp::Or => Ok(Value::Boolean(if !lv.try_extract_into::<bool>()? {
                    rv()?.try_extract_into::<bool>()?
                } else {
                    true
                })),
                LogicalOp::Xor => Ok(Value::Boolean(
                    lv.try_extract_into::<bool>()? ^ rv()?.try_extract_into::<bool>()?,
                )),
            },
            BinOpKind::Relation(op) => match op {
                // `EQ`/`NEQ` check each operand's value against its type, the left one before
                // the right is evaluated (`trees.scala:1204-1209`, `:1224-1229`)
                RelationOp::Eq => {
                    crate::eval::check_value_of_type(&self.left.tpe(), &lv, ctx.tree_version())?;
                    let rv_val = rv()?;
                    crate::eval::check_value_of_type(
                        &self.right.tpe(),
                        &rv_val,
                        ctx.tree_version(),
                    )?;
                    Ok(Value::Boolean(
                        crate::eval::data_value_comparer::eq_with_cost(&lv, &rv_val, ctx)?,
                    ))
                }
                RelationOp::NEq => {
                    crate::eval::check_value_of_type(&self.left.tpe(), &lv, ctx.tree_version())?;
                    let rv_val = rv()?;
                    crate::eval::check_value_of_type(
                        &self.right.tpe(),
                        &rv_val,
                        ctx.tree_version(),
                    )?;
                    Ok(Value::Boolean(
                        !crate::eval::data_value_comparer::eq_with_cost(&lv, &rv_val, ctx)?,
                    ))
                }
                RelationOp::Gt => eval_gt(lv, rv()?),
                RelationOp::Lt => eval_lt(lv, rv()?),
                RelationOp::Ge => eval_ge(lv, rv()?),
                RelationOp::Le => eval_le(lv, rv()?),
            },
            BinOpKind::Arith(op) => {
                let rv = rv()?;
                // sigmastate looks the operation up by the LEFT operand's static type, once both
                // operands are evaluated (`ArithOp.impls(tpe.typeCode)`, `tpe = left.tpe`, v6.0.6
                // `trees.scala:703-739`, the table `:861-869`): a type with no arithmetic, such as
                // the `Any` of a tuple's item read by index, throws there whatever the values are
                // (SANTA's tuple-values reply, N14: `KeyNotFoundException: key 97`). The
                // implementation found there casts both values to its own type, at every tree
                // version (SANTA `ArithOp.mixed_width`)
                let op_tpe = self.left.tpe();
                match op_tpe {
                    SType::SByte => eval_arith::<i8>(op, lv, rv, false, ctx),
                    SType::SShort => eval_arith::<i16>(op, lv, rv, false, ctx),
                    SType::SInt => eval_arith::<i32>(op, lv, rv, false, ctx),
                    SType::SLong => eval_arith::<i64>(op, lv, rv, false, ctx),
                    SType::SBigInt => eval_arith::<BigInt256>(op, lv, rv, true, ctx),
                    SType::SUnsignedBigInt => eval_arith::<UnsignedBigInt>(op, lv, rv, false, ctx),
                    _ => Err(EvalError::UnexpectedValue(format!(
                        "{op:?}: no arithmetic for the left operand's type {op_tpe:?}"
                    ))),
                }
            }
            BinOpKind::Bit(_) => Err(not_evaluated("BitOp")),
        }
    }
}

#[cfg(test)]
#[cfg(feature = "arbitrary")]
#[allow(clippy::unwrap_used)]
mod arith_op_type_tests {
    use ergotree_ir::bigint256::BigInt256;
    use ergotree_ir::chain::context::Context;
    use ergotree_ir::ergo_tree::ErgoTreeVersion;
    use ergotree_ir::mir::expr::Expr;
    use ergotree_ir::mir::value::Value;
    use ergotree_ir::serialization::sigma_byte_reader::from_bytes;
    use ergotree_ir::serialization::sigma_byte_reader::SigmaByteRead;
    use ergotree_ir::serialization::SigmaSerializable;
    use ergotree_ir::unsignedbigint256::UnsignedBigInt;
    use sigma_test_util::force_any_val;

    use crate::eval::env::Env;
    use crate::eval::test_util::{assert_accepts, eval_v3_body};
    use crate::eval::EvalError;
    use crate::eval::Evaluable;

    /// The value and JIT cost of a SANTA eval tree at activated version 3: a v3 sized tree
    /// (`0b`, [`eval_v3_body`]) or a v0 tree (`00`), whose root is the expression
    fn eval_tree(tree_hex: &str) -> (Result<Value<'static>, EvalError>, u64) {
        if tree_hex.starts_with("0b") {
            let (_, res, cost) = eval_v3_body(tree_hex);
            return (res, cost);
        }
        let bytes = base16::decode(tree_hex.as_bytes()).unwrap();
        assert_eq!(bytes[0], 0x00, "{tree_hex}");
        let expr = from_bytes(&bytes[1..])
            .with_tree_version(ErgoTreeVersion::V0, Expr::sigma_parse)
            .unwrap();
        let mut ctx = force_any_val::<Context>();
        ctx.activated_script_version_byte = 3;
        ctx.tree_version.set(ErgoTreeVersion::V0);
        ctx.jit_cost.set(0);
        let mut env = Env::empty();
        let res = expr.eval(&mut env, &ctx).map(|v| v.to_static());
        (res, ctx.jit_cost_value())
    }

    /// Each tree's value and cost, or that it errors where no value is given
    fn check(rows: &[(&str, Option<(Value<'static>, u64)>)]) {
        for (tree_hex, expected) in rows {
            let (res, cost) = eval_tree(tree_hex);
            match expected {
                Some((value, expected_cost)) => {
                    assert_eq!(res.as_ref().ok(), Some(value), "{tree_hex}: {res:?}");
                    assert_eq!(cost, *expected_cost, "{tree_hex}");
                }
                None => assert!(res.is_err(), "{tree_hex}: {res:?}"),
            }
        }
    }

    fn bigint(n: i32) -> Value<'static> {
        Value::BigInt(BigInt256::from(n))
    }

    #[test]
    fn santa_mixed_width_rows() {
        // SANTA `ArithOp.mixed_width` (sigma-state 6.0.7), each script as a v3 sized tree and
        // as a v0 tree. sigmastate takes the implementation of the LEFT operand's static type
        // and casts both values to it (v6.0.6 `trees.scala:703-739`): a value of another
        // numeric type throws `ClassCastException`. Below tree version 3 the parser wraps the
        // narrower of two operands of different static types in an `Upcast`, which costs 10,
        // or 30 to `BigInt` (`SigmaBuilder.scala:674-683`, `:757-763`, `CostKind.scala:60-65`)
        check(&[
            // A1, A2: `1L + 2L`, the control
            ("0b059a05020504", Some((Value::Long(3), 25))),
            ("009a05020504", Some((Value::Long(3), 25))),
            // A3, A4: `1L + 2`
            ("0b059a05020404", None),
            ("009a05020404", Some((Value::Long(3), 35))),
            // A5, A6: `1L + ByIndex((1, 2, 3), 0)`, an `Any` holding an `Int`: nothing to upcast
            ("0b0e9a0502b248040404020406040000", None),
            ("009a0502b248040404020406040000", None),
            // B1, B2: `2 + 1L`, the `Long` is cast down to the left operand's `Int`
            ("0b059a04040502", None),
            ("009a04040502", Some((Value::Long(3), 35))),
            // B3, B4: `BigInt(1) + 2L`
            ("0b069a0601010504", None),
            ("009a0601010504", Some((bigint(3), 60))),
            // C1, C2: `BigInt(1) + BigInt(2)`, the control for the 30
            ("0b079a060101060102", Some((bigint(3), 30))),
            ("009a060101060102", Some((bigint(3), 30))),
            // B5, B6: `If(false, 1L, 2)` alone: typed by its true branch, the value an `Int`
            ("0b0795010005020404", Some((Value::Int(2), 20))),
            ("0095010005020404", Some((Value::Int(2), 20))),
            // B7, B8: `1L + If(false, 1L, 2)`: the static types agree, the right value is narrow
            ("0b0a9a050295010005020404", None),
            ("009a050295010005020404", None),
            // B9, B10: `If(false, 1L, 2) + 2`: the left value is narrower than its static type
            ("0b0a9a950100050204040404", None),
            ("009a950100050204040404", None),
        ]);
    }

    #[test]
    fn every_operation_takes_the_static_type() {
        // The refusals are by reading (the one table serves every operation, `min`/`max`
        // through its ordering, `trees.scala:746-848`), except `min` of two narrow values,
        // which SANTA measured (`Comparison.mixed_width` R8: throws at both versions). The
        // values and costs are sigma-rust's own at `a7656774`, not JVM measurements;
        // sigma-state 6.0.3 measured the mirrored v0 pairs (`Int * Long`, `max(Int, Long)`,
        // `min(Int, Long)`, in `arith_mismatched_numeric_coercion_vectors`) at the same costs
        check(&[
            // `1L * 2`
            ("0b059c05020404", None),
            ("009c05020404", Some((Value::Long(2), 35))),
            // `max(1L, 2)`
            ("0b05a205020404", None),
            ("00a205020404", Some((Value::Long(2), 25))),
            // `min(1L, 2)`
            ("0b05a105020404", None),
            ("00a105020404", Some((Value::Long(1), 25))),
            // `min(If(false, 1L, 2), If(false, 1L, 3))`
            ("0b0fa19501000502040495010005020406", None),
            ("00a19501000502040495010005020406", None),
            // `If(false, 1L, 2) + 2L`: a narrow left value, the right one fits
            ("0b0a9a950100050204040504", None),
            ("009a950100050204040504", None),
            // `(2 + 1L) * 2`: below v3 two `Upcast`s, the outer by the inner sum's static type
            ("0b089c9a040405020404", None),
            ("009c9a040405020404", Some((Value::Long(6), 65))),
            // `UnsignedBigInt`: the default cost (only `SBigInt` is special); no mixing with
            // `BigInt` in either order
            (
                "0b079a090101090102",
                Some((Value::UnsignedBigInt(UnsignedBigInt::from(3u64)), 25)),
            ),
            ("0b079a060101090102", None),
            ("0b079a090101060102", None),
            // `1 / 0` and `Int.MaxValue + 1` stay errors
            ("0b059d04020400", None),
            ("009d04020400", None),
            ("0b0e9a04feffffffffffffffff010402", None),
            ("009a04feffffffffffffffff010402", None),
        ]);
    }

    #[test]
    fn producers_keep_their_arithmetic() {
        // An operand whose static type comes from another node keeps its arithmetic. Values and
        // costs are sigma-rust's own at `a7656774`, not JVM measurements: a regression guard
        // for a static type that disagrees with the value it types
        check(&[
            // `ByIndex(Coll[Long](1, 2), 0) + 1L`
            ("0b0b9ab2110202040400000502", Some((Value::Long(2), 60))),
            ("009ab2110202040400000502", Some((Value::Long(2), 60))),
            // `(1L, 2L)._1 + 1L`
            ("0b089a8c590204010502", Some((Value::Long(2), 35))),
            // `{ val x = 1L; x + x }`
            ("0b0bd801d60105029a72017201", Some((Value::Long(2), 37))),
            // `Coll[Long](1, 2).fold(0L, { (p: (Long, Long)) => p._1 + p._2 })`
            (
                "0b14b0110202040500d90101599a8c7201018c720102",
                Some((Value::Long(3), 119)),
            ),
            // `SizeOf(Coll[Long](1, 2)) + 1`, and the same through the `size` property
            ("0b089ab1110202040402", Some((Value::Int(3), 39))),
            ("0b0a9adb0c01110202040402", Some((Value::Int(3), 43))),
            // `OptionGet(Some(1L)) + 1L`
            ("0b079ae42901020502", Some((Value::Long(2), 40))),
        ]);
    }

    #[test]
    fn the_left_operand_s_type_selects_the_arithmetic() {
        // SANTA `Tuple.non_pair_value_gates` G3-G8 with their controls C3-C5 (sigma-state
        // 6.0.7). An item of the tuple `(1, 2, 3)` is typed `Any` and holds an `Int`
        // (`Coll.runtime_shapes_methodcall` C2: `(1, 2, 3).apply(0)` is 1 @44)
        assert_accepts("0b0ddc0c0a48040404020406010400", Value::Int(1), 44);
        // on the LEFT there is no arithmetic for `Any` (`KeyNotFoundException: key 97`):
        // `ByIndex(t, 0) + 1`, `t.apply(0) + 1`, `min(t.apply(0), 1)`
        for tree_hex in [
            "0b0e9ab2480404040204060400000402",
            "0b109adc0c0a480404040204060104000402",
            "0b10a1dc0c0a480404040204060104000402",
        ] {
            assert!(eval_v3_body(tree_hex).1.is_err(), "{tree_hex}");
        }
        // on the RIGHT it is `Int` arithmetic at the plain collection's cost:
        // `1 + ByIndex(t, 0)`, `1 + t.apply(0)`, `min(1, t.apply(0))`
        assert_accepts("0b0e9a0402b248040404020406040000", Value::Int(2), 60);
        assert_accepts("0b109a0402dc0c0a48040404020406010400", Value::Int(2), 64);
        assert_accepts("0b10a10402dc0c0a48040404020406010400", Value::Int(1), 54);
        // the controls over `Coll[Int](1, 2, 3)`
        assert_accepts("0b0c9ab210030204060400000402", Value::Int(2), 60);
        assert_accepts("0b0e9adc0c0a10030204060104000402", Value::Int(2), 64);
        assert_accepts("0b0ea1dc0c0a10030204060104000402", Value::Int(1), 54);
    }
}

#[allow(clippy::panic)]
#[cfg(test)]
#[allow(clippy::unwrap_used)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use crate::eval::test_util::eval_out_wo_ctx;
    use crate::eval::test_util::try_eval_out_with_version;
    use crate::eval::test_util::try_eval_out_wo_ctx;
    use alloc::boxed::Box;
    use ergotree_ir::ergo_tree::ErgoTree;
    use ergotree_ir::mir::constant::Constant;
    use ergotree_ir::mir::expr::Expr;
    use ergotree_ir::serialization::SigmaSerializable;
    use ergotree_ir::sigma_protocol::sigma_boolean::cand::Cand;
    use ergotree_ir::sigma_protocol::sigma_boolean::cthreshold::Cthreshold;
    use ergotree_ir::sigma_protocol::sigma_boolean::{
        ProveDlog, SigmaBoolean, SigmaConjecture, SigmaProp,
    };
    use ergotree_ir::unsignedbigint256::UnsignedBigInt;
    use num_traits::Bounded;
    use proptest::prelude::*;
    use sigma_test_util::force_any_val;

    #[test]
    fn arith_mismatched_numeric_coercion_vectors() {
        // Zero-tolerance vectors from the sigma-state 6.0.3 sweep (eval value:type +
        // raw JIT cost, regime activatedScriptVersion=3 / ergoTree v0). Below tree
        // version 3 the parser wraps the narrower of two operands of different numeric
        // static types in an `Upcast` to the WIDER type (Byte<Short<Int<Long<BigInt), so
        // the operation computes checked in the wider type and rejects only on genuine
        // wider-type overflow. Cost = same-type arith + the Upcast's (bigint=30 else 10).
        use crate::eval::test_util::try_eval_out;
        use ergotree_ir::chain::context::Context;
        use ergotree_ir::ergo_tree::ErgoTree;
        use sigma_test_util::force_any_val;
        fn hx(s: &str) -> alloc::vec::Vec<u8> {
            (0..s.len())
                .step_by(2)
                .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
                .collect()
        }
        // (bare tree hex, Some(expected value) or None=reject, expected raw JIT cost)
        let cases: &[(&str, Option<Value>, u64)] = &[
            ("009a04020504", Some(Value::Long(3)), 35), // Plus(Int 1, Long 2)
            ("009a05040402", Some(Value::Long(3)), 35), // Plus(Long 2, Int 1)
            (
                "009a04feffffffffffffffff010502",
                Some(Value::Long(2147483648)),
                35,
            ), // Plus(IntMax, Long 1)
            ("009a05feffffffffffffffff010402", None, 35), // Plus(LongMax, Int 1) -> reject
            ("009904020504", Some(Value::Long(-1)), 35), // Minus(Int 1, Long 2)
            ("009c04020504", Some(Value::Long(2)), 35), // Multiply(Int 1, Long 2)
            ("009d04020504", Some(Value::Long(0)), 35), // Divide(Int 1, Long 2)
            ("009e04020504", Some(Value::Long(1)), 35), // Modulo(Int 1, Long 2)
            ("00a104020504", Some(Value::Long(1)), 25), // Min(Int 1, Long 2)
            ("00a204020504", Some(Value::Long(2)), 25), // Max(Int 1, Long 2)
            ("009a02010404", Some(Value::Int(3)), 35),  // Plus(Byte 1, Int 2)
            ("009a03020404", Some(Value::Int(3)), 35),  // Plus(Short 1, Int 2)
            (
                "009a0402060102",
                Some(Value::BigInt(BigInt256::from(3i32))),
                60,
            ), // Plus(Int 1, BigInt 2)
            (
                "009a0502060102",
                Some(Value::BigInt(BigInt256::from(3i32))),
                60,
            ), // Plus(Long 1, BigInt 2)
        ];
        for (hex, expected, expected_cost) in cases {
            let tree = ErgoTree::sigma_parse_bytes_lenient(&hx(hex)).expect("parse tree");
            let expr = tree.proposition().expect("tree proposition");
            let ctx = force_any_val::<Context>();
            let before = ctx.jit_cost_value();
            let res = try_eval_out::<Value>(&expr, &ctx);
            let cost = ctx.jit_cost_value() - before;
            match expected {
                Some(v) => {
                    assert_eq!(res.as_ref().ok(), Some(v), "{hex}: value mismatch");
                    assert_eq!(cost, *expected_cost, "{hex}: cost mismatch");
                }
                None => assert!(res.is_err(), "{hex}: expected reject, got {res:?}"),
            }
        }
    }

    // SANTA tx-tier regression (captured testnet tx at height 28,474): Scala's
    // arith cost table special-cases ONLY `SBigInt` (every
    // `ArithOpCompanion.costKind` in trees.scala is `case SBigInt => …;
    // case _ => …`), so UnsignedBigInt arithmetic must ride the default arm —
    // Plus/Minus 15, Multiply/Divide/Modulo 15, Max/Min 5 — not the BigInt
    // 20/25/10. Pre-fix, the cost arm routed UBI alongside BigInt,
    // overcharging every UBI arith op (+10 per division in the captured tx).
    #[test]
    fn arith_unsigned_bigint_costs_default_arm_not_bigint() {
        use crate::eval::test_util::try_eval_out;
        use ergotree_ir::chain::context::Context;
        use sigma_test_util::force_any_val;

        let cost_of = |kind: ArithOp, left: Constant, right: Constant| -> u64 {
            let expr: Expr = BinOp {
                kind: BinOpKind::Arith(kind),
                left: Box::new(left.into()),
                right: Box::new(right.into()),
            }
            .into();
            let ctx = force_any_val::<Context>();
            let before = ctx.jit_cost_value();
            let _ = try_eval_out::<Value>(&expr, &ctx).unwrap();
            ctx.jit_cost_value() - before
        };
        let ubi = |v: u32| Constant::from(UnsignedBigInt::from(v));
        let bi = |v: i32| Constant::from(BigInt256::from(v));

        // Two Const evals (5 each) + the op cost.
        for (op, ubi_op_cost, bigint_op_cost) in [
            (ArithOp::Plus, 15, 20),
            (ArithOp::Minus, 15, 20),
            (ArithOp::Multiply, 15, 25),
            (ArithOp::Divide, 15, 25),
            (ArithOp::Modulo, 15, 25),
            (ArithOp::Max, 5, 10),
            (ArithOp::Min, 5, 10),
        ] {
            assert_eq!(
                cost_of(op, ubi(6), ubi(3)),
                10 + ubi_op_cost,
                "{op:?} over UnsignedBigInt must cost the default arm ({ubi_op_cost})"
            );
            assert_eq!(
                cost_of(op, bi(6), bi(3)),
                10 + bigint_op_cost,
                "{op:?} over BigInt must cost the SBigInt arm ({bigint_op_cost})"
            );
        }
    }

    fn check_eq_neq(left: Constant, right: Constant) -> bool {
        let eq_op: Expr = BinOp {
            kind: BinOpKind::Relation(RelationOp::Eq),
            left: Box::new(left.clone().into()),
            right: Box::new(right.clone().into()),
        }
        .into();
        let neq_op: Expr = BinOp {
            kind: BinOpKind::Relation(RelationOp::NEq),
            left: Box::new(left.into()),
            right: Box::new(right.into()),
        }
        .into();
        eval_out_wo_ctx::<bool>(&eq_op) && !eval_out_wo_ctx::<bool>(&neq_op)
    }

    #[test]
    fn num_eq() {
        assert!(check_eq_neq(1i64.into(), 1i64.into()));
    }

    #[test]
    fn num_neq() {
        assert!(!check_eq_neq(2i64.into(), 1i64.into()));
    }

    #[test]
    fn option_eq() {
        assert!(check_eq_neq(Some(1i64).into(), Some(1i64).into()));
        let none: Option<i64> = None;
        assert!(check_eq_neq(none.into(), none.into()));
        // Option<Vec<i8>>
        assert!(check_eq_neq(
            Some(vec![1i8, 2i8]).into(),
            Some(vec![1i8, 2i8]).into()
        ));
        // Vec<Option<i64>>
        assert!(check_eq_neq(
            vec![Some(1i64), Some(1i64)].into(),
            vec![Some(1i64), Some(1i64)].into()
        ));
    }

    #[test]
    fn option_neq() {
        assert!(!check_eq_neq(Some(2i64).into(), Some(1i64).into()));
        let none: Option<i64> = None;
        assert!(!check_eq_neq(none.into(), Some(1i64).into()));
        // Option<Vec<i8>>
        assert!(!check_eq_neq(
            Some(vec![1i8, 2i8]).into(),
            Some(vec![2i8, 2i8]).into()
        ));
        // Vec<Option<i64>>
        assert!(!check_eq_neq(
            vec![Some(1i64), Some(1i64)].into(),
            vec![Some(2i64), Some(1i64)].into()
        ));
    }

    #[test]
    fn tuple_eq() {
        assert!(check_eq_neq((1i64, true).into(), (1i64, true).into()));
    }

    #[test]
    fn bin_or_eval_laziness() {
        let e: Expr = BinOp {
            kind: BinOpKind::Logical(LogicalOp::Or),
            left: Box::new(Expr::Const(true.into())),
            // something that should blow-up the evaluation
            right: Box::new(
                BinOp {
                    kind: ArithOp::Divide.into(),
                    left: Box::new(Expr::Const(1i32.into())),
                    right: Box::new(Expr::Const(0i32.into())),
                }
                .into(),
            ),
        }
        .into();
        assert!(eval_out_wo_ctx::<bool>(&e));
    }

    #[test]
    fn bin_and_eval_laziness() {
        let e: Expr = BinOp {
            kind: BinOpKind::Logical(LogicalOp::And),
            left: Box::new(Expr::Const(false.into())),
            // something that should blow-up the evaluation
            right: Box::new(
                BinOp {
                    kind: ArithOp::Divide.into(),
                    left: Box::new(Expr::Const(1i32.into())),
                    right: Box::new(Expr::Const(0i32.into())),
                }
                .into(),
            ),
        }
        .into();
        assert!(!eval_out_wo_ctx::<bool>(&e));
    }

    fn eval_arith_op<T: TryExtractFrom<Value<'static>> + Into<Constant> + 'static>(
        op: ArithOp,
        left: T,
        right: T,
    ) -> Result<T, EvalError> {
        let expr: Expr = BinOp {
            kind: BinOpKind::Arith(op),
            left: Box::new(left.into().into()),
            right: Box::new(right.into().into()),
        }
        .into();
        try_eval_out_wo_ctx::<T>(&expr)
    }

    fn eval_relation_op<T: Into<Constant>>(op: RelationOp, left: T, right: T) -> bool {
        let expr: Expr = BinOp {
            kind: BinOpKind::Relation(op),
            left: Box::new(left.into().into()),
            right: Box::new(right.into().into()),
        }
        .into();
        eval_out_wo_ctx::<bool>(&expr)
    }

    fn eval_logical_op<T: Into<Constant>>(op: LogicalOp, left: T, right: T) -> bool {
        let expr: Expr = BinOp {
            kind: BinOpKind::Logical(op),
            left: Box::new(left.into().into()),
            right: Box::new(right.into().into()),
        }
        .into();
        eval_out_wo_ctx::<bool>(&expr)
    }

    #[test]
    fn test_bigint_extremes() {
        let b = BigInt256::from;
        // Our BigInt should behave like a 256 bit signed (two's complement) integer according to
        // the language spec. These are the max and min values representable:
        let max = BigInt256::max_value;
        let min = BigInt256::min_value;

        assert!(eval_arith_op(ArithOp::Multiply, max(), b(2)).is_err());
        assert_eq!(eval_arith_op(ArithOp::Multiply, max(), b(1)), Ok(max()));
        assert!(eval_arith_op(ArithOp::Multiply, min(), b(2)).is_err());
        assert_eq!(eval_arith_op(ArithOp::Multiply, min(), b(1)), Ok(min()));

        assert!(eval_arith_op(ArithOp::Divide, min(), b(-1)).is_err());
        assert_eq!(
            eval_arith_op(ArithOp::Divide, min() + b(1), b(-1)),
            Ok(max())
        );
        assert!(eval_arith_op(ArithOp::Divide, b(20), b(0)).is_err());

        assert!(eval_arith_op(ArithOp::Modulo, b(20), b(-1)).is_err());
        assert!(eval_arith_op(ArithOp::Modulo, b(20), b(0)).is_err());
        assert_eq!(eval_arith_op(ArithOp::Modulo, max(), b(1)), Ok(b(0)));
        assert_eq!(eval_arith_op(ArithOp::Modulo, min(), b(1)), Ok(b(0)));

        assert!(eval_arith_op(ArithOp::Plus, max(), b(1)).is_err());
        assert_eq!(eval_arith_op(ArithOp::Plus, max(), b(0)), Ok(max()));
        assert!(eval_arith_op(ArithOp::Plus, min(), b(-1)).is_err());
        assert_eq!(eval_arith_op(ArithOp::Plus, min(), b(0)), Ok(min()));

        assert!(eval_arith_op(ArithOp::Minus, max(), b(-1)).is_err());
        assert_eq!(eval_arith_op(ArithOp::Minus, max(), b(0)), Ok(max()));
        assert!(eval_arith_op(ArithOp::Minus, min(), b(1)).is_err());
        assert_eq!(eval_arith_op(ArithOp::Minus, min(), b(0)), Ok(min()));
    }

    proptest! {

        #![proptest_config(ProptestConfig::with_cases(32))]

        #[test]
        fn test_eq(v in any::<Constant>()) {
            // `EQ`/`NEQ` check each operand's value against its type, and sigmastate's
            // `isValueOfType` has no case for a tuple of other than two items
            // (`SType.scala:200-202`): comparing one fails
            if matches!(&v.tpe, ergotree_ir::types::stype::SType::STuple(t) if t.items.len() != 2) {
                let eq_op: Expr = BinOp {
                    kind: BinOpKind::Relation(RelationOp::Eq),
                    left: Box::new(v.clone().into()),
                    right: Box::new(v.into()),
                }
                .into();
                prop_assert!(try_eval_out_wo_ctx::<bool>(&eq_op).is_err());
            } else {
                prop_assert![check_eq_neq(v.clone(), v)];
            }
        }

        #[test]
        fn test_num_slong(l in any::<i64>(), r in any::<i64>()) {
            prop_assert_eq!(eval_arith_op(ArithOp::Plus, l, r).ok(), l.checked_add(r));
            prop_assert_eq!(eval_arith_op(ArithOp::Minus, l, r).ok(), l.checked_sub(r));
            prop_assert_eq!(eval_arith_op(ArithOp::Multiply, l, r).ok(), l.checked_mul(r));
            prop_assert_eq!(eval_arith_op(ArithOp::Divide, l, r).ok(), l.checked_div(r));
            prop_assert_eq!(eval_arith_op(ArithOp::Modulo, l, r).ok(), l.checked_rem(r));
            prop_assert_eq!(eval_arith_op::<i64>(ArithOp::Max, l, r).unwrap(), l.max(r));
            prop_assert_eq!(eval_arith_op::<i64>(ArithOp::Min, l, r).unwrap(), l.min(r));


            prop_assert_eq!(eval_relation_op(RelationOp::Gt, l, r), l > r);
            prop_assert_eq!(eval_relation_op(RelationOp::Lt, l, r), l < r);
            prop_assert_eq!(eval_relation_op(RelationOp::Ge, l, r), l >= r);
            prop_assert_eq!(eval_relation_op(RelationOp::Le, l, r), l <= r);
        }

        #[test]
        fn test_num_sint(l in any::<i32>(), r in any::<i32>()) {
            prop_assert_eq!(eval_arith_op(ArithOp::Plus, l, r).ok(), l.checked_add(r));
            prop_assert_eq!(eval_arith_op(ArithOp::Minus, l, r).ok(), l.checked_sub(r));
            prop_assert_eq!(eval_arith_op(ArithOp::Multiply, l, r).ok(), l.checked_mul(r));
            prop_assert_eq!(eval_arith_op(ArithOp::Divide, l, r).ok(), l.checked_div(r));
            prop_assert_eq!(eval_arith_op(ArithOp::Modulo, l, r).ok(), l.checked_rem(r));
            prop_assert_eq!(eval_arith_op::<i32>(ArithOp::Max, l, r).unwrap(), l.max(r));
            prop_assert_eq!(eval_arith_op::<i32>(ArithOp::Min, l, r).unwrap(), l.min(r));


            prop_assert_eq!(eval_relation_op(RelationOp::Gt, l, r), l > r);
            prop_assert_eq!(eval_relation_op(RelationOp::Lt, l, r), l < r);
            prop_assert_eq!(eval_relation_op(RelationOp::Ge, l, r), l >= r);
            prop_assert_eq!(eval_relation_op(RelationOp::Le, l, r), l <= r);
        }

        #[test]
        fn test_num_sshort(l in any::<i16>(), r in any::<i16>()) {
            prop_assert_eq!(eval_arith_op(ArithOp::Plus, l, r).ok(), l.checked_add(r));
            prop_assert_eq!(eval_arith_op(ArithOp::Minus, l, r).ok(), l.checked_sub(r));
            prop_assert_eq!(eval_arith_op(ArithOp::Multiply, l, r).ok(), l.checked_mul(r));
            prop_assert_eq!(eval_arith_op(ArithOp::Divide, l, r).ok(), l.checked_div(r));
            prop_assert_eq!(eval_arith_op(ArithOp::Modulo, l, r).ok(), l.checked_rem(r));
            prop_assert_eq!(eval_arith_op::<i16>(ArithOp::Max, l, r).unwrap(), l.max(r));
            prop_assert_eq!(eval_arith_op::<i16>(ArithOp::Min, l, r).unwrap(), l.min(r));


            prop_assert_eq!(eval_relation_op(RelationOp::Gt, l, r), l > r);
            prop_assert_eq!(eval_relation_op(RelationOp::Lt, l, r), l < r);
            prop_assert_eq!(eval_relation_op(RelationOp::Ge, l, r), l >= r);
            prop_assert_eq!(eval_relation_op(RelationOp::Le, l, r), l <= r);
        }

        #[test]
        fn test_num_sbyte(l in any::<i8>(), r in any::<i8>()) {
            prop_assert_eq!(eval_arith_op(ArithOp::Plus, l, r).ok(), l.checked_add(r));
            prop_assert_eq!(eval_arith_op(ArithOp::Minus, l, r).ok(), l.checked_sub(r));
            prop_assert_eq!(eval_arith_op(ArithOp::Multiply, l, r).ok(), l.checked_mul(r));
            prop_assert_eq!(eval_arith_op(ArithOp::Divide, l, r).ok(), l.checked_div(r));
            prop_assert_eq!(eval_arith_op(ArithOp::Modulo, l, r).ok(), l.checked_rem(r));
            prop_assert_eq!(eval_arith_op::<i8>(ArithOp::Max, l, r).unwrap(), l.max(r));
            prop_assert_eq!(eval_arith_op::<i8>(ArithOp::Min, l, r).unwrap(), l.min(r));


            prop_assert_eq!(eval_relation_op(RelationOp::Gt, l, r), l > r);
            prop_assert_eq!(eval_relation_op(RelationOp::Lt, l, r), l < r);
            prop_assert_eq!(eval_relation_op(RelationOp::Ge, l, r), l >= r);
            prop_assert_eq!(eval_relation_op(RelationOp::Le, l, r), l <= r);
        }

        #[test]
        fn test_num_bigint(l_long in any::<i64>(), r_long in any::<i64>()) {
            let l = BigInt256::from(l_long);
            let r = BigInt256::from(r_long);
            prop_assert_eq!(eval_arith_op(ArithOp::Plus, l, r).ok(), l.checked_add(&r));
            prop_assert_eq!(eval_arith_op(ArithOp::Minus, l, r).ok(), l.checked_sub(&r));
            prop_assert_eq!(eval_arith_op(ArithOp::Multiply, l, r).ok(), l.checked_mul(&r));
            prop_assert_eq!(eval_arith_op(ArithOp::Divide, l, r).ok(), l.checked_div(&r));
            prop_assert_eq!(eval_arith_op(ArithOp::Modulo, l, r).ok(), l.checked_rem(&r));
            prop_assert_eq!(eval_arith_op::<BigInt256>(ArithOp::Max, l,
                    r).unwrap(), l.max(r));
            prop_assert_eq!(eval_arith_op::<BigInt256>(ArithOp::Min, l,
                    r).unwrap(), l.min(r));


            prop_assert_eq!(eval_relation_op(RelationOp::Gt, l, r), l > r);
            prop_assert_eq!(eval_relation_op(RelationOp::Lt, l, r), l < r);
            prop_assert_eq!(eval_relation_op(RelationOp::Ge, l, r), l >= r);
            prop_assert_eq!(eval_relation_op(RelationOp::Le, l, r), l <= r);
        }

        #[test]
        fn test_num_unsigned_bigint(l in any::<UnsignedBigInt>(), r in any::<UnsignedBigInt>()) {
            prop_assert_eq!(eval_arith_op(ArithOp::Plus, l, r).ok(), l.checked_add(&r));
            prop_assert_eq!(eval_arith_op(ArithOp::Minus, l, r).ok(), l.checked_sub(&r));
            prop_assert_eq!(eval_arith_op(ArithOp::Multiply, l, r).ok(), l.checked_mul(&r));
            prop_assert_eq!(eval_arith_op(ArithOp::Divide, l, r).ok(), l.checked_div(&r));
            prop_assert_eq!(eval_arith_op(ArithOp::Modulo, l, r).ok(), l.checked_rem(&r));
            prop_assert_eq!(eval_arith_op::<UnsignedBigInt>(ArithOp::Max, l, r).unwrap(), l.max(r));
            prop_assert_eq!(eval_arith_op::<UnsignedBigInt>(ArithOp::Min, l, r).unwrap(), l.min(r));


            prop_assert_eq!(eval_relation_op(RelationOp::Gt, l, r), l > r);
            prop_assert_eq!(eval_relation_op(RelationOp::Lt, l, r), l < r);
            prop_assert_eq!(eval_relation_op(RelationOp::Ge, l, r), l >= r);
            prop_assert_eq!(eval_relation_op(RelationOp::Le, l, r), l <= r);
        }

        #[test]
        fn test_and_or_xor(l in any::<bool>(), r in any::<bool>()) {
            prop_assert_eq!(eval_logical_op(LogicalOp::And, l, r), l && r);
            prop_assert_eq!(eval_logical_op(LogicalOp::Or, l, r), l || r);
            prop_assert_eq!(eval_logical_op(LogicalOp::Xor, l, r), l ^ r);
        }
    }

    // --- SigmaProp EQ/NEQ: Scala `DataValueComparer.equalSigmaBoolean` parity ---

    // The two ProveDlog points embedded in the blessed vectors below
    // (secp256k1 G and 2G).
    const PK_A_HEX: &str = "0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798";
    const PK_B_HEX: &str = "02c6047f9441ed7d6d3045406e95c07cd85c778e4b8cef3ca7abac09b95c709ee5";

    fn pk(point_hex: &str) -> SigmaBoolean {
        let p = ergo_chain_types::EcPoint::sigma_parse_bytes(&base16::decode(point_hex).unwrap())
            .unwrap();
        ProveDlog::new(p).into()
    }

    fn cand(items: Vec<SigmaBoolean>) -> SigmaBoolean {
        SigmaBoolean::SigmaConjecture(SigmaConjecture::Cand(Cand { items }))
    }

    fn cthreshold(k: u8, children: Vec<SigmaBoolean>) -> SigmaBoolean {
        SigmaBoolean::SigmaConjecture(SigmaConjecture::Cthreshold(Cthreshold {
            k,
            children: children.try_into().unwrap(),
        }))
    }

    fn eval_sigmaprop_relation(
        op: RelationOp,
        l: SigmaBoolean,
        r: SigmaBoolean,
    ) -> Result<bool, EvalError> {
        let expr: Expr = BinOp {
            kind: BinOpKind::Relation(op),
            left: Box::new(Constant::from(SigmaProp::new(l)).into()),
            right: Box::new(Constant::from(SigmaProp::new(r)).into()),
        }
        .into();
        try_eval_out_wo_ctx::<bool>(&expr)
    }

    fn assert_throws_unknown_type(res: Result<bool, EvalError>) {
        let err = res.unwrap_err();
        assert!(
            format!("{err:?}").contains("Cannot compare SigmaBoolean"),
            "expected the equalSigmaBoolean sys.error mirror, got: {err:?}"
        );
    }

    // JVM-blessed byte vectors (santa-eval `EQ_of_SigmaProp_conjecture_mismatch`,
    // eval/v5/authored): closed v2 trees, `EQ(CP(0), CP(1))` over two segregated
    // SigmaProp constants (pkA/pkB above). The blessed sized header (`1a` + size
    // VLQ) is rewritten to the non-sized `12` (size bit cleared, size bytes
    // dropped) because the sized parse path rejects non-SigmaProp roots — the
    // same lenient deserialize the conformance runner applies to
    // expression-rooted corpus trees; body bytes verbatim.
    fn eval_blessed_eq_tree(tree_hex: &str) -> Result<bool, EvalError> {
        let tree_bytes = base16::decode(tree_hex).unwrap();
        // The hex is the unsized (size-bit-cleared) form; the lenient entry restores
        // the size slot (Rule-1012-valid) and skips the SigmaProp-root check.
        let tree = ErgoTree::sigma_parse_bytes_lenient_from_unsized(&tree_bytes).unwrap();
        let expr = tree.proposition().unwrap();
        let ctx = force_any_val::<Context>();
        try_eval_out_with_version::<bool>(&expr, &ctx, 2, 2)
    }

    #[test]
    fn eq_sigmaprop_cand_left_dlog_right_throws_blessed_bytes() {
        // `{ (pkA && pkB) == pkA }` (cand-vs-dlog#0): conjecture left, leaf
        // right — every guarded conjecture arm fails and the top-level
        // `case _` is sys.error → eval throws.
        assert_throws_unknown_type(eval_blessed_eq_tree(
            "1202089602cd0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798cd02c6047f9441ed7d6d3045406e95c07cd85c778e4b8cef3ca7abac09b95c709ee508cd0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f817989373007301",
        ));
    }

    #[test]
    fn eq_sigmaprop_dlog_left_cand_right_false_blessed_bytes() {
        // `{ pkA == (pkA && pkB) }` (dlog-vs-cand#1): leaf left — the
        // ProveDlog arm's inner `case _ => false`. The asymmetry twin of the
        // throw above.
        assert!(!eval_blessed_eq_tree(
            "120208cd0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798089602cd0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798cd02c6047f9441ed7d6d3045406e95c07cd85c778e4b8cef3ca7abac09b95c709ee59373007301",
        )
        .unwrap());
    }

    #[test]
    fn eq_sigmaprop_trivial_left_dlog_right_false_blessed_bytes() {
        // `{ sigmaProp(true) == pkA }` (trivial-vs-dlog#2): TrivialProp left →
        // false, no throw.
        assert!(!eval_blessed_eq_tree(
            "120208d308cd0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f817989373007301",
        )
        .unwrap());
    }

    #[test]
    fn eq_sigmaprop_cthreshold_left_cand_right_throws_blessed_bytes() {
        // `{ cthreshold(1, pkA, pkB) == (pkA && pkB) }` (cthreshold-vs-cand#3):
        // conjecture left vs a different conjecture kind → throws.
        assert_throws_unknown_type(eval_blessed_eq_tree(
            "120208980102cd0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798cd02c6047f9441ed7d6d3045406e95c07cd85c778e4b8cef3ca7abac09b95c709ee5089602cd0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798cd02c6047f9441ed7d6d3045406e95c07cd85c778e4b8cef3ca7abac09b95c709ee59373007301",
        ));
    }

    #[test]
    fn neq_sigmaprop_conjecture_mismatch_throws() {
        // NEQ is EQ-negated over the same comparer, so the mismatch throws
        // under `!=` too.
        let (a, b) = (pk(PK_A_HEX), pk(PK_B_HEX));
        assert_throws_unknown_type(eval_sigmaprop_relation(
            RelationOp::NEq,
            cand(vec![a.clone(), b]),
            a,
        ));
    }

    #[test]
    fn eq_sigmaprop_nested_conjecture_mismatch_throws() {
        // Matching CAND tops (same length) recurse into the children, where
        // child 0 is CAND-vs-ProveDlog → the nested mismatch throws.
        let (a, b) = (pk(PK_A_HEX), pk(PK_B_HEX));
        let l = cand(vec![cand(vec![a.clone(), b.clone()]), a.clone()]);
        let r = cand(vec![a.clone(), a]);
        assert_throws_unknown_type(eval_sigmaprop_relation(RelationOp::Eq, l, r));
    }

    #[test]
    fn eq_sigmaprop_matching_conjectures_compare_structurally() {
        let (a, b) = (pk(PK_A_HEX), pk(PK_B_HEX));
        // Equal CANDs are true under EQ and false under NEQ.
        assert!(check_eq_neq(
            SigmaProp::new(cand(vec![a.clone(), b.clone()])).into(),
            SigmaProp::new(cand(vec![a.clone(), b.clone()])).into(),
        ));
        // A leaf-level child mismatch inside matching tops is plain false.
        assert!(!eval_sigmaprop_relation(
            RelationOp::Eq,
            cand(vec![a.clone(), a.clone()]),
            cand(vec![a.clone(), b.clone()]),
        )
        .unwrap());
        // A length mismatch is false without comparing children (child 0
        // would throw if it were reached).
        assert!(!eval_sigmaprop_relation(
            RelationOp::Eq,
            cand(vec![cand(vec![a.clone(), b.clone()]), a.clone()]),
            cand(vec![a.clone(), b.clone(), a.clone()]),
        )
        .unwrap());
    }

    #[test]
    fn eq_sigmaprop_cthreshold_k_mismatch_short_circuits() {
        // Scala: `k == sb2.k && equalSigmaBooleans(...)` — a k mismatch
        // returns false before the children (which here would throw) are
        // compared.
        let (a, b) = (pk(PK_A_HEX), pk(PK_B_HEX));
        let l = cthreshold(1, vec![cand(vec![a.clone(), b.clone()]), a.clone()]);
        let r = cthreshold(2, vec![a.clone(), a]);
        assert!(!eval_sigmaprop_relation(RelationOp::Eq, l, r).unwrap());
    }
}

#[cfg(test)]
#[cfg(feature = "arbitrary")]
#[allow(clippy::unwrap_used)]
mod pair_value_tests {
    //! JVM parity: a tuple expression's data is a collection even where its type is a pair
    //! (`values.scala:818-822`), and `EQ`/`NEQ` check each operand's value against its type
    //! (`trees.scala:1204-1229`), so comparing a pair read from such a variable fails, as does
    //! `SelectField` on it (`transformers.scala:300-307`).
    use super::*;
    use crate::eval::test_util::try_eval_out;
    use ergotree_ir::chain::context::Context;
    use ergotree_ir::chain::context_extension::ContextExtension;
    use ergotree_ir::chain::evaluated_value::EvaluatedValue;
    use ergotree_ir::mir::constant::Constant;
    use ergotree_ir::mir::expr::Expr;
    use ergotree_ir::mir::get_var::GetVar;
    use ergotree_ir::mir::option_get::OptionGet;
    use ergotree_ir::mir::select_field::SelectField;
    use ergotree_ir::mir::unary_op::OneArgOpTryBuild;
    use ergotree_ir::serialization::SigmaSerializable;
    use ergotree_ir::types::stuple::STuple;
    use ergotree_ir::types::stype::SType;
    use sigma_test_util::force_any_val;

    /// var 0 = `hex`
    fn ctx_with_var0(hex: &str) -> Context<'static> {
        let value = EvaluatedValue::sigma_parse_bytes(&base16::decode(hex).unwrap()).unwrap();
        let ext = ContextExtension {
            values: [(0u8, value)].into_iter().collect(),
        };
        Context {
            extension: Box::leak(Box::new(ext)),
            ..force_any_val::<Context>()
        }
    }

    fn var0() -> Expr {
        GetVar {
            var_id: 0,
            var_tpe: SType::STuple(STuple::pair(SType::SInt, SType::SInt)),
        }
        .into()
    }

    fn relation(op: RelationOp, left: Expr, right: Expr) -> Expr {
        BinOp {
            kind: BinOpKind::Relation(op),
            left: left.into(),
            right: right.into(),
        }
        .into()
    }

    #[test]
    fn a_tuple_expression_read_as_a_pair_fails_where_its_value_is_checked() {
        // SANTA T2, `getVar[(Int, Int)](0).get != (0, 0)`, and V3,
        // `getVar[(Int, Int)](0).get._1 == 1`: invalid with var 0 = `Tuple(1, 2)`, valid with its
        // constant twin `58 02 04`
        let pair: Expr = OptionGet::try_build(var0()).unwrap().into();
        let zeros = Constant::sigma_parse_bytes(&[0x58, 0x00, 0x00]).unwrap();
        let ne = relation(RelationOp::NEq, pair.clone(), zeros.into());
        let first: Expr = SelectField::new(pair, 1u8.try_into().unwrap())
            .unwrap()
            .into();
        let first_is_1 = relation(RelationOp::Eq, first, Expr::Const(1i32.into()));
        for expr in [ne, first_is_1] {
            assert!(try_eval_out::<bool>(&expr, &ctx_with_var0("860204020404")).is_err());
            assert!(try_eval_out::<bool>(&expr, &ctx_with_var0("580204")).unwrap());
        }
    }

    #[test]
    fn a_tuple_expression_read_as_a_pair_passes_where_nothing_checks_it() {
        // SANTA T4, `getVar[(Int, Int)](0) == getVar[(Int, Int)](0)`: the operands are options,
        // and `isValueOfType` checks only an option's class (`SType.scala:199`)
        let eq = relation(RelationOp::Eq, var0(), var0());
        assert!(try_eval_out::<bool>(&eq, &ctx_with_var0("860204020404")).unwrap());
    }
}
