//! Operators in ErgoTree

use core::cmp::Ordering;

use ergotree_ir::ergo_tree::ErgoTreeVersion;
use ergotree_ir::mir::bin_op::BinOp;
use ergotree_ir::mir::bin_op::BinOpKind;
use ergotree_ir::mir::bin_op::RelationOp;
use ergotree_ir::mir::bin_op::{ArithOp, LogicalOp};
use ergotree_ir::mir::constant::TryExtractFrom;
use ergotree_ir::mir::constant::TryExtractInto;
use ergotree_ir::mir::value::Value;
use ergotree_ir::types::stype::SType;
use ergotree_ir::unsignedbigint256::UnsignedBigInt;
use num_bigint::BigInt;
use num_traits::CheckedAdd;
use num_traits::CheckedDiv;
use num_traits::CheckedMul;
use num_traits::CheckedRem;
use num_traits::CheckedSub;
use num_traits::Num;
use num_traits::Signed;
use num_traits::Zero;

use crate::eval::env::Env;
use crate::eval::fits_256_bits;
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

/// `/` and `%` as the operand type's entry of sigmastate's table does them (`Division`:
/// `impl.i.quot`, `Modulo`: `impl.i.divisionRemainder`, v6.0.6 `trees.scala:792`, `:809`).
/// `None` is the JVM's throw
trait JvmDivision: Sized {
    fn jvm_quot(&self, divisor: &Self) -> Option<Self>;
    fn jvm_rem(&self, divisor: &Self) -> Option<Self>;
}

/// `Byte`, `Short`, `Int` and `Long` divide with Scala's own `Integral.quot` and `rem`: their
/// `ExactIntegral` instances override neither (`ExactIntegral.scala:22`, `:30`, the instances
/// from `:36`). That is the JVM's plain division: only a zero divisor throws, `MIN / -1` wraps
/// to `MIN` and `MIN % -1` is 0 (SANTA `ArithOp.division_overflow`). The zero check comes
/// first: Rust's wrapping division panics on a zero divisor
macro_rules! impl_jvm_division_fixed_width {
    ($($t:ty),*) => {$(
        impl JvmDivision for $t {
            fn jvm_quot(&self, divisor: &Self) -> Option<Self> {
                if *divisor == 0 {
                    return None;
                }
                Some(self.wrapping_div(*divisor))
            }
            fn jvm_rem(&self, divisor: &Self) -> Option<Self> {
                if *divisor == 0 {
                    return None;
                }
                Some(self.wrapping_rem(*divisor))
            }
        }
    )*};
}
impl_jvm_division_fixed_width!(i8, i16, i32, i64);

/// `UnsignedBigInt` has its own `quot` and `divisionRemainder` (`UnsignedBigIntegerOps.scala:71`,
/// `:87`). `BigInt`'s are in [`eval_bigint_arith`]
impl JvmDivision for UnsignedBigInt {
    fn jvm_quot(&self, divisor: &Self) -> Option<Self> {
        self.checked_div(divisor)
    }
    fn jvm_rem(&self, divisor: &Self) -> Option<Self> {
        self.checked_rem(divisor)
    }
}

fn eval_div<'ctx, T>(lv_raw: T, rv_raw: T) -> Result<Value<'ctx>, EvalError>
where
    T: Num + JvmDivision + Into<Value<'ctx>> + core::fmt::Display,
{
    lv_raw
        .jvm_quot(&rv_raw)
        .ok_or_else(|| arithmetic_err("/", lv_raw, rv_raw, "exception"))
        .map(|t| t.into()) // convert T to Value
}

fn eval_mod<'ctx, T>(lv_raw: T, rv_raw: T) -> Result<Value<'ctx>, EvalError>
where
    T: Num + JvmDivision + Into<Value<'ctx>> + core::fmt::Display,
{
    lv_raw
        .jvm_rem(&rv_raw)
        .ok_or_else(|| arithmetic_err("%", lv_raw, rv_raw, "exception"))
        .map(|t| t.into()) // convert T to Value
}

/// The ordering of a comparison's operands. sigmastate adds the comparison's cost once both
/// operands are evaluated, then takes the ordering of the LEFT operand's static type, from the
/// table arithmetic uses, and casts BOTH operand values to it (`SimpleRelation.opImpl`, v6.0.6
/// `trees.scala:1075-1078`; `LT.eval`, `:1092-1098`, and `LE`, `GT`, `GE` alike). A value of
/// another type throws at every tree version, also when both values are of one kind (SANTA
/// `Comparison.mixed_width`)
fn eval_ordering<'ctx>(
    op: &BinOp,
    lv: Value<'ctx>,
    rv: Value<'ctx>,
    ctx: &Context<'ctx>,
) -> Result<Ordering, EvalError> {
    ctx.add_jit_cost(20)?; // LT, LE, GT, GE: 20 for every type
    let op_tpe = op.left.tpe();
    match op_tpe {
        SType::SByte => ordering_at::<i8>(lv, rv),
        SType::SShort => ordering_at::<i16>(lv, rv),
        SType::SInt => ordering_at::<i32>(lv, rv),
        SType::SLong => ordering_at::<i64>(lv, rv),
        SType::SBigInt => ordering_at::<BigInt>(lv, rv),
        SType::SUnsignedBigInt => ordering_at::<UnsignedBigInt>(lv, rv),
        // No parsed comparison gets here: the builder requires numeric operands
        // (`comparisonOp`, `check_build`). The JVM's table has no entry for another type
        _ => Err(EvalError::UnexpectedValue(format!(
            "{:?}: no ordering for the left operand's type {op_tpe:?}",
            op.kind
        ))),
    }
}

/// The ordering of two values at `T`: each must be a `T`
fn ordering_at<'ctx, T>(lv: Value<'ctx>, rv: Value<'ctx>) -> Result<Ordering, EvalError>
where
    T: Ord + TryExtractFrom<Value<'ctx>>,
{
    let lv_raw = lv.try_extract_into::<T>()?;
    let rv_raw = rv.try_extract_into::<T>()?;
    Ok(lv_raw.cmp(&rv_raw))
}

/// An arithmetic operation at `T`, the Rust type of the left operand's static type. sigmastate
/// charges the operation's cost for that type, then casts BOTH operand values to it (v6.0.6
/// `trees.scala:729-739`): a value of another type, numeric or not, is an error. Every type but
/// `SBigInt` takes these costs (every `ArithOpCompanion.costKind`, `:746-848`); `BigInt` is
/// [`eval_bigint_arith`]
fn eval_arith<'ctx, T>(
    op: ArithOp,
    lv: Value<'ctx>,
    rv: Value<'ctx>,
    ctx: &Context<'ctx>,
) -> Result<Value<'ctx>, EvalError>
where
    T: Num
        + CheckedAdd
        + CheckedSub
        + CheckedMul
        + JvmDivision
        + Ord
        + TryExtractFrom<Value<'ctx>>
        + Into<Value<'ctx>>
        + core::fmt::Display,
{
    ctx.add_jit_cost(match op {
        ArithOp::Plus | ArithOp::Minus => 15,
        ArithOp::Multiply | ArithOp::Divide | ArithOp::Modulo => 15,
        ArithOp::Max | ArithOp::Min => 5,
    })?;
    let lv_raw = lv.try_extract_into::<T>()?;
    let rv_raw = rv.try_extract_into::<T>()?;
    match op {
        ArithOp::Plus => eval_plus(lv_raw, rv_raw),
        ArithOp::Minus => eval_minus(lv_raw, rv_raw),
        ArithOp::Multiply => eval_mul(lv_raw, rv_raw),
        ArithOp::Divide => eval_div(lv_raw, rv_raw),
        ArithOp::Modulo => eval_mod(lv_raw, rv_raw),
        ArithOp::Max => Ok(lv_raw.max(rv_raw).into()),
        ArithOp::Min => Ok(lv_raw.min(rv_raw).into()),
    }
}

/// `BigInt` arithmetic, on sigmastate's own width. Its `BigInt` wraps a `BigInteger`, and the
/// 256-bit range is a check whose place depends on the operation and on the TREE version
/// (v6.0.6 `CBigInt.scala`). `add`, `subtract` and `multiply` check their RESULT at every version
/// (`:39-43`). `divide`, `mod`, `min` and `max` check nothing (`:45-53`), so below tree version 3
/// they take and return a value outside the range: `MIN / -1` is 2^255. From tree version 3 the
/// constructor refuses any result above 255 bits (`:18-20`). SANTA
/// `BigInt.range_check_tree_version` and the rows of its reply `sigma-rust-bigint-range-reply.md`
fn eval_bigint_arith<'ctx>(
    op: ArithOp,
    lv: Value<'ctx>,
    rv: Value<'ctx>,
    ctx: &Context<'ctx>,
) -> Result<Value<'ctx>, EvalError> {
    ctx.add_jit_cost(match op {
        ArithOp::Plus | ArithOp::Minus => 20,
        ArithOp::Multiply | ArithOp::Divide | ArithOp::Modulo => 25,
        ArithOp::Max | ArithOp::Min => 10,
    })?;
    let l = lv.try_extract_into::<BigInt>()?;
    let r = rv.try_extract_into::<BigInt>()?;
    // the operation's symbol, its result, and whether the operation checks its own result
    let (symbol, result, checks_its_result) = match op {
        ArithOp::Plus => ("+", &l + &r, true),
        ArithOp::Minus => ("-", &l - &r, true),
        ArithOp::Multiply => ("*", &l * &r, true),
        // `BigInteger.divide`: a zero divisor throws; the quotient truncates toward zero. The
        // zero check comes first: the `/` of the wide integer panics on a zero divisor
        ArithOp::Divide => {
            if r.is_zero() {
                return Err(arithmetic_err("/", l, r, "exception"));
            }
            ("/", &l / &r, false)
        }
        // `BigInteger.mod`: a modulus of zero or below throws; the result is not negative
        ArithOp::Modulo => {
            if !r.is_positive() {
                return Err(arithmetic_err("%", l, r, "exception"));
            }
            let rem = &l % &r;
            ("%", if rem.is_negative() { rem + &r } else { rem }, false)
        }
        ArithOp::Max => ("max", if l >= r { l.clone() } else { r.clone() }, false),
        ArithOp::Min => ("min", if l <= r { l.clone() } else { r.clone() }, false),
    };
    let checked = checks_its_result || ctx.tree_version() >= ErgoTreeVersion::V3;
    if checked && !fits_256_bits(&result) {
        return Err(arithmetic_err(symbol, l, r, "overflow"));
    }
    Ok(Value::BigInt(result))
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
            // Eq/NEq cost is charged by eq_with_cost during the eval
            // dispatch below (per-type + per-coll-element), since the cost
            // depends on the runtime value type. LT, LE, GT, GE are charged in
            // `eval_ordering`, once the right operand is evaluated
            BinOpKind::Relation(_) => {}
            BinOpKind::Logical(_) => {
                ctx.add_jit_cost(20)?; // BinOr, BinAnd, BinXor = Fixed(20)
            }
            // never reached: `eval` returned for a `Bit` kind above, before its operands
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
                RelationOp::Gt => Ok(eval_ordering(self, lv, rv()?, ctx)?.is_gt().into()),
                RelationOp::Lt => Ok(eval_ordering(self, lv, rv()?, ctx)?.is_lt().into()),
                RelationOp::Ge => Ok(eval_ordering(self, lv, rv()?, ctx)?.is_ge().into()),
                RelationOp::Le => Ok(eval_ordering(self, lv, rv()?, ctx)?.is_le().into()),
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
                    SType::SByte => eval_arith::<i8>(op, lv, rv, ctx),
                    SType::SShort => eval_arith::<i16>(op, lv, rv, ctx),
                    SType::SInt => eval_arith::<i32>(op, lv, rv, ctx),
                    SType::SLong => eval_arith::<i64>(op, lv, rv, ctx),
                    SType::SBigInt => eval_bigint_arith(op, lv, rv, ctx),
                    SType::SUnsignedBigInt => eval_arith::<UnsignedBigInt>(op, lv, rv, ctx),
                    _ => Err(EvalError::UnexpectedValue(format!(
                        "{op:?}: no arithmetic for the left operand's type {op_tpe:?}"
                    ))),
                }
            }
            // never reached: `eval` returned for a `Bit` kind above, before its operands
            BinOpKind::Bit(_) => Err(not_evaluated("BitOp")),
        }
    }
}

#[cfg(test)]
#[cfg(feature = "arbitrary")]
#[allow(clippy::unwrap_used)]
mod arith_op_type_tests {
    use alloc::boxed::Box;
    use ergotree_ir::bigint256::BigInt256;
    use ergotree_ir::chain::context::Context;
    use ergotree_ir::chain::context_extension::ContextExtension;
    use ergotree_ir::ergo_tree::ErgoTreeVersion;
    use ergotree_ir::mir::bin_op::{ArithOp, BinOp};
    use ergotree_ir::mir::expr::Expr;
    use ergotree_ir::mir::value::Value;
    use ergotree_ir::serialization::sigma_byte_reader::from_bytes;
    use ergotree_ir::serialization::sigma_byte_reader::SigmaByteRead;
    use ergotree_ir::serialization::SigmaSerializable;
    use ergotree_ir::unsignedbigint256::UnsignedBigInt;
    use sigma_test_util::force_any_val;

    use crate::eval::test_util::{
        assert_accepts, assert_trees, eval_v3_body, try_eval_out_wo_ctx, try_eval_with_deserialize,
    };
    use crate::eval::EvalError;

    fn bigint(n: i32) -> Value<'static> {
        Value::from(BigInt256::from(n))
    }

    #[test]
    fn santa_mixed_width_rows() {
        // SANTA `ArithOp.mixed_width` (sigma-state 6.0.7), each script as a v3 sized tree and
        // as a v0 tree. sigmastate takes the implementation of the LEFT operand's static type
        // and casts both values to it (v6.0.6 `trees.scala:703-739`): a value of another
        // numeric type throws `ClassCastException`. Below tree version 3 the parser wraps the
        // narrower of two operands of different static types in an `Upcast`, which costs 10,
        // or 30 to `BigInt` (`SigmaBuilder.scala:674-683`, `:757-763`, `CostKind.scala:60-65`)
        assert_trees(&[
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
        assert_trees(&[
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
    fn santa_division_overflow_rows() {
        // SANTA `ArithOp.division_overflow` (`67da856`, sigma-state 6.0.7), each script as a v3
        // sized tree and a v0 tree: the fixed-width types divide with the JVM's plain division,
        // so `MIN / -1` is `MIN`, `MIN % -1` is 0, and only a zero divisor throws
        assert_trees(&[
            // Q0 `7 / 2`, the control
            ("0b059d040e0404", Some((Value::Int(3), 25))),
            ("009d040e0404", Some((Value::Int(3), 25))),
            // Q1 `Int.MinValue / -1`
            (
                "0b0e9d04ffffffffffffffffff010401",
                Some((Value::Int(i32::MIN), 25)),
            ),
            (
                "009d04ffffffffffffffffff010401",
                Some((Value::Int(i32::MIN), 25)),
            ),
            // Q2 `Int.MinValue % -1`
            (
                "0b0e9e04ffffffffffffffffff010401",
                Some((Value::Int(0), 25)),
            ),
            ("009e04ffffffffffffffffff010401", Some((Value::Int(0), 25))),
            // Q3 `Long.MinValue / -1L`
            (
                "0b0e9d05ffffffffffffffffff010501",
                Some((Value::Long(i64::MIN), 25)),
            ),
            (
                "009d05ffffffffffffffffff010501",
                Some((Value::Long(i64::MIN), 25)),
            ),
            // Q5 `Long.MinValue % -1L`
            (
                "0b0e9e05ffffffffffffffffff010501",
                Some((Value::Long(0), 25)),
            ),
            ("009e05ffffffffffffffffff010501", Some((Value::Long(0), 25))),
            // Q4 `(-128: Byte) / (-1: Byte)`
            ("0b059d028002ff", Some((Value::Byte(i8::MIN), 25))),
            ("009d028002ff", Some((Value::Byte(i8::MIN), 25))),
            // Q6 `(-128: Byte) % (-1: Byte)`
            ("0b059e028002ff", Some((Value::Byte(0), 25))),
            ("009e028002ff", Some((Value::Byte(0), 25))),
            // Q7 `Short.MinValue / (-1: Short)`
            ("0b079d03ffff030301", Some((Value::Short(i16::MIN), 25))),
            ("009d03ffff030301", Some((Value::Short(i16::MIN), 25))),
            // Q8 `Short.MinValue % (-1: Short)`
            ("0b079e03ffff030301", Some((Value::Short(0), 25))),
            ("009e03ffff030301", Some((Value::Short(0), 25))),
            // Z1 `7 / 0`
            ("0b059d040e0400", None),
            ("009d040e0400", None),
            // Z2 `7 % 0`
            ("0b059e040e0400", None),
            ("009e040e0400", None),
        ]);
    }

    #[test]
    fn a_zero_divisor_is_an_error_at_every_width() {
        // By reading: the JVM's division throws `ArithmeticException` on a zero divisor whatever
        // the dividend (SANTA measured `7 / 0` and `7 % 0`). `MIN / 0` and `MIN % 0` at each
        // width: an error here, and not the panic of a wrapping division by zero
        assert_trees(&[
            // Byte
            ("0b059d02800200", None),
            ("009d02800200", None),
            ("0b059e02800200", None),
            ("009e02800200", None),
            // Short
            ("0b079d03ffff030300", None),
            ("009d03ffff030300", None),
            ("0b079e03ffff030300", None),
            ("009e03ffff030300", None),
            // Int
            ("0b0e9d04ffffffffffffffffff010400", None),
            ("009d04ffffffffffffffffff010400", None),
            ("0b0e9e04ffffffffffffffffff010400", None),
            ("009e04ffffffffffffffffff010400", None),
            // Long
            ("0b0e9d05ffffffffffffffffff010500", None),
            ("009d05ffffffffffffffffff010500", None),
            ("0b0e9e05ffffffffffffffffff010500", None),
            ("009e05ffffffffffffffffff010500", None),
        ]);
    }

    #[test]
    fn division_keeps_its_other_values() {
        // sigma-rust's own values and costs at `e922604d`, not JVM measurements: every pair
        // other than `(MIN, -1)` divides as it did
        let i = |v, cost| Some((Value::Int(v), cost));
        assert_trees(&[
            // `Int.MinValue / 1`
            ("0b0e9d04ffffffffffffffffff010402", i(i32::MIN, 25)),
            // `Int.MinValue % 1`
            ("0b0e9e04ffffffffffffffffff010402", i(0, 25)),
            // `-1 / Int.MinValue`
            ("0b0e9d040104ffffffffffffffffff01", i(0, 25)),
            // `Int.MinValue / Int.MinValue`
            (
                "0b179d04ffffffffffffffffff0104ffffffffffffffffff01",
                i(1, 25),
            ),
            // `Int.MinValue % Int.MinValue`
            (
                "0b179e04ffffffffffffffffff0104ffffffffffffffffff01",
                i(0, 25),
            ),
            // `-7 / 2`
            ("0b059d040d0404", i(-3, 25)),
            // `-7 % 2`
            ("0b059e040d0404", i(-1, 25)),
            // `7 % -2`
            ("0b059e040e0403", i(1, 25)),
            // `7 / -1`
            ("0b059d040e0401", i(-7, 25)),
            // `7 % -1`
            ("0b059e040e0401", i(0, 25)),
        ]);
    }

    #[test]
    fn producers_keep_their_arithmetic() {
        // An operand whose static type comes from another node keeps its arithmetic. Values and
        // costs are sigma-rust's own at `a7656774`, not JVM measurements: a regression guard
        // for a static type that disagrees with the value it types
        assert_trees(&[
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
    fn a_substituted_operand_keeps_the_parser_s_upcast() {
        // By reading, not measured (SANTA's eval tier binds no context variable). Below tree
        // version 3 the parser wraps `executeFromVar[Int](1)` in an `Upcast` to `Long` when it
        // builds the sum. The substitution at spend replaces the node under that `Upcast` and
        // rebuilds its ancestors through their constructors, not the builder (v6.0.7
        // `Rewriter.scala:241-260`), so the `Upcast` stays and no new one is inserted. From
        // v3 there is none, and the `Int` meets the `Long` implementation
        fn sum(version: ErgoTreeVersion, body_hex: &str, var_hex: &str) -> Result<i64, EvalError> {
            let body = base16::decode(body_hex.as_bytes()).unwrap();
            let expr = from_bytes(&body)
                .with_tree_version(version, Expr::sigma_parse)
                .unwrap();
            let var = base16::decode(var_hex.as_bytes()).unwrap();
            let ext = ContextExtension {
                values: [(1u8, var.into())].iter().cloned().collect(),
            };
            let mut ctx = force_any_val::<Context>().with_extension(&ext);
            ctx.activated_script_version_byte = 3;
            ctx.tree_version.set(version);
            try_eval_with_deserialize::<i64>(&expr, &ctx)
        }
        // `1L + executeFromVar[Int](1)`, the variable holding the constant `2`
        assert_eq!(sum(ErgoTreeVersion::V0, "9a0502d40401", "0404").unwrap(), 3);
        assert!(sum(ErgoTreeVersion::V3, "9a0502d40401", "0404").is_err());
        // the control: `1L + executeFromVar[Long](1)`, the variable holding `2L`
        assert_eq!(sum(ErgoTreeVersion::V3, "9a0502d40501", "0504").unwrap(), 3);
    }

    #[test]
    fn a_tree_built_in_memory_gets_no_upcast() {
        // By reading: only the deserializing builder inserts an `Upcast`. A sum built without
        // the parser, as a tree from sigmastate's `StdSigmaBuilder`, keeps its operands as
        // given, and a `Long` with an `Int` throws at evaluation
        let plus = |right: Expr| -> Expr {
            BinOp {
                kind: ArithOp::Plus.into(),
                left: Box::new(Expr::Const(1i64.into())),
                right: Box::new(right),
            }
            .into()
        };
        assert!(try_eval_out_wo_ctx::<Value>(&plus(Expr::Const(2i32.into()))).is_err());
        assert_eq!(
            try_eval_out_wo_ctx::<Value>(&plus(Expr::Const(2i64.into()))).unwrap(),
            Value::Long(3)
        );
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

#[cfg(test)]
#[cfg(feature = "arbitrary")]
#[allow(clippy::unwrap_used)]
mod comparison_type_tests {
    use alloc::boxed::Box;
    use ergotree_ir::mir::bin_op::{BinOp, RelationOp};
    use ergotree_ir::mir::expr::Expr;
    use ergotree_ir::mir::value::Value;

    use crate::eval::test_util::{assert_trees, eval_tree, try_eval_out_wo_ctx};

    #[test]
    fn santa_comparison_rows() {
        // SANTA `Comparison.mixed_width` (`78cb343`, sigma-state 6.0.7), each script as a v3
        // sized tree and a v0 tree. `I2` = `If(false, 1L, 2)`, `I3` = `If(false, 1L, 3)`: the
        // static type is `Long`, the value an `Int`
        let f = |cost| Some((Value::Boolean(false), cost));
        assert_trees(&[
            // R0 `1L > 2L`, the control
            ("0b059105020504", f(30)),
            ("009105020504", f(30)),
            // R1 `I3 > I2`
            ("0b0f919501000502040695010005020404", None),
            ("00919501000502040695010005020404", None),
            // R2 `I2 > 1L`
            ("0b0a91950100050204040502", None),
            ("0091950100050204040502", None),
            // R3 `3L > I2`
            ("0b0a91050695010005020404", None),
            ("0091050695010005020404", None),
            // R4 `I2 < I3`
            ("0b0f8f9501000502040495010005020406", None),
            ("008f9501000502040495010005020406", None),
            // R5 `I2 >= I2`
            ("0b0f929501000502040495010005020404", None),
            ("00929501000502040495010005020404", None),
            // R6 `I2 <= I2`
            ("0b0f909501000502040495010005020404", None),
            ("00909501000502040495010005020404", None),
            // R7 `I2 == I2`: refused by `EQ`'s own check of each value against its operand's
            // type, a control and not the ordering's refusal
            ("0b0f939501000502040495010005020404", None),
            ("00939501000502040495010005020404", None),
        ]);
    }

    #[test]
    fn a_comparison_takes_the_left_operand_s_static_type() {
        // The refusals are by reading: the lookup is the one SANTA measured for a `Long`
        // operand holding an `Int` (`SimpleRelation.opImpl`, v6.0.6 `trees.scala:1075-1078`),
        // here for other types and for the other direction. The values and costs are
        // sigma-rust's own at `d34f774b`, not JVM measurements
        let b = |v, cost| Some((Value::Boolean(v), cost));
        assert_trees(&[
            // `If(false, 1, 3L) > If(false, 1, 2L)`: `Long` values in `Int` operands
            ("0b0f919501000402050695010004020504", None),
            ("00919501000402050695010004020504", None),
            // `If(false, BigInt 1, 3L) > If(false, BigInt 1, 2L)`: `Long` values in `BigInt`
            // operands
            ("0b119195010006010105069501000601010504", None),
            ("009195010006010105069501000601010504", None),
            // `If(false, 1L, 3L) > If(false, 1L, 2L)`, the honest twin of R1
            ("0b0f919501000502050695010005020504", b(true, 60)),
            ("00919501000502050695010005020504", b(true, 60)),
            // `UnsignedBigInt 1 > UnsignedBigInt 2`
            ("0b0791090101090102", b(false, 30)),
            // `BigInt 1 > BigInt 2`
            ("0b0791060101060102", b(false, 30)),
            // `(1: Byte) > (2: Byte)`
            ("0b059102010202", b(false, 30)),
        ]);
    }

    #[test]
    fn producers_keep_their_comparison() {
        // A comparison whose left operand's type comes from another node keeps its value:
        // sigma-rust's own values and costs at `d34f774b`, not JVM measurements. A `tpe()` that
        // named another type than the value's would refuse a tree the JVM accepts
        let b = |v, cost| Some((Value::Boolean(v), cost));
        assert_trees(&[
            // `ByIndex(Coll[Long](1, 2), 0) > 1L`
            ("0b0b91b2110202040400000502", b(false, 65)),
            ("0091b2110202040400000502", b(false, 65)),
            // `(1L, 2L)._1 > 1L`
            ("0b08918c590204010502", b(false, 40)),
            // `{ val x = 1L; x > x }`
            ("0b0bd801d60105029172017201", b(false, 42)),
            // `SizeOf(Coll[Long](1, 2)) > 1`
            ("0b0891b1110202040402", b(true, 44)),
            // `Coll[Long](1, 2).size > 1`
            ("0b0a91db0c01110202040402", b(true, 48)),
            // `OptionGet(Some(1L)) > 1L`
            ("0b0791e42901020502", b(false, 45)),
        ]);
    }

    #[test]
    fn a_comparison_charges_after_both_operands() {
        // By reading: `LT.eval` evaluates both operands, then adds its cost (v6.0.6
        // `trees.scala:1092-1098`). `1L > 2L / 0L` fails in the division: the three constants
        // and the division's 15 are charged, the comparison's 20 is not
        for tree in ["0b089105029d05040500", "009105029d05040500"] {
            let (res, cost) = eval_tree(tree);
            assert!(res.is_err(), "{tree}: {res:?}");
            assert_eq!(cost, 30, "{tree}");
        }
    }

    #[test]
    fn a_comparison_built_in_memory_over_a_non_numeric_type_errors() {
        // The parser refuses a comparison of non-numeric operands (`comparisonOp`); a tree
        // built without it has no entry in the JVM's table and must be an error here, not a
        // panic
        for op in [
            RelationOp::Gt,
            RelationOp::Lt,
            RelationOp::Ge,
            RelationOp::Le,
        ] {
            let expr: Expr = BinOp {
                kind: op.into(),
                left: Box::new(Expr::Const(true.into())),
                right: Box::new(Expr::Const(true.into())),
            }
            .into();
            assert!(try_eval_out_wo_ctx::<Value>(&expr).is_err(), "{op:?}");
        }
    }
}

#[cfg(test)]
#[cfg(feature = "arbitrary")]
#[allow(clippy::unwrap_used)]
mod bigint_range_tests {
    use alloc::string::String;
    use ergotree_ir::ergo_tree::ErgoTreeVersion;
    use ergotree_ir::mir::value::Value;
    use num_bigint::BigInt;

    use crate::eval::test_util::{bigint_x, bigint_y, eval_body, BIGINT_MAX_HEX};

    const V0: ErgoTreeVersion = ErgoTreeVersion::V0;
    const V1: ErgoTreeVersion = ErgoTreeVersion::V1;
    const V2: ErgoTreeVersion = ErgoTreeVersion::V2;
    const V3: ErgoTreeVersion = ErgoTreeVersion::V3;

    /// 2^255: one above the 256-bit range
    fn p() -> BigInt {
        BigInt::from(1) << 255usize
    }

    fn big(v: BigInt) -> Option<Value<'static>> {
        Some(Value::BigInt(v))
    }

    fn boolean(v: bool) -> Option<Value<'static>> {
        Some(Value::Boolean(v))
    }

    /// A one-byte `BigInt` constant
    fn b(n: i8) -> String {
        format!("0601{:02x}", n as u8)
    }

    /// The body's value and cost at tree version 0, or that it errors there
    fn at_v0(body: &str, expected: Option<Value<'static>>, cost: u64) {
        let (res, c) = eval_body(body, V0);
        match expected {
            Some(v) => {
                assert_eq!(res.as_ref().ok(), Some(&v), "{body}: {res:?}");
                assert_eq!(c, cost, "{body}");
            }
            None => assert!(res.is_err(), "{body}: {res:?}"),
        }
    }

    /// [`at_v0`], and an error at tree version 3: there sigmastate's `BigInt` constructor
    /// throws at `X` itself
    fn v0_only(body: &str, expected: Option<Value<'static>>, cost: u64) {
        at_v0(body, expected, cost);
        let (res, _) = eval_body(body, V3);
        assert!(res.is_err(), "{body} at v3: {res:?}");
    }

    #[test]
    fn santa_bigint_range_rows() {
        // SANTA's reply `sigma-rust-bigint-range-reply.md` (sigma-state 6.0.7), measured; G0 and
        // G1 are also `BigInt.range_check_tree_version` (`8798807`). `X` = `BigInt.MIN / -1`:
        // 2^255 below tree version 3, where only the results of `+ - *` and negation are
        // range-checked (v6.0.6 `CBigInt.scala:39-55`)
        let x = bigint_x();
        let x = x.as_str();
        // G1 `X`
        v0_only(x, big(p()), 35);
        // G0 `X - 1`
        v0_only(&format!("99{x}{}", b(1)), big(p() - 1), 60);
        // B1 `X / 1`
        v0_only(&format!("9d{x}{}", b(1)), big(p()), 65);
        // B2 `X / -1`
        v0_only(&format!("9d{x}{}", b(-1)), big(-p()), 65);
        // B3 `X * 1`: the result is checked
        v0_only(&format!("9c{x}{}", b(1)), None, 0);
        // B4 `X * -1`
        v0_only(&format!("9c{x}{}", b(-1)), big(-p()), 65);
        // B5 `X * 0`
        v0_only(&format!("9c{x}{}", b(0)), big(BigInt::from(0)), 65);
        // B6 `X + 0`: the result is checked
        v0_only(&format!("9a{x}{}", b(0)), None, 0);
        // B7 `X + (-1)`
        v0_only(&format!("9a{x}{}", b(-1)), big(p() - 1), 60);
        // B8 `-X`
        v0_only(&format!("f0{x}"), big(-p()), 65);
        // B9 `min(X, 0)`
        v0_only(&format!("a1{x}{}", b(0)), big(BigInt::from(0)), 50);
        // B10 `max(X, 0)`
        v0_only(&format!("a2{x}{}", b(0)), big(p()), 50);
        // B11 `X > 0`
        v0_only(&format!("91{x}{}", b(0)), boolean(true), 60);
        // B12 `X > BigInt.MAX`
        v0_only(&format!("91{x}{BIGINT_MAX_HEX}"), boolean(true), 60);
        // B13 `X == X`
        v0_only(&format!("93{x}{x}"), boolean(true), 75);
        // B14 `X == BigInt.MAX`
        v0_only(&format!("93{x}{BIGINT_MAX_HEX}"), boolean(false), 45);
        // B15 `X % 7`
        v0_only(&format!("9e{x}{}", b(7)), big(BigInt::from(1)), 65);
        // B16 `7 % X`
        v0_only(&format!("9e{}{x}", b(7)), big(BigInt::from(7)), 65);
        // B17 `X % X`
        v0_only(&format!("9e{x}{x}"), big(BigInt::from(0)), 95);

        // The gate is the tree version: 2^255 at tree versions 1 and 2 as well
        for version in [V1, V2] {
            let (res, cost) = eval_body(x, version);
            assert_eq!(res.ok(), big(p()), "{version:?}");
            assert_eq!(cost, 35, "{version:?}");
        }
    }

    #[test]
    fn santa_bigint_range_rows_through_other_nodes() {
        // SANTA's reply, section 2 (measured): nothing on the way refuses the value, and each
        // row costs what its in-range twin costs. `Y` = `6 / -1` is the twin: -7 at both
        // versions
        const F: &str = "d9010106997201060101"; // `{ (a: BigInt) => a - 1 }`
        type Body = fn(&str) -> String;
        let rows: [(Body, u64); 5] = [
            // `{ val x = X; x - 1 }`
            (|v| format!("d801d601{v}997201{}", b(1)), 72),
            // `(X, 1)._1 - 1`
            (|v| format!("998c8602{v}040201{}", b(1)), 90),
            // `Coll(X)(0) - 1`
            (|v| format!("99b2830106{v}040000{}", b(1)), 115),
            // `{ (a: BigInt) => a - 1 }(X)`
            (|v| format!("da{F}01{v}"), 105),
            // `Coll(X).map({ (a: BigInt) => a - 1 })(0)`
            (|v| format!("b2ad830106{v}{F}040000"), 151),
        ];
        for (body, cost) in rows {
            v0_only(&body(&bigint_x()), big(p() - 1), cost);
            for version in [V0, V3] {
                let (res, c) = eval_body(&body(&bigint_y()), version);
                assert_eq!(res.ok(), big(BigInt::from(-7)), "{version:?}");
                assert_eq!(c, cost, "{version:?}");
            }
        }

        // A cast never takes a `BigInt` input below tree version 3, in range or not
        for v in [bigint_x(), bigint_y()] {
            // `Downcast(v, Long)`, `Upcast(v, BigInt)`
            at_v0(&format!("7d{v}05"), None, 0);
            at_v0(&format!("7e{v}06"), None, 0);
        }
        let (res, cost) = eval_body(&format!("7d{}05", bigint_y()), V3);
        assert_eq!((res.ok(), cost), (Some(Value::Long(-6)), 45));
        let (res, cost) = eval_body(&format!("7e{}06", bigint_y()), V3);
        assert_eq!((res.ok(), cost), (big(BigInt::from(-6)), 65));
    }

    #[test]
    fn a_wide_bigint_moves_through_other_nodes() {
        // By reading: SANTA did not measure these one by one. A node that only moves a value
        // passes the wide one through; the costs are the in-range twins', sigma-rust's own
        let x = bigint_x();
        // `If(true, X, 0) - 1`
        v0_only(&format!("99950101{x}{}{}", b(0), b(1)), big(p() - 1), 75);
        // `Coll(X) == Coll(X)`
        v0_only(&format!("93830106{x}830106{x}"), boolean(true), 133);
        // `min(max(X, 0), X)`
        v0_only(&format!("a1a2{x}{}{x}", b(0)), big(p()), 95);
    }

    #[test]
    fn a_wide_bigint_divides_without_panic() {
        // By reading: `BigInteger.divide` throws on a zero divisor and `BigInteger.mod` on a
        // modulus of zero or below. An error here, and not the panic of a division by zero
        for v in [bigint_x(), bigint_y()] {
            for body in [
                format!("9d{v}{}", b(0)),  // `v / 0`
                format!("9e{v}{}", b(0)),  // `v % 0`
                format!("9e{v}{}", b(-1)), // `v % -1`
            ] {
                for version in [V0, V3] {
                    let (res, _) = eval_body(&body, version);
                    assert!(res.is_err(), "{body} at {version:?}: {res:?}");
                }
            }
        }
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
    use ergotree_ir::bigint256::BigInt256;
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
                Some(Value::from(BigInt256::from(3i32))),
                60,
            ), // Plus(Int 1, BigInt 2)
            (
                "009a0502060102",
                Some(Value::from(BigInt256::from(3i32))),
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
    fn the_minimum_divided_by_minus_one_wraps_at_every_width() {
        // SANTA `ArithOp.division_overflow` (sigma-state 6.0.7): the JVM returns `MIN` and 0 at
        // `Byte`, `Short`, `Int` and `Long`. The zero divisor is by reading at `MIN`
        macro_rules! check {
            ($t:ty) => {{
                assert_eq!(eval_arith_op(ArithOp::Divide, <$t>::MIN, -1), Ok(<$t>::MIN));
                assert_eq!(eval_arith_op(ArithOp::Modulo, <$t>::MIN, -1), Ok(0));
                assert!(eval_arith_op(ArithOp::Divide, <$t>::MIN, 0).is_err());
                assert!(eval_arith_op(ArithOp::Modulo, <$t>::MIN, 0).is_err());
            }};
        }
        check!(i8);
        check!(i16);
        check!(i32);
        check!(i64);
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

        // `MIN / -1`: sigmastate's `divide` does not check its result (v6.0.6 `CBigInt.scala:45`),
        // so below tree version 3 it is 2^255; from version 3 the constructor refuses it
        // (`:18-20`). SANTA `BigInt.range_check_tree_version` G1
        let min_div_minus_one = crate::eval::test_util::bigint_x();
        let (res, cost) = crate::eval::test_util::eval_body(
            &min_div_minus_one,
            ergotree_ir::ergo_tree::ErgoTreeVersion::V0,
        );
        assert_eq!(
            res.ok(),
            Some(Value::BigInt(num_bigint::BigInt::from(1) << 255usize))
        );
        assert_eq!(cost, 35);
        let (res, _) = crate::eval::test_util::eval_body(
            &min_div_minus_one,
            ergotree_ir::ergo_tree::ErgoTreeVersion::V3,
        );
        assert!(res.is_err());
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
            prop_assert_eq!(eval_arith_op(ArithOp::Divide, l, r).ok(), (r != 0).then(|| l.wrapping_div(r)));
            prop_assert_eq!(eval_arith_op(ArithOp::Modulo, l, r).ok(), (r != 0).then(|| l.wrapping_rem(r)));
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
            prop_assert_eq!(eval_arith_op(ArithOp::Divide, l, r).ok(), (r != 0).then(|| l.wrapping_div(r)));
            prop_assert_eq!(eval_arith_op(ArithOp::Modulo, l, r).ok(), (r != 0).then(|| l.wrapping_rem(r)));
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
            prop_assert_eq!(eval_arith_op(ArithOp::Divide, l, r).ok(), (r != 0).then(|| l.wrapping_div(r)));
            prop_assert_eq!(eval_arith_op(ArithOp::Modulo, l, r).ok(), (r != 0).then(|| l.wrapping_rem(r)));
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
            prop_assert_eq!(eval_arith_op(ArithOp::Divide, l, r).ok(), (r != 0).then(|| l.wrapping_div(r)));
            prop_assert_eq!(eval_arith_op(ArithOp::Modulo, l, r).ok(), (r != 0).then(|| l.wrapping_rem(r)));
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
