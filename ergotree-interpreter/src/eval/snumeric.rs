use alloc::vec::Vec;
use ergotree_ir::{
    bigint256::BigInt256,
    mir::{coll_value::CollValue, constant::TryExtractInto, value::Value},
    types::{
        smethod::SMethod,
        snumeric::{
            self,
            sbigint::{TO_UNSIGNED_METHOD_ID, TO_UNSIGNED_MOD_METHOD_ID},
            sunsignedbigint::{
                MOD_INVERSE_METHOD_ID, MOD_METHOD_ID, MULTIPLY_MOD_METHOD_ID, PLUS_MOD_METHOD_ID,
                SUBTRACT_MOD_METHOD_ID, TO_SIGNED_METHOD_ID,
            },
        },
        stype::SType,
        stype_companion::STypeCompanion,
    },
    unsignedbigint256::UnsignedBigInt,
};
use num_traits::{CheckedRem, CheckedShr};

use super::{fits_256_bits, EvalError, EvalFn};

/// Scala `FixedCost(JitCost(5))` charged per numeric 6.0 method call —
/// `SNumericTypeMethods.{ToBytes,ToBits,BitwiseOp}_CostKind` and
/// `SBigIntMethods.ToUnsignedCostKind` in sigma-state `ast/methods.scala`.
/// Charged inside each eval fn below, mirroring how every other method family
/// charges its own costKind (scoll PerItemCost; sbox/savltree/sglobal FixedCost).
const NUMERIC_METHOD_COST_KIND: u64 = 5;

const TO_BYTES_EVAL_FN: EvalFn = |_mc, _env, ctx, obj, _args| {
    ctx.add_jit_cost(NUMERIC_METHOD_COST_KIND)?;
    // the four fixed-width containers build a generic `Coll[Byte]`, the two arbitrary-precision
    // ones a specialised one (SANTA `Coll.producer_specialisation_witness` H3, H4, H12)
    fn generic_bytes(bytes: &[u8]) -> Value<'static> {
        let items: Vec<Value> = bytes.iter().map(|&b| Value::Byte(b as i8)).collect();
        Value::Coll(CollValue::generic(SType::SByte, items))
    }
    Ok(match obj {
        Value::Byte(obj) => generic_bytes(&obj.to_be_bytes()),
        Value::Short(obj) => generic_bytes(&obj.to_be_bytes()),
        Value::Int(obj) => generic_bytes(&obj.to_be_bytes()),
        Value::Long(obj) => generic_bytes(&obj.to_be_bytes()),
        // BigInt/UnsignedBigInt use a minimal-length big-endian encoding, not a
        // fixed 32 bytes: JVM `BigInt.toBytes` = `BigInteger.toByteArray` (signed
        // two's-complement, minimal length, `0 -> [0]`) and
        // `UnsignedBigInt.toBytes` = `BigIntegers.asUnsignedByteArray` (unsigned,
        // minimal length, `0 -> []`). `to_be_vec` already implements both (it is
        // what serialization uses); `to_be_bytes` is the fixed-width form.
        Value::BigInt(obj) => bigint256(obj)?.to_be_vec().into(),
        Value::UnsignedBigInt(obj) => obj.to_be_vec().into(),
        other => {
            return Err(EvalError::UnexpectedValue(format!(
                "Expected numeric type, got {other:?}"
            )))
        }
    })
};

static TO_BITS_EVAL_FN: EvalFn = |_mc, _env, ctx, obj, _args| {
    ctx.add_jit_cost(NUMERIC_METHOD_COST_KIND)?;
    fn byte_to_bits(mut byte: u8) -> [bool; 8] {
        let mut res = [false; 8];
        let mut i = 8;
        while byte != 0 {
            i -= 1;
            res[i] = (byte & 1) == 1;
            byte >>= 1;
        }
        res
    }
    // every `toBits` builds a specialised `Coll[Boolean]` (SANTA H5)
    fn to_bits(bytes: &[u8]) -> Value<'static> {
        Value::Coll(CollValue::Booleans(
            bytes.iter().copied().flat_map(byte_to_bits).collect(),
        ))
    }
    Ok(match obj {
        Value::Byte(obj) => to_bits(obj.to_be_bytes().as_slice()),
        Value::Short(obj) => to_bits(obj.to_be_bytes().as_slice()),
        Value::Int(obj) => to_bits(obj.to_be_bytes().as_slice()),
        Value::Long(obj) => to_bits(obj.to_be_bytes().as_slice()),
        // Minimal-length bytes, as in TO_BYTES_EVAL_FN: JVM `toBits` derives from
        // the same big-endian byte sequence, so it is minimal-length too.
        Value::BigInt(obj) => to_bits(bigint256(obj)?.to_be_vec().as_slice()),
        Value::UnsignedBigInt(obj) => to_bits(obj.to_be_vec().as_slice()),
        other => {
            return Err(EvalError::UnexpectedValue(format!(
                "Expected numeric type, got {other:?}"
            )))
        }
    })
};

static BITWISE_INVERSE_EVAL_FN: EvalFn = |_mc, _env, ctx, obj, _args| {
    ctx.add_jit_cost(NUMERIC_METHOD_COST_KIND)?;
    Ok(match obj {
        Value::Byte(obj) => (!obj).into(),
        Value::Short(obj) => (!obj).into(),
        Value::Int(obj) => (!obj).into(),
        Value::Long(obj) => (!obj).into(),
        Value::BigInt(obj) => (!bigint256(obj)?).into(),
        Value::UnsignedBigInt(obj) => (!obj).into(),
        other => {
            return Err(EvalError::UnexpectedValue(format!(
                "Expected numeric type, got {other:?}"
            )))
        }
    })
};

static BITWISE_OR_EVAL_FN: EvalFn = |_mc, _env, ctx, obj, args| {
    ctx.add_jit_cost(NUMERIC_METHOD_COST_KIND)?;
    // Checked rather than indexed: the argument count is the parser's to decide, and
    // sigmastate parses a zero-argument MethodCall below ErgoTree v3 (writing it back as a
    // PropertyCall). Indexing aborted the process on one.
    let rhs = args
        .first()
        .cloned()
        .ok_or_else(|| EvalError::NotFound("bitwiseOr: missing right argument".into()))?;
    Ok(match _mc.obj_type {
        STypeCompanion::SByte => {
            (obj.try_extract_into::<i8>()? | rhs.try_extract_into::<i8>()?).into()
        }
        STypeCompanion::SShort => {
            (obj.try_extract_into::<i16>()? | rhs.try_extract_into::<i16>()?).into()
        }
        STypeCompanion::SInt => {
            (obj.try_extract_into::<i32>()? | rhs.try_extract_into::<i32>()?).into()
        }
        STypeCompanion::SLong => {
            (obj.try_extract_into::<i64>()? | rhs.try_extract_into::<i64>()?).into()
        }
        STypeCompanion::SBigInt => {
            (obj.try_extract_into::<BigInt256>()? | rhs.try_extract_into::<BigInt256>()?).into()
        }
        STypeCompanion::SUnsignedBigInt => (obj.try_extract_into::<UnsignedBigInt>()?
            | rhs.try_extract_into::<UnsignedBigInt>()?)
        .into(),
        other => {
            return Err(EvalError::UnexpectedValue(format!(
                "Expected numeric type, got {other:?}"
            )))
        }
    })
};

static BITWISE_AND_EVAL_FN: EvalFn = |_mc, _env, ctx, obj, args| {
    ctx.add_jit_cost(NUMERIC_METHOD_COST_KIND)?;
    // Checked rather than indexed: the argument count is the parser's to decide, and
    // sigmastate parses a zero-argument MethodCall below ErgoTree v3 (writing it back as a
    // PropertyCall). Indexing aborted the process on one.
    let rhs = args
        .first()
        .cloned()
        .ok_or_else(|| EvalError::NotFound("bitwiseAnd: missing right argument".into()))?;
    Ok(match _mc.obj_type {
        STypeCompanion::SByte => {
            (obj.try_extract_into::<i8>()? & rhs.try_extract_into::<i8>()?).into()
        }
        STypeCompanion::SShort => {
            (obj.try_extract_into::<i16>()? & rhs.try_extract_into::<i16>()?).into()
        }
        STypeCompanion::SInt => {
            (obj.try_extract_into::<i32>()? & rhs.try_extract_into::<i32>()?).into()
        }
        STypeCompanion::SLong => {
            (obj.try_extract_into::<i64>()? & rhs.try_extract_into::<i64>()?).into()
        }
        STypeCompanion::SBigInt => {
            (obj.try_extract_into::<BigInt256>()? & rhs.try_extract_into::<BigInt256>()?).into()
        }
        STypeCompanion::SUnsignedBigInt => (obj.try_extract_into::<UnsignedBigInt>()?
            & rhs.try_extract_into::<UnsignedBigInt>()?)
        .into(),
        other => {
            return Err(EvalError::UnexpectedValue(format!(
                "Expected numeric type, got {other:?}"
            )))
        }
    })
};

static BITWISE_XOR_EVAL_FN: EvalFn = |_mc, _env, ctx, obj, args| {
    ctx.add_jit_cost(NUMERIC_METHOD_COST_KIND)?;
    let rhs = args
        .first()
        .ok_or_else(|| EvalError::UnexpectedValue("rhs missing".into()))?
        .clone();
    Ok(match _mc.obj_type {
        STypeCompanion::SByte => {
            (obj.try_extract_into::<i8>()? ^ rhs.try_extract_into::<i8>()?).into()
        }
        STypeCompanion::SShort => {
            (obj.try_extract_into::<i16>()? ^ rhs.try_extract_into::<i16>()?).into()
        }
        STypeCompanion::SInt => {
            (obj.try_extract_into::<i32>()? ^ rhs.try_extract_into::<i32>()?).into()
        }
        STypeCompanion::SLong => {
            (obj.try_extract_into::<i64>()? ^ rhs.try_extract_into::<i64>()?).into()
        }
        STypeCompanion::SBigInt => {
            (obj.try_extract_into::<BigInt256>()? ^ rhs.try_extract_into::<BigInt256>()?).into()
        }
        STypeCompanion::SUnsignedBigInt => (obj.try_extract_into::<UnsignedBigInt>()?
            ^ rhs.try_extract_into::<UnsignedBigInt>()?)
        .into(),
        other => {
            return Err(EvalError::UnexpectedValue(format!(
                "Expected numeric type, got {other:?}"
            )))
        }
    })
};

/// The `BigInt` receiver of a method as the 256-bit type its operations are written for.
/// sigmastate declares no `BigInt` method below tree version 3, and from version 3 its
/// constructor refuses a value above 255 bits, so the receiver always fits; a wider one is
/// an error
fn bigint256(v: num_bigint::BigInt) -> Result<BigInt256, EvalError> {
    BigInt256::try_from(v).map_err(EvalError::Misc)
}

#[inline(never)]
fn invalid_shift_err() -> EvalError {
    EvalError::Misc("shift value is out of bounds".into())
}

static SHIFT_LEFT_EVAL_FN: EvalFn = |_mc, _env, ctx, obj, args| {
    ctx.add_jit_cost(NUMERIC_METHOD_COST_KIND)?;
    let shift_value: u32 = args
        .first()
        .ok_or_else(|| EvalError::UnexpectedValue("shift arg missing".into()))?
        .clone()
        .try_extract_into::<i32>()?
        .try_into()
        .map_err(|_| EvalError::UnexpectedValue("expected non-negative shift value".into()))?;
    Ok(match obj {
        Value::Byte(obj) => obj
            .checked_shl(shift_value)
            .ok_or_else(invalid_shift_err)?
            .into(),
        Value::Short(obj) => obj
            .checked_shl(shift_value)
            .ok_or_else(invalid_shift_err)?
            .into(),
        Value::Int(obj) => obj
            .checked_shl(shift_value)
            .ok_or_else(invalid_shift_err)?
            .into(),
        Value::Long(obj) => obj
            .checked_shl(shift_value)
            .ok_or_else(invalid_shift_err)?
            .into(),
        Value::BigInt(obj) => bigint_shift_left(obj, shift_value)?,
        Value::UnsignedBigInt(obj) => unsigned_bigint_shift_left(obj, shift_value)?,
        other => {
            return Err(EvalError::UnexpectedValue(format!(
                "Expected numeric type, got {other:?}"
            )))
        }
    })
};

/// sigmastate's `BigInt.shiftLeft` (v6.0.6): the shift amount is refused from 256
/// (`BigIntegerOps.scala:109-115`), then the value is shifted as an unbounded integer and
/// refused if it left the 256-bit range (`CBigInt.scala:67`,
/// `wrappedValue.shiftLeft(n).toSignedBigIntValueExact`; `Extensions.scala:219-223`). SANTA
/// `BigInt.shift_and_inverse_range_edges`
fn bigint_shift_left(
    obj: num_bigint::BigInt,
    shift_value: u32,
) -> Result<Value<'static>, EvalError> {
    let obj = bigint256(obj)?;
    // A `u8` holds exactly the amounts the JVM accepts. The wide integer has no width of its
    // own to bound a shift, so the amount it is shifted by is of this type: a larger one
    // cannot reach it
    let shift = u8::try_from(shift_value).map_err(|_| invalid_shift_err())?;
    let shifted = num_bigint::BigInt::from(obj) << shift;
    if !fits_256_bits(&shifted) {
        return Err(EvalError::ArithmeticException(format!(
            "BigInt.shiftLeft: {obj} << {shift} is out of the 256 bit range"
        )));
    }
    Ok(Value::BigInt(shifted))
}

/// sigmastate's `UnsignedBigInt.shiftLeft` (v6.0.6): the shift amount is refused from 256
/// (`UnsignedBigIntegerOps.scala:131-137`), then the value is shifted as an unbounded integer
/// (`CUnsignedBigInt.scala:86`, `CUnsignedBigInt(wrappedValue.shiftLeft(n))`) and the
/// constructor refuses one above 256 bits (`:20-22`). SANTA
/// `UnsignedBigInt.shiftLeft_past_256_bits`
fn unsigned_bigint_shift_left(
    obj: UnsignedBigInt,
    shift_value: u32,
) -> Result<Value<'static>, EvalError> {
    // The amount as a `u8`, as in `bigint_shift_left`: the wide integer has no width of its
    // own to bound a shift
    let shift = u8::try_from(shift_value).map_err(|_| invalid_shift_err())?;
    let shifted = num_bigint::BigUint::from_bytes_be(&obj.to_be_bytes()) << shift;
    UnsignedBigInt::try_from(shifted)
        .map(Value::from)
        .map_err(|_| {
            EvalError::ArithmeticException(format!(
                "UnsignedBigInt.shiftLeft: {obj} << {shift} is out of the 256 bit range"
            ))
        })
}

static SHIFT_RIGHT_EVAL_FN: EvalFn = |_mc, _env, ctx, obj, args| {
    ctx.add_jit_cost(NUMERIC_METHOD_COST_KIND)?;
    let shift_value: u32 = args
        .first()
        .ok_or_else(|| EvalError::UnexpectedValue("shift arg missing".into()))?
        .clone()
        .try_extract_into::<i32>()?
        .try_into()
        .map_err(|_| EvalError::UnexpectedValue("expected non-negative shift value".into()))?;
    Ok(match obj {
        Value::Byte(obj) => obj
            .checked_shr(shift_value)
            .ok_or_else(invalid_shift_err)?
            .into(),
        Value::Short(obj) => obj
            .checked_shr(shift_value)
            .ok_or_else(invalid_shift_err)?
            .into(),
        Value::Int(obj) => obj
            .checked_shr(shift_value)
            .ok_or_else(invalid_shift_err)?
            .into(),
        Value::Long(obj) => obj
            .checked_shr(shift_value)
            .ok_or_else(invalid_shift_err)?
            .into(),
        Value::BigInt(obj) => bigint256(obj)?
            .checked_shr(shift_value)
            .ok_or_else(invalid_shift_err)?
            .into(),
        Value::UnsignedBigInt(obj) => obj
            .checked_shr(shift_value)
            .ok_or_else(invalid_shift_err)?
            .into(),
        other => {
            return Err(EvalError::UnexpectedValue(format!(
                "Expected numeric type, got {other:?}"
            )))
        }
    })
};

static TO_UNSIGNED_EVAL_FN: EvalFn = |_mc, _env, ctx, obj, _args| {
    ctx.add_jit_cost(NUMERIC_METHOD_COST_KIND)?;
    let signed = obj.try_extract_into::<BigInt256>()?;
    UnsignedBigInt::try_from(signed)
        .map_err(|err| EvalError::ArithmeticException(err.into()))
        .map(Value::from)
};

// UnsignedBigInt modular-method costKinds -- each a Scala FixedCost(JitCost(n))
// (SBigIntMethods.toUnsignedMod / SUnsignedBigIntMethods.*, methods.scala
// 551-609). Each eval fn charges its FixedCost per call, mirroring
// NUMERIC_METHOD_COST_KIND above.
const TO_UNSIGNED_MOD_COST: u64 = 15;
const MOD_INVERSE_COST: u64 = 150;
const PLUS_MOD_COST: u64 = 30;
const SUBTRACT_MOD_COST: u64 = 30;
const MULTIPLY_MOD_COST: u64 = 40;
const MOD_COST: u64 = 20;
const TO_SIGNED_COST: u64 = 10;

static TO_UNSIGNED_MOD_EVAL_FN: EvalFn = |_mc, _env, ctx, obj, args| {
    ctx.add_jit_cost(TO_UNSIGNED_MOD_COST)?;
    let signed = obj.try_extract_into::<BigInt256>()?;
    let modulus = args
        .first()
        .cloned()
        .ok_or_else(|| EvalError::UnexpectedValue("toUnsignedMod: missing first argument".into()))?
        .try_extract_into::<UnsignedBigInt>()?;
    UnsignedBigInt::from_signed_mod(signed, modulus)
        .map(Value::from)
        .ok_or_else(|| EvalError::ArithmeticException("toUnsignedMod: can't divide by 0".into()))
};

static MOD_INVERSE_EVAL_FN: EvalFn = |_mc, _env, ctx, obj, args| {
    ctx.add_jit_cost(MOD_INVERSE_COST)?;
    let obj = obj.try_extract_into::<UnsignedBigInt>()?;
    let modulus = args
        .first()
        .cloned()
        .ok_or_else(|| EvalError::UnexpectedValue("modInv: missing first argument".into()))?
        .try_extract_into::<UnsignedBigInt>()?;
    obj.mod_inv(modulus)
        .map(Value::from)
        .ok_or_else(|| EvalError::ArithmeticException("modInv: can't divide by 0".into()))
};

static PLUS_MOD_EVAL_FN: EvalFn = |_mc, _env, ctx, obj, args| {
    ctx.add_jit_cost(PLUS_MOD_COST)?;
    let obj = obj.try_extract_into::<UnsignedBigInt>()?;
    let b = args
        .first()
        .cloned()
        .ok_or_else(|| EvalError::UnexpectedValue("plusMod: missing first argument".into()))?
        .try_extract_into::<UnsignedBigInt>()?;
    let modulus = args
        .get(1)
        .cloned()
        .ok_or_else(|| EvalError::UnexpectedValue("plusMod: missing first argument".into()))?
        .try_extract_into::<UnsignedBigInt>()?;
    obj.checked_mod_add(b, modulus)
        .map(Value::from)
        .ok_or_else(|| EvalError::ArithmeticException("plusMod: can't divide by 0".into()))
};

static SUBTRACT_MOD_EVAL_FN: EvalFn = |_mc, _env, ctx, obj, args| {
    ctx.add_jit_cost(SUBTRACT_MOD_COST)?;
    let obj = obj.try_extract_into::<UnsignedBigInt>()?;
    let b = args
        .first()
        .cloned()
        .ok_or_else(|| EvalError::UnexpectedValue("subtractMod: missing first argument".into()))?
        .try_extract_into::<UnsignedBigInt>()?;
    let modulus = args
        .get(1)
        .cloned()
        .ok_or_else(|| EvalError::UnexpectedValue("subtractMod: missing first argument".into()))?
        .try_extract_into::<UnsignedBigInt>()?;
    obj.checked_mod_sub(b, modulus)
        .map(Value::from)
        .ok_or_else(|| EvalError::ArithmeticException("subtractMod: can't divide by 0".into()))
};

static MULTIPLY_MOD_EVAL_FN: EvalFn = |_mc, _env, ctx, obj, args| {
    ctx.add_jit_cost(MULTIPLY_MOD_COST)?;
    let obj = obj.try_extract_into::<UnsignedBigInt>()?;
    let b = args
        .first()
        .cloned()
        .ok_or_else(|| EvalError::UnexpectedValue("multiplyMod: missing first argument".into()))?
        .try_extract_into::<UnsignedBigInt>()?;
    let modulus = args
        .get(1)
        .cloned()
        .ok_or_else(|| EvalError::UnexpectedValue("multiplyMod: missing first argument".into()))?
        .try_extract_into::<UnsignedBigInt>()?;
    obj.checked_mod_mul(b, modulus)
        .map(Value::from)
        .ok_or_else(|| EvalError::ArithmeticException("multiplyMod: can't divide by 0".into()))
};

static MOD_EVAL_FN: EvalFn = |_mc, _env, ctx, obj, args| {
    ctx.add_jit_cost(MOD_COST)?;
    let obj = obj.try_extract_into::<UnsignedBigInt>()?;
    let modulus = args
        .first()
        .cloned()
        .ok_or_else(|| EvalError::UnexpectedValue("mod: missing first argument".into()))?
        .try_extract_into::<UnsignedBigInt>()?;
    obj.checked_rem(&modulus)
        .map(Value::from)
        .ok_or_else(|| EvalError::ArithmeticException("mod: can't divide by 0".into()))
};

static TO_SIGNED_EVAL_FN: EvalFn = |_mc, _env, ctx, obj, _args| {
    ctx.add_jit_cost(TO_SIGNED_COST)?;
    let obj = obj.try_extract_into::<UnsignedBigInt>()?;
    BigInt256::try_from(obj)
        .map_err(|e| EvalError::ArithmeticException(e.into()))
        .map(Value::from)
};

// List of methods that are available for all numeric types
fn snumeric_evalfn(method: &SMethod) -> Result<EvalFn, EvalError> {
    match method.method_id() {
        // The following methods are explicitly not supported
        snumeric::TO_BYTE_METHOD_ID
        | snumeric::TO_INT_METHOD_ID
        | snumeric::TO_SHORT_METHOD_ID
        | snumeric::TO_LONG_METHOD_ID
        | snumeric::TO_BIGINT_METHOD_ID => Err(EvalError::NotFound("Not implemented".into())),
        snumeric::TO_BYTES_METHOD_ID => Ok(TO_BYTES_EVAL_FN),
        snumeric::TO_BITS_METHOD_ID => Ok(TO_BITS_EVAL_FN),
        snumeric::BITWISE_INVERSE_METHOD_ID => Ok(BITWISE_INVERSE_EVAL_FN),
        snumeric::BITWISE_OR_METHOD_ID => Ok(BITWISE_OR_EVAL_FN),
        snumeric::BITWISE_AND_METHOD_ID => Ok(BITWISE_AND_EVAL_FN),
        snumeric::BITWISE_XOR_METHOD_ID => Ok(BITWISE_XOR_EVAL_FN),
        snumeric::SHIFT_LEFT_METHOD_ID => Ok(SHIFT_LEFT_EVAL_FN),
        snumeric::SHIFT_RIGHT_METHOD_ID => Ok(SHIFT_RIGHT_EVAL_FN),
        _ => Err(EvalError::NotFound(format!(
            "Method {:?} not found",
            method.method_id()
        ))),
    }
}

fn bigint_evalfn(method: &SMethod) -> Result<EvalFn, EvalError> {
    match snumeric_evalfn(method) {
        Ok(eval_fn) => Ok(eval_fn),
        Err(_) => match method.method_id() {
            TO_UNSIGNED_METHOD_ID => Ok(TO_UNSIGNED_EVAL_FN),
            TO_UNSIGNED_MOD_METHOD_ID => Ok(TO_UNSIGNED_MOD_EVAL_FN),
            _ => Err(EvalError::NotFound(format!(
                "SBigInt: Method id {:?} not found",
                method.method_id()
            ))),
        },
    }
}

fn unsigned_bigint_evalfn(method: &SMethod) -> Result<EvalFn, EvalError> {
    match snumeric_evalfn(method) {
        Ok(eval_fn) => Ok(eval_fn),
        Err(_) => match method.method_id() {
            MOD_INVERSE_METHOD_ID => Ok(MOD_INVERSE_EVAL_FN),
            PLUS_MOD_METHOD_ID => Ok(PLUS_MOD_EVAL_FN),
            SUBTRACT_MOD_METHOD_ID => Ok(SUBTRACT_MOD_EVAL_FN),
            MULTIPLY_MOD_METHOD_ID => Ok(MULTIPLY_MOD_EVAL_FN),
            MOD_METHOD_ID => Ok(MOD_EVAL_FN),
            TO_SIGNED_METHOD_ID => Ok(TO_SIGNED_EVAL_FN),
            _ => Err(EvalError::NotFound(format!(
                "SUnsignedBigInt: Method id {:?} not found",
                method.method_id()
            ))),
        },
    }
}

pub(crate) fn numeric_method_evalfn(method: &SMethod) -> Result<EvalFn, EvalError> {
    match method.obj_type.type_code() {
        snumeric::sbyte::TYPE_CODE
        | snumeric::sshort::TYPE_CODE
        | snumeric::sint::TYPE_CODE
        | snumeric::slong::TYPE_CODE => snumeric_evalfn(method),
        snumeric::sbigint::TYPE_CODE => bigint_evalfn(method),
        snumeric::sunsignedbigint::TYPE_CODE => unsigned_bigint_evalfn(method),
        _ => Err(EvalError::UnexpectedValue(format!(
            "Expected numeric type, found {:?}",
            method.obj_type.type_code()
        ))),
    }
}

#[cfg(test)]
#[cfg(feature = "arbitrary")]
#[allow(clippy::unwrap_used)]
mod test {
    use core::{iter::Sum, ops::Not};

    use ergotree_ir::{
        bigint256::BigInt256,
        mir::{
            constant::{Constant, TryExtractFrom},
            expr::Expr,
            method_call::MethodCall,
            value::Value,
        },
        types::{
            sglobal,
            smethod::{SMethod, SMethodDesc},
            snumeric::{
                self,
                sbigint::{TO_UNSIGNED_METHOD_DESC, TO_UNSIGNED_MOD_METHOD_DESC},
                sunsignedbigint::{
                    MOD_INVERSE_METHOD_DESC, MOD_METHOD_DESC, MULTIPLY_MOD_METHOD_DESC,
                    PLUS_MOD_METHOD_DESC, SUBTRACT_MOD_METHOD_DESC, TO_SIGNED_METHOD_DESC,
                },
                BITWISE_AND_METHOD_ID, BITWISE_INVERSE_METHOD_ID, BITWISE_OR_METHOD_ID,
                BITWISE_XOR_METHOD_ID, SHIFT_LEFT_METHOD_ID, SHIFT_RIGHT_METHOD_ID,
                TO_BITS_METHOD_ID, TO_BYTES_METHOD_ID,
            },
            stype::LiftIntoSType,
            stype_companion::STypeCompanion,
            stype_param::STypeVar,
        },
        unsignedbigint256::UnsignedBigInt,
    };
    use num_traits::{CheckedRem, CheckedShl, CheckedShr, One, Zero};
    use proptest::prelude::*;

    use crate::eval::{
        test_util::{
            assert_trees, eval_out_wo_ctx, eval_tree, innermost, try_eval_out_wo_ctx,
            BIGINT_MAX_HEX, BIGINT_MIN_HEX,
        },
        EvalError,
    };

    fn eval_modular_op(
        desc: &SMethodDesc,
        args: &[UnsignedBigInt],
    ) -> Result<UnsignedBigInt, EvalError> {
        let mc: Expr = MethodCall::new(
            Constant::from(args[0]).into(),
            SMethod::new(STypeCompanion::SUnsignedBigInt, desc.clone()),
            args[1..]
                .iter()
                .copied()
                .map(Constant::from)
                .map(Expr::from)
                .collect(),
        )
        .unwrap()
        .into();
        try_eval_out_wo_ctx(&mc)
    }
    #[test]
    fn eval_subtract_mod_zero_modulus_arithmetic_exception() {
        assert!(matches!(
            eval_modular_op(
                &SUBTRACT_MOD_METHOD_DESC,
                &[5u32.into(), 3u32.into(), 0u32.into()]
            ),
            Err(EvalError::Spanned(err)) if matches!(*err.error, EvalError::ArithmeticException(_))
        ));
    }

    #[test]
    fn eval_mod_inverse_zero_modulus_arithmetic_exception() {
        assert!(matches!(
            eval_modular_op(&MOD_INVERSE_METHOD_DESC, &[3u32.into(), 0u32.into()]),
            Err(EvalError::Spanned(err)) if matches!(*err.error, EvalError::ArithmeticException(_))
        ));
    }

    trait Numeric:
        LiftIntoSType
        + TryExtractFrom<Value<'static>>
        + PartialEq
        + Into<Constant>
        + core::fmt::Debug
        + Clone
        + 'static
        + One
        + Zero
        + CheckedShl
        + CheckedShr
        + Not<Output = Self>
        + Sum
    {
    }
    impl Numeric for i8 {}
    impl Numeric for i16 {}
    impl Numeric for i32 {}
    impl Numeric for i64 {}
    impl Numeric for BigInt256 {}
    impl Numeric for UnsignedBigInt {}
    fn big_endian_roundtrip<T: Numeric>(v: T, methods: &[SMethod]) {
        let type_args = std::iter::once((STypeVar::t(), T::stype())).collect();
        let to_be_bytes_expr: Expr = MethodCall::new(
            <T as Into<Constant>>::into(v.clone()).into(),
            methods
                .iter()
                .find(|method| method.method_id() == TO_BYTES_METHOD_ID)
                .unwrap()
                .clone(),
            vec![],
        )
        .unwrap()
        .into();
        let expr: Expr = MethodCall::with_type_args(
            Expr::Global,
            sglobal::FROM_BIGENDIAN_BYTES_METHOD
                .clone()
                .with_concrete_types(&type_args),
            vec![to_be_bytes_expr],
            type_args,
        )
        .unwrap()
        .into();
        assert_eq!(eval_out_wo_ctx::<T>(&expr), v);
    }
    fn bits_roundtrip<T: Numeric>(v: T, methods: &[SMethod]) {
        let to_bits: Expr = MethodCall::new(
            <T as Into<Constant>>::into(v.clone()).into(),
            methods
                .iter()
                .find(|method| method.method_id() == TO_BITS_METHOD_ID)
                .unwrap()
                .clone(),
            vec![],
        )
        .unwrap()
        .into();
        let res: Vec<bool> = eval_out_wo_ctx(&to_bits);
        // perform binary multiplication to make sure bits equals long
        let res = res
            .into_iter()
            .rev()
            .map(|bit| if bit { T::one() } else { T::zero() })
            .enumerate()
            .map(|(i, bit)| bit << i as u32)
            .sum::<T>();

        assert_eq!(res, v);
    }

    fn bitwise_inverse_test<T: Numeric>(v: T, methods: &[SMethod]) {
        let inverse_mc: Expr = MethodCall::new(
            <T as Into<Constant>>::into(v.clone()).into(),
            methods
                .iter()
                .find(|method| method.method_id() == BITWISE_INVERSE_METHOD_ID)
                .unwrap()
                .clone(),
            vec![],
        )
        .unwrap()
        .into();
        let res: T = eval_out_wo_ctx(&inverse_mc);
        assert_eq!(res, !v);
    }

    /// A zero-argument `MethodCall` for a binary numeric method. `MethodCall`'s fields are all
    /// public, so a struct literal bypasses the arity check `MethodCall::new` applies — which is
    /// what the parser does once it stops checking, as sigmastate's does. sigmastate parses such
    /// a call below ErgoTree v3 and writes it back as a `PropertyCall`.
    fn no_arg_call(method_id: ergotree_ir::types::smethod::MethodId) -> Expr {
        let method = snumeric::sbyte::METHODS
            .iter()
            .find(|m| m.method_id() == method_id)
            .unwrap()
            .clone();
        MethodCall {
            obj: Box::new(Expr::Const(Constant::from(1i8))),
            method,
            args: alloc::vec::Vec::new(),
            explicit_type_args: Default::default(),
        }
        .into()
    }

    #[test]
    fn bitwise_or_with_no_argument_errors() {
        let mc = no_arg_call(BITWISE_OR_METHOD_ID);
        assert!(try_eval_out_wo_ctx::<i8>(&mc).is_err());
    }

    #[test]
    fn bitwise_and_with_no_argument_errors() {
        let mc = no_arg_call(BITWISE_AND_METHOD_ID);
        assert!(try_eval_out_wo_ctx::<i8>(&mc).is_err());
    }

    fn bitwise_or<T: Numeric>(v: T, rhs: T, methods: &[SMethod]) -> T {
        let or_mc: Expr = MethodCall::new(
            <T as Into<Constant>>::into(v.clone()).into(),
            methods
                .iter()
                .find(|method| method.method_id() == BITWISE_OR_METHOD_ID)
                .unwrap()
                .clone(),
            vec![<T as Into<Constant>>::into(rhs).into()],
        )
        .unwrap()
        .into();
        eval_out_wo_ctx(&or_mc)
    }

    fn bitwise_and<T: Numeric>(v: T, rhs: T, methods: &[SMethod]) -> T {
        let and_mc: Expr = MethodCall::new(
            <T as Into<Constant>>::into(v.clone()).into(),
            methods
                .iter()
                .find(|method| method.method_id() == BITWISE_AND_METHOD_ID)
                .unwrap()
                .clone(),
            vec![<T as Into<Constant>>::into(rhs).into()],
        )
        .unwrap()
        .into();
        eval_out_wo_ctx(&and_mc)
    }
    fn bitwise_xor<T: Numeric>(v: T, rhs: T, methods: &[SMethod]) -> T {
        let xor_mc: Expr = MethodCall::new(
            <T as Into<Constant>>::into(v.clone()).into(),
            methods
                .iter()
                .find(|method| method.method_id() == BITWISE_XOR_METHOD_ID)
                .unwrap()
                .clone(),
            vec![<T as Into<Constant>>::into(rhs).into()],
        )
        .unwrap()
        .into();
        eval_out_wo_ctx(&xor_mc)
    }
    fn try_shl<T: Numeric>(v: T, rhs: u32, methods: &[SMethod]) -> Result<T, EvalError> {
        let shl_mc: Expr = MethodCall::new(
            <T as Into<Constant>>::into(v.clone()).into(),
            methods
                .iter()
                .find(|method| method.method_id() == SHIFT_LEFT_METHOD_ID)
                .unwrap()
                .clone(),
            vec![Constant::from(rhs as i32).into()],
        )
        .unwrap()
        .into();
        try_eval_out_wo_ctx(&shl_mc)
    }
    fn shl<T: Numeric>(v: T, rhs: u32, methods: &[SMethod]) -> T {
        try_shl(v, rhs, methods).unwrap()
    }
    fn shr<T: Numeric>(v: T, rhs: u32, methods: &[SMethod]) -> T {
        let shr_mc: Expr = MethodCall::new(
            <T as Into<Constant>>::into(v.clone()).into(),
            methods
                .iter()
                .find(|method| method.method_id() == SHIFT_RIGHT_METHOD_ID)
                .unwrap()
                .clone(),
            vec![Constant::from(rhs as i32).into()],
        )
        .unwrap()
        .into();
        eval_out_wo_ctx(&shr_mc)
    }
    fn two_pow(n: u32) -> num_bigint::BigInt {
        num_bigint::BigInt::from(1) << n
    }

    /// A `BigInt` value at a cost, as a row of `assert_trees`
    fn big(n: num_bigint::BigInt, cost: u64) -> Option<(Value<'static>, u64)> {
        Some((Value::BigInt(n), cost))
    }

    /// The error a tree fails with, under the span wrapper each eval node adds
    fn refusal(tree_hex: &str) -> EvalError {
        innermost(&eval_tree(tree_hex).0.unwrap_err()).clone()
    }

    /// SANTA `BigInt.shift_and_inverse_range_edges` (`3870df6`, sigma-state 6.0.7, tree v3)
    #[test]
    fn santa_shift_and_inverse_range_edges() {
        // H1 `BigInt(1).shiftLeft(255)`, H2 `BigInt(2).shiftLeft(255)`, H3 `BigInt.MAX.shiftLeft(1)`:
        // the JVM throws `ArithmeticException: BigInteger out of 256 bit range`
        for tree in [
            "0b0adc060c0601010104fe03".to_string(),
            "0b0adc060c0601020104fe03".to_string(),
            format!("0b28dc060c{BIGINT_MAX_HEX}010402"),
        ] {
            assert!(
                matches!(refusal(&tree), EvalError::ArithmeticException(_)),
                "{tree}"
            );
        }
        let min_shr = format!("0b29dc060d{BIGINT_MIN_HEX}0104fe03");
        let max_shr = format!("0b29dc060d{BIGINT_MAX_HEX}0104fe03");
        let min_inv = format!("0b25db0608{BIGINT_MIN_HEX}");
        let max_inv = format!("0b25db0608{BIGINT_MAX_HEX}");
        assert_trees(&[
            // H4 `BigInt(1).shiftLeft(254)`
            ("0b0adc060c0601010104fc03", big(two_pow(254), 19)),
            // H5 `BigInt(-1).shiftLeft(255)`: `BigInt.MIN` fits
            ("0b0adc060c0601ff0104fe03", big(-two_pow(255), 19)),
            // R1 `BigInt.MIN.shiftRight(255)`, R2 `BigInt.MAX.shiftRight(255)`
            (min_shr.as_str(), big((-1).into(), 19)),
            (max_shr.as_str(), big(0.into(), 19)),
            // I1 `BigInt.MIN.bitwiseInverse`, I2 `BigInt.MAX.bitwiseInverse`
            (min_inv.as_str(), big(two_pow(255) - 1, 14)),
            (max_inv.as_str(), big(-two_pow(255), 14)),
        ]);
    }

    /// By reading (`CBigInt.scala:67`, `BigIntegerOps.scala:109-115`), not measured on the JVM;
    /// the Rust side was measured with blitzen
    #[test]
    fn shift_left_by_reading_rows() {
        // a negative receiver shifted out of the range
        for tree in [
            // `BigInt.MIN.shiftLeft(1)`
            format!("0b28dc060c{BIGINT_MIN_HEX}010402"),
            // `BigInt(-2).shiftLeft(255)`
            "0b0adc060c0601fe0104fe03".to_string(),
        ] {
            assert!(
                matches!(refusal(&tree), EvalError::ArithmeticException(_)),
                "{tree}"
            );
        }
        // the shift amount is refused first, with its own error, on a receiver whose shift
        // would leave the range too: `BigInt.MAX` by 256, by `Int.MaxValue`, by -1
        assert!(matches!(
            refusal(&format!("0b29dc060c{BIGINT_MAX_HEX}01048004")),
            EvalError::Misc(_)
        ));
        assert!(matches!(
            refusal(&format!("0b2cdc060c{BIGINT_MAX_HEX}0104feffffff0f")),
            EvalError::Misc(_)
        ));
        assert!(matches!(
            refusal(&format!("0b28dc060c{BIGINT_MAX_HEX}010401")),
            EvalError::UnexpectedValue(_)
        ));
        // values a shift keeps: both edges by 0, zero by 255
        let min_by_0 = format!("0b28dc060c{BIGINT_MIN_HEX}010400");
        let max_by_0 = format!("0b28dc060c{BIGINT_MAX_HEX}010400");
        assert_trees(&[
            (min_by_0.as_str(), big(-two_pow(255), 19)),
            (max_by_0.as_str(), big(two_pow(255) - 1, 19)),
            ("0b0adc060c0601000104fe03", big(0.into(), 19)),
        ]);
    }

    /// SANTA's `UnsignedBigInt_methods` row 16 tree,
    /// `{ (x: (UnsignedBigInt, Int)) => x._1.shiftLeft(x._2) }`, on the input `(v, n)` bound as
    /// context variable 1: the value and the JIT cost at activated version 3
    fn unsigned_shl_row(v: UnsignedBigInt, n: i32) -> (Result<Value<'static>, EvalError>, u64) {
        use crate::eval::{Env, Evaluable};
        use ergotree_ir::chain::context::Context;
        use ergotree_ir::chain::context_extension::ContextExtension;
        use ergotree_ir::ergo_tree::{ErgoTree, ErgoTreeVersion};
        use sigma_test_util::force_any_val;
        let bytes =
            base16::decode("1b1900dad901014504dc090c8c720101018c72010201e4e3014504").unwrap();
        let expr = ErgoTree::sigma_parse_bytes_lenient(&bytes)
            .unwrap()
            .proposition()
            .unwrap();
        let input: Constant = (v, n).into();
        let mut ext = ContextExtension::empty();
        ext.values.insert(1u8, input.into());
        let mut ctx = Context {
            extension: Box::leak(Box::new(ext)),
            ..force_any_val::<Context>()
        };
        ctx.activated_script_version_byte = 3;
        ctx.tree_version.set(ErgoTreeVersion::V3);
        ctx.jit_cost.set(0);
        let res = expr.eval(&mut Env::empty(), &ctx).map(|v| v.to_static());
        (res, ctx.jit_cost_value())
    }

    fn unsigned(n: num_bigint::BigUint) -> UnsignedBigInt {
        UnsignedBigInt::try_from(n).unwrap()
    }

    fn two_pow_u(n: u32) -> num_bigint::BigUint {
        num_bigint::BigUint::from(1u8) << n
    }

    /// SANTA `UnsignedBigInt.shiftLeft_past_256_bits` (`99a0c75`, sigma-state 6.0.7, tree v3)
    #[test]
    fn santa_unsigned_shift_left_past_256_bits() {
        let max = two_pow_u(256) - 1u8;
        // U1 `(2^255, 1)`, U2 `(2^256 - 1, 1)`, U3 `(3, 255)`: the JVM throws
        // `ArithmeticException: Too big unsigned big int value`
        for (v, n) in [(two_pow_u(255), 1), (max, 1), (3u8.into(), 255)] {
            let (res, _) = unsigned_shl_row(unsigned(v.clone()), n);
            assert!(
                matches!(
                    res.as_ref().map_err(innermost),
                    Err(EvalError::ArithmeticException(_))
                ),
                "{v} << {n}: {res:?}"
            );
        }
        // U4 `(1, 255)`, U5 `(2^254, 1)`, U6 `(0, 255)`
        for (v, n, expected) in [
            (1u8.into(), 255, two_pow_u(255)),
            (two_pow_u(254), 1, two_pow_u(255)),
            (0u8.into(), 255, 0u8.into()),
        ] {
            let (res, cost) = unsigned_shl_row(unsigned(v), n);
            assert_eq!(res.unwrap(), Value::from(unsigned(expected)));
            assert_eq!(cost, 104);
        }
        // L1 `(1, 256)`, L2 `(1, -1)`: the amount's bound
        let (res, _) = unsigned_shl_row(unsigned(1u8.into()), 256);
        assert!(
            matches!(res.as_ref().map_err(innermost), Err(EvalError::Misc(_))),
            "{res:?}"
        );
        let (res, _) = unsigned_shl_row(unsigned(1u8.into()), -1);
        assert!(
            matches!(
                res.as_ref().map_err(innermost),
                Err(EvalError::UnexpectedValue(_))
            ),
            "{res:?}"
        );
    }

    /// By reading (v6.0.6 `CUnsignedBigInt.scala:20-22`, `:86`,
    /// `UnsignedBigIntegerOps.scala:131-137`), not measured on the JVM; the Rust side was
    /// measured with blitzen. SANTA's tree, other inputs
    #[test]
    fn unsigned_shift_left_by_reading_rows() {
        let max = two_pow_u(256) - 1u8;
        // far past the range: `(2^256 - 1, 255)`
        let (res, _) = unsigned_shl_row(unsigned(max.clone()), 255);
        assert!(
            matches!(
                res.as_ref().map_err(innermost),
                Err(EvalError::ArithmeticException(_))
            ),
            "{res:?}"
        );
        // the shift amount is refused first, with its own error, on a receiver whose shift
        // would leave the range too: by 256, by `Int.MaxValue`, by -1
        for n in [256, i32::MAX] {
            let (res, _) = unsigned_shl_row(unsigned(max.clone()), n);
            assert!(
                matches!(res.as_ref().map_err(innermost), Err(EvalError::Misc(_))),
                "{n}: {res:?}"
            );
        }
        let (res, _) = unsigned_shl_row(unsigned(max.clone()), -1);
        assert!(
            matches!(
                res.as_ref().map_err(innermost),
                Err(EvalError::UnexpectedValue(_))
            ),
            "{res:?}"
        );
        // the top bit set and kept: the range ends at 256 bits, not at 255 as `BigInt`'s
        for (v, n, expected) in [
            (max.clone(), 0, max.clone()),
            (two_pow_u(255), 0, two_pow_u(255)),
            (two_pow_u(255) - 1u8, 1, two_pow_u(256) - 2u8),
        ] {
            let (res, cost) = unsigned_shl_row(unsigned(v), n);
            assert_eq!(res.unwrap(), Value::from(unsigned(expected)));
            assert_eq!(cost, 104);
        }
    }

    #[test]
    fn bitwise_or_byte() {
        assert_eq!(bitwise_or(127i8, -128i8, &snumeric::sbyte::METHODS), -1i8);
    }

    /// Regression: a numeric 6.0 method call must charge its
    /// `FixedCost(JitCost(5))` costKind (sigma-state `ToBytes_CostKind` etc.).
    /// Isolated by subtracting the receiver-eval cost: a `MethodCall` evaluates its
    /// object exactly once, so `mc_cost - obj_cost == MethodCall Fixed(4) + 5`.
    #[test]
    fn numeric_method_charges_costkind() {
        use crate::eval::test_util::eval_out;
        use ergotree_ir::chain::context::Context;
        use sigma_test_util::force_any_val;

        let obj_expr: Expr = Constant::from(5i64).into();
        let mc: Expr = MethodCall::new(
            obj_expr.clone(),
            snumeric::slong::METHODS
                .iter()
                .find(|m| m.method_id() == TO_BYTES_METHOD_ID)
                .unwrap()
                .clone(),
            vec![],
        )
        .unwrap()
        .into();

        let ctx = force_any_val::<Context>();
        let c0 = ctx.jit_cost_value();
        let _ = eval_out::<i64>(&obj_expr, &ctx);
        let obj_cost = ctx.jit_cost_value() - c0;
        let _ = eval_out::<Vec<u8>>(&mc, &ctx);
        let mc_cost = ctx.jit_cost_value() - c0 - obj_cost;
        // MethodCall Fixed(4) + ToBytes_CostKind FixedCost(JitCost(5)) = 9.
        assert_eq!(mc_cost - obj_cost, 9);
    }

    #[test]
    fn unsigned_bigint_modular_methods_charge_scala_costkinds() {
        use crate::eval::test_util::try_eval_out;
        use ergotree_ir::chain::context::Context;
        use sigma_test_util::force_any_val;

        // Each UnsignedBigInt modular method is a Scala FixedCost (methods.scala
        // 551-609): toUnsignedMod 15, modInverse 150, plusMod 30, subtractMod 30,
        // multiplyMod 40, mod 20, toSigned 10. Isolate it like
        // `numeric_method_charges_costkind`: the MethodCall node charges Fixed(4)
        // and evaluates the receiver + each arg separately, so
        // `full - receiver - args` leaves exactly `4 + costKind`. A receiver/arg
        // Constant costs the same standalone as inside the call, so it cancels
        // regardless of value (and of the cost-only extraction type below).
        // Pre-fix every modular eval fn took `_ctx` and charged nothing.
        let charge_of = |receiver: Expr, method: SMethod, args: Vec<Expr>| -> u64 {
            let mc: Expr = MethodCall::new(receiver.clone(), method, args.clone())
                .unwrap()
                .into();
            let ctx = force_any_val::<Context>();
            let base = ctx.jit_cost_value();
            let _ = try_eval_out::<UnsignedBigInt>(&receiver, &ctx);
            for a in &args {
                let _ = try_eval_out::<UnsignedBigInt>(a, &ctx);
            }
            let receiver_and_args = ctx.jit_cost_value() - base;
            let before_mc = ctx.jit_cost_value();
            let _ = try_eval_out::<UnsignedBigInt>(&mc, &ctx);
            (ctx.jit_cost_value() - before_mc) - receiver_and_args
        };

        let ubi = |n: u64| -> Expr { Constant::from(UnsignedBigInt::from(n)).into() };
        let on_ubi = |desc| SMethod::new(STypeCompanion::SUnsignedBigInt, desc);

        assert_eq!(
            charge_of(
                Constant::from(BigInt256::from(7i64)).into(),
                SMethod::new(STypeCompanion::SBigInt, TO_UNSIGNED_MOD_METHOD_DESC.clone()),
                vec![ubi(5)],
            ),
            4 + 15,
            "toUnsignedMod: MethodCall Fixed(4) + ToUnsignedMod FixedCost(15)",
        );
        assert_eq!(
            charge_of(
                ubi(7),
                on_ubi(MOD_INVERSE_METHOD_DESC.clone()),
                vec![ubi(5)]
            ),
            4 + 150,
            "modInverse: Fixed(4) + ModInverse FixedCost(150)",
        );
        assert_eq!(
            charge_of(
                ubi(7),
                on_ubi(PLUS_MOD_METHOD_DESC.clone()),
                vec![ubi(3), ubi(5)]
            ),
            4 + 30,
            "plusMod: Fixed(4) + PlusMod FixedCost(30)",
        );
        assert_eq!(
            charge_of(
                ubi(7),
                on_ubi(SUBTRACT_MOD_METHOD_DESC.clone()),
                vec![ubi(3), ubi(5)]
            ),
            4 + 30,
            "subtractMod: Fixed(4) + SubtractMod FixedCost(30)",
        );
        assert_eq!(
            charge_of(
                ubi(7),
                on_ubi(MULTIPLY_MOD_METHOD_DESC.clone()),
                vec![ubi(3), ubi(5)]
            ),
            4 + 40,
            "multiplyMod: Fixed(4) + MultiplyMod FixedCost(40)",
        );
        assert_eq!(
            charge_of(ubi(7), on_ubi(MOD_METHOD_DESC.clone()), vec![ubi(5)]),
            4 + 20,
            "mod: Fixed(4) + Mod FixedCost(20)",
        );
        assert_eq!(
            charge_of(ubi(7), on_ubi(TO_SIGNED_METHOD_DESC.clone()), vec![]),
            4 + 10,
            "toSigned: Fixed(4) + ToSigned FixedCost(10)",
        );
    }

    fn eval_to_bytes<T: Numeric>(v: T, methods: &[SMethod]) -> Vec<i8> {
        let mc: Expr = MethodCall::new(
            <T as Into<Constant>>::into(v).into(),
            methods
                .iter()
                .find(|method| method.method_id() == TO_BYTES_METHOD_ID)
                .unwrap()
                .clone(),
            vec![],
        )
        .unwrap()
        .into();
        eval_out_wo_ctx::<Vec<i8>>(&mc)
    }
    fn eval_to_bits<T: Numeric>(v: T, methods: &[SMethod]) -> Vec<bool> {
        let mc: Expr = MethodCall::new(
            <T as Into<Constant>>::into(v).into(),
            methods
                .iter()
                .find(|method| method.method_id() == TO_BITS_METHOD_ID)
                .unwrap()
                .clone(),
            vec![],
        )
        .unwrap()
        .into();
        eval_out_wo_ctx::<Vec<bool>>(&mc)
    }

    // Byte/bit-exact `toBytes`/`toBits` against the blessed v6 vectors
    // (BigInt_6.0_features / UnsignedBigInt_methods). The JVM emits a
    // minimal-length big-endian encoding: signed two's-complement for BigInt
    // (`BigInteger.toByteArray`, `0 -> [0]`) and unsigned for UnsignedBigInt
    // (`asUnsignedByteArray`, `0 -> []`).
    #[test]
    fn bigint_to_bytes_minimal_length() {
        let cases: &[(i64, &[i8])] = &[
            (0, &[0]),
            (127, &[127]),        // vector #17
            (32767, &[127, -1]),  // vector #18
            (-32768, &[-128, 0]), // vector #19
            (255, &[0, -1]),      // sign-pad: high bit set -> leading 0x00
            (-1, &[-1]),
            (-55, &[-55]), // 0xc9
        ];
        for (input, expected) in cases {
            assert_eq!(
                eval_to_bytes(BigInt256::from(*input), &snumeric::sbigint::METHODS),
                expected.to_vec(),
                "BigInt({}).toBytes",
                input,
            );
        }
    }
    #[test]
    fn bigint_to_bits_minimal_length() {
        // vector #21 CBigInt(83) and #22 CBigInt(-55).
        assert_eq!(
            eval_to_bits(BigInt256::from(83i64), &snumeric::sbigint::METHODS),
            vec![false, true, false, true, false, false, true, true], // 0x53
        );
        assert_eq!(
            eval_to_bits(BigInt256::from(-55i64), &snumeric::sbigint::METHODS),
            vec![true, true, false, false, true, false, false, true], // 0xc9
        );
        assert_eq!(
            eval_to_bits(BigInt256::from(0i64), &snumeric::sbigint::METHODS),
            vec![false; 8], // 0x00
        );
    }
    #[test]
    fn unsigned_bigint_to_bytes_minimal_length() {
        let cases: &[(u64, &[i8])] = &[
            (0, &[]),            // empty: asUnsignedByteArray drops the zero
            (127, &[127]),       // vector #12
            (255, &[-1]),        // unsigned: no sign byte (contrast BigInt 255)
            (32767, &[127, -1]), // vector #13
        ];
        for (input, expected) in cases {
            assert_eq!(
                eval_to_bytes(
                    UnsignedBigInt::from(*input),
                    &snumeric::sunsignedbigint::METHODS,
                ),
                expected.to_vec(),
                "UnsignedBigInt({}).toBytes",
                input,
            );
        }
    }
    #[test]
    fn unsigned_bigint_to_bits_minimal_length() {
        // vector #15 CUnsignedBigInt(83); zero -> empty.
        assert_eq!(
            eval_to_bits(
                UnsignedBigInt::from(83u64),
                &snumeric::sunsignedbigint::METHODS,
            ),
            vec![false, true, false, true, false, false, true, true],
        );
        assert_eq!(
            eval_to_bits(
                UnsignedBigInt::from(0u64),
                &snumeric::sunsignedbigint::METHODS,
            ),
            Vec::<bool>::new(),
        );
    }
    proptest! {
        #[test]
        fn byte_big_endian_roundtrip(byte in any::<i8>()) {
            big_endian_roundtrip(byte, &snumeric::sbyte::METHODS);
        }
        #[test]
        fn short_big_endian_roundtrip(short in any::<i16>()) {
            big_endian_roundtrip(short, &snumeric::sshort::METHODS);
        }
        #[test]
        fn int_big_endian_roundtrip(int in any::<i32>()) {
            big_endian_roundtrip(int, &snumeric::sint::METHODS);
        }
        #[test]
        fn long_big_endian_roundtrip(long in any::<i64>()) {
            big_endian_roundtrip(long, &snumeric::slong::METHODS);
        }
        #[test]
        fn bigint_big_endian_roundtrip(bigint in any::<BigInt256>()) {
            big_endian_roundtrip(bigint, &snumeric::sbigint::METHODS);
        }
        #[test]
        fn unsigned_bigint_big_endian_roundtrip(bigint in any::<UnsignedBigInt>()) {
            big_endian_roundtrip(bigint, &snumeric::sunsignedbigint::METHODS);
        }
        #[test]
        fn to_bits_byte(byte in any::<i8>()) {
            bits_roundtrip(byte, &snumeric::sbyte::METHODS);
        }
        #[test]
        fn to_bits_short(short in any::<i16>()) {
            bits_roundtrip(short, &snumeric::sshort::METHODS);
        }
        #[test]
        fn to_bits_int(int in any::<i32>()) {
            bits_roundtrip(int, &snumeric::sint::METHODS);
        }
        #[test]
        fn to_bits_long(long in any::<i64>()) {
            bits_roundtrip(long, &snumeric::slong::METHODS);
        }
        #[test]
        fn to_bits_bigint(
            // Non-negative only: bits_roundtrip reconstructs via an unsigned
            // positional sum, which matches signed minimal two's-complement bits
            // only for non-negative values. Negatives are covered byte-exactly by
            // `bigint_to_bits_minimal_length`.
            bigint in any::<[u8; 32]>().prop_map(|mut b| {
                b[0] &= 0x7f;
                BigInt256::from_be_slice(&b).unwrap()
            })
        ) {
            bits_roundtrip(bigint, &snumeric::sbigint::METHODS);
        }
        #[test]
        fn to_bits_unsigned_bigint(bigint in any::<UnsignedBigInt>()) {
            bits_roundtrip(bigint, &snumeric::sunsignedbigint::METHODS);
        }
        #[test]
        fn inverse_byte(b in any::<i8>()) {
            bitwise_inverse_test(b, &snumeric::sbyte::METHODS);
        }
        #[test]
        fn inverse_short(short in any::<i16>()) {
            bitwise_inverse_test(short, &snumeric::sshort::METHODS);
        }
        #[test]
        fn inverse_int(int in any::<i32>()) {
            bitwise_inverse_test(int, &snumeric::sint::METHODS);
        }
        #[test]
        fn inverse_long(long in any::<i64>()) {
            bitwise_inverse_test(long, &snumeric::slong::METHODS);
        }
        #[test]
        fn inverse_bigint(bigint in any::<BigInt256>()) {
            bitwise_inverse_test(bigint, &snumeric::sbigint::METHODS);
        }
        #[test]
        fn inverse_unsigned_bigint(bigint in any::<UnsignedBigInt>()) {
            bitwise_inverse_test(bigint, &snumeric::sunsignedbigint::METHODS);
        }

        #[test]
        fn bitwise_or_byte_arbitrary(a in any::<i8>(), b in any::<i8>()) {
            assert_eq!(bitwise_or(a, b, &snumeric::sbyte::METHODS), a | b);
        }
        #[test]
        fn bitwise_or_short_arbitrary(a in any::<i16>(), b in any::<i16>()) {
            assert_eq!(bitwise_or(a, b, &snumeric::sshort::METHODS), a | b);
        }
        #[test]
        fn bitwise_or_int_arbitrary(a in any::<i32>(), b in any::<i32>()) {
            assert_eq!(bitwise_or(a, b, &snumeric::sint::METHODS), a | b);
        }
        #[test]
        fn bitwise_or_long_arbitrary(a in any::<i64>(), b in any::<i64>()) {
            assert_eq!(bitwise_or(a, b, &snumeric::slong::METHODS), a | b);
        }
        #[test]
        fn bitwise_or_bigint_arbitrary(a in any::<BigInt256>(), b in any::<BigInt256>()) {
            assert_eq!(bitwise_or(a, b, &snumeric::sbigint::METHODS), a | b);
        }
        #[test]
        fn bitwise_or_unsigned_bigint_arbitrary(a in any::<UnsignedBigInt>(), b in any::<UnsignedBigInt>()) {
            assert_eq!(bitwise_or(a, b, &snumeric::sunsignedbigint::METHODS), a | b);
        }
        #[test]
        fn bitwise_and_byte_arbitrary(a in any::<i8>(), b in any::<i8>()) {
            assert_eq!(bitwise_and(a, b, &snumeric::sbyte::METHODS), a & b);
        }
        #[test]
        fn bitwise_and_short_arbitrary(a in any::<i16>(), b in any::<i16>()) {
            assert_eq!(bitwise_and(a, b, &snumeric::sshort::METHODS), a & b);
        }
        #[test]
        fn bitwise_and_int_arbitrary(a in any::<i32>(), b in any::<i32>()) {
            assert_eq!(bitwise_and(a, b, &snumeric::sint::METHODS), a & b);
        }
        #[test]
        fn bitwise_and_long_arbitrary(a in any::<i64>(), b in any::<i64>()) {
            assert_eq!(bitwise_and(a, b, &snumeric::slong::METHODS), a & b);
        }
        #[test]
        fn bitwise_and_bigint_arbitrary(a in any::<BigInt256>(), b in any::<BigInt256>()) {
            assert_eq!(bitwise_and(a, b, &snumeric::sbigint::METHODS), a & b);
        }
        #[test]
        fn bitwise_and_unsigned_bigint_arbitrary(a in any::<UnsignedBigInt>(), b in any::<UnsignedBigInt>()) {
            assert_eq!(bitwise_and(a, b, &snumeric::sunsignedbigint::METHODS), a & b);
        }
        #[test]
        fn bitwise_xor_byte_arbitrary(a in any::<i8>(), b in any::<i8>()) {
            assert_eq!(bitwise_xor(a, b, &snumeric::sbyte::METHODS), a ^ b);
        }
        #[test]
        fn bitwise_xor_short_arbitrary(a in any::<i16>(), b in any::<i16>()) {
            assert_eq!(bitwise_xor(a, b, &snumeric::sshort::METHODS), a ^ b);
        }
        #[test]
        fn bitwise_xor_int_arbitrary(a in any::<i32>(), b in any::<i32>()) {
            assert_eq!(bitwise_xor(a, b, &snumeric::sint::METHODS), a ^ b);
        }
        #[test]
        fn bitwise_xor_long_arbitrary(a in any::<i64>(), b in any::<i64>()) {
            assert_eq!(bitwise_xor(a, b, &snumeric::slong::METHODS), a ^ b);
        }
        #[test]
        fn bitwise_xor_bigint_arbitrary(a in any::<BigInt256>(), b in any::<BigInt256>()) {
            assert_eq!(bitwise_xor(a, b, &snumeric::sbigint::METHODS), a ^ b);
        }
        #[test]
        fn bitwise_xor_unsigned_bigint_arbitrary(a in any::<UnsignedBigInt>(), b in any::<UnsignedBigInt>()) {
            assert_eq!(bitwise_xor(a, b, &snumeric::sunsignedbigint::METHODS), a ^ b);
        }
        #[test]
        fn shl_byte_arbitrary(a in any::<i8>(), shift in 0u32..8) {
            assert_eq!(shl(a, shift, &snumeric::sbyte::METHODS), a << shift);
        }
        #[test]
        fn shl_short_arbitrary(a in any::<i16>(), shift in 0u32..16) {
            assert_eq!(shl(a, shift, &snumeric::sshort::METHODS), a << shift);
        }
        #[test]
        fn shl_int_arbitrary(a in any::<i32>(), shift in 0u32..32) {
            assert_eq!(shl(a, shift, &snumeric::sint::METHODS), a << shift);
        }
        #[test]
        fn shl_long_arbitrary(a in any::<i64>(), shift in 0u32..64) {
            assert_eq!(shl(a, shift, &snumeric::slong::METHODS), a << shift);
        }
        #[test]
        fn shl_bigint_arbitrary(a in any::<BigInt256>(), shift in 0u32..256) {
            // the JVM's rule: the exact product where it fits 256 bits, a throw otherwise. The
            // throw is matched by its variant: a wide value handed back would fail the test's
            // extraction as `BigInt256`, an error of another kind
            let res = try_shl(a, shift, &snumeric::sbigint::METHODS);
            match BigInt256::try_from(num_bigint::BigInt::from(a) << shift) {
                Ok(exact) => prop_assert_eq!(res.ok(), Some(exact)),
                Err(_) => prop_assert!(
                    matches!(
                        res.as_ref().map_err(innermost),
                        Err(EvalError::ArithmeticException(_))
                    ),
                    "{:?}",
                    res
                ),
            }
        }
        #[test]
        fn shl_unsigned_bigint_arbitrary(a in any::<UnsignedBigInt>(), shift in 0u32..256) {
            // the JVM's rule: the exact product where it fits 256 bits, a throw otherwise,
            // matched by its variant
            let res = try_shl(a, shift, &snumeric::sunsignedbigint::METHODS);
            let exact = num_bigint::BigUint::from_bytes_be(&a.to_be_bytes()) << shift;
            match UnsignedBigInt::try_from(exact) {
                Ok(exact) => prop_assert_eq!(res.ok(), Some(exact)),
                Err(_) => prop_assert!(
                    matches!(
                        res.as_ref().map_err(innermost),
                        Err(EvalError::ArithmeticException(_))
                    ),
                    "{:?}",
                    res
                ),
            }
        }
        #[test]
        #[should_panic]
        fn shl_byte_arbitrary_invalid(a in any::<i8>(), shift in 8u32..) {
            assert_eq!(shl(a, shift, &snumeric::sbyte::METHODS), a << shift);
        }
        #[test]
        #[should_panic]
        fn shl_short_arbitrary_invalid(a in any::<i16>(), shift in 16u32..) {
            assert_eq!(shl(a, shift, &snumeric::sshort::METHODS), a << shift);
        }
        #[test]
        #[should_panic]
        fn shl_int_arbitrary_invalid(a in any::<i32>(), shift in 32u32..) {
            assert_eq!(shl(a, shift, &snumeric::sint::METHODS), a << shift);
        }
        #[test]
        #[should_panic]
        fn shl_long_arbitrary_invalid(a in any::<i64>(), shift in 64u32..) {
            assert_eq!(shl(a, shift, &snumeric::slong::METHODS), a << shift);
        }
        #[test]
        #[should_panic]
        fn shl_bigint_arbitrary_invalid(a in any::<BigInt256>(), shift in 256u32..) {
            assert_eq!(shl(a, shift, &snumeric::sbigint::METHODS), a << shift);
        }
        #[test]
        #[should_panic]
        fn shl_unsigned_bigint_arbitrary_invalid(a in any::<UnsignedBigInt>(), shift in 256u32..) {
            assert_eq!(shl(a, shift, &snumeric::sunsignedbigint::METHODS), a << shift);
        }
        #[test]
        fn shr_byte_arbitrary(a in any::<i8>(), shift in 0u32..8) {
            assert_eq!(shr(a, shift, &snumeric::sbyte::METHODS), a >> shift);
        }
        #[test]
        fn shr_short_arbitrary(a in any::<i16>(), shift in 0u32..16) {
            assert_eq!(shr(a, shift, &snumeric::sshort::METHODS), a >> shift);
        }
        #[test]
        fn shr_int_arbitrary(a in any::<i32>(), shift in 0u32..32) {
            assert_eq!(shr(a, shift, &snumeric::sint::METHODS), a >> shift);
        }
        #[test]
        fn shr_long_arbitrary(a in any::<i64>(), shift in 0u32..64) {
            assert_eq!(shr(a, shift, &snumeric::slong::METHODS), a >> shift);
        }
        #[test]
        fn shr_bigint_arbitrary(a in any::<BigInt256>(), shift in 0u32..256) {
            assert_eq!(shr(a, shift, &snumeric::sbigint::METHODS), a >> shift);
        }
        #[test]
        fn shr_unsigned_bigint_arbitrary(a in any::<UnsignedBigInt>(), shift in 0u32..256) {
            assert_eq!(shr(a, shift, &snumeric::sunsignedbigint::METHODS), a >> shift);
        }
        #[test]
        #[should_panic]
        fn shr_byte_arbitrary_invalid(a in any::<i8>(), shift in 8u32..) {
            assert_eq!(shr(a, shift, &snumeric::sbyte::METHODS), a >> shift);
        }
        #[test]
        #[should_panic]
        fn shr_short_arbitrary_invalid(a in any::<i16>(), shift in 16u32..) {
            assert_eq!(shr(a, shift, &snumeric::sshort::METHODS), a >> shift);
        }
        #[test]
        #[should_panic]
        fn shr_int_arbitrary_invalid(a in any::<i32>(), shift in 32u32..) {
            assert_eq!(shr(a, shift, &snumeric::sint::METHODS), a >> shift);
        }
        #[test]
        #[should_panic]
        fn shr_long_arbitrary_invalid(a in any::<i64>(), shift in 64u32..) {
            assert_eq!(shr(a, shift, &snumeric::slong::METHODS), a >> shift);
        }
        #[test]
        #[should_panic]
        fn shr_bigint_arbitrary_invalid(a in any::<BigInt256>(), shift in 256u32..) {
            assert_eq!(shr(a, shift, &snumeric::sbigint::METHODS), a >> shift);
        }
        #[test]
        #[should_panic]
        fn shr_unsigned_bigint_arbitrary_invalid(a in any::<UnsignedBigInt>(), shift in 256u32..) {
            assert_eq!(shr(a, shift, &snumeric::sunsignedbigint::METHODS), a >> shift);
        }

        #[test]
        fn eval_to_unsigned(signed in any::<BigInt256>()) {
            let mc: Expr = MethodCall::new(
                Constant::from(signed).into(),
                SMethod::new(STypeCompanion::SBigInt, TO_UNSIGNED_METHOD_DESC.clone()),
                vec![],
            )
            .unwrap()
            .into();
            let res = try_eval_out_wo_ctx::<UnsignedBigInt>(&mc);
            if signed < 0.into() {
                assert!(res.is_err())
            } else {
                assert_eq!(res.unwrap().to_string(), signed.to_string());
            }
        }
        #[test]
        fn eval_to_unsigned_mod(signed in any::<BigInt256>(), modulus in any::<UnsignedBigInt>()) {
            let mc: Expr = MethodCall::new(
                Constant::from(signed).into(),
                SMethod::new(STypeCompanion::SBigInt, TO_UNSIGNED_MOD_METHOD_DESC.clone()),
                vec![Constant::from(modulus).into()],
            )
            .unwrap()
            .into();
            assert_eq!(try_eval_out_wo_ctx::<UnsignedBigInt>(&mc).ok(), UnsignedBigInt::from_signed_mod(signed, modulus));
        }

        #[test]
        fn eval_to_signed(unsigned in any::<UnsignedBigInt>()) {
            let mc: Expr = MethodCall::new(
                Constant::from(unsigned).into(),
                SMethod::new(STypeCompanion::SUnsignedBigInt, TO_SIGNED_METHOD_DESC.clone()),
                vec![],
            )
            .unwrap()
            .into();
            let res = try_eval_out_wo_ctx::<BigInt256>(&mc);
            assert_eq!(res.ok(), BigInt256::try_from(unsigned).ok());
        }

        #[test]
        fn eval_mod_ops(a in any::<UnsignedBigInt>(), b in any::<UnsignedBigInt>(), modulus in any::<UnsignedBigInt>()) {
            assert_eq!(eval_modular_op(&MOD_METHOD_DESC, &[a, modulus]).ok(), a.checked_rem(&modulus));
            assert_eq!(eval_modular_op(&PLUS_MOD_METHOD_DESC, &[a, b, modulus]).ok(), a.checked_mod_add(b, modulus));
            assert_eq!(eval_modular_op(&SUBTRACT_MOD_METHOD_DESC, &[a, b, modulus]).ok(), a.checked_mod_sub(b, modulus));
            assert_eq!(eval_modular_op(&MULTIPLY_MOD_METHOD_DESC, &[a, b, modulus]).ok(), a.checked_mod_mul(b, modulus));
            assert_eq!(eval_modular_op(&MOD_INVERSE_METHOD_DESC, &[a, modulus]).ok(), a.mod_inv(modulus));
        }
    }
}
