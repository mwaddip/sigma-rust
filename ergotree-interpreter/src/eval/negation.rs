use ergotree_ir::mir::constant::TryExtractInto;
use ergotree_ir::mir::negation::Negation;
use ergotree_ir::mir::value::Value;
use ergotree_ir::types::stype::SType;

use crate::eval::env::Env;
use crate::eval::fits_256_bits;
use crate::eval::Context;
use crate::eval::EvalError;
use crate::eval::Evaluable;

/// The entry of sigmastate's `ArithOp.impls` for a type, as `Negation` uses it (v6.0.6
/// `trees.scala:885-890`)
enum NegationImpl {
    Byte,
    Short,
    Int,
    Long,
    BigInt,
    UnsignedBigInt,
}

/// The table's entry for a type, or `None` for a type that has none
fn negation_impl(tpe: &SType) -> Option<NegationImpl> {
    match tpe {
        SType::SByte => Some(NegationImpl::Byte),
        SType::SShort => Some(NegationImpl::Short),
        SType::SInt => Some(NegationImpl::Int),
        SType::SLong => Some(NegationImpl::Long),
        SType::SBigInt => Some(NegationImpl::BigInt),
        SType::SUnsignedBigInt => Some(NegationImpl::UnsignedBigInt),
        _ => None,
    }
}

impl Evaluable for Negation {
    fn eval<'ctx>(
        &self,
        env: &mut Env<'ctx>,
        ctx: &Context<'ctx>,
    ) -> Result<Value<'ctx>, EvalError> {
        let input_v = self.input.eval(env, ctx)?;

        fn overflow_err<T: core::fmt::Display>(v: &T) -> EvalError {
            EvalError::ArithmeticException(format!("Overflow on Negation of value {}", *v))
        }
        // sigmastate takes the implementation from the table arithmetic uses, by the INPUT's
        // static type, once the input is evaluated; then it adds the cost, and the
        // implementation casts the value to its type (`ArithOp.impls(input.tpe.typeCode).i`,
        // v6.0.6 `trees.scala:885-890`). A value of another numeric type throws at every tree
        // version, a narrower one and a wider one alike (SANTA `Negation.mixed_width`)
        let tpe = self.input.tpe();
        let Some(implementation) = negation_impl(&tpe) else {
            return Err(EvalError::UnexpectedValue(format!(
                "Negation: no arithmetic for the input's type {tpe:?}"
            )));
        };
        ctx.add_jit_cost(30)?; // Negation = Fixed(30)
        match implementation {
            // Fixed-width signed ints negate with two's-complement wrap
            // (`-MIN == MIN`), matching sigma-state's unchecked numeric
            // `negate` (`ast/trees.scala` Negation.eval) — negating MIN is
            // not an error on the JVM.
            NegationImpl::Byte => Ok(input_v.try_extract_into::<i8>()?.wrapping_neg().into()),
            NegationImpl::Short => Ok(input_v.try_extract_into::<i16>()?.wrapping_neg().into()),
            NegationImpl::Int => Ok(input_v.try_extract_into::<i32>()?.wrapping_neg().into()),
            NegationImpl::Long => Ok(input_v.try_extract_into::<i64>()?.wrapping_neg().into()),
            // sigmastate's `BigInt.negate` checks its RESULT against the 256-bit range at every
            // tree version (v6.0.6 `CBigInt.scala:55`): negating `MIN` is refused, and negating
            // 2^255, a value a tree below version 3 can hold, is `MIN` (SANTA's reply
            // `sigma-rust-bigint-range-reply.md`, B8)
            NegationImpl::BigInt => {
                let v = input_v.try_extract_into::<num_bigint::BigInt>()?;
                let negated = -&v;
                if fits_256_bits(&negated) {
                    Ok(Value::BigInt(negated))
                } else {
                    Err(overflow_err(&v))
                }
            }
            // `UnsignedBigInt` has its entry, and its `negate` is `???`
            // (`UnsignedBigIntegerOps.scala:48`): it throws once the cost is added
            NegationImpl::UnsignedBigInt => Err(EvalError::UnexpectedValue(format!(
                "Negation is not implemented for {tpe:?}"
            ))),
        }
    }
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod tests {

    use super::*;
    use crate::eval::test_util::try_eval_out_wo_ctx;
    use ergotree_ir::bigint256::BigInt256;
    use ergotree_ir::mir::constant::Constant;
    use ergotree_ir::mir::constant::TryExtractFrom;
    use ergotree_ir::mir::expr::Expr;
    use ergotree_ir::mir::unary_op::OneArgOpTryBuild;
    use num_traits::{Bounded, Num};

    fn try_run_eval<T: Num + Into<Constant> + TryExtractFrom<Value<'static>> + 'static>(
        input: T,
    ) -> Result<T, EvalError> {
        let expr: Expr = Negation::try_build(Expr::Const(input.into()))
            .unwrap()
            .into();
        try_eval_out_wo_ctx::<T>(&expr)
    }
    fn run_eval<T: Num + Into<Constant> + TryExtractFrom<Value<'static>> + 'static>(input: T) -> T {
        try_run_eval(input).unwrap()
    }

    #[test]
    fn eval() {
        assert_eq!(run_eval(1i8), -1i8);
        assert_eq!(run_eval(1i16), -1i16);
        assert_eq!(run_eval(1i32), -1i32);
        assert_eq!(run_eval(1i64), -1i64);
        assert_eq!(run_eval(BigInt256::from(1i64)), BigInt256::from(-1i64));
    }

    // JVM parity (sigma-state 6.0.3, LanguageSpecificationV5): negating a
    // fixed-width MIN_VALUE two's-complement-wraps to itself (`-MIN == MIN`)
    // with no error — sigma-state's `Negation.eval` uses the unchecked
    // numeric `negate`. Repros: santa `vectors/eval/v5/Numeric_Negation_
    // equivalence.json` (`-128#0`, `-32768#9`, `-2147483648#18`,
    // `-9223372036854775808#26`). `BigInt256` is the exception — its 256-bit
    // MIN (`-(2^255)`) overflows on both sides, so it still errors.
    #[test]
    fn negation_of_min_value_wraps_to_self() {
        assert_eq!(run_eval(i8::MIN), i8::MIN);
        assert_eq!(run_eval(i16::MIN), i16::MIN);
        assert_eq!(run_eval(i32::MIN), i32::MIN);
        assert_eq!(run_eval(i64::MIN), i64::MIN);
        assert!(try_run_eval(BigInt256::min_value()).is_err());
    }
}

#[cfg(test)]
#[cfg(feature = "arbitrary")]
#[allow(clippy::unwrap_used)]
mod negation_type_tests {
    use alloc::boxed::Box;
    use ergotree_ir::bigint256::BigInt256;
    use ergotree_ir::chain::context::Context;
    use ergotree_ir::ergo_tree::ErgoTreeVersion;
    use ergotree_ir::mir::expr::Expr;
    use ergotree_ir::mir::negation::Negation;
    use ergotree_ir::mir::value::Value;
    use sigma_test_util::force_any_val;

    use crate::eval::env::Env;
    use crate::eval::test_util::{assert_trees, eval_tree};
    use crate::eval::Evaluable;

    #[test]
    fn santa_negation_rows() {
        // SANTA `Negation.mixed_width` (`056df13`, sigma-state 6.0.7), each script as a v3
        // sized tree and a v0 tree
        assert_trees(&[
            // N0 `-(1L)`, the control
            ("0b03f00502", Some((Value::Long(-1), 35))),
            ("00f00502", Some((Value::Long(-1), 35))),
            // N1 `-If(false, 1L, 2)`: an `Int` value in a `Long` input
            ("0b08f095010005020404", None),
            ("00f095010005020404", None),
            // N2 `-If(false, 1, 2L)`, the mirror: a `Long` value in an `Int` input
            ("0b08f095010004020504", None),
            ("00f095010004020504", None),
            // N3 `-If(false, 1L, 2L)`, the honest twin of N1
            ("0b08f095010005020504", Some((Value::Long(-2), 50))),
            ("00f095010005020504", Some((Value::Long(-2), 50))),
            // L1 `If(false, 1, 2L)` alone: nothing checks the branch that is taken
            ("0b0795010004020504", Some((Value::Long(2), 20))),
            ("0095010004020504", Some((Value::Long(2), 20))),
        ]);
    }

    #[test]
    fn negation_takes_the_input_s_static_type() {
        // The refusals are by reading: the lookup SANTA measured for `Long` and `Int` inputs
        // (v6.0.6 `trees.scala:885-890`), here for a `BigInt` input, and the `UnsignedBigInt`
        // entry, whose `negate` is `???` (`UnsignedBigIntegerOps.scala:48`). The value and its
        // cost are sigma-rust's own at `d34f774b`, not a JVM measurement
        assert_trees(&[
            // `-(UnsignedBigInt 1)`
            ("0b04f0090101", None),
            // `-If(false, BigInt 1, 2L)`: a `Long` value in a `BigInt` input
            ("0b09f09501000601010504", None),
            ("00f09501000601010504", None),
            // `-(BigInt 1)`
            (
                "0b04f0060101",
                Some((Value::from(BigInt256::from(-1i32)), 35)),
            ),
        ]);
        // `UnsignedBigInt` has its entry in the table, so the cost is charged before `negate`
        // throws: the constant's 5 and the 30
        let (res, cost) = eval_tree("0b04f0090101");
        assert!(res.is_err(), "{res:?}");
        assert_eq!(cost, 35);
    }

    #[test]
    fn producers_keep_their_negation() {
        // A negation whose input's type comes from another node keeps its value: sigma-rust's
        // own values and costs at `d34f774b`, not JVM measurements. A `tpe()` that named another
        // type than the value's would refuse a tree the JVM accepts
        assert_trees(&[
            // `-ByIndex(Coll[Long](1, 2), 0)`
            ("0b09f0b211020204040000", Some((Value::Long(-1), 70))),
            // `-(1L, 2L)._1`
            ("0b06f08c59020401", Some((Value::Long(-1), 45))),
            // `-SizeOf(Coll[Long](1, 2))`
            ("0b06f0b111020204", Some((Value::Int(-2), 49))),
            // `-OptionGet(Some(1L))`
            ("0b05f0e4290102", Some((Value::Long(-1), 50))),
        ]);
    }

    #[test]
    fn negation_charges_after_its_input() {
        // By reading: `Negation.eval` evaluates its input, then adds its cost (v6.0.6
        // `trees.scala:885-890`). `-(2L / 0L)` fails in the division: the two constants and the
        // division's 15 are charged, the negation's 30 is not
        for tree in ["0b06f09d05040500", "00f09d05040500"] {
            let (res, cost) = eval_tree(tree);
            assert!(res.is_err(), "{tree}: {res:?}");
            assert_eq!(cost, 25, "{tree}");
        }
    }

    #[test]
    fn negation_built_in_memory_over_a_non_numeric_type_errors() {
        // By reading: the JVM looks the input's type up before it adds the cost, and the table
        // has no entry for a non-numeric type. The parser refuses such a node (`Negation`'s
        // `require`); one built without it is an error here, charged the constant's 5 only
        let expr: Expr = Negation {
            input: Box::new(Expr::Const(true.into())),
        }
        .into();
        let mut ctx = force_any_val::<Context>();
        ctx.activated_script_version_byte = 3;
        ctx.tree_version.set(ErgoTreeVersion::V3);
        ctx.jit_cost.set(0);
        let res = expr.eval(&mut Env::empty(), &ctx);
        assert!(res.is_err(), "{res:?}");
        assert_eq!(ctx.jit_cost_value(), 5);
    }
}
