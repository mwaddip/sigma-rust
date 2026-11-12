use ergo_chain_types::EcPoint;
use ergotree_ir::mir::constant::TryExtractInto;
use ergotree_ir::mir::exponentiate::Exponentiate;
use ergotree_ir::mir::value::Value;
use ergotree_ir::sigma_protocol::dlog_group;
use ergotree_ir::unsignedbigint256::UnsignedBigInt;
use k256::Scalar;
use num_bigint::BigInt;
use num_bigint::Sign;
use num_traits::Signed;

use crate::eval::env::Env;
use crate::eval::Context;
use crate::eval::EvalError;
use crate::eval::Evaluable;

impl Evaluable for Exponentiate {
    fn eval<'ctx>(
        &self,
        env: &mut Env<'ctx>,
        ctx: &Context<'ctx>,
    ) -> Result<Value<'ctx>, EvalError> {
        ctx.add_jit_cost(900)?; // Exponentiate = Fixed(900)
        let left_v = self.left.eval(env, ctx)?.try_extract_into()?;
        let right_v = self.right.eval(env, ctx)?.try_extract_into::<BigInt>()?;

        exponentiate(left_v, &right_v)
    }
}

/// `base` to the power of `exponent`: the `Exponentiate` node and the `GroupElement.exp` method
pub(crate) fn exponentiate(base: EcPoint, exponent: &BigInt) -> Result<Value<'static>, EvalError> {
    let exp_scalar = exponent_scalar(exponent)?;
    Ok(ergo_chain_types::ec_point::exponentiate(&base, &exp_scalar).into())
}

/// The exponent as a scalar: reduced modulo the group order, to a non-negative value.
/// sigmastate multiplies the point by the `BigInteger` as it is, which is `p^(k mod order)` for
/// a negative `k` and for one at or above the order (v6.0.6 `CGroupElement.scala:22-28`). A
/// tree below version 3 can hold an exponent outside the 256-bit range (`CBigInt.scala:18-20`):
/// `g.exp(BigInt.MIN / -1)` is g^(2^255) there (SANTA's reply
/// `sigma-rust-bigint-range-reply.md`)
fn exponent_scalar(exponent: &BigInt) -> Result<Scalar, EvalError> {
    let order = BigInt::from_bytes_be(Sign::Plus, &dlog_group::order().to_be_vec());
    let mut reduced = exponent % &order;
    if reduced.is_negative() {
        reduced += &order;
    }
    let (_, bytes) = reduced.to_bytes_be();
    UnsignedBigInt::from_be_slice(&bytes)
        .map(Scalar::from)
        .ok_or_else(|| EvalError::Misc("Exponentiate: the exponent does not fit a scalar".into()))
}

#[allow(clippy::unwrap_used)]
#[allow(clippy::panic)]
#[cfg(test)]
#[cfg(feature = "arbitrary")]
mod tests {
    use super::*;
    use crate::eval::test_util::eval_out;
    use crate::sigma_protocol::private_input::DlogProverInput;
    use ergotree_ir::chain::context::Context;

    use ergo_chain_types::EcPoint;
    use ergotree_ir::bigint256::BigInt256;
    use ergotree_ir::mir::expr::Expr;
    use ergotree_ir::sigma_protocol::dlog_group::scalar_to_bigint256;
    use num_traits::Num;
    use proptest::prelude::*;
    use sigma_test_util::force_any_val;

    proptest! {

        #[test]
        fn eval_any(left in any::<EcPoint>(), pi in any::<DlogProverInput>()) {
            // Shift right to make sure that the MSB is 0, so that the Scalar
            // can be converted to a BigInt256 and back
            let right: BigInt256 = scalar_to_bigint256(pi.w.as_scalar_ref() >> 1).unwrap();

            let expected_exp = ergo_chain_types::ec_point::exponentiate(
                &left,
                &dlog_group::bigint256_to_scalar(right)
            );

            let expr: Expr = Exponentiate {
                left: Box::new(Expr::Const(left.into())),
                right: Box::new(Expr::Const(right.into())),
            }
            .into();

            let ctx = force_any_val::<Context>();
            assert_eq!(eval_out::<EcPoint>(&expr, &ctx), expected_exp);
        }
    }

    #[test]
    fn eval_exponent_negative() {
        let left = force_any_val::<EcPoint>();
        let right = BigInt256::from_str_radix("-1", 10).unwrap();

        let expected_exp = ergo_chain_types::ec_point::exponentiate(
            &left,
            &dlog_group::bigint256_to_scalar(right),
        );

        let expr: Expr = Exponentiate {
            left: Box::new(Expr::Const(left.into())),
            right: Box::new(Expr::Const(right.into())),
        }
        .into();

        let ctx = force_any_val::<Context>();
        assert_eq!(eval_out::<EcPoint>(&expr, &ctx), expected_exp);
    }
}

#[cfg(test)]
#[cfg(feature = "arbitrary")]
#[allow(clippy::unwrap_used)]
mod wide_exponent_tests {
    use ergo_chain_types::ec_point;
    use ergo_chain_types::EcPoint;
    use ergotree_ir::ergo_tree::ErgoTreeVersion;
    use ergotree_ir::mir::value::Value;
    use ergotree_ir::sigma_protocol::dlog_group;
    use num_bigint::BigInt;
    use num_bigint::Sign;

    use super::exponent_scalar;
    use crate::eval::test_util::{bigint_x, eval_body, BIGINT_MAX_HEX};

    #[test]
    fn santa_exponentiate_takes_a_wide_exponent() {
        // SANTA's reply `sigma-rust-bigint-range-reply.md` (sigma-state 6.0.7), measured: below
        // tree version 3 `groupGenerator.exp(BigInt.MIN / -1)` is a point, and it is g^(2^255)
        let x = bigint_x();
        let g_to_x = format!("9f82{x}");
        let (res, cost) = eval_body(&g_to_x, ErgoTreeVersion::V0);
        let expected = EcPoint::from_base16_str(
            "02b23790a42be63e1b251ad6c94fdef07271ec0aada31db6c3e8bd32043f8be384".into(),
        )
        .unwrap();
        assert_eq!(res.ok(), Some(Value::from(expected)));
        assert_eq!(cost, 945);

        // `g.exp(X) == g.exp(BigInt.MAX).multiply(g)`
        let same = format!("939f82{x}a09f82{BIGINT_MAX_HEX}82");
        let (res, cost) = eval_body(&same, ErgoTreeVersion::V0);
        assert_eq!(res.ok(), Some(Value::Boolean(true)));
        assert_eq!(cost, 2082);

        // from tree version 3 sigmastate throws at `X` itself
        for body in [g_to_x, same] {
            let (res, _) = eval_body(&body, ErgoTreeVersion::V3);
            assert!(res.is_err(), "{body}: {res:?}");
        }
    }

    #[test]
    fn a_wide_exponent_is_reduced_modulo_the_group_order() {
        // By reading: sigmastate multiplies the point by the `BigInteger` as it is, which is
        // `p^(k mod order)` for a negative `k` and for one at or above the order (v6.0.6
        // `CGroupElement.scala:22-28`). Not measured above the order
        let order = BigInt::from_bytes_be(Sign::Plus, &dlog_group::order().to_be_vec());
        let g = ec_point::generator();
        let g_to = |k: BigInt| ec_point::exponentiate(&g, &exponent_scalar(&k).unwrap());
        assert_eq!(g_to(order.clone() + 1), g);
        assert!(ec_point::is_identity(&g_to(order.clone())));
        assert!(ec_point::is_identity(&g_to(order.clone() << 1800usize)));
        assert_eq!(g_to(BigInt::from(-1)), ec_point::inverse(&g));
        assert_eq!(g_to(-order.clone() - 1), ec_point::inverse(&g));
    }
}
