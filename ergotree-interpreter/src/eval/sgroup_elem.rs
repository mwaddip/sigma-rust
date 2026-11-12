use crate::eval::EvalError;

use alloc::vec::Vec;
use ergo_chain_types::ec_point::exponentiate;
use ergo_chain_types::EcPoint;
use ergotree_ir::mir::constant::TryExtractInto;
use ergotree_ir::mir::value::Value;
use ergotree_ir::reference::Ref;
use ergotree_ir::serialization::SigmaSerializable;
use ergotree_ir::unsignedbigint256::UnsignedBigInt;
use k256::Scalar;

use super::EvalFn;

pub(crate) static GET_ENCODED_EVAL_FN: EvalFn = |_mc, _env, ctx, obj, _args| {
    ctx.add_jit_cost(250)?;
    let encoded: Vec<u8> = match obj {
        Value::GroupElement(ec_point) => Ok(ec_point.sigma_serialize_bytes()?),
        _ => Err(EvalError::UnexpectedValue(format!(
            "expected obj to be Value::GroupElement, got: {0:?}",
            obj
        ))),
    }?;

    Ok(Value::from(encoded))
};

pub(crate) static NEGATE_EVAL_FN: EvalFn = |_mc, _env, ctx, obj, _args| {
    ctx.add_jit_cost(45)?;
    let negated: EcPoint = match obj {
        Value::GroupElement(ec_point) => Ok(-(*ec_point)),
        _ => Err(EvalError::UnexpectedValue(format!(
            "expected obj to be Value::GroupElement, got: {0:?}",
            obj
        ))),
    }?;
    Ok(Value::GroupElement(Ref::from(negated)))
};

pub(crate) static EXPONENTIATE_EVAL_FN: EvalFn = |_mc, _env, ctx, obj, mut args| {
    ctx.add_jit_cost(900)?;
    // the exponent as it is: this method is declared below tree version 3, where a `BigInt` can
    // be outside the 256-bit range
    let exponent = args
        .pop()
        .ok_or_else(|| EvalError::UnexpectedValue("exponentiate: first argument not found".into()))?
        .try_extract_into::<num_bigint::BigInt>()?;
    crate::eval::exponentiate::exponentiate(obj.try_extract_into()?, &exponent)
};

pub(crate) static MULTIPLY_EVAL_FN: EvalFn = |_mc, _env, ctx, obj, mut args| {
    ctx.add_jit_cost(40)?;
    let obj = obj.try_extract_into::<EcPoint>()?;
    let right = args
        .pop()
        .ok_or_else(|| EvalError::UnexpectedValue("exponentiate: first argument not found".into()))?
        .try_extract_into::<EcPoint>()?;
    Ok((obj * &right).into())
};

pub(crate) static EXPONENTIATE_UNSIGNED_EVAL_FN: EvalFn = |_mc, _env, ctx, obj, mut args| {
    ctx.add_jit_cost(900)?;
    let exponent: Scalar = args
        .pop()
        .ok_or_else(|| EvalError::UnexpectedValue("exponentiate: first argument not found".into()))?
        .try_extract_into::<UnsignedBigInt>()?
        .into();
    Ok(exponentiate(&obj.try_extract_into()?, &exponent).into())
};

#[allow(clippy::unwrap_used)]
#[cfg(test)]
#[cfg(feature = "arbitrary")]
mod tests {
    use alloc::vec::Vec;
    use ergotree_ir::bigint256::BigInt256;
    use ergotree_ir::mir::constant::Constant;
    use ergotree_ir::mir::exponentiate::Exponentiate;
    use ergotree_ir::mir::expr::Expr;
    use ergotree_ir::mir::method_call::MethodCall;
    use ergotree_ir::mir::multiply_group::MultiplyGroup;
    use ergotree_ir::mir::value::Value;
    use ergotree_ir::types::sgroup_elem;
    use ergotree_ir::unsignedbigint256::UnsignedBigInt;
    use proptest::prelude::*;
    use proptest::proptest;

    use crate::eval::test_util::eval_out_wo_ctx;
    use crate::eval::test_util::try_eval_out_wo_ctx;
    use crate::sigma_protocol::private_input::DlogProverInput;
    use ergo_chain_types::EcPoint;
    use ergotree_ir::serialization::SigmaSerializable;
    use sigma_test_util::force_any_val;

    #[test]
    fn eval_get_encoded() {
        let input = force_any_val::<EcPoint>();
        let expr: Expr = MethodCall::new(
            input.into(),
            sgroup_elem::GET_ENCODED_METHOD.clone(),
            vec![],
        )
        .unwrap()
        .into();

        let res: Vec<u8> = eval_out_wo_ctx::<Vec<u8>>(&expr);
        let roundtrip_res: EcPoint = SigmaSerializable::sigma_parse_bytes(&res).unwrap();

        assert!(!res.is_empty());
        assert_eq!(input, roundtrip_res)
    }

    #[test]
    fn eval_negate() {
        let input = force_any_val::<EcPoint>();
        let expr: Expr = MethodCall::new(input.into(), sgroup_elem::NEGATE_METHOD.clone(), vec![])
            .unwrap()
            .into();
        assert_eq!(-input, eval_out_wo_ctx::<EcPoint>(&expr))
    }

    proptest! {
        #[test]
        fn eval_exponentiate(a in any::<EcPoint>(), b in any::<BigInt256>()) {
            let mc: Expr = MethodCall::new(Constant::from(a).into(), sgroup_elem::EXPONENTIATE_METHOD.clone(), vec![Constant::from(b).into()]).unwrap().into();
            let exponentiate_node: Expr = Exponentiate::new(Constant::from(a).into(), Constant::from(b).into()).unwrap().into();
            assert_eq!(try_eval_out_wo_ctx::<Value>(&mc), try_eval_out_wo_ctx::<Value>(&exponentiate_node))
        }

        #[test]
        fn eval_exponentiate_unsigned(left in any::<EcPoint>(), pi in any::<DlogProverInput>()) {
            let right = UnsignedBigInt::from_be_slice(&pi.w.as_scalar_ref().to_bytes()[..]).unwrap();
            let expected_exp = ergo_chain_types::ec_point::exponentiate(
                &left,
                pi.w.as_scalar_ref()
            );
            let mc: Expr = MethodCall::new(Constant::from(left).into(), sgroup_elem::EXPONENTIATE_UNSIGNED_METHOD.clone(), vec![Constant::from(right).into()]).unwrap().into();
            assert_eq!(eval_out_wo_ctx::<EcPoint>(&mc), expected_exp);
        }
        #[test]
        fn eval_multiply(a in any::<EcPoint>(), b in any::<EcPoint>()) {
            let mc: Expr = MethodCall::new(Constant::from(a).into(), sgroup_elem::MULTIPLY_METHOD.clone(), vec![Constant::from(b).into()]).unwrap().into();
            let multiply_node: Expr = MultiplyGroup::new(Constant::from(a).into(), Constant::from(b).into()).unwrap().into();
            assert_eq!(try_eval_out_wo_ctx::<Value>(&mc), try_eval_out_wo_ctx::<Value>(&multiply_node))
        }
    }
}

#[cfg(test)]
#[cfg(feature = "arbitrary")]
#[allow(clippy::unwrap_used)]
mod wide_exponent_tests {
    use ergo_chain_types::EcPoint;
    use ergotree_ir::ergo_tree::ErgoTreeVersion;
    use ergotree_ir::mir::value::Value;

    use crate::eval::test_util::{bigint_x, bigint_y, eval_body};

    #[test]
    fn the_exp_method_takes_a_wide_exponent() {
        // By reading: `GroupElement.exp` is declared at every tree version (v6.0.6
        // `methods.scala:647-654`, `:679-688`) and calls the `CGroupElement.exp` the
        // `Exponentiate` node calls, so a tree below version 3 can hand it a `BigInt` outside
        // 256 bits. SANTA measured the node (g^(2^255)); the method call is not measured, and
        // its cost here is sigma-rust's own, that of the in-range twin
        let point = |hex: &str| Some(Value::from(EcPoint::from_base16_str(hex.into()).unwrap()));
        // `MethodCall(groupGenerator, GroupElement.exp, BigInt.MIN / -1)`
        let wide = format!("dc07038201{}", bigint_x());
        let (res, cost) = eval_body(&wide, ErgoTreeVersion::V0);
        assert_eq!(
            res.ok(),
            point("02b23790a42be63e1b251ad6c94fdef07271ec0aada31db6c3e8bd32043f8be384")
        );
        assert_eq!(cost, 949);
        let (res, _) = eval_body(&wide, ErgoTreeVersion::V3);
        assert!(res.is_err(), "{res:?}");
        // the twin, `6 / -1`
        let (res, cost) = eval_body(&format!("dc07038201{}", bigint_y()), ErgoTreeVersion::V0);
        assert_eq!(
            res.ok(),
            point("02fff97bd5755eeea420453a14355235d382f6472f8568a18b2f057a1460297556")
        );
        assert_eq!(cost, 949);
    }
}
