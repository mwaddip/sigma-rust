use alloc::string::ToString;
use alloc::sync::Arc;

use ergotree_ir::mir::coll_map::Map;
use ergotree_ir::mir::coll_value::CollValue;
use ergotree_ir::mir::value::Value;
use ergotree_ir::types::stype::SType;

use crate::eval::env::Env;
use crate::eval::Context;
use crate::eval::EvalError;
use crate::eval::Evaluable;
use crate::eval::LambdaInvoker;

impl Evaluable for Map {
    fn eval<'ctx>(
        &self,
        env: &mut Env<'ctx>,
        ctx: &Context<'ctx>,
    ) -> Result<Value<'ctx>, EvalError> {
        let input_v = self.input.eval(env, ctx)?;
        let mapper_v = self.mapper.eval(env, ctx)?;
        map_coll(ctx, input_v, &mapper_v, self.out_elem_tpe())
    }
}

/// `MapCollection` once its input and mapper are evaluated, shared with the `Coll.map` method
/// (sigmastate's `map_eval` charges the same per-item cost and maps the same way, v6.0.7
/// `methods.scala:862-867`)
pub(crate) fn map_coll<'ctx>(
    ctx: &Context<'ctx>,
    input_v: Value<'ctx>,
    mapper_v: &Value<'ctx>,
    out_elem_tpe: SType,
) -> Result<Value<'ctx>, EvalError> {
    let input_v_clone = input_v.clone();
    let mut invoker = match &mapper_v {
        Value::Lambda(func_value) => {
            func_value.args.first().ok_or_else(|| {
                EvalError::NotFound("Map: evaluated mapper has empty arguments list".to_string())
            })?;
            // The body evaluates in the lambda's CAPTURED environment
            // (JVM closure semantics) — not in the caller's env.
            LambdaInvoker::new(func_value)
        }
        _ => {
            return Err(EvalError::UnexpectedValue(format!(
                "expected mapper to be Value::FuncValue got: {0:?}",
                input_v_clone
            )))
        }
    };
    let mut mapper_call = |arg: Value<'ctx>| invoker.invoke(ctx, vec![arg]);
    // sigmastate reads no element type here: the mapper checks each element against its
    // argument's type (v6.0.7 `transformers.scala:33-46`, `values.scala:1074`)
    let (normalized_input_vals, over_pairs) = match input_v {
        Value::Coll(coll) => Ok((coll.as_vec(), matches!(coll, CollValue::Pairs { .. }))),
        _ => Err(EvalError::UnexpectedValue(format!(
            "expected Map input to be Value::Coll, got: {0:?}",
            input_v
        ))),
    }?;
    // `stypeToRType(tRange)` before the charge, in `MapCollection.eval` and `map_eval` alike
    super::runtime_type::check_runtime_type(&out_elem_tpe, "map")?;
    ctx.add_per_item_jit_cost(20, 1, 10, normalized_input_vals.len() as u32)?;
    normalized_input_vals
        .iter()
        .map(|item| mapper_call(item.clone()))
        .collect::<Result<Arc<[Value]>, EvalError>>()
        .map(|values| {
            // `PairOfCols.map` builds a generic collection directly (v6.0.7
            // `CollsOverArrays.scala:317-324`); any other receiver's goes through `builder.fromArray`
            // at an abstract type (`:31-34`), a `PairOfCols` for pairs (SANTA
            // `Coll.pair_shape_restored` Y1). Both store each result into an array of the STATIC
            // result's class, which refuses another class: the lambda's own result check does not
            // cover a mapper that is an `If` of two lambdas (SANTA `Coll.reference_store_ops` and
            // the third reply's P1-P4)
            let is_pair_tpe = matches!(&out_elem_tpe, SType::STuple(t) if t.items.len() == 2);
            if over_pairs && is_pair_tpe {
                // a `Tuple2[]`: any pair is stored, its components unchecked (SANTA G2, P3)
                match values.iter().find(|v| !matches!(v, Value::Pair(_))) {
                    Some(v) => Err(EvalError::UnexpectedValue(format!(
                        "map: a collection of {out_elem_tpe:?} cannot hold {v:?}"
                    ))),
                    None => Ok(CollValue::generic(out_elem_tpe, values)),
                }
            } else {
                // not a pair type from a `PairOfCols`: a generic collection, as `from_array` builds
                CollValue::from_array(out_elem_tpe, values.to_vec())
                    .map_err(EvalError::TryExtractFrom)
            }
        })
        .and_then(|v| v) // flatten <Result<Result<Value, _>, _>
        .map(Value::Coll)
}

#[allow(clippy::panic)]
#[allow(clippy::unwrap_used)]
#[cfg(test)]
#[cfg(feature = "arbitrary")]
mod tests {

    use crate::eval::test_util::eval_out;
    use crate::eval::test_util::eval_out_wo_ctx;
    use ergotree_ir::chain::context::Context;
    use ergotree_ir::chain::context::TxIoVec;
    use ergotree_ir::ergo_tree::ErgoTree;
    use ergotree_ir::ergo_tree::ErgoTreeVersion;
    use ergotree_ir::mir::bin_op::ArithOp;
    use ergotree_ir::mir::bin_op::BinOp;
    use ergotree_ir::mir::expr::Expr;
    use ergotree_ir::mir::extract_amount::ExtractAmount;
    use ergotree_ir::mir::func_value::FuncArg;
    use ergotree_ir::mir::func_value::FuncValue;
    use ergotree_ir::mir::property_call::PropertyCall;
    use ergotree_ir::mir::unary_op::OneArgOpTryBuild;
    use ergotree_ir::mir::val_use::ValUse;
    use ergotree_ir::serialization::roundtrip_new_feature;
    use ergotree_ir::serialization::SigmaSerializable;
    use ergotree_ir::sigma_protocol::sigma_boolean::SigmaProp;
    use ergotree_ir::types::scontext;
    use ergotree_ir::types::stype::SType;

    use super::*;

    use proptest::prelude::*;

    proptest! {

        #![proptest_config(ProptestConfig::with_cases(16))]

        #[test]
        fn eval_box_value(ctx in any::<Context>()) {
            let data_inputs: Expr = PropertyCall::new(Expr::Context, scontext::DATA_INPUTS_PROPERTY.clone()).unwrap()
            .into();
            let val_use: Expr = ValUse {
                val_id: 1.into(),
                tpe: SType::SBox,
            }
            .into();
            let mapper_body: Expr = BinOp {
                kind: ArithOp::Plus.into(),
                left: Box::new(Expr::Const(1i64.into())),
                right: Box::new(Expr::ExtractAmount(
                        ExtractAmount::try_build(val_use)
                    .unwrap(),
                )),
            }
            .into();
            let expr: Expr = Map::new(
                data_inputs,
                FuncValue::new(
                    vec![FuncArg {
                        idx: 1.into(),
                        tpe: SType::SBox,
                    }],
                    mapper_body,
                )
                .into(),
            )
            .unwrap()
            .into();
            let output = {
                let e = eval_out::<Vec<i64>>(&expr, &ctx);
                if e.is_empty() {
                  None
                } else {
                  Some(TxIoVec::from_vec(e).unwrap())
                }
            };

            assert_eq!(
                output,
                ctx.data_inputs.clone().map(|d| d.iter().map(| b| b.value.as_i64() + 1).collect::<Vec<_>>().try_into().unwrap())
            );
        }

    }
    // test higher order lambda script from https://github.com/ergoplatform/sigmastate-interpreter/blob/eb1477178348d19736bf3b3d6a2013cc81ae0f67/sc/shared/src/test/scala/sigmastate/utxo/BasicOpsSpecification.scala#L2737
    #[test]
    fn test_higher_order_lambda() {
        let tree_bytes = base16::decode("1b4403040201010402d1ae8301108301047300d9010110ed730194ad7201d9010304dad901054c7001040400da8c720501018c720502018602d90105049a7205730272037201").unwrap();
        let tree = ErgoTree::sigma_parse_bytes(&tree_bytes).unwrap();
        let expr = tree.proposition().unwrap();
        roundtrip_new_feature(&expr, ErgoTreeVersion::V3);
        assert_eq!(
            eval_out_wo_ctx::<SigmaProp>(&expr),
            SigmaProp::new(true.into())
        );
    }
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod element_type_tests {
    //! sigmastate's `MapCollection` reads no element type: the mapper checks each element against
    //! its argument's type (v6.0.7 `transformers.scala:33-46`, `values.scala:1074`), so an empty
    //! input checks nothing.
    use super::*;
    use crate::eval::test_util::{eval_out_wo_ctx, try_eval_out_wo_ctx};
    use alloc::boxed::Box;
    use alloc::vec;
    use alloc::vec::Vec;
    use ergotree_ir::mir::coll_size::SizeOf;
    use ergotree_ir::mir::expr::Expr;
    use ergotree_ir::mir::func_value::{FuncArg, FuncValue};
    use ergotree_ir::mir::if_op::If;
    use ergotree_ir::mir::unary_op::OneArgOpTryBuild;
    use ergotree_ir::mir::val_use::ValUse;
    use ergotree_ir::types::sfunc::SFunc;
    use ergotree_ir::types::stype::SType;

    /// `if (false) Coll[Int](1) else <longs>`: typed `Coll[Int]` (an `If` is typed by its true
    /// branch), holding `Long`s
    fn int_coll_holding(longs: Vec<i64>) -> Expr {
        If {
            condition: Box::new(Expr::Const(false.into())),
            true_branch: Box::new(vec![1i32].into()),
            false_branch: Box::new(longs.into()),
        }
        .into()
    }

    fn lambda(arg_tpe: SType, body: Expr) -> Expr {
        FuncValue::new(
            vec![FuncArg {
                idx: 1.into(),
                tpe: arg_tpe,
            }],
            body,
        )
        .into()
    }

    /// `input.map({ (x: arg_tpe) => x })`
    fn map(input: Expr, arg_tpe: SType) -> Expr {
        let x: Expr = ValUse {
            val_id: 1.into(),
            tpe: arg_tpe.clone(),
        }
        .into();
        Map {
            input: Box::new(input),
            mapper: Box::new(lambda(arg_tpe.clone(), x)),
            mapper_sfunc: SFunc::new(vec![arg_tpe.clone()], arg_tpe),
        }
        .into()
    }

    #[test]
    fn the_lambda_checks_each_element_as_the_jvm_closure_does() {
        // (x: Long) => x maps the 2L; (x: Int) => x refuses it
        assert_eq!(
            eval_out_wo_ctx::<Vec<i64>>(&map(int_coll_holding(vec![2]), SType::SLong)),
            vec![2i64]
        );
        assert!(
            try_eval_out_wo_ctx::<Vec<i32>>(&map(int_coll_holding(vec![2]), SType::SInt)).is_err()
        );
    }

    #[test]
    fn an_empty_input_of_another_item_type_is_not_checked() {
        let size: Expr = SizeOf::try_build(map(int_coll_holding(vec![]), SType::SInt))
            .unwrap()
            .into();
        assert_eq!(eval_out_wo_ctx::<i32>(&size), 0);
    }
}
