use alloc::string::ToString;

use alloc::vec::Vec;
use ergotree_ir::mir::coll_filter::Filter;
use ergotree_ir::mir::coll_value::CollValue;
use ergotree_ir::mir::constant::TryExtractInto;
use ergotree_ir::mir::value::Value;

use crate::eval::env::Env;
use crate::eval::Context;
use crate::eval::EvalError;
use crate::eval::Evaluable;
use crate::eval::LambdaInvoker;

impl Evaluable for Filter {
    fn eval<'ctx>(
        &self,
        env: &mut Env<'ctx>,
        ctx: &Context<'ctx>,
    ) -> Result<Value<'ctx>, EvalError> {
        let input_v = self.input.eval(env, ctx)?;
        let condition_v = self.condition.eval(env, ctx)?;
        let input_v_clone = input_v.clone();
        let mut invoker = match &condition_v {
            Value::Lambda(func_value) => {
                func_value.args.first().ok_or_else(|| {
                    EvalError::NotFound(
                        "Filter: evaluated condition has empty arguments list".to_string(),
                    )
                })?;
                // The body evaluates in the lambda's CAPTURED environment
                // (JVM closure semantics) — not in the caller's env.
                LambdaInvoker::new(func_value)
            }
            _ => {
                return Err(EvalError::UnexpectedValue(format!(
                    "expected Filter::condition to be Value::FuncValue got: {0:?}",
                    input_v_clone
                )))
            }
        };
        let mut condition_call = |arg: Value<'ctx>| invoker.invoke(ctx, vec![arg]);
        // sigmastate reads no element type here: the condition checks each element against its
        // argument's type, and the result is `inputV.filter(...)`, with the input's item type
        // (v6.0.7 `transformers.scala:117-128`, `values.scala:1074`)
        let (normalized_input_vals, receiver) = match input_v {
            Value::Coll(coll) => Ok((coll.as_vec(), coll)),
            _ => Err(EvalError::UnexpectedValue(format!(
                "expected Map input to be Value::Coll, got: {0:?}",
                input_v
            ))),
        }?;
        ctx.add_per_item_jit_cost(20, 1, 10, normalized_input_vals.len() as u32)?;

        let items_conditions: Vec<bool> = normalized_input_vals
            .clone()
            .into_iter()
            .map(|item| {
                condition_call(item).and_then(|res| {
                    res.try_extract_into::<bool>()
                        .map_err(EvalError::TryExtractFrom)
                })
            })
            .collect::<Result<Vec<bool>, EvalError>>()?;
        let filtered_items = normalized_input_vals
            .into_iter()
            .zip(items_conditions)
            .filter(|(_, condition)| *condition)
            .map(|(item, _)| item)
            .collect::<Vec<_>>();
        // the receiver's class (`builder.fromArray` inside the receiver), but a `PairOfCols`
        // rebuilds its sides with `pairCollFromArrays` at an abstract type (v6.0.7
        // `CollsOverArrays.scala:348-370`; SANTA `Coll.pair_shape_restored` Y11)
        Ok(Value::Coll(match receiver {
            CollValue::Pairs { .. } => CollValue::from_array(receiver.elem_tpe(), filtered_items)?,
            _ => receiver.rebuild_like(filtered_items)?,
        }))
    }
}

#[allow(clippy::unwrap_used)]
#[allow(clippy::panic)]
#[cfg(test)]
#[cfg(feature = "arbitrary")]
mod tests {

    use crate::eval::test_util::eval_out;
    use ergotree_ir::chain::context::Context;

    use super::*;

    use ergotree_ir::chain::ergo_box::ErgoBox;
    use ergotree_ir::mir::bin_op::BinOp;
    use ergotree_ir::mir::bin_op::RelationOp;
    use ergotree_ir::mir::expr::Expr;
    use ergotree_ir::mir::extract_amount::ExtractAmount;
    use ergotree_ir::mir::func_value::FuncArg;
    use ergotree_ir::mir::func_value::FuncValue;
    use ergotree_ir::mir::property_call::PropertyCall;
    use ergotree_ir::mir::unary_op::OneArgOpTryBuild;
    use ergotree_ir::mir::val_use::ValUse;
    use ergotree_ir::reference::Ref;
    use ergotree_ir::types::scontext;
    use ergotree_ir::types::stype::SType;
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
            let body: Expr = BinOp {
                kind: RelationOp::Le.into(),
                left: Box::new(Expr::Const(1i64.into())),
                right: Box::new(Expr::ExtractAmount(
                        ExtractAmount::try_build(val_use)
                    .unwrap(),
                )),
            }
            .into();
            let expr: Expr = Filter::new(
                data_inputs,
                FuncValue::new(
                    vec![FuncArg {
                        idx: 1.into(),
                        tpe: SType::SBox,
                    }],
                    body,
                )
                .into(),
            )
            .unwrap()
            .into();
            let expected: Vec<_> = ctx
                .data_inputs
                .clone()
                .map_or(
                     vec![],
                     |d| d
                         .iter()
                         .cloned()
                         .filter(|b| 1 <= b.value.as_i64())
                         .collect()
                );


            eval_out::<Vec<Ref<'_, ErgoBox>>>(&expr, &ctx)
                .into_iter()
                .zip(expected)
                .for_each(|(left, right)| assert_eq!(&*left, right));
       }
    }
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod element_type_tests {
    //! sigmastate's `Filter` reads no element type: the condition checks each element against its
    //! argument's type (v6.0.7 `transformers.scala:117-128`, `values.scala:1074`), so an empty
    //! input checks nothing, and its result is `inputV.filter(...)`, with the input's item type.
    use super::*;
    use crate::eval::test_util::{eval_out_wo_ctx, try_eval_out_wo_ctx};
    use alloc::boxed::Box;
    use alloc::vec;
    use alloc::vec::Vec;
    use ergotree_ir::mir::bin_op::{BinOp, RelationOp};
    use ergotree_ir::mir::coll_size::SizeOf;
    use ergotree_ir::mir::expr::Expr;
    use ergotree_ir::mir::func_value::{FuncArg, FuncValue};
    use ergotree_ir::mir::if_op::If;
    use ergotree_ir::mir::unary_op::OneArgOpTryBuild;
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

    fn filter(input: Expr, arg_tpe: SType, keep: bool) -> Expr {
        Filter {
            input: Box::new(input),
            condition: Box::new(lambda(arg_tpe, Expr::Const(keep.into()))),
            elem_tpe: SType::SInt.into(),
        }
        .into()
    }

    fn size(coll: Expr) -> Expr {
        SizeOf::try_build(coll).unwrap().into()
    }

    #[test]
    fn the_lambda_checks_each_element_as_the_jvm_closure_does() {
        // (x: Long) => true keeps the 2L; (x: Int) => true refuses it
        let long_filter = filter(int_coll_holding(vec![2]), SType::SLong, true);
        assert_eq!(eval_out_wo_ctx::<i32>(&size(long_filter)), 1);
        let int_filter = filter(int_coll_holding(vec![2]), SType::SInt, true);
        assert!(try_eval_out_wo_ctx::<i32>(&size(int_filter)).is_err());
    }

    #[test]
    fn an_empty_input_of_another_item_type_is_not_checked() {
        let empty = filter(int_coll_holding(vec![]), SType::SInt, true);
        assert_eq!(eval_out_wo_ctx::<i32>(&size(empty)), 0);
    }

    #[test]
    fn a_filter_result_keeps_the_input_item_type() {
        // `CX.filter({ (x: Long) => false })` is an empty `Coll[Long]`, so it is not
        // `Coll[Int]()` (`DataValueComparer.scala:318-326`)
        let none_kept = || filter(int_coll_holding(vec![2]), SType::SLong, false);
        let kept = try_eval_out_wo_ctx::<Value>(&none_kept()).unwrap();
        assert!(
            matches!(&kept, Value::Coll(coll) if coll.elem_tpe() == SType::SLong),
            "expected an empty Coll[Long], got {kept:?}"
        );
        let eq: Expr = BinOp {
            kind: RelationOp::Eq.into(),
            left: Box::new(none_kept()),
            right: Box::new(Vec::<i32>::new().into()),
        }
        .into();
        assert!(!eval_out_wo_ctx::<bool>(&eq));
    }
}
