use crate::eval::EvalError;
use crate::eval::LambdaInvoker;

use alloc::boxed::Box;
use alloc::string::ToString;
use alloc::vec::Vec;
use ergotree_ir::mir::coll_value::CollValue;
use ergotree_ir::mir::constant::TryExtractFromError;
use ergotree_ir::mir::constant::TryExtractInto;
use ergotree_ir::mir::expr::Expr;
use ergotree_ir::mir::value::Value;
use ergotree_ir::types::stype::SType;

use super::coll_by_index::item_at;
use super::coll_map::map_coll;
use super::env::Env;
use super::Context;
use super::EvalFn;
use super::MethodCallSite;
use alloc::sync::Arc;
use core::convert::TryFrom;

pub(crate) static INDEX_OF_EVAL_FN: EvalFn = |_mc, _env, ctx, obj, args| {
    Ok(Value::Int({
        let normalized_input_vals: Vec<Value> = match obj {
            Value::Coll(coll) => Ok(coll.as_vec()),
            _ => Err(EvalError::UnexpectedValue(format!(
                "expected obj to be Value::Coll, got: {0:?}",
                obj
            ))),
        }?;
        let target_element = args
            .first()
            .cloned()
            .ok_or_else(|| EvalError::NotFound("indexOf: missing first arg".to_string()))?;
        let from = args
            .get(1)
            .cloned()
            .ok_or_else(|| EvalError::NotFound("indexOf: missing second arg".to_string()))?
            .try_extract_into::<i32>()?
            .max(0);
        let len = normalized_input_vals.len();
        let start = (from as usize).min(len);
        // Scan from `start`, comparing each element with the target through
        // the cost-charging DataValueComparer: JVM `indexOf_eval` calls
        // `equalDataValues(xs(i), elem)` per step, charging the element-type
        // equality cost (EQ_PRIM=3, EQ_BIGINT=5, EQ_GroupElement=172, ...). A
        // bare `==` here left that per-comparison cost uncharged.
        let mut found = None;
        for (offset, it) in normalized_input_vals.iter().skip(start).enumerate() {
            if crate::eval::data_value_comparer::eq_with_cost(it, &target_element, ctx)? {
                found = Some(offset);
                break;
            }
        }
        // indexOf also charges PerItemCost(20, 10, 2) over the iterations
        // actually performed (Scala's `i - start`): from `start` to the found
        // index (inclusive) or to the collection end -- not the full length.
        let iterations = match found {
            Some(off) => off + 1,
            None => len - start,
        };
        ctx.add_per_item_jit_cost(20, 10, 2, iterations as u32)?;
        found.map(|off| (start + off) as i32).unwrap_or(-1)
    }))
};

/// The concatenation of `colls`, each a collection of the type `colls_tpe`, as `flatMap` builds
/// it: `builder.fromArray` at an abstract item type (v6.0.7 `CollsOverArrays.scala:72-75`,
/// `:426-427`), so generic, or a `PairOfCols` for pairs (SANTA `Coll.pair_shape_restored` Y3)
fn flatten<'ctx>(
    colls_tpe: SType,
    colls: Vec<Value<'ctx>>,
) -> Result<CollValue<'ctx>, TryExtractFromError> {
    let SType::SColl(item_tpe) = colls_tpe else {
        return Err(TryExtractFromError(format!(
            "Expected Value::Coll, got: {colls_tpe:?}"
        )));
    };
    let items: Vec<Value> = if *item_tpe == SType::SByte {
        colls
            .into_iter()
            .map(|v| v.try_extract_into::<Vec<i8>>())
            .collect::<Result<Vec<_>, _>>()?
            .concat()
            .into_iter()
            .map(Value::Byte)
            .collect()
    } else {
        colls
            .into_iter()
            .map(|v| v.try_extract_into::<Vec<Value>>())
            .collect::<Result<Vec<_>, _>>()?
            .concat()
    };
    CollValue::from_array((*item_tpe).clone(), items)
}

pub(crate) fn flatmap_eval<'ctx>(
    _mc: &MethodCallSite<'_>,
    _env: &mut Env<'ctx>,
    ctx: &Context<'ctx>,
    obj: Value<'ctx>,
    args: Vec<Value<'ctx>>,
) -> Result<Value<'ctx>, EvalError> {
    let input_v = obj;
    let lambda_v = args
        .first()
        .cloned()
        .ok_or_else(|| EvalError::NotFound("flatmap: eval is missing first arg".to_string()))?;
    let input_v_clone = input_v.clone();
    let lambda = match &lambda_v {
        Value::Lambda(l) => Ok(l),
        _ => Err(EvalError::UnexpectedValue(format!(
            "expected lambda to be Value::FuncValue got: {0:?}",
            input_v_clone
        ))),
    }?;
    if lambda.args.len() > 1 {
        return Err(EvalError::UnexpectedValue(format!(
            "flatmap: expected lambda taking 1 arg but got {} args",
            lambda.args.len()
        )));
    }
    let unsupported_msg =
        "unsupported lambda in flatMap: allowed usage `xs.flatMap(x => x.property)".to_string();
    if let Expr::MethodCall(mc) = &*lambda.body {
        if !mc.expr().args.is_empty() {
            return Err(EvalError::UnexpectedValue(unsupported_msg));
        }
    }
    lambda.args.first().ok_or_else(|| {
        EvalError::NotFound("flatmap: lambda has empty arguments list".to_string())
    })?;
    // The body evaluates in the lambda's CAPTURED environment (JVM closure
    // semantics) — not in the caller's env. ADD_TO_ENV_COST (5) per input
    // element is charged by the invoker (AddToEnvironmentDesc,
    // values.scala:1047), distinct from the output-length per-item charge
    // below.
    let mut invoker = LambdaInvoker::new(lambda);
    let mut lambda_call = |arg: Value<'ctx>| invoker.invoke(ctx, vec![arg]);
    // sigmastate reads no element type here: the lambda checks each element against its
    // argument's type (v6.0.7 `methods.scala` `flatMap_eval`, `values.scala:1074`)
    let normalized_input_vals: Vec<Value> = match input_v {
        Value::Coll(coll) => Ok(coll.as_vec()),
        _ => Err(EvalError::UnexpectedValue(format!(
            "expected Flatmap input to be Value::Coll, got: {0:?}",
            input_v
        ))),
    }?;
    let values = normalized_input_vals
        .iter()
        .map(|item| lambda_call(item.clone()))
        .collect::<Result<Vec<Value>, EvalError>>()?;
    let coll = flatten(lambda.body.tpe(), values).map_err(EvalError::TryExtractFrom)?;
    // flatMap cost scales with the OUTPUT (flattened) length, not the input
    // length, matching Scala's `flatMap_eval` (addSeqCost over `res.length`
    // after building the result, post the per-item lambda calls).
    ctx.add_per_item_jit_cost(60, 10, 8, coll.len() as u32)?;
    Ok(Value::Coll(coll))
}

pub(crate) static ZIP_EVAL_FN: EvalFn = |_mc, _env, ctx, obj, args| {
    let Value::Coll(ls) = obj else {
        return Err(EvalError::UnexpectedValue(format!(
            "expected obj to be Value::Coll, got: {0:?}",
            obj
        )));
    };
    ctx.add_per_item_jit_cost(10, 1, 10, ls.len() as u32)?;
    let arg_1 = args
        .first()
        .cloned()
        .ok_or_else(|| EvalError::NotFound("zip: missing first arg".to_string()))?;
    let Value::Coll(rs) = arg_1 else {
        return Err(EvalError::UnexpectedValue(format!(
            "expected first arg to be Value::Coll, got: {0:?}",
            arg_1
        )));
    };
    // a `PairOfCols` of both sides as they are, the longer sliced to the shorter's length, which
    // keeps its class (`pairColl`, v6.0.7 `CollsOverArrays.scala:183-196`)
    fn shorten(side: CollValue<'_>, n: usize) -> Result<CollValue<'_>, EvalError> {
        if side.len() > n {
            Ok(side.rebuild_like(side.as_vec().into_iter().take(n).collect())?)
        } else {
            Ok(side)
        }
    }
    let n = ls.len().min(rs.len());
    Ok(Value::Coll(CollValue::pairs(
        shorten(ls, n)?,
        shorten(rs, n)?,
    )))
};

pub(crate) static INDICES_EVAL_FN: EvalFn = |_mc, _env, ctx, obj, _args| {
    let input_len = match obj {
        Value::Coll(coll) => Ok(coll.len()),
        _ => Err(EvalError::UnexpectedValue(format!(
            "expected obj to be Value::Coll, got: {0:?}",
            obj
        ))),
    }?;
    ctx.add_per_item_jit_cost(20, 2, 16, input_len as u32)?;
    // a specialised `Coll[Int]` from any receiver (`builder.fromArray(toArray.indices.toArray)`,
    // v6.0.7 `CollsOverArrays.scala:68`)
    let indices = (0..input_len)
        .map(i32::try_from)
        .collect::<Result<Arc<[_]>, core::num::TryFromIntError>>()
        .map_err(|e| EvalError::UnexpectedValue(format!("Coll length overflow: {0:?}", e)))?;
    Ok(Value::Coll(CollValue::Ints(indices)))
};

pub(crate) static PATCH_EVAL_FN: EvalFn = |_mc, _env, ctx, obj, args| {
    let (normalized_input_vals, receiver) = match obj {
        Value::Coll(coll) => Ok((coll.as_vec(), coll)),
        _ => Err(EvalError::UnexpectedValue(format!(
            "expected obj to be Value::Coll, got: {0:?}",
            obj
        ))),
    }?;
    let n = normalized_input_vals.len() as u32;
    ctx.add_per_item_jit_cost(30, 2, 10, n)?;
    let from_index_val = args
        .first()
        .cloned()
        .ok_or_else(|| EvalError::NotFound("patch: missing first arg (from)".to_string()))?;
    let patch_val = args
        .get(1)
        .cloned()
        .ok_or_else(|| EvalError::NotFound("patch: missing second arg (patch)".to_string()))?;
    let replaced_val = args
        .get(2)
        .cloned()
        .ok_or_else(|| EvalError::NotFound("patch: missing third arg (replaced)".to_string()))?;

    let from = from_index_val.try_extract_into::<i32>()?.max(0) as usize;
    let replaced = replaced_val.try_extract_into::<i32>()?.max(0) as usize;
    let patch = match patch_val {
        Value::Coll(coll) => Ok(coll.as_vec()),
        _ => Err(EvalError::UnexpectedValue(format!(
            "expected patch arg to be Value::Coll, got: {0:?}",
            patch_val
        ))),
    }?;

    let res = normalized_input_vals
        .iter()
        .take(from)
        .chain(patch.iter())
        .chain(normalized_input_vals.iter().skip(from + replaced))
        .cloned()
        .collect::<Arc<[_]>>();
    // the receiver's class (`builder.fromArray` inside the receiver)
    Ok(Value::Coll(receiver.rebuild_like(res.to_vec())?))
};

pub(crate) static UPDATED_EVAL_FN: EvalFn = |_mc, _env, ctx, obj, args| {
    let (normalized_input_vals, receiver) = match obj {
        Value::Coll(coll) => Ok((coll.as_vec(), coll)),
        _ => Err(EvalError::UnexpectedValue(format!(
            "expected obj to be Value::Coll, got: {0:?}",
            obj
        ))),
    }?;
    let n = normalized_input_vals.len() as u32;
    ctx.add_per_item_jit_cost(20, 1, 10, n)?;
    let target_index_val = args
        .first()
        .cloned()
        .ok_or_else(|| EvalError::NotFound("updated: missing first arg (index)".to_string()))?;
    let update_val = args
        .get(1)
        .cloned()
        .ok_or_else(|| EvalError::NotFound("updated: missing second arg (update)".to_string()))?;

    let target_index_usize = target_index_val.clone().try_extract_into::<i32>()? as usize;
    let mut res = normalized_input_vals;

    match res.get_mut(target_index_usize) {
        Some(elem) => {
            *elem = update_val;
            // the receiver's class (`builder.fromArray` inside the receiver)
            Ok(Value::Coll(receiver.rebuild_like(res)?))
        }
        None => Err(EvalError::UnexpectedValue(format!(
            "updated: target index out of bounds, got: {:?}",
            target_index_val
        ))),
    }
};

pub(crate) static UPDATE_MANY_EVAL_FN: EvalFn =
    |_mc, _env, ctx, obj, args| {
        let (input_tpe, normalized_input_vals, receiver) = match obj {
            Value::Coll(coll) => Ok((coll.elem_tpe(), coll.as_vec(), coll)),
            _ => Err(EvalError::UnexpectedValue(format!(
                "expected obj to be Value::Coll, got: {0:?}",
                obj
            ))),
        }?;
        let n = normalized_input_vals.len() as u32;
        ctx.add_per_item_jit_cost(20, 2, 10, n)?;
        let indexes_arg = args.first().cloned().ok_or_else(|| {
            EvalError::NotFound("updated: missing first arg (indexes)".to_string())
        })?;
        let updates_arg = args.get(1).cloned().ok_or_else(|| {
            EvalError::NotFound("updated: missing second arg (updates)".to_string())
        })?;

        let updates_val = match updates_arg {
            Value::Coll(coll) => Ok(coll.as_vec()),
            _ => Err(EvalError::UnexpectedValue(format!(
                "expected first arg to be Value::Coll, got: {0:?}",
                updates_arg
            ))),
        }?;

        let indexes_usize = indexes_arg
            .try_extract_into::<Vec<i32>>()?
            .into_iter()
            .map(|i| i as usize)
            .collect::<Vec<usize>>();

        let inputs_len = normalized_input_vals.len();
        let indexes_len = indexes_usize.len();
        let updates_len = updates_val.len();

        if indexes_len != updates_len {
            return Err(EvalError::UnexpectedValue(format!(
                "Collections should have same length but was: \
            {0:?} and {1:?}. \n Indexes: {2:?} \n Updates: {3:?}",
                indexes_len, updates_len, indexes_usize, updates_val
            )));
        };
        // sigmastate compares no item types (v6.0.6 `CollsOverArrays.scala:106-117`, `:436-450`):
        // each value is stored into a clone of the receiver's array, and the rebuild below
        // refuses the ones that array would not take (SANTA `Tuple.non_pair_value_ops` T23, U3,
        // `Coll.reference_store_ops`)
        let mut i = 0;
        let mut res = normalized_input_vals;

        while i < indexes_len {
            let pos = indexes_usize[i];
            if pos >= inputs_len {
                return Err(EvalError::UnexpectedValue(format!(
                    "updateMany index out of bounds, got: {0:?}",
                    pos
                )));
            }
            let update = updates_val[i].clone();
            match res.get_mut(pos) {
                Some(elem) => *elem = update,
                None => {
                    return Err(EvalError::UnexpectedValue(format!(
                        "updateMany index out of bounds, got: {0:?}",
                        pos
                    )))
                }
            }
            i += 1;
        }
        // the receiver's class (`builder.fromArray` inside the receiver), but a `PairOfCols`
        // rebuilds its sides with `fromArray` at an abstract type (v6.0.7 `CollsOverArrays.scala:436-450`;
        // SANTA `Coll.pair_shape_restored` Y8)
        Ok(Value::Coll(match receiver {
            CollValue::Pairs { .. } => CollValue::from_array(input_tpe, res)?,
            _ => receiver.rebuild_like(res)?,
        }))
    };

pub(crate) static REVERSE_EVAL_FN: EvalFn = |_mc, _env, ctx, obj, _args| {
    let Value::Coll(coll) = obj else {
        return Err(EvalError::UnexpectedValue(format!(
            "Reverse: expected Coll, found {obj:?}"
        )));
    };
    // Scala `reverse_eval` charges `Append.costKind` = PerItemCost(20, 2, 100)
    // over the receiver `xs.length` via `addSeqCost` (methods.scala ~1135;
    // transformers.scala:74). A flat 20 omitted the per-chunk term (JVM charges
    // 20+2 = 22 for a non-empty single chunk) -> 2 undercharge.
    ctx.add_per_item_jit_cost(20, 2, 100, coll.len() as u32)?;
    // the receiver's class (`builder.fromArray` inside the receiver): a generic collection of
    // pairs comes back a `PairOfCols` (SANTA `Coll.pair_shape_restored` Y4)
    let reversed: Vec<Value> = coll.as_vec().into_iter().rev().collect();
    Ok(Value::Coll(coll.rebuild_like(reversed)?))
};

pub(crate) static STARTS_WITH_EVAL_FN: EvalFn = |_mc, _env, ctx, obj, args| {
    let Value::Coll(coll) = obj else {
        return Err(EvalError::UnexpectedValue(format!(
            "endsWith: expected Coll, found {obj:?}"
        )));
    };
    // Scala `startsWith_eval` charges `Zip_CostKind` = PerItemCost(10, 1, 10)
    // over the receiver `xs.length` via `addSeqCost` (methods.scala ~1155).
    // A flat 20 overcharged by 9 for n<=10 (JVM charges 10+1 = 11).
    ctx.add_per_item_jit_cost(10, 1, 10, coll.len() as u32)?;
    let Some(Value::Coll(prefix)) = args.first() else {
        return Err(EvalError::UnexpectedValue(format!(
            "startsWith: expected Coll argument, found {:?}",
            args.first(),
        )));
    };
    // the receiver's class decides; no item-type check (SANTA
    // `Coll.prefix_suffix_runtime_item_type`)
    Ok(Value::from(super::coll_shape::starts_with(&coll, prefix)?))
};

pub(crate) static ENDS_WITH_EVAL_FN: EvalFn = |_mc, _env, ctx, obj, args| {
    let Value::Coll(coll) = obj else {
        return Err(EvalError::UnexpectedValue(format!(
            "endsWith: expected Coll, found {obj:?}"
        )));
    };
    // Scala `endsWith_eval` charges `Zip_CostKind` = PerItemCost(10, 1, 10)
    // over the receiver `xs.length` via `addSeqCost` (methods.scala ~1175).
    // A flat 20 overcharged by 9 for n<=10 (JVM charges 10+1 = 11).
    ctx.add_per_item_jit_cost(10, 1, 10, coll.len() as u32)?;
    let Some(Value::Coll(suffix)) = args.first() else {
        return Err(EvalError::UnexpectedValue(format!(
            "endsWith: expected Coll argument, found {:?}",
            args.first(),
        )));
    };
    // the receiver's class decides; no item-type check (SANTA
    // `Coll.prefix_suffix_runtime_item_type`)
    Ok(Value::from(super::coll_shape::ends_with(&coll, suffix)?))
};

pub(crate) static GET_EVAL_FN: EvalFn = |_mc, _env, ctx, obj, args| {
    ctx.add_jit_cost(30)?;
    let Value::Coll(coll) = obj else {
        return Err(EvalError::UnexpectedValue(format!(
            "get: expected Coll, found {obj:?}"
        )));
    };
    let index = args
        .first()
        .cloned()
        .ok_or_else(|| EvalError::UnexpectedValue("Get: index argument not found".into()))?
        .try_extract_into::<i32>()?;
    Ok(Value::Opt(
        index
            .try_into()
            .ok()
            .and_then(|index| coll.get_val(index))
            .map(Box::new),
    ))
};

/// The receiver as sigmastate's `Coll` methods see it: a collection, which a tuple of other
/// than two items is (a `Coll[Any]`, `Tuple.value`, v6.0.7 `values.scala:818-822`). Anything
/// else, a pair (a `Tuple2`) included, is not a `Coll`, and the reflective call fails.
fn coll_receiver<'ctx>(obj: Value<'ctx>, method: &str) -> Result<CollValue<'ctx>, EvalError> {
    match obj {
        Value::Coll(coll) => Ok(coll),
        _ => Err(EvalError::UnexpectedValue(format!(
            "{method}: expected Coll, found {obj:?}"
        ))),
    }
}

/// `Coll.size` (id 1): sigmastate calls `Coll.size` reflectively at `SizeOf`'s fixed cost
/// (`invokeFixed`, v6.0.7 `methods.scala:821`)
pub(crate) static SIZE_EVAL_FN: EvalFn = |_mc, _env, ctx, obj, _args| {
    ctx.add_jit_cost(14)?;
    let coll = coll_receiver(obj, "size")?;
    Ok(Value::Int(coll.len() as i32))
};

/// `Coll.apply` (id 10): sigmastate calls `Coll.apply` reflectively at `ByIndex`'s fixed cost
/// (`methods.scala:940`); out of range it throws
pub(crate) static APPLY_EVAL_FN: EvalFn = |_mc, _env, ctx, obj, args| {
    ctx.add_jit_cost(30)?;
    let coll = coll_receiver(obj, "apply")?;
    let index = args
        .first()
        .cloned()
        .ok_or_else(|| EvalError::UnexpectedValue("apply: index argument not found".into()))?
        .try_extract_into::<i32>()?;
    item_at(&coll, index).ok_or_else(|| {
        EvalError::Misc(format!(
            "apply: index {index} out of bounds for collection size {}",
            coll.len()
        ))
    })
};

/// `Coll.getOrElse` (id 2), as sigmastate's `getOrElse_eval` (`methods.scala:839-844`):
/// `ByIndex`'s cost, then the default, already evaluated with every other argument, is checked
/// against the default ARGUMENT's declared type whatever the index (`mc.args.last.tpe`: the
/// method's domain stays generic when the call's types don't unify), then the item or the
/// default
pub(crate) static GET_OR_ELSE_EVAL_FN: EvalFn = |mc, _env, ctx, obj, args| {
    ctx.add_jit_cost(30)?;
    let coll = coll_receiver(obj, "getOrElse")?;
    let [index, default]: [Value; 2] = args.try_into().map_err(|_| {
        EvalError::UnexpectedValue("getOrElse: expected an index and a default".into())
    })?;
    let default_tpe = mc
        .args
        .get(1)
        .ok_or_else(|| EvalError::UnexpectedValue("getOrElse: default argument not found".into()))?
        .tpe();
    crate::eval::check_value_of_type(&default_tpe, &default, ctx.tree_version())?;
    // then the receiver's class decides whether it takes the default
    super::coll_shape::get_or_else(&coll, index.try_extract_into::<i32>()?, default)
};

/// `Coll.map` (id 3), as sigmastate's `map_eval` (`methods.scala:862-867`): the result's item
/// type is the call's, `mc.tpe` (when the call's types didn't unify it still holds a type
/// variable, which [`map_coll`] refuses before charging, as `stypeToRType` throws)
pub(crate) static MAP_EVAL_FN: EvalFn = |mc, _env, ctx, obj, args| {
    let SType::SColl(item_tpe) = &*mc.tpe().t_range else {
        return Err(EvalError::UnexpectedValue(format!(
            "map: expected a collection range, found {:?}",
            mc.tpe().t_range
        )));
    };
    let mapper = args
        .first()
        .ok_or_else(|| EvalError::UnexpectedValue("map: mapper argument not found".into()))?;
    map_coll(
        ctx,
        Value::Coll(coll_receiver(obj, "map")?),
        mapper,
        (**item_tpe).clone(),
    )
};

#[allow(clippy::unwrap_used)]
#[cfg(test)]
#[cfg(feature = "arbitrary")]
mod tests {
    use alloc::sync::Arc;

    use alloc::vec::Vec;
    use ergotree_ir::mir::constant::Constant;
    use ergotree_ir::mir::constant::Literal;
    use ergotree_ir::mir::expr::Expr;
    use ergotree_ir::mir::func_value::FuncArg;
    use ergotree_ir::mir::func_value::FuncValue;
    use ergotree_ir::mir::method_call::MethodCall;
    use ergotree_ir::mir::val_use::ValUse;
    use ergotree_ir::mir::value::CollKind;
    use ergotree_ir::types::scoll;
    use ergotree_ir::types::stype::SType;
    use ergotree_ir::types::stype_param::STypeVar;

    use crate::eval::test_util::{eval_out_wo_ctx, try_eval_out_wo_ctx};

    #[test]
    fn eval_index_of() {
        let index_of_expr = |coll: Vec<i64>, elem: i64, from: i32| -> Expr {
            MethodCall::new(
                coll.into(),
                scoll::INDEX_OF_METHOD.clone().with_concrete_types(
                    &[(STypeVar::t(), SType::SLong)].iter().cloned().collect(),
                ),
                vec![elem.into(), from.into()],
            )
            .unwrap()
            .into()
        };
        let res = eval_out_wo_ctx::<i32>(&index_of_expr(vec![1i64, 2i64], 2, 0));
        assert_eq!(res, 1);
        // Test searching in array starting from 1st index
        let res = eval_out_wo_ctx::<i32>(&index_of_expr(vec![1i64, 2i64], 2, 1));
        assert_eq!(res, 1);

        // Test searching in array starting from 1st index
        let res = eval_out_wo_ctx::<i32>(&index_of_expr(vec![1i64, 2i64], 2, 1));
        assert_eq!(res, 1);
        // Test searching in array starting from index greater than array length
        let res = eval_out_wo_ctx::<i32>(&index_of_expr(vec![1i64, 2i64], 2, 10000));
        assert_eq!(res, -1);
        // Test element that doesn't exist
        let res = eval_out_wo_ctx::<i32>(&index_of_expr(vec![1i64, 2i64], 3, 0));
        assert_eq!(res, -1);
    }

    #[test]
    fn eval_index_of_default() {
        let coll_const: Constant = vec![1i64, 2i64].into();
        let expr: Expr = MethodCall::new(
            coll_const.into(),
            scoll::INDEX_OF_METHOD
                .clone()
                .with_concrete_types(&[(STypeVar::t(), SType::SLong)].iter().cloned().collect()),
            vec![3i64.into(), 0i32.into()],
        )
        .unwrap()
        .into();
        let res = eval_out_wo_ctx::<i32>(&expr);
        assert_eq!(res, -1);
    }

    #[test]
    fn index_of_cost_scales_with_iterations() {
        use crate::eval::test_util::try_eval_out;
        use ergotree_ir::chain::context::Context;
        use sigma_test_util::force_any_val;

        let coll: Vec<i64> = (1..=16).collect();
        let index_of = |target: i64| -> Expr {
            MethodCall::new(
                coll.clone().into(),
                scoll::INDEX_OF_METHOD.clone().with_concrete_types(
                    &[(STypeVar::t(), SType::SLong)].iter().cloned().collect(),
                ),
                vec![target.into(), 0i32.into()],
            )
            .unwrap()
            .into()
        };
        let cost_of = |target: i64| -> u64 {
            let ctx = force_any_val::<Context>();
            let before = ctx.jit_cost_value();
            let _: i32 = try_eval_out(&index_of(target), &ctx).unwrap();
            ctx.jit_cost_value() - before
        };
        // indexOf charges, per iteration performed (Scala's `i - start`), both
        // the loop's PerItemCost(20, 10, 2) and the element-type equality cost
        // (EQ_PRIM_COST=3 for Long, via eq_with_cost). Finding 1 at index 0 is
        // 1 iteration; finding 16 at index 15 is 16. The rest of the tree is
        // identical, so the delta isolates the per-iteration scaling:
        //   PerItemCost: (20+10*8) - (20+10*1) = 70
        //   element eq:  3 * (16 - 1)          = 45   (EQ_PRIM_COST per compare)
        //   total                              = 115
        // Pre-iterations-fix the PerItemCost delta was 0; pre-eq-fix (bare ==)
        // the eq delta was 0.
        let delta = cost_of(16) - cost_of(1);
        assert_eq!(
            delta, 115,
            "indexOf cost must scale with iterations: PerItemCost 70 + \
             per-comparison EQ_PRIM_COST 3*15 = 45 -> 115; got {}",
            delta,
        );
    }

    #[test]
    fn index_of_charges_element_eq_cost() {
        use crate::eval::test_util::try_eval_out;
        use ergotree_ir::bigint256::BigInt256;
        use ergotree_ir::chain::context::Context;
        use sigma_test_util::force_any_val;

        // B3: each indexOf comparison is charged the element type's equality
        // cost via eq_with_cost. Holding the iteration count fixed (target at
        // index 0 => 1 comparison) and varying only the element type isolates
        // that charge: everything else (the single PerItemCost iteration, the
        // coll and target Constants, the method-call tree) is type-independent,
        // so the cost difference is exactly EQ_BIGINT_COST(5) - EQ_PRIM_COST(3)
        // = 2. Pre-fix (bare ==, uncharged) the difference was 0.
        let cost_of = |coll: Constant, target: Constant, t: SType| -> u64 {
            let expr: Expr = MethodCall::new(
                coll.into(),
                scoll::INDEX_OF_METHOD
                    .clone()
                    .with_concrete_types(&[(STypeVar::t(), t)].iter().cloned().collect()),
                vec![target.into(), 0i32.into()],
            )
            .unwrap()
            .into();
            let ctx = force_any_val::<Context>();
            let before = ctx.jit_cost_value();
            let _: i32 = try_eval_out(&expr, &ctx).unwrap();
            ctx.jit_cost_value() - before
        };
        let long_cost = cost_of(vec![5i64].into(), 5i64.into(), SType::SLong);
        let bigint = BigInt256::from(5i64);
        let bigint_coll = Constant {
            tpe: SType::SColl(Arc::new(SType::SBigInt)),
            v: Literal::Coll(CollKind::WrappedColl {
                items: Arc::from(vec![Constant::from(bigint).v]),
                elem_tpe: SType::SBigInt,
            }),
        };
        let bigint_cost = cost_of(bigint_coll, bigint.into(), SType::SBigInt);
        assert_eq!(
            bigint_cost - long_cost,
            2,
            "indexOf must charge the element-type eq cost per comparison \
             (EQ_BIGINT_COST 5 - EQ_PRIM_COST 3 = 2); got {}",
            bigint_cost - long_cost,
        );
    }

    #[test]
    fn eval_flatmap() {
        let coll_const = Constant {
            tpe: SType::SColl(Arc::new(SType::SColl(Arc::new(SType::SLong)))),
            v: Literal::Coll(CollKind::WrappedColl {
                items: Arc::new([vec![4i64, 5i64].into(), vec![3i64].into()]),
                elem_tpe: SType::SColl(Arc::new(SType::SLong)),
            }),
        };
        let body: Expr = MethodCall::new(
            ValUse {
                val_id: 1.into(),
                tpe: SType::SColl(Arc::new(SType::SLong)),
            }
            .into(),
            scoll::INDICES_METHOD
                .clone()
                .with_concrete_types(&[(STypeVar::t(), SType::SLong)].iter().cloned().collect()),
            vec![],
        )
        .unwrap()
        .into();
        let expr: Expr = MethodCall::new(
            coll_const.into(),
            scoll::FLATMAP_METHOD.clone().with_concrete_types(
                &[
                    (STypeVar::iv(), SType::SColl(Arc::new(SType::SLong))),
                    (STypeVar::ov(), SType::SInt),
                ]
                .iter()
                .cloned()
                .collect(),
            ),
            vec![FuncValue::new(
                vec![FuncArg {
                    idx: 1.into(),
                    tpe: SType::SColl(Arc::new(SType::SLong)),
                }],
                body,
            )
            .into()],
        )
        .unwrap()
        .into();
        let res = eval_out_wo_ctx::<Vec<i32>>(&expr);
        assert_eq!(res, vec![0, 1, 0]);
    }

    #[test]
    fn flatmap_charges_add_to_env_per_input() {
        use crate::eval::test_util::try_eval_out;
        use ergotree_ir::chain::context::Context;
        use sigma_test_util::force_any_val;

        // `inner.flatMap(x => x.reverse)` over a Coll[Coll[Long]].
        let flatmap_reverse = |inner: Vec<Vec<i64>>| -> Expr {
            let items: Arc<[Literal]> = Arc::from(
                inner
                    .into_iter()
                    .map(|v| Constant::from(v).v)
                    .collect::<Vec<_>>(),
            );
            let coll_const = Constant {
                tpe: SType::SColl(Arc::new(SType::SColl(Arc::new(SType::SLong)))),
                v: Literal::Coll(CollKind::WrappedColl {
                    items,
                    elem_tpe: SType::SColl(Arc::new(SType::SLong)),
                }),
            };
            let body: Expr = MethodCall::new(
                ValUse {
                    val_id: 1.into(),
                    tpe: SType::SColl(Arc::new(SType::SLong)),
                }
                .into(),
                scoll::REVERSE_METHOD.clone().with_concrete_types(
                    &[(STypeVar::t(), SType::SLong)].iter().cloned().collect(),
                ),
                vec![],
            )
            .unwrap()
            .into();
            MethodCall::new(
                coll_const.into(),
                scoll::FLATMAP_METHOD.clone().with_concrete_types(
                    &[
                        (STypeVar::iv(), SType::SColl(Arc::new(SType::SLong))),
                        (STypeVar::ov(), SType::SLong),
                    ]
                    .iter()
                    .cloned()
                    .collect(),
                ),
                vec![FuncValue::new(
                    vec![FuncArg {
                        idx: 1.into(),
                        tpe: SType::SColl(Arc::new(SType::SLong)),
                    }],
                    body,
                )
                .into()],
            )
            .unwrap()
            .into()
        };

        let cost_of = |expr: &Expr| -> u64 {
            let ctx = force_any_val::<Context>();
            let before = ctx.jit_cost_value();
            let _ = try_eval_out::<Vec<i64>>(expr, &ctx).unwrap();
            ctx.jit_cost_value() - before
        };

        // Standalone `[7].reverse` measures one lambda-body evaluation `r`:
        // inside flatMap the body's only difference is a ValUse vs Constant
        // receiver, and both are Fixed(5), so the per-input body cost equals
        // this.
        let reverse_standalone: Expr = MethodCall::new(
            Constant::from(vec![7i64]).into(),
            scoll::REVERSE_METHOD
                .clone()
                .with_concrete_types(&[(STypeVar::t(), SType::SLong)].iter().cloned().collect()),
            vec![],
        )
        .unwrap()
        .into();
        let r = cost_of(&reverse_standalone);

        // 1 -> 2 input elements adds exactly one body eval (`r`) + one
        // ADD_TO_ENV_COST(5). Output length 1 -> 2 stays in the first chunk of
        // PerItemCost(60,10,8), so that charge is unchanged. Pre-fix the delta
        // was just `r`; the +5 is the per-input ADD_TO_ENV this fix adds.
        let one = cost_of(&flatmap_reverse(vec![vec![7]]));
        let two = cost_of(&flatmap_reverse(vec![vec![7], vec![8]]));
        assert_eq!(
            two - one,
            r + 5,
            "flatMap must charge ADD_TO_ENV_COST(5) per input element: delta {} \
             should equal one body eval {} + 5",
            two - one,
            r,
        );
    }

    #[test]
    fn eval_zip_empty() {
        // Both empty
        let empty_coll_const: Constant = Vec::<i64>::new().into();
        let empty_input: Constant = Vec::<bool>::new().into();
        let expr: Expr = MethodCall::new(
            empty_coll_const.into(),
            scoll::ZIP_METHOD.clone().with_concrete_types(
                &[
                    (STypeVar::t(), SType::SLong),
                    (STypeVar::iv(), SType::SBoolean),
                ]
                .iter()
                .cloned()
                .collect(),
            ),
            vec![empty_input.into()],
        )
        .unwrap()
        .into();
        let res = eval_out_wo_ctx::<Vec<(i64, bool)>>(&expr);
        assert_eq!(res, Vec::<(i64, bool)>::new());

        // Only obj empty
        let empty_coll_const: Constant = Vec::<i64>::new().into();
        let input: Constant = vec![true, false].into();
        let expr: Expr = MethodCall::new(
            empty_coll_const.into(),
            scoll::ZIP_METHOD.clone().with_concrete_types(
                &[
                    (STypeVar::t(), SType::SLong),
                    (STypeVar::iv(), SType::SBoolean),
                ]
                .iter()
                .cloned()
                .collect(),
            ),
            vec![input.into()],
        )
        .unwrap()
        .into();
        let res = eval_out_wo_ctx::<Vec<(i64, bool)>>(&expr);
        assert_eq!(res, Vec::<(i64, bool)>::new());

        // Only input empty
        let coll_const: Constant = vec![1i64, 2i64].into();
        let empty_input: Constant = Vec::<bool>::new().into();
        let expr: Expr = MethodCall::new(
            coll_const.into(),
            scoll::ZIP_METHOD.clone().with_concrete_types(
                &[
                    (STypeVar::t(), SType::SLong),
                    (STypeVar::iv(), SType::SBoolean),
                ]
                .iter()
                .cloned()
                .collect(),
            ),
            vec![empty_input.into()],
        )
        .unwrap()
        .into();
        let res = eval_out_wo_ctx::<Vec<(i64, bool)>>(&expr);
        assert_eq!(res, Vec::<(i64, bool)>::new());
    }

    #[test]
    fn eval_zip_same_length() {
        let coll_const: Constant = vec![1i64, 2i64].into();
        let input: Constant = vec![true, false].into();
        let expr: Expr = MethodCall::new(
            coll_const.into(),
            scoll::ZIP_METHOD.clone().with_concrete_types(
                &[
                    (STypeVar::t(), SType::SLong),
                    (STypeVar::iv(), SType::SBoolean),
                ]
                .iter()
                .cloned()
                .collect(),
            ),
            vec![input.into()],
        )
        .unwrap()
        .into();
        let res = eval_out_wo_ctx::<Vec<(i64, bool)>>(&expr);
        assert_eq!(res, vec![(1i64, true), (2, false)]);
    }

    #[test]
    fn eval_zip_different_lengths() {
        // Input shorter than obj
        let coll_const: Constant = vec![1i64, 2i64].into();
        let short_input: Constant = vec![true].into();
        let expr: Expr = MethodCall::new(
            coll_const.into(),
            scoll::ZIP_METHOD.clone().with_concrete_types(
                &[
                    (STypeVar::t(), SType::SLong),
                    (STypeVar::iv(), SType::SBoolean),
                ]
                .iter()
                .cloned()
                .collect(),
            ),
            vec![short_input.into()],
        )
        .unwrap()
        .into();
        let res = eval_out_wo_ctx::<Vec<(i64, bool)>>(&expr);
        assert_eq!(res, vec![(1i64, true)]);

        // Input longer than obj
        let coll_const: Constant = vec![1i64, 2i64].into();
        let long_input: Constant = vec![true, false, true].into();
        let expr: Expr = MethodCall::new(
            coll_const.into(),
            scoll::ZIP_METHOD.clone().with_concrete_types(
                &[
                    (STypeVar::t(), SType::SLong),
                    (STypeVar::iv(), SType::SBoolean),
                ]
                .iter()
                .cloned()
                .collect(),
            ),
            vec![long_input.into()],
        )
        .unwrap()
        .into();
        let res = eval_out_wo_ctx::<Vec<(i64, bool)>>(&expr);
        assert_eq!(res, vec![(1i64, true), (2, false)]);
    }

    #[test]
    fn eval_indices() {
        let coll_const: Constant = vec![1i64, 2i64, 3i64].into();
        let expr: Expr = MethodCall::new(
            coll_const.into(),
            scoll::INDICES_METHOD
                .clone()
                .with_concrete_types(&[(STypeVar::t(), SType::SLong)].iter().cloned().collect()),
            vec![],
        )
        .unwrap()
        .into();
        let res = eval_out_wo_ctx::<Vec<i32>>(&expr);
        assert_eq!(res, vec![0i32, 1i32, 2i32]);
    }

    #[test]
    fn eval_indices_empty_coll() {
        let coll_const: Constant = Vec::<i64>::new().into();
        let expr: Expr = MethodCall::new(
            coll_const.into(),
            scoll::INDICES_METHOD
                .clone()
                .with_concrete_types(&[(STypeVar::t(), SType::SLong)].iter().cloned().collect()),
            vec![],
        )
        .unwrap()
        .into();
        let res = eval_out_wo_ctx::<Vec<i32>>(&expr);
        assert_eq!(res, Vec::<i32>::new());
    }

    #[test]
    fn eval_patch() {
        let coll_const: Constant = vec![1i64, 5i64, 5i64].into();
        let patch_input: Vec<i64> = vec![2i64, 3i64];

        let expr: Expr = MethodCall::new(
            coll_const.into(),
            scoll::PATCH_METHOD
                .clone()
                .with_concrete_types(&[(STypeVar::t(), SType::SLong)].iter().cloned().collect()),
            vec![1i32.into(), patch_input.into(), 2i32.into()],
        )
        .unwrap()
        .into();
        let res = eval_out_wo_ctx::<Vec<i64>>(&expr);
        assert_eq!(res, vec![1i64, 2i64, 3i64]);
    }

    #[test]
    fn eval_patch_addition() {
        let coll_const: Constant = vec![1i64, 2i64, 4i64, 5i64].into();
        let patch_input: Vec<i64> = vec![3i64];

        let expr: Expr = MethodCall::new(
            coll_const.into(),
            scoll::PATCH_METHOD
                .clone()
                .with_concrete_types(&[(STypeVar::t(), SType::SLong)].iter().cloned().collect()),
            vec![2i32.into(), patch_input.into(), 0i32.into()],
        )
        .unwrap()
        .into();
        let res = eval_out_wo_ctx::<Vec<i64>>(&expr);
        assert_eq!(res, vec![1i64, 2i64, 3i64, 4i64, 5i64]);
    }

    #[test]
    fn eval_patch_subtraction() {
        let coll_const: Constant = vec![1i64, 2i64, 5i64, 5i64, 4i64, 5i64].into();
        let patch_input: Vec<i64> = vec![3i64];

        let expr: Expr = MethodCall::new(
            coll_const.into(),
            scoll::PATCH_METHOD
                .clone()
                .with_concrete_types(&[(STypeVar::t(), SType::SLong)].iter().cloned().collect()),
            vec![2i32.into(), patch_input.into(), 2i32.into()],
        )
        .unwrap()
        .into();
        let res = eval_out_wo_ctx::<Vec<i64>>(&expr);
        assert_eq!(res, vec![1i64, 2i64, 3i64, 4i64, 5i64]);
    }

    #[test]
    fn eval_patch_index_oob() {
        let coll_const: Constant = vec![1i64, 2i64, 3i64].into();
        let patch_input: Vec<i64> = vec![4i64, 5i64];

        let expr: Expr = MethodCall::new(
            coll_const.into(),
            scoll::PATCH_METHOD
                .clone()
                .with_concrete_types(&[(STypeVar::t(), SType::SLong)].iter().cloned().collect()),
            vec![9i32.into(), patch_input.into(), 9i32.into()],
        )
        .unwrap()
        .into();
        let res = eval_out_wo_ctx::<Vec<i64>>(&expr);
        assert_eq!(res, vec![1i64, 2i64, 3i64, 4i64, 5i64]);
    }

    #[test]
    fn eval_patch_index_negative() {
        let coll_const: Constant = vec![1i64, 2i64, 3i64].into();
        let patch_input: Vec<i64> = vec![4i64, 5i64];

        let expr: Expr = MethodCall::new(
            coll_const.into(),
            scoll::PATCH_METHOD
                .clone()
                .with_concrete_types(&[(STypeVar::t(), SType::SLong)].iter().cloned().collect()),
            vec![(-1i32).into(), patch_input.into(), (-1i32).into()],
        )
        .unwrap()
        .into();
        let res = eval_out_wo_ctx::<Vec<i64>>(&expr);
        assert_eq!(res, vec![4i64, 5i64, 1i64, 2i64, 3i64]);
    }

    #[test]
    fn eval_update() {
        let coll_const: Constant = vec![1i64, 2i64, 3i64].into();
        let expr: Expr = MethodCall::new(
            coll_const.into(),
            scoll::UPDATED_METHOD
                .clone()
                .with_concrete_types(&[(STypeVar::t(), SType::SLong)].iter().cloned().collect()),
            vec![1i32.into(), 5i64.into()],
        )
        .unwrap()
        .into();
        let res = eval_out_wo_ctx::<Vec<i64>>(&expr);
        assert_eq!(res, vec![1i64, 5i64, 3i64]);
    }

    #[test]
    fn eval_update_oob() {
        let coll_const: Constant = vec![1i64, 2i64, 3i64].into();
        let expr: Expr = MethodCall::new(
            coll_const.into(),
            scoll::UPDATED_METHOD
                .clone()
                .with_concrete_types(&[(STypeVar::t(), SType::SLong)].iter().cloned().collect()),
            vec![5i32.into(), 5i64.into()],
        )
        .unwrap()
        .into();
        assert!(try_eval_out_wo_ctx::<Vec<i64>>(&expr).is_err());
    }

    #[test]
    fn eval_update_many() {
        let coll_const: Constant = vec![1i64, 2i64, 3i64].into();

        let indexes_input: Vec<i32> = vec![1i32, 2i32];
        let updates_input: Vec<i64> = vec![5i64, 6i64];
        let expr: Expr = MethodCall::new(
            coll_const.into(),
            scoll::UPDATE_MANY_METHOD
                .clone()
                .with_concrete_types(&[(STypeVar::t(), SType::SLong)].iter().cloned().collect()),
            vec![indexes_input.into(), updates_input.into()],
        )
        .unwrap()
        .into();
        let res = eval_out_wo_ctx::<Vec<i64>>(&expr);
        assert_eq!(res, vec![1i64, 5i64, 6i64]);
    }

    #[test]
    fn eval_update_many_index_oob() {
        let coll_const: Constant = vec![1i64, 2i64, 3i64].into();

        let indexes_input: Vec<i32> = vec![1i32, 5i32];
        let updates_input: Vec<i64> = vec![5i64, 6i64];
        let expr: Expr = MethodCall::new(
            coll_const.into(),
            scoll::UPDATE_MANY_METHOD
                .clone()
                .with_concrete_types(&[(STypeVar::t(), SType::SLong)].iter().cloned().collect()),
            vec![indexes_input.into(), updates_input.into()],
        )
        .unwrap()
        .into();
        assert!(try_eval_out_wo_ctx::<Vec<i64>>(&expr).is_err());
    }

    #[test]
    fn eval_update_many_len_mismatch() {
        let coll_const: Constant = vec![1i64, 2i64, 3i64].into();

        let indexes_input: Vec<i32> = vec![1i32];
        let updates_input: Vec<i64> = vec![5i64, 6i64];
        let expr: Expr = MethodCall::new(
            coll_const.into(),
            scoll::UPDATE_MANY_METHOD
                .clone()
                .with_concrete_types(&[(STypeVar::t(), SType::SLong)].iter().cloned().collect()),
            vec![indexes_input.into(), updates_input.into()],
        )
        .unwrap()
        .into();
        assert!(try_eval_out_wo_ctx::<Vec<i64>>(&expr).is_err());
    }

    #[test]
    fn eval_reverse() {
        let arr = vec![1i64, 2i64, 3i64];
        let coll_const: Constant = arr.clone().into();
        let expr: Expr = MethodCall::new(
            coll_const.into(),
            scoll::REVERSE_METHOD
                .clone()
                .with_concrete_types(&[(STypeVar::t(), SType::SLong)].into_iter().collect()),
            vec![],
        )
        .unwrap()
        .into();
        assert_eq!(
            eval_out_wo_ctx::<Vec<i64>>(&expr),
            arr.into_iter().rev().collect::<Vec<_>>(),
        );
        let arr = vec![1i8, 2i8, 3i8];
        let coll_const: Constant = arr.clone().into();
        let expr: Expr = MethodCall::new(
            coll_const.into(),
            scoll::REVERSE_METHOD
                .clone()
                .with_concrete_types(&[(STypeVar::t(), SType::SByte)].into_iter().collect()),
            vec![],
        )
        .unwrap()
        .into();
        assert_eq!(
            eval_out_wo_ctx::<Vec<i8>>(&expr),
            arr.into_iter().rev().collect::<Vec<_>>(),
        );
    }
    #[test]
    fn eval_starts_with() {
        fn starts_with(input: Vec<i64>, prefix: Vec<i64>) -> bool {
            let mc: Expr = MethodCall::new(
                Constant::from(input).into(),
                scoll::STARTS_WITH_METHOD
                    .clone()
                    .with_concrete_types(&[(STypeVar::t(), SType::SLong)].into_iter().collect()),
                vec![Constant::from(prefix).into()],
            )
            .unwrap()
            .into();
            eval_out_wo_ctx(&mc)
        }
        fn starts_with_byte_array(input: Vec<i8>, prefix: Vec<i8>) -> bool {
            let mc: Expr = MethodCall::new(
                Constant::from(input).into(),
                scoll::STARTS_WITH_METHOD
                    .clone()
                    .with_concrete_types(&[(STypeVar::t(), SType::SByte)].into_iter().collect()),
                vec![Constant::from(prefix).into()],
            )
            .unwrap()
            .into();
            eval_out_wo_ctx(&mc)
        }
        assert!(starts_with(vec![1, 2, 3], vec![1, 2]));
        assert!(starts_with(vec![1, 2, 3], vec![1, 2, 3]));
        assert!(!starts_with(vec![1, 2, 3], vec![1, 2, 4]));
        assert!(!starts_with(vec![1, 2, 3], vec![1, 2, 3, 4]));
        assert!(starts_with(vec![], vec![]));
        assert!(starts_with(vec![1, 2], vec![]));
        assert!(starts_with_byte_array(vec![1, 2, 3], vec![1, 2]));
        assert!(starts_with_byte_array(vec![1, 2, 3], vec![1, 2, 3]));
        assert!(!starts_with_byte_array(vec![1, 2, 3], vec![1, 2, 4]));
        assert!(!starts_with_byte_array(vec![1, 2, 3], vec![1, 2, 3, 4]));
        assert!(starts_with_byte_array(vec![], vec![]));
        assert!(starts_with_byte_array(vec![1, 2], vec![]));
    }
    #[test]
    fn eval_ends_with() {
        fn ends_with(input: Vec<i64>, suffix: Vec<i64>) -> bool {
            let mc: Expr = MethodCall::new(
                Constant::from(input).into(),
                scoll::ENDS_WITH_METHOD
                    .clone()
                    .with_concrete_types(&[(STypeVar::t(), SType::SLong)].into_iter().collect()),
                vec![Constant::from(suffix).into()],
            )
            .unwrap()
            .into();
            eval_out_wo_ctx(&mc)
        }
        fn ends_with_byte_array(input: Vec<i8>, suffix: Vec<i8>) -> bool {
            let mc: Expr = MethodCall::new(
                Constant::from(input).into(),
                scoll::ENDS_WITH_METHOD
                    .clone()
                    .with_concrete_types(&[(STypeVar::t(), SType::SByte)].into_iter().collect()),
                vec![Constant::from(suffix).into()],
            )
            .unwrap()
            .into();
            eval_out_wo_ctx(&mc)
        }
        assert!(!ends_with(vec![1, 2, 3], vec![1, 2]));
        assert!(ends_with(vec![1, 2, 3], vec![2, 3]));
        assert!(!ends_with(vec![1, 2, 3], vec![2, 3, 4]));
        assert!(ends_with(vec![1, 2, 3], vec![1, 2, 3]));
        assert!(ends_with(vec![], vec![]));
        assert!(ends_with(vec![1, 2], vec![]));
        assert!(!ends_with_byte_array(vec![1, 2, 3], vec![1, 2]));
        assert!(ends_with_byte_array(vec![1, 2, 3], vec![2, 3]));
        assert!(!ends_with_byte_array(vec![1, 2, 3], vec![2, 3, 4]));
        assert!(ends_with_byte_array(vec![1, 2, 3], vec![1, 2, 3]));
        assert!(ends_with_byte_array(vec![], vec![]));
        assert!(ends_with_byte_array(vec![1, 2], vec![]));
    }
    #[test]
    fn eval_get() {
        fn get(input: Vec<i64>, index: i32) -> Option<i64> {
            let mc: Expr = MethodCall::new(
                Constant::from(input).into(),
                scoll::GET_METHOD
                    .clone()
                    .with_concrete_types(&[(STypeVar::t(), SType::SLong)].into_iter().collect()),
                vec![Constant::from(index).into()],
            )
            .unwrap()
            .into();
            eval_out_wo_ctx(&mc)
        }
        assert_eq!(get(vec![1, 2], 0), Some(1));
        assert_eq!(get(vec![1, 2], 1), Some(2));
        assert_eq!(get(vec![1, 2], -1), None);
        assert_eq!(get(vec![1, 2], 2), None);
        assert_eq!(get(vec![], 0), None);
    }

    #[test]
    fn scoll_methods_charge_scala_costkinds() {
        use crate::eval::test_util::eval_out;
        use ergotree_ir::chain::context::Context;
        use sigma_test_util::force_any_val;

        // Isolate a method's costKind exactly as snumeric's
        // `numeric_method_charges_costkind`: a MethodCall node charges Fixed(4)
        // (method_call.rs) and evaluates the receiver and each arg separately,
        // so `full_mc_cost - receiver_cost - arg_cost` leaves `4 + costKind`.
        //
        // reverse uses Append.costKind = PerItemCost(20, 2, 100); startsWith and
        // endsWith use Zip_CostKind = PerItemCost(10, 1, 10) -- both over the
        // receiver length `xs.length` (methods.scala 1124/1143/1163). For a
        // 3-element receiver each is one chunk: reverse 20+2 = 22, starts/ends
        // 10+1 = 11. Pre-fix all three charged a flat 20.
        let obj_expr: Expr = Constant::from(vec![1i64, 2i64, 3i64]).into();

        let ctx = force_any_val::<Context>();
        // Standalone receiver eval cost -- the same Constant is re-evaluated
        // inside each MethodCall below, so this delta is reused as the subtrahend.
        let mark = ctx.jit_cost_value();
        let _ = eval_out::<Vec<i64>>(&obj_expr, &ctx);
        let obj_cost = ctx.jit_cost_value() - mark;

        // reverse (no args): 4 + Append.costKind(20,2,100) over len 3 = 4 + 22
        let reverse_mc: Expr = MethodCall::new(
            obj_expr.clone(),
            scoll::REVERSE_METHOD
                .clone()
                .with_concrete_types(&[(STypeVar::t(), SType::SLong)].into_iter().collect()),
            vec![],
        )
        .unwrap()
        .into();
        let mark = ctx.jit_cost_value();
        let _ = eval_out::<Vec<i64>>(&reverse_mc, &ctx);
        assert_eq!(
            (ctx.jit_cost_value() - mark) - obj_cost,
            4 + 22,
            "reverse must charge MethodCall Fixed(4) + Append.costKind PerItemCost(20,2,100)",
        );

        // startsWith (1 arg): 4 + Zip_CostKind(10,1,10) over len 3 = 4 + 11
        let prefix_expr: Expr = Constant::from(vec![1i64, 2i64]).into();
        let mark = ctx.jit_cost_value();
        let _ = eval_out::<Vec<i64>>(&prefix_expr, &ctx);
        let prefix_cost = ctx.jit_cost_value() - mark;
        let starts_mc: Expr = MethodCall::new(
            obj_expr.clone(),
            scoll::STARTS_WITH_METHOD
                .clone()
                .with_concrete_types(&[(STypeVar::t(), SType::SLong)].into_iter().collect()),
            vec![prefix_expr],
        )
        .unwrap()
        .into();
        let mark = ctx.jit_cost_value();
        let _ = eval_out::<bool>(&starts_mc, &ctx);
        assert_eq!(
            (ctx.jit_cost_value() - mark) - obj_cost - prefix_cost,
            4 + 11,
            "startsWith must charge MethodCall Fixed(4) + Zip_CostKind PerItemCost(10,1,10)",
        );

        // endsWith (1 arg): 4 + Zip_CostKind(10,1,10) over len 3 = 4 + 11
        let suffix_expr: Expr = Constant::from(vec![2i64, 3i64]).into();
        let mark = ctx.jit_cost_value();
        let _ = eval_out::<Vec<i64>>(&suffix_expr, &ctx);
        let suffix_cost = ctx.jit_cost_value() - mark;
        let ends_mc: Expr = MethodCall::new(
            obj_expr.clone(),
            scoll::ENDS_WITH_METHOD
                .clone()
                .with_concrete_types(&[(STypeVar::t(), SType::SLong)].into_iter().collect()),
            vec![suffix_expr],
        )
        .unwrap()
        .into();
        let mark = ctx.jit_cost_value();
        let _ = eval_out::<bool>(&ends_mc, &ctx);
        assert_eq!(
            (ctx.jit_cost_value() - mark) - obj_cost - suffix_cost,
            4 + 11,
            "endsWith must charge MethodCall Fixed(4) + Zip_CostKind PerItemCost(10,1,10)",
        );
    }
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod flat_map_element_type_tests {
    //! sigmastate's `flatMap_eval` reads no element type (v6.0.7 `methods.scala`, `flatMap_eval`):
    //! the lambda checks each element against its argument's type (`values.scala:1074`), a shallow
    //! check, so a `Coll[Long]` element passes a `Coll[Int]` argument and an empty input checks
    //! nothing.
    use crate::eval::test_util::{eval_out_wo_ctx, try_eval_out_wo_ctx};
    use alloc::boxed::Box;
    use alloc::sync::Arc;
    use alloc::vec;
    use alloc::vec::Vec;
    use ergotree_ir::mir::expr::Expr;
    use ergotree_ir::mir::func_value::{FuncArg, FuncValue};
    use ergotree_ir::mir::if_op::If;
    use ergotree_ir::mir::method_call::MethodCall;
    use ergotree_ir::mir::val_use::ValUse;
    use ergotree_ir::types::scoll;
    use ergotree_ir::types::stype::SType;
    use ergotree_ir::types::stype_param::STypeVar;

    fn coll_of(t: SType) -> SType {
        SType::SColl(Arc::new(t))
    }

    /// `if (false) Coll(Coll(1)) else <other>`: typed `Coll[Coll[Int]]` (an `If` is typed by its
    /// true branch), holding whatever `other` holds
    fn int_colls_holding(other: Expr) -> Expr {
        If {
            condition: Box::new(Expr::Const(false.into())),
            true_branch: Box::new(vec![vec![1i32]].into()),
            false_branch: Box::new(other),
        }
        .into()
    }

    /// `input.flatMap({ (x: Coll[Int]) => x.indices })`, built as the parser builds it
    fn flat_map_indices(input: Expr) -> Expr {
        let arg = coll_of(SType::SInt);
        let x: Expr = ValUse {
            val_id: 1.into(),
            tpe: arg.clone(),
        }
        .into();
        let indices: Expr = MethodCall {
            obj: Box::new(x),
            method: scoll::INDICES_METHOD
                .clone()
                .with_concrete_types(&[(STypeVar::t(), SType::SInt)].iter().cloned().collect()),
            args: vec![],
            explicit_type_args: Default::default(),
        }
        .into();
        MethodCall {
            obj: Box::new(input),
            method: scoll::FLATMAP_METHOD.clone().with_concrete_types(
                &[(STypeVar::iv(), arg.clone()), (STypeVar::ov(), SType::SInt)]
                    .iter()
                    .cloned()
                    .collect(),
            ),
            args: vec![FuncValue::new(
                vec![FuncArg {
                    idx: 1.into(),
                    tpe: arg,
                }],
                indices,
            )
            .into()],
            explicit_type_args: Default::default(),
        }
        .into()
    }

    #[test]
    fn the_lambda_checks_each_element_as_the_jvm_closure_does() {
        // a `Coll[Long]` element passes the `Coll[Int]` argument: `Coll(2L).indices`
        let longs: Expr = vec![vec![2i64]].into();
        assert_eq!(
            eval_out_wo_ctx::<Vec<i32>>(&flat_map_indices(int_colls_holding(longs))),
            vec![0]
        );
        // a `Long` element does not
        let bare_long: Expr = vec![2i64].into();
        assert!(
            try_eval_out_wo_ctx::<Vec<i32>>(&flat_map_indices(int_colls_holding(bare_long)))
                .is_err()
        );
    }

    #[test]
    fn an_empty_input_of_another_item_type_is_not_checked() {
        let no_longs: Expr = Vec::<Vec<i64>>::new().into();
        assert_eq!(
            eval_out_wo_ctx::<Vec<i32>>(&flat_map_indices(int_colls_holding(no_longs))),
            Vec::<i32>::new()
        );
    }
}
