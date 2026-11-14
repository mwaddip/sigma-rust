use alloc::boxed::Box;
use alloc::string::ToString;
use alloc::sync::Arc;
use alloc::vec::Vec;
use ergotree_ir::ergo_tree::ErgoTreeVersion;

use bytes::Bytes;
use ergo_avltree_rust::authenticated_tree_ops::AuthenticatedTreeOps;
use ergo_avltree_rust::batch_avl_verifier::BatchAVLVerifier;
use ergo_avltree_rust::batch_node::AVLTree;
use ergo_avltree_rust::batch_node::Node;
use ergo_avltree_rust::batch_node::NodeHeader;
use ergo_avltree_rust::operation::KeyValue;
use ergo_avltree_rust::operation::Operation;
use ergotree_ir::mir::avl_tree_data::AvlTreeData;
use ergotree_ir::mir::avl_tree_data::AvlTreeFlags;
use ergotree_ir::mir::avl_tree_data::AvlTreeValue;
use ergotree_ir::mir::coll_value::CollValue;
use ergotree_ir::mir::constant::TryExtractInto;
use ergotree_ir::mir::value::Value;

use super::Context;
use super::EvalError;
use super::EvalFn;
use ergotree_ir::types::stype::SType;

/// Tree height as recorded in the digest's last byte — the same source Scala's
/// `BatchAVLVerifier.rootNodeHeight` reads (`startingDigest.last & 0xff`), used
/// to scale per-operation verifier costs.
///
/// `rootNodeHeight` is assigned only AFTER reconstruction's up-front
/// `require`s pass (`keyLength > 0`, digest length); when they fail no root is
/// built and the height stays 0, so the JVM charges degenerate trees a
/// zero-height walk (`CAvlTreeVerifier.treeHeight`). `keyLength` is a signed
/// `Int` on the JVM — wire values with the high bit set are negative there.
/// (Failures *during* proof parsing — malformed proof bytes, wrong value
/// length — happen after the assignment and keep the digest-derived height.)
fn tree_height(avl_tree_data: &AvlTreeData) -> u32 {
    // 33 = ADDigest (32-byte root hash + height byte); a non-33 in-memory
    // digest (producible via updateDigest) cannot seed reconstruction either.
    if avl_tree_data.digest.len() != 33 || (avl_tree_data.key_length as i32) <= 0 {
        return 0;
    }
    avl_tree_data.digest.last().copied().unwrap_or(0) as u32
}

/// The items of a keys or entries argument, which the call has checked to be a collection.
/// sigmastate reads such an argument item by item, each where it uses it
/// (`CErgoTreeEvaluator.scala:78-241`), so nothing is cast here.
fn coll_items(v: Value<'_>) -> Result<Vec<Value<'_>>, EvalError> {
    match v {
        Value::Coll(coll) => Ok(coll.as_vec()),
        _ => Err(EvalError::UnexpectedValue(format!(
            "expected a collection, found {v:?}"
        ))),
    }
}

/// One entry of a key-value collection. sigmastate takes it apart with `case (key, value)`
/// BEFORE the operation's cost, so an entry that is no pair fails there
/// (`CErgoTreeEvaluator.scala:140`, `:178`, `:209`).
fn entry_of(entry: Value<'_>) -> Result<(Value<'_>, Value<'_>), EvalError> {
    match entry {
        Value::Pair(pair) => Ok(*pair),
        _ => Err(EvalError::UnexpectedValue(format!(
            "expected a pair of a key and a value, found {entry:?}"
        ))),
    }
}

/// A key's or a value's bytes: sigmastate's `toArray`, cast to a byte array where the
/// operation is performed, so AFTER that item's cost. A collection of another item type
/// fails here.
fn bytes_of(v: Value<'_>) -> Result<Bytes, EvalError> {
    Ok(Bytes::from(v.try_extract_into::<Vec<u8>>()?))
}

/// What the verifier is built from. sigmastate adds the verifier's cost over the proof's length
/// (`CreateAvlVerifier_Info`, `CErgoTreeEvaluator.scala:71-75`), then casts the proof's array
/// and then the digest's to bytes (`CAvlTreeVerifier.scala:57-59`)
fn verifier_inputs(
    ctx: &Context,
    tree: AvlTreeValue<'_>,
    proof: Value,
) -> Result<(AvlTreeData, Bytes), EvalError> {
    let proof_len = match &proof {
        Value::Coll(coll) => coll.len(),
        _ => {
            return Err(EvalError::UnexpectedValue(format!(
                "expected the proof to be a collection, found {proof:?}"
            )))
        }
    };
    ctx.add_per_item_jit_cost(110, 20, 64, proof_len as u32)?;
    let proof = Bytes::from(proof.try_extract_into::<Vec<u8>>()?);
    Ok((AvlTreeData::try_from(tree)?, proof))
}

/// Build the batch AVL verifier for `avl_tree_data` over `proof`
fn create_verifier(
    avl_tree_data: &AvlTreeData,
    proof: &Bytes,
) -> Result<BatchAVLVerifier, EvalError> {
    let starting_digest = Bytes::from(avl_tree_data.digest.clone());
    BatchAVLVerifier::new(
        &starting_digest,
        proof,
        AVLTree::new(
            |digest| Node::LabelOnly(NodeHeader::new(Some(*digest), None)),
            avl_tree_data.key_length as usize,
            avl_tree_data
                .value_length_opt
                .as_ref()
                .map(|v| **v as usize),
        ),
        None,
        None,
    )
    .map_err(map_eval_err)
}

pub(crate) static DIGEST_EVAL_FN: EvalFn = |_mc, _env, _ctx, obj, _args| {
    let tree = obj.try_extract_into::<AvlTreeValue>()?;
    Ok(Value::Coll(tree.digest))
};

pub(crate) static ENABLED_OPERATIONS_EVAL_FN: EvalFn = |_mc, _env, _ctx, obj, _args| {
    let tree = obj.try_extract_into::<AvlTreeValue>()?;
    Ok(Value::Byte(tree.tree_flags.serialize() as i8))
};

pub(crate) static KEY_LENGTH_EVAL_FN: EvalFn = |_mc, _env, _ctx, obj, _args| {
    let tree = obj.try_extract_into::<AvlTreeValue>()?;
    Ok(Value::Int(tree.key_length as i32))
};

pub(crate) static VALUE_LENGTH_OPT_EVAL_FN: EvalFn = |_mc, _env, _ctx, obj, _args| {
    let tree = obj.try_extract_into::<AvlTreeValue>()?;
    Ok(Value::Opt(
        tree.value_length_opt
            .map(|v| Value::Int(*v as i32))
            .map(Box::new),
    ))
};

pub(crate) static IS_INSERT_ALLOWED_EVAL_FN: EvalFn = |_mc, _env, _ctx, obj, _args| {
    let tree = obj.try_extract_into::<AvlTreeValue>()?;
    Ok(Value::Boolean(tree.tree_flags.insert_allowed()))
};

pub(crate) static IS_UPDATE_ALLOWED_EVAL_FN: EvalFn = |_mc, _env, _ctx, obj, _args| {
    let tree = obj.try_extract_into::<AvlTreeValue>()?;
    Ok(Value::Boolean(tree.tree_flags.update_allowed()))
};

pub(crate) static IS_REMOVE_ALLOWED_EVAL_FN: EvalFn = |_mc, _env, _ctx, obj, _args| {
    let tree = obj.try_extract_into::<AvlTreeValue>()?;
    Ok(Value::Boolean(tree.tree_flags.remove_allowed()))
};

pub(crate) static UPDATE_OPERATIONS_EVAL_FN: EvalFn = |_mc, _env, _ctx, obj, args| {
    let mut tree = obj.try_extract_into::<AvlTreeValue>()?;
    let new_operations = {
        let v = args.first().cloned().ok_or_else(|| {
            EvalError::AvlTree("eval is missing first arg (new_operations)".to_string())
        })?;
        v.try_extract_into::<i8>()? as u8
    };
    tree.tree_flags = AvlTreeFlags::parse(new_operations);
    Ok(Value::AvlTree(Box::new(tree)))
};

pub(crate) static UPDATE_DIGEST_EVAL_FN: EvalFn = |_mc, _env, _ctx, obj, args| {
    let mut tree = obj.try_extract_into::<AvlTreeValue>()?;
    // `CAvlTree.updateDigest` stores the new digest with no length check; mirror
    // that — accept any-length `Coll[Byte]`. A non-33 digest exists only in
    // memory (it cannot round-trip through serialization), same as the reference.
    let new_digest = {
        let v = args.first().cloned().ok_or_else(|| {
            EvalError::AvlTree("eval is missing first arg (new_digest)".to_string())
        })?;
        v.try_extract_into::<Vec<i8>>()?
    };
    tree.digest = CollValue::Bytes(new_digest.into());
    Ok(Value::AvlTree(Box::new(tree)))
};

pub(crate) static GET_EVAL_FN: EvalFn = |_mc, _env, ctx, obj, args| {
    let tree = obj.try_extract_into::<AvlTreeValue>()?;
    let key = args
        .first()
        .cloned()
        .ok_or_else(|| EvalError::AvlTree("eval is missing first arg (entries)".to_string()))?;
    let proof = args
        .get(1)
        .cloned()
        .ok_or_else(|| EvalError::AvlTree("eval is missing second arg (proof)".to_string()))?;

    let (avl_tree_data, proof) = verifier_inputs(ctx, tree, proof)?;
    let mut bv = match create_verifier(&avl_tree_data, &proof) {
        Ok(bv) => bv,
        // The reference impl's verifier construction never throws: a proof that
        // does not match the tree digest yields a verifier with no reconstructed
        // tree, the lookup fails, and `get_eval` raises the same "Tree proof is
        // incorrect" error as an op-level lookup failure
        // (`CErgoTreeEvaluator.get_eval`). The JVM charges that lookup before
        // raising — charge-then-throw. Cost-limit errors from the construction
        // charge are not construction failures — they propagate.
        Err(EvalError::AvlTree(_)) => {
            ctx.add_per_item_jit_cost(40, 10, 1, tree_height(&avl_tree_data))?;
            bytes_of(key)?;
            return Err(EvalError::AvlTree(format!(
                "Tree proof is incorrect {:?}",
                avl_tree_data
            )));
        }
        Err(e) => return Err(e),
    };
    // The cost of a tree lookup is O(treeHeight): `LookupAvlTree_Info` =
    // PerItemCost(40, 10, 1) per height item (Scala `get_eval`).
    ctx.add_per_item_jit_cost(40, 10, 1, tree_height(&avl_tree_data))?;
    match bv.perform_one_operation(&Operation::Lookup(bytes_of(key)?)) {
        Ok(opt) => match opt {
            Some(v) => Ok(Value::Opt(Some(Box::new(Value::Coll(CollValue::Bytes(
                v.iter().map(|&b| b as i8).collect(),
            )))))),
            _ => Ok(Value::Opt(None)),
        },
        Err(_) => Err(EvalError::AvlTree(format!(
            "Tree proof is incorrect {:?}",
            avl_tree_data
        ))),
    }
};

pub(crate) static GET_MANY_EVAL_FN: EvalFn = |_mc, _env, ctx, obj, args| {
    let tree = obj.try_extract_into::<AvlTreeValue>()?;

    let keys =
        coll_items(args.first().cloned().ok_or_else(|| {
            EvalError::AvlTree("eval is missing first arg (entries)".to_string())
        })?)?;
    let proof = args
        .get(1)
        .cloned()
        .ok_or_else(|| EvalError::AvlTree("eval is missing second arg (proof)".to_string()))?;

    let (avl_tree_data, proof) = verifier_inputs(ctx, tree, proof)?;
    let mut bv = match create_verifier(&avl_tree_data, &proof) {
        Ok(bv) => bv,
        // The reference impl's verifier construction never throws: lookups
        // fail on the no-tree verifier, so the first key raises the same
        // "Tree proof is incorrect" error as an op-level lookup failure
        // (charged like any first lookup — charge-then-throw) — and with no
        // keys no lookup runs at all, leaving the empty collection
        // (`CErgoTreeEvaluator.getMany_eval`). Cost-limit errors from the
        // construction charge are not construction failures — they
        // propagate.
        Err(EvalError::AvlTree(_)) => {
            return if keys.is_empty() {
                Ok(Value::Coll(CollValue::Generic {
                    elem_tpe: SType::SOption(Arc::new(SType::SColl(Arc::new(SType::SByte)))),
                    items: Arc::new([]),
                }))
            } else {
                ctx.add_per_item_jit_cost(40, 10, 1, tree_height(&avl_tree_data))?;
                bytes_of(keys[0].clone())?;
                Err(EvalError::AvlTree(format!(
                    "Tree proof is incorrect {:?}",
                    avl_tree_data
                )))
            }
        }
        Err(e) => return Err(e),
    };
    let height = tree_height(&avl_tree_data);
    let res = keys
        .into_iter()
        .map(|key| {
            // Per key, the cost of a tree lookup is O(treeHeight):
            // `LookupAvlTree_Info` = PerItemCost(40, 10, 1) per height item
            // (Scala `getMany_eval`).
            ctx.add_per_item_jit_cost(40, 10, 1, height)?;
            if let Ok(r) = bv.perform_one_operation(&Operation::Lookup(bytes_of(key)?)) {
                if let Some(v) = r {
                    Ok(Value::Opt(Some(Box::new(Value::Coll(CollValue::Bytes(
                        v.iter().map(|&b| b as i8).collect(),
                    ))))))
                } else {
                    Ok(Value::Opt(None))
                }
            } else {
                Err(EvalError::AvlTree(format!(
                    "Tree proof is incorrect {:?}",
                    avl_tree_data
                )))
            }
        })
        .collect::<Result<Arc<[_]>, _>>()?;

    Ok(Value::Coll(CollValue::Generic {
        elem_tpe: SType::SOption(Arc::new(SType::SColl(Arc::new(SType::SByte)))),
        items: res,
    }))
};

pub(crate) static INSERT_EVAL_FN: EvalFn = |_mc, _env, ctx, obj, args| {
    let tree = obj.try_extract_into::<AvlTreeValue>()?;

    // Flag-check cost: `isInsertAllowed_Info` = FixedCost(15) (Scala `insert_eval`).
    ctx.add_jit_cost(15)?;
    if !tree.tree_flags.insert_allowed() {
        return Ok(Value::Opt(None));
    }

    let entries =
        coll_items(args.first().cloned().ok_or_else(|| {
            EvalError::AvlTree("eval is missing first arg (entries)".to_string())
        })?)?;

    let proof = args
        .get(1)
        .cloned()
        .ok_or_else(|| EvalError::AvlTree("eval is missing second arg (proof)".to_string()))?;

    let (mut avl_tree_data, proof) = verifier_inputs(ctx, tree, proof)?;
    let mut bv = match create_verifier(&avl_tree_data, &proof) {
        Ok(bv) => bv,
        // The reference impl's verifier construction never throws: every
        // insert fails on the no-tree verifier, which pre-v3 raises at the
        // first failed op and from v3 fast-breaks (issue #908), leaving the
        // None digest — so the method evaluates to None. With no entries no
        // op runs and the digest is still None at every version
        // (`CErgoTreeEvaluator.insert_eval`). The JVM charges that first
        // (failing) insert before raising/breaking, so mirror the charge.
        // Cost-limit errors from the construction charge are not
        // construction failures — they propagate.
        Err(EvalError::AvlTree(_)) => {
            if entries.is_empty() {
                return Ok(Value::Opt(None));
            }
            let (key, value) = entry_of(entries[0].clone())?;
            let n_items = tree_height(&avl_tree_data).max(1);
            ctx.add_per_item_jit_cost(40, 10, 1, n_items)?;
            bytes_of(key)?;
            bytes_of(value)?;
            return if ctx.tree_version() >= ErgoTreeVersion::V3 {
                Ok(Value::Opt(None))
            } else {
                Err(EvalError::AvlTree(format!(
                    "Incorrect insert for {:?}",
                    avl_tree_data
                )))
            };
        }
        Err(e) => return Err(e),
    };
    // When the tree is empty we still need to add the insert cost (Scala
    // `insert_eval`: `nItems = Math.max(bv.treeHeight, 1)`).
    let n_items = tree_height(&avl_tree_data).max(1);
    for entry in entries {
        let (key, value) = entry_of(entry)?;
        // The cost of a tree insert is O(treeHeight): `InsertIntoAvlTree_Info`
        // = PerItemCost(40, 10, 1) per height item.
        ctx.add_per_item_jit_cost(40, 10, 1, n_items)?;
        let (key, value) = (bytes_of(key)?, bytes_of(value)?);
        if bv
            .perform_one_operation(&Operation::Insert(KeyValue { key, value }))
            .is_err()
        {
            if ctx.tree_version() >= ErgoTreeVersion::V3 {
                break;
            } else {
                return Err(EvalError::AvlTree(format!(
                    "Incorrect insert for {:?}",
                    avl_tree_data
                )));
            }
        }
    }
    Ok(if let Some(new_digest) = bv.digest() {
        // Digest extraction after mutation: `updateDigest_Info` = FixedCost(40).
        ctx.add_jit_cost(40)?;
        avl_tree_data.digest = new_digest.to_vec();
        Value::Opt(Some(Box::new(Value::from(avl_tree_data))))
    } else {
        Value::Opt(None)
    })
};

pub(crate) static REMOVE_EVAL_FN: EvalFn = |_mc, _env, ctx, obj, args| {
    let tree = obj.try_extract_into::<AvlTreeValue>()?;

    // Flag-check cost: `isRemoveAllowed_Info` = FixedCost(15) (Scala `remove_eval`).
    ctx.add_jit_cost(15)?;
    if !tree.tree_flags.remove_allowed() {
        return Ok(Value::Opt(None));
    }

    let keys = coll_items(
        args.first()
            .cloned()
            .ok_or_else(|| EvalError::AvlTree("eval is missing first arg (keys)".to_string()))?,
    )?;

    let proof = args
        .get(1)
        .cloned()
        .ok_or_else(|| EvalError::AvlTree("eval is missing second arg (proof)".to_string()))?;

    let (mut avl_tree_data, proof) = verifier_inputs(ctx, tree, proof)?;
    let mut bv = match create_verifier(&avl_tree_data, &proof) {
        Ok(bv) => bv,
        // The reference impl's verifier construction never throws: every
        // remove fails on the no-tree verifier (op results are ignored —
        // `remove_eval` loops with `cfor`, no break), and the final digest
        // is None — so the method evaluates to None
        // (`CErgoTreeEvaluator.remove_eval`). The JVM still attempts and
        // charges EVERY remove plus the unconditional digest read, so
        // mirror those charges. Cost-limit errors from the construction
        // charge are not construction failures — they propagate.
        Err(EvalError::AvlTree(_)) => {
            let n_items = tree_height(&avl_tree_data).max(1);
            for key in keys {
                ctx.add_per_item_jit_cost(100, 15, 1, n_items)?;
                bytes_of(key)?;
            }
            ctx.add_jit_cost(15)?;
            return Ok(Value::Opt(None));
        }
        Err(e) => return Err(e),
    };
    // When the tree is empty we still need to add the remove cost (Scala
    // `remove_eval`: `nItems = Math.max(bv.treeHeight, 1)`).
    let n_items = tree_height(&avl_tree_data).max(1);
    for key in keys {
        // The cost of a tree remove is O(treeHeight): `RemoveAvlTree_Info`
        // = PerItemCost(100, 15, 1) per height item.
        ctx.add_per_item_jit_cost(100, 15, 1, n_items)?;
        // op results are ignored — the reference impl loops with `cfor`,
        // no break, no check (`CErgoTreeEvaluator.remove_eval`); a failed
        // op invalidates the verifier, so the digest below decides the
        // outcome
        let _ = bv.perform_one_operation(&Operation::Remove(bytes_of(key)?));
    }
    // Digest read after the operation loop: `digest_Info` = FixedCost(15)
    // (Scala `remove_eval` charges it unconditionally, unlike insert/update).
    ctx.add_jit_cost(15)?;
    if let Some(new_digest) = bv.digest() {
        // Digest extraction after mutation: `updateDigest_Info` = FixedCost(40).
        ctx.add_jit_cost(40)?;
        avl_tree_data.digest = new_digest.to_vec();
        Ok(Value::Opt(Some(Box::new(Value::from(avl_tree_data)))))
    } else {
        Ok(Value::Opt(None))
    }
};

pub(crate) static CONTAINS_EVAL_FN: EvalFn = |_mc, _env, ctx, obj, args| {
    let tree = obj.try_extract_into::<AvlTreeValue>()?;
    let key = args
        .first()
        .cloned()
        .ok_or_else(|| EvalError::AvlTree("eval is missing first arg (key)".to_string()))?;

    let proof = args
        .get(1)
        .cloned()
        .ok_or_else(|| EvalError::AvlTree("eval is missing second arg (proof)".to_string()))?;

    let (avl_tree_data, proof) = verifier_inputs(ctx, tree, proof)?;
    let mut bv = match create_verifier(&avl_tree_data, &proof) {
        Ok(bv) => bv,
        // The reference impl's verifier construction never throws: the lookup
        // fails on the no-tree verifier and `contains_eval` maps the failure to
        // false (`CErgoTreeEvaluator.contains_eval`). The JVM charges that
        // lookup before its failure maps to false, so mirror the charge.
        // Cost-limit errors from the construction charge are not construction
        // failures — they propagate.
        Err(EvalError::AvlTree(_)) => {
            ctx.add_per_item_jit_cost(40, 10, 1, tree_height(&avl_tree_data))?;
            bytes_of(key)?;
            return Ok(Value::Boolean(false));
        }
        Err(e) => return Err(e),
    };
    // The cost of a tree lookup is O(treeHeight): `LookupAvlTree_Info` =
    // PerItemCost(40, 10, 1) per height item (Scala `contains_eval`).
    ctx.add_per_item_jit_cost(40, 10, 1, tree_height(&avl_tree_data))?;
    Ok(
        match bv.perform_one_operation(&Operation::Lookup(bytes_of(key)?)) {
            Ok(s) => match s {
                Some(_e) => Value::Boolean(true),
                _ => Value::Boolean(false),
            },
            Err(_) => Value::Boolean(false),
        },
    )
};

pub(crate) static UPDATE_EVAL_FN: EvalFn = |_mc, _env, ctx, obj, args| {
    let tree = obj.try_extract_into::<AvlTreeValue>()?;

    // Flag-check cost: `isUpdateAllowed_Info` = FixedCost(15) (Scala `update_eval`).
    ctx.add_jit_cost(15)?;
    if !tree.tree_flags.update_allowed() {
        return Ok(Value::Opt(None));
    }

    let entries =
        coll_items(args.first().cloned().ok_or_else(|| {
            EvalError::AvlTree("eval is missing first arg (entries)".to_string())
        })?)?;

    let proof = args
        .get(1)
        .cloned()
        .ok_or_else(|| EvalError::AvlTree("eval is missing second arg (proof)".to_string()))?;

    let (mut avl_tree_data, proof) = verifier_inputs(ctx, tree, proof)?;
    let mut bv = match create_verifier(&avl_tree_data, &proof) {
        Ok(bv) => bv,
        // The reference impl's verifier construction never throws: every
        // update fails on the no-tree verifier (`update_eval`'s forall
        // fast-breaks at the first failure), and the final digest is None —
        // so the method evaluates to None
        // (`CErgoTreeEvaluator.update_eval`). The JVM still charges the
        // first (failing) update before its forall fast-breaks, so mirror
        // that charge. Cost-limit errors from the construction charge are
        // not construction failures — they propagate.
        Err(EvalError::AvlTree(_)) => {
            if !entries.is_empty() {
                let (key, value) = entry_of(entries[0].clone())?;
                let n_items = tree_height(&avl_tree_data).max(1);
                ctx.add_per_item_jit_cost(120, 20, 1, n_items)?;
                bytes_of(key)?;
                bytes_of(value)?;
            }
            return Ok(Value::Opt(None));
        }
        Err(e) => return Err(e),
    };
    // When the tree is empty we still need to add the update cost (Scala
    // `update_eval`: `nItems = Math.max(bv.treeHeight, 1)`).
    let n_items = tree_height(&avl_tree_data).max(1);
    for entry in entries {
        let (key, value) = entry_of(entry)?;
        // The cost of a tree update is O(treeHeight): `UpdateAvlTree_Info`
        // = PerItemCost(120, 20, 1) per height item.
        ctx.add_per_item_jit_cost(120, 20, 1, n_items)?;
        let (key, value) = (bytes_of(key)?, bytes_of(value)?);
        if bv
            .perform_one_operation(&Operation::Update(KeyValue { key, value }))
            .is_err()
        {
            break;
        }
    }
    Ok(if let Some(new_digest) = bv.digest() {
        // Digest extraction after mutation: `updateDigest_Info` = FixedCost(40).
        ctx.add_jit_cost(40)?;
        avl_tree_data.digest = new_digest.to_vec();
        Value::Opt(Some(Value::from(avl_tree_data).into()))
    } else {
        Value::Opt(None)
    })
};

pub(crate) static INSERT_OR_UPDATE_EVAL_FN: EvalFn = |_mc, _env, ctx, obj, args| {
    let tree = obj.try_extract_into::<AvlTreeValue>()?;

    // Flag-check costs: `isUpdateAllowed_Info` + `isInsertAllowed_Info` =
    // FixedCost(15) each (Scala `insertOrUpdate_eval` charges both).
    ctx.add_jit_cost(15)?;
    ctx.add_jit_cost(15)?;
    if !tree.tree_flags.insert_allowed() || !tree.tree_flags.update_allowed() {
        return Ok(Value::Opt(None));
    }

    let entries =
        coll_items(args.first().cloned().ok_or_else(|| {
            EvalError::AvlTree("eval is missing first arg (entries)".to_string())
        })?)?;

    let proof = args
        .get(1)
        .cloned()
        .ok_or_else(|| EvalError::AvlTree("eval is missing second arg (proof)".to_string()))?;

    let (mut avl_tree_data, proof) = verifier_inputs(ctx, tree, proof)?;
    let mut bv = match create_verifier(&avl_tree_data, &proof) {
        Ok(bv) => bv,
        // The reference impl's verifier construction never throws: with a proof
        // that does not match the tree digest it yields a verifier with no
        // reconstructed tree, every operation fails (and `insertOrUpdate_eval`
        // discards per-op failures), and the final digest is None — so the
        // method returns None rather than erroring
        // (`CErgoTreeEvaluator.insertOrUpdate_eval`). The JVM still charges the
        // first (failing) operation before its forall fast-breaks, so mirror
        // that charge. Cost-limit errors from the construction charge are not
        // construction failures — they propagate.
        Err(EvalError::AvlTree(_)) => {
            if !entries.is_empty() {
                let (key, value) = entry_of(entries[0].clone())?;
                let n_items = tree_height(&avl_tree_data).max(1);
                ctx.add_per_item_jit_cost(120, 20, 1, n_items)?;
                bytes_of(key)?;
                bytes_of(value)?;
            }
            return Ok(Value::Opt(None));
        }
        Err(e) => return Err(e),
    };
    // When the tree is empty we still need to add the insert cost (Scala
    // `insertOrUpdate_eval`: `nItems = Math.max(bv.treeHeight, 1)`).
    let n_items = tree_height(&avl_tree_data).max(1);
    for entry in entries {
        let (key, value) = entry_of(entry)?;
        // Per entry, Scala `insertOrUpdate_eval` charges `UpdateAvlTree_Info`
        // = PerItemCost(120, 20, 1) per height item.
        ctx.add_per_item_jit_cost(120, 20, 1, n_items)?;
        let (key, value) = (bytes_of(key)?, bytes_of(value)?);
        if bv
            .perform_one_operation(&Operation::InsertOrUpdate(KeyValue { key, value }))
            .is_err()
        {
            break;
        }
    }
    Ok(if let Some(new_digest) = bv.digest() {
        // Digest extraction after mutation: `updateDigest_Info` = FixedCost(40).
        ctx.add_jit_cost(40)?;
        avl_tree_data.digest = new_digest.to_vec();
        Value::Opt(Some(Box::new(Value::from(avl_tree_data))))
    } else {
        Value::Opt(None)
    })
};

fn map_eval_err<T: core::fmt::Debug>(e: T) -> EvalError {
    EvalError::AvlTree(format!("{:?}", e))
}

#[allow(clippy::unwrap_used, clippy::panic, clippy::unreachable)]
#[cfg(test)]
#[cfg(feature = "arbitrary")]
mod tests {
    use alloc::sync::Arc;

    use ergo_avltree_rust::batch_avl_prover::BatchAVLProver;
    use ergotree_ir::{
        chain::context::Context,
        ergo_tree::ErgoTree,
        mir::{
            avl_tree_data::{AvlTreeData, AvlTreeFlags},
            constant::{Constant, Literal},
            expr::Expr,
            method_call::MethodCall,
            value::CollKind,
        },
        types::{savltree, stuple::STuple, stype::SType},
    };
    use proptest::prelude::*;
    use sigma_test_util::force_any_val;

    use crate::eval::test_util::{eval_out_wo_ctx, try_eval_out_with_version, try_eval_out_wo_ctx};

    use super::*;
    use ergo_chain_types::ADDigest;
    use sigma_ser::ScorexSerializable;
    use sigma_util::{AsVecI8, AsVecU8};

    #[test]
    fn eval_avl_get() {
        let mut prover = populate_tree(vec![(vec![1u8], 10u64.to_be_bytes().to_vec())]);
        let initial_digest = ADDigest::scorex_parse_bytes(&prover.digest().unwrap()).unwrap();

        let key1 = Bytes::from(vec![1u8]);
        let key2 = Bytes::from(vec![2u8]);
        let op1 = Operation::Lookup(key1);
        let op2 = Operation::Lookup(key2);
        let lookup_found = prover.perform_one_operation(&op1).unwrap();
        let lookup_not_found = prover.perform_one_operation(&op2).unwrap();
        let proof: Constant = prover
            .generate_proof()
            .into_iter()
            .collect::<Vec<_>>()
            .into();

        let tree_flags = AvlTreeFlags::new(false, false, false);
        let obj = Expr::Const(
            AvlTreeData {
                digest: initial_digest.0.to_vec(),
                tree_flags,
                key_length: 1,
                value_length_opt: None,
            }
            .into(),
        );

        let search_key_found = vec![1i8];
        let search_key_not_found = vec![2i8];
        let expr_found: Expr = MethodCall::new(
            obj.clone(),
            savltree::GET_METHOD.clone(),
            vec![search_key_found.into(), proof.clone().into()],
        )
        .unwrap()
        .into();
        let expr_not_found: Expr = MethodCall::new(
            obj,
            savltree::GET_METHOD.clone(),
            vec![search_key_not_found.into(), proof.into()],
        )
        .unwrap()
        .into();

        let res_found = eval_out_wo_ctx::<Value>(&expr_found);
        let res_not_found = eval_out_wo_ctx::<Value>(&expr_not_found);

        if let Value::Opt(opt) = res_found {
            if let Some(Value::Coll(CollValue::Bytes(b))) = opt.as_deref() {
                assert!(lookup_found.unwrap().eq(&b.as_vec_u8()));
            } else {
                unreachable!();
            }
        } else {
            unreachable!();
        }

        if let Value::Opt(opt) = res_not_found {
            assert!(lookup_not_found.is_none() && opt.is_none())
        } else {
            unreachable!();
        }
    }

    #[test]
    fn eval_avl_get_many() {
        let mut prover = populate_tree(vec![
            (vec![1u8], 10u64.to_be_bytes().to_vec()),
            (vec![2u8], 20u64.to_be_bytes().to_vec()),
        ]);

        let initial_digest = ADDigest::scorex_parse_bytes(&prover.digest().unwrap()).unwrap();

        let key1 = Bytes::from(vec![1u8]);
        let key2 = Bytes::from(vec![2u8]);
        let key3 = Bytes::from(vec![3u8]);
        let op1 = Operation::Lookup(key1);
        let op2 = Operation::Lookup(key2);
        let op3 = Operation::Lookup(key3);
        let lookups = vec![
            prover.perform_one_operation(&op1).unwrap(),
            prover.perform_one_operation(&op2).unwrap(),
            prover.perform_one_operation(&op3).unwrap(),
        ];

        let proof: Constant = prover
            .generate_proof()
            .into_iter()
            .collect::<Vec<_>>()
            .into();

        let tree_flags = AvlTreeFlags::new(false, false, false);
        let obj = Expr::Const(
            AvlTreeData {
                digest: initial_digest.0.to_vec(),
                tree_flags,
                key_length: 1,
                value_length_opt: None,
            }
            .into(),
        );

        let search_key_1 = Literal::from(vec![1u8]);
        let search_key_2 = Literal::from(vec![2u8]);
        let search_key_3 = Literal::from(vec![3u8]);

        let keys = Constant {
            tpe: SType::SColl(Arc::new(SType::SColl(Arc::new(SType::SByte)))),
            v: Literal::Coll(CollKind::WrappedColl {
                items: Arc::new([search_key_1, search_key_2, search_key_3]),
                elem_tpe: SType::SColl(Arc::new(SType::SByte)),
            }),
        };

        let expr: Expr = MethodCall::new(
            obj,
            savltree::GET_MANY_METHOD.clone(),
            vec![keys.into(), proof.into()],
        )
        .unwrap()
        .into();

        let res = eval_out_wo_ctx::<Value>(&expr);

        if let Value::Coll(CollValue::Generic { items, .. }) = res {
            for (item, expected) in items.iter().zip(lookups) {
                if let Value::Opt(opt) = item.clone() {
                    match opt.as_deref() {
                        None => assert!(expected.is_none()),
                        Some(Value::Coll(CollValue::Bytes(b))) => {
                            assert_eq!(&b[..], &expected.unwrap().to_vec().as_vec_i8()[..]);
                        }
                        Some(_) => unreachable!(),
                    }
                } else {
                    unreachable!();
                }
            }
        } else {
            unreachable!();
        }
    }

    #[test]
    fn eval_avl_insert() {
        // This example taken from `ergo_avltree_rust` README
        let mut prover = BatchAVLProver::new(
            AVLTree::new(
                |digest| Node::LabelOnly(NodeHeader::new(Some(*digest), None)),
                1,
                None,
            ),
            true,
        );
        let initial_digest = ADDigest::scorex_parse_bytes(&prover.digest().unwrap()).unwrap();
        let key1 = Bytes::from(vec![1u8]);
        let key2 = Bytes::from(vec![2u8; 1]);
        let key3 = Bytes::from(vec![3u8; 1]);
        let op1 = Operation::Insert(KeyValue {
            key: key1,
            value: Bytes::from(10u64.to_be_bytes().to_vec()),
        });
        let op2 = Operation::Insert(KeyValue {
            key: key2,
            value: Bytes::from(20u64.to_be_bytes().to_vec()),
        });
        let op3 = Operation::Insert(KeyValue {
            key: key3,
            value: Bytes::from(30u64.to_be_bytes().to_vec()),
        });
        prover.perform_one_operation(&op1).unwrap();
        prover.perform_one_operation(&op2).unwrap();
        prover.perform_one_operation(&op3).unwrap();
        let final_digest = ADDigest::scorex_parse_bytes(&prover.digest().unwrap()).unwrap();
        let proof: Constant = prover
            .generate_proof()
            .into_iter()
            .collect::<Vec<_>>()
            .into();

        let tree_flags = AvlTreeFlags::new(true, false, false);
        let obj = Expr::Const(
            AvlTreeData {
                digest: initial_digest.0.to_vec(),
                tree_flags,
                key_length: 1,
                value_length_opt: None,
            }
            .into(),
        );
        let pair1 = Literal::Tup(mk_pair(1u8, 10u64).into());
        let pair2 = Literal::Tup(mk_pair(2u8, 20u64).into());
        let pair3 = Literal::Tup(mk_pair(3u8, 30u64).into());
        let entries = Constant {
            tpe: SType::SColl(Arc::new(SType::STuple(STuple::pair(
                SType::SColl(Arc::new(SType::SByte)),
                SType::SColl(Arc::new(SType::SByte)),
            )))),
            v: Literal::Coll(CollKind::WrappedColl {
                items: Arc::new([pair1.clone(), pair2.clone(), pair3.clone()]),
                elem_tpe: SType::STuple(STuple::pair(
                    SType::SColl(Arc::new(SType::SByte)),
                    SType::SColl(Arc::new(SType::SByte)),
                )),
            }),
        };
        let expr: Expr = MethodCall::new(
            obj.clone(),
            savltree::INSERT_METHOD.clone(),
            vec![entries.clone().into(), proof.clone().into()],
        )
        .unwrap()
        .into();

        let res = eval_out_wo_ctx::<Value>(&expr);
        if let Value::Opt(opt) = res {
            if let Some(Value::AvlTree(avl)) = opt.as_deref() {
                assert_eq!(
                    Value::Coll(avl.digest.clone()),
                    Value::from(final_digest.0.to_vec())
                );
            } else {
                unreachable!();
            }
        } else {
            unreachable!();
        }

        // perform invalid insertion (duplicate entries). Before v6.0 this would return an error. After 6.0 this will return None
        let duplicate_entries = Constant {
            v: Literal::Coll(CollKind::WrappedColl {
                items: Arc::new([pair1, pair2, pair3.clone(), pair3]),
                elem_tpe: SType::STuple(STuple::pair(
                    SType::SColl(Arc::new(SType::SByte)),
                    SType::SColl(Arc::new(SType::SByte)),
                )),
            }),
            ..entries
        };
        let expr: Expr = MethodCall::new(
            obj,
            savltree::INSERT_METHOD.clone(),
            vec![duplicate_entries.into(), proof.into()],
        )
        .unwrap()
        .into();
        assert!(try_eval_out_with_version::<Value<'_>>(&expr, &force_any_val(), 0, 3).is_err());
        assert_eq!(
            try_eval_out_with_version::<Value<'_>>(&expr, &force_any_val(), 3, 3).unwrap(),
            Value::Opt(None)
        );
    }

    #[test]
    fn eval_avl_insert_or_update() {
        let mut prover = BatchAVLProver::new(
            AVLTree::new(
                |digest| Node::LabelOnly(NodeHeader::new(Some(*digest), None)),
                1,
                None,
            ),
            true,
        );
        let initial_digest = ADDigest::scorex_parse_bytes(&prover.digest().unwrap()).unwrap();
        let key1 = Bytes::from(vec![1u8]);
        let op = Operation::InsertOrUpdate(KeyValue {
            key: key1,
            value: Bytes::from(10u64.to_be_bytes().to_vec()),
        });
        prover.perform_one_operation(&op).unwrap();
        prover.perform_one_operation(&op).unwrap();
        let final_digest = ADDigest::scorex_parse_bytes(&prover.digest().unwrap()).unwrap();
        let proof: Constant = prover
            .generate_proof()
            .into_iter()
            .collect::<Vec<_>>()
            .into();

        let tree_flags = AvlTreeFlags::new(true, true, false);
        let obj = Expr::Const(
            AvlTreeData {
                digest: initial_digest.0.to_vec(),
                tree_flags,
                key_length: 1,
                value_length_opt: None,
            }
            .into(),
        );
        let pair1 = Literal::Tup(mk_pair(1u8, 10u64).into());
        let entries = Constant {
            tpe: SType::SColl(Arc::new(SType::STuple(STuple::pair(
                SType::SColl(Arc::new(SType::SByte)),
                SType::SColl(Arc::new(SType::SByte)),
            )))),
            v: Literal::Coll(CollKind::WrappedColl {
                items: Arc::new([pair1.clone(), pair1.clone()]),
                elem_tpe: SType::STuple(STuple::pair(
                    SType::SColl(Arc::new(SType::SByte)),
                    SType::SColl(Arc::new(SType::SByte)),
                )),
            }),
        };
        let expr: Expr = MethodCall::new(
            obj.clone(),
            savltree::INSERT_OR_UPDATE_METHOD.clone(),
            vec![entries.clone().into(), proof.clone().into()],
        )
        .unwrap()
        .into();

        let res = eval_out_wo_ctx::<Value>(&expr);
        if let Value::Opt(opt) = res {
            if let Some(Value::AvlTree(avl)) = opt.as_deref() {
                assert_eq!(
                    Value::Coll(avl.digest.clone()),
                    Value::from(final_digest.0.to_vec())
                );
            } else {
                unreachable!();
            }
        } else {
            unreachable!();
        }
    }

    // A structurally-valid proof generated from a DIFFERENT tree: the reference
    // impl's verifier construction never throws — it yields a verifier with no
    // reconstructed tree, every operation fails (discarded), the final digest is
    // None, and `insertOrUpdate` returns None (`insertOrUpdate_eval`).
    #[test]
    fn eval_avl_insert_or_update_bad_proof() {
        // tree A: non-empty (one committed insert), so its digest cannot match
        // the empty-tree digest the wrong proof starts from
        let mut prover_a = BatchAVLProver::new(
            AVLTree::new(
                |digest| Node::LabelOnly(NodeHeader::new(Some(*digest), None)),
                1,
                None,
            ),
            true,
        );
        prover_a
            .perform_one_operation(&Operation::Insert(KeyValue {
                key: Bytes::from(vec![5u8]),
                value: Bytes::from(50u64.to_be_bytes().to_vec()),
            }))
            .unwrap();
        prover_a.generate_proof();
        let tree_a_digest = ADDigest::scorex_parse_bytes(&prover_a.digest().unwrap()).unwrap();

        // proof from tree B (empty start) inserting key 1
        let mut prover_b = BatchAVLProver::new(
            AVLTree::new(
                |digest| Node::LabelOnly(NodeHeader::new(Some(*digest), None)),
                1,
                None,
            ),
            true,
        );
        prover_b
            .perform_one_operation(&Operation::InsertOrUpdate(KeyValue {
                key: Bytes::from(vec![1u8]),
                value: Bytes::from(10u64.to_be_bytes().to_vec()),
            }))
            .unwrap();
        let wrong_proof: Constant = prover_b
            .generate_proof()
            .into_iter()
            .collect::<Vec<_>>()
            .into();

        let obj = Expr::Const(
            AvlTreeData {
                digest: tree_a_digest.0.to_vec(),
                tree_flags: AvlTreeFlags::new(true, true, false),
                key_length: 1,
                value_length_opt: None,
            }
            .into(),
        );
        let pair1 = Literal::Tup(mk_pair(1u8, 10u64).into());
        let entries = Constant {
            tpe: SType::SColl(Arc::new(SType::STuple(STuple::pair(
                SType::SColl(Arc::new(SType::SByte)),
                SType::SColl(Arc::new(SType::SByte)),
            )))),
            v: Literal::Coll(CollKind::WrappedColl {
                items: Arc::new([pair1]),
                elem_tpe: SType::STuple(STuple::pair(
                    SType::SColl(Arc::new(SType::SByte)),
                    SType::SColl(Arc::new(SType::SByte)),
                )),
            }),
        };
        let expr: Expr = MethodCall::new(
            obj,
            savltree::INSERT_OR_UPDATE_METHOD.clone(),
            vec![entries.into(), wrong_proof.into()],
        )
        .unwrap()
        .into();

        let res = eval_out_wo_ctx::<Value>(&expr);
        assert!(matches!(res, Value::Opt(None)));
    }

    // JVM-blessed byte vector (santa-eval `AvlTree.insertOrUpdate`, entry
    // `insertOrUpdate#bad-proof`): v3 tree carrying the AvlTree, the entries,
    // and the wrong-tree proof as segregated constants; blessed None. The
    // blessed header `1b` + size VLQ `f801` is rewritten to the non-sized `13`
    // (size-bit cleared, size dropped) because the sized parse path rejects
    // non-SigmaProp roots — the same lenient deserialize the conformance
    // runner applies to expression-rooted corpus trees; body bytes verbatim.
    #[test]
    fn eval_avl_insert_or_update_bad_proof_blessed_bytes() {
        let tree_bytes = base16::decode("130464fb2b77372d81da43ce2d72714aec79ae5fcac20a9aff426fe6afb476a6fbc02c04072001080e209a39c57d13039b50bfe4aa21b2c8238be31d133db1fbe4549b16d9b94338b4e20e08000000005a17a0320e8f0103fcbfd9e0c4781263bb161625674719024acef64654799a00c41cc148bdc4a891028f23c5ee24a49084f24812ee9dbcdc9e11974b05cad682c21e88fca866bc85cec4e67333f14133fce4ed7bbf147642b3e7b2b8324268ea6ded1be0caa2da237f000000005a17a000039594489172346c6b22ac52210d14be7fdb02f6ec52b9e525b3a9c77114374cd900ff0402dc641073000283013c0e0e8602730173027303").unwrap();
        let tree = ErgoTree::sigma_parse_bytes_lenient_from_unsized(&tree_bytes).unwrap();
        let expr = tree.proposition().unwrap();
        let ctx = force_any_val::<Context>();
        let res = try_eval_out_with_version::<Value>(&expr, &ctx, 3, 3).unwrap();
        assert!(matches!(res, Value::Opt(None)));
    }

    proptest! {
        #[test]
        fn eval_avl_digest(v in any::<AvlTreeData>()) {
            let digest: Vec<i8> = v.digest.as_vec_i8();
            let obj = Expr::Const(v.into());

            let expr: Expr = MethodCall::new(
                obj,
                savltree::DIGEST_METHOD.clone(),
                vec![],
            )
            .unwrap()
            .into();

            let res = eval_out_wo_ctx::<Value>(&expr);
            if let Value::Coll(CollValue::Bytes(b)) = res {
                assert_eq!(&b[..], digest.as_slice());
            } else {
                unreachable!();
            }
        }

        #[test]
        fn eval_avl_enabled_operations(v in any::<AvlTreeData>()) {
            let enabled_ops = v.tree_flags.serialize() as i8;
            let obj = Expr::Const(v.into());

            let expr: Expr = MethodCall::new(
                obj,
                savltree::ENABLED_OPERATIONS_METHOD.clone(),
                vec![],
            )
            .unwrap()
            .into();

            let res = eval_out_wo_ctx::<Value>(&expr);
            if let Value::Byte(b) = res {
                assert_eq!(b, enabled_ops);
            } else {
                unreachable!();
            }
        }

        #[test]
        fn eval_avl_key_length(v in any::<AvlTreeData>()) {
            let key_length = v.key_length as i32;
            let obj = Expr::Const(v.into());

            let expr: Expr = MethodCall::new(
                obj,
                savltree::KEY_LENGTH_METHOD.clone(),
                vec![],
            )
            .unwrap()
            .into();

            let res = eval_out_wo_ctx::<Value>(&expr);
            if let Value::Int(i) = res {
                assert_eq!(key_length, i);
            } else {
                unreachable!();
            }
        }

        #[test]
        fn eval_avl_value_length_opt(v in any::<AvlTreeData>()) {
            let value_length_opt = v.value_length_opt.clone().map(|v| Value::Int(*v as i32));
            let obj = Expr::Const(v.into());

            let expr: Expr = MethodCall::new(
                obj,
                savltree::VALUE_LENGTH_OPT_METHOD.clone(),
                vec![],
            )
            .unwrap()
            .into();

            let res = eval_out_wo_ctx::<Value>(&expr);
            if let Value::Opt(opt) = res {
                assert_eq!(opt.as_deref().cloned(), value_length_opt);
            } else {
                unreachable!();
            }
        }

        #[test]
        fn eval_avl_insert_allowed(v in any::<AvlTreeData>()) {
            let insert_allowed = v.tree_flags.insert_allowed();
            let obj = Expr::Const(v.into());

            let expr: Expr = MethodCall::new(
                obj,
                savltree::IS_INSERT_ALLOWED_METHOD.clone(),
                vec![],
            )
            .unwrap()
            .into();
            let res = eval_out_wo_ctx::<Value>(&expr);
            if let Value::Boolean(i) = res {
                assert_eq!(insert_allowed, i);
            } else {
                unreachable!();
            }
        }

        #[test]
        fn eval_avl_update_allowed(v in any::<AvlTreeData>()) {
            let update_allowed = v.tree_flags.update_allowed();
            let obj = Expr::Const(v.into());

            let expr: Expr = MethodCall::new(
                obj,
                savltree::IS_UPDATE_ALLOWED_METHOD.clone(),
                vec![],
            )
            .unwrap()
            .into();
            let res = eval_out_wo_ctx::<Value>(&expr);
            if let Value::Boolean(i) = res {
                assert_eq!(update_allowed, i);
            } else {
                unreachable!();
            }
        }

        #[test]
        fn eval_avl_remove_allowed(v in any::<AvlTreeData>()) {
            let remove_allowed = v.tree_flags.remove_allowed();
            let obj = Expr::Const(v.into());

            let expr: Expr = MethodCall::new(
                obj,
                savltree::IS_REMOVE_ALLOWED_METHOD.clone(),
                vec![],
            )
            .unwrap()
            .into();
            let res = eval_out_wo_ctx::<Value>(&expr);
            if let Value::Boolean(i) = res {
                assert_eq!(remove_allowed, i);
            } else {
                unreachable!();
            }
        }

        #[test]
        fn eval_avl_update_operations(v in any::<AvlTreeData>(), new_ops in any::<AvlTreeFlags>()) {
            // Test updateOperations method
            let obj = Expr::Const(v.into());
            let expr: Expr = MethodCall::new(
                obj,
                savltree::UPDATE_OPERATIONS_METHOD.clone(),
                vec![Constant::from(new_ops.serialize() as i8).into()],
            )
            .unwrap()
            .into();
            let res = eval_out_wo_ctx::<Value>(&expr);
            if let Value::AvlTree(a) = res {
                assert_eq!(a.tree_flags, new_ops);
            } else {
                unreachable!();
            }
        }

        #[test]
        fn eval_avl_update_digest(v in any::<AvlTreeData>(), new_digest in any::<ADDigest>()) {
            let obj = Expr::Const(v.into());
            let expr: Expr = MethodCall::new(
                obj,
                savltree::UPDATE_DIGEST_METHOD.clone(),
                vec![Constant::from(new_digest.scorex_serialize_bytes()?).into()],
            )
            .unwrap()
            .into();
            let res = eval_out_wo_ctx::<Value>(&expr);
            if let Value::AvlTree(a) = res {
                assert_eq!(Value::Coll(a.digest.clone()), Value::from(new_digest.0.to_vec()));
            } else {
                unreachable!();
            }
        }
    }

    #[test]
    fn eval_avl_update_digest_any_length() {
        // `CAvlTree.updateDigest` stores any-length `Coll[Byte]` with no length
        // check (santa-eval `AvlTree.updateDigest_any_length`): the new digest is
        // kept verbatim and read back by `.digest`, regardless of length. Such a
        // value is in-memory only — it does not round-trip through serialization.
        for new_digest in [
            vec![1u8, 2, 3],
            Vec::<u8>::new(),
            (1u8..=40).collect::<Vec<u8>>(),
        ] {
            let tree = AvlTreeData {
                digest: vec![0u8; 33],
                tree_flags: AvlTreeFlags::new(true, true, true),
                key_length: 32,
                value_length_opt: None,
            };
            let updated: Expr = MethodCall::new(
                Expr::Const(tree.into()),
                savltree::UPDATE_DIGEST_METHOD.clone(),
                vec![Constant::from(new_digest.clone()).into()],
            )
            .unwrap()
            .into();
            // stored verbatim
            assert_eq!(
                eval_out_wo_ctx::<AvlTreeData>(&updated.clone()).digest,
                new_digest
            );
            // and read back via `.digest`, whatever the length
            let digest_back: Expr =
                MethodCall::new(updated, savltree::DIGEST_METHOD.clone(), vec![])
                    .unwrap()
                    .into();
            if let Value::Coll(CollValue::Bytes(b)) = eval_out_wo_ctx::<Value>(&digest_back) {
                assert_eq!(b.as_vec_u8(), new_digest);
            } else {
                unreachable!();
            }
        }
    }

    #[test]
    fn eval_avl_contains() {
        let mut prover = populate_tree(vec![
            (vec![1u8], 10u64.to_be_bytes().to_vec()),
            (vec![2u8], 20u64.to_be_bytes().to_vec()),
            (vec![3u8], 30u64.to_be_bytes().to_vec()),
        ]);
        let digest = ADDigest::scorex_parse_bytes(&prover.digest().unwrap()).unwrap();

        let op = Operation::Lookup(Bytes::from(vec![2u8]));
        prover.perform_one_operation(&op).unwrap();

        let key = Constant::from(vec![2u8]);
        let proof: Constant = prover
            .generate_proof()
            .into_iter()
            .collect::<Vec<_>>()
            .into();
        let tree_flags = AvlTreeFlags::new(false, false, false);
        let obj = Expr::Const(
            AvlTreeData {
                digest: digest.0.to_vec(),
                tree_flags,
                key_length: 1,
                value_length_opt: None,
            }
            .into(),
        );
        let expr: Expr = MethodCall::new(
            obj,
            savltree::CONTAINS_METHOD.clone(),
            vec![key.into(), proof.into()],
        )
        .unwrap()
        .into();

        assert!(eval_out_wo_ctx::<bool>(&expr));
    }

    #[test]
    fn eval_avl_remove() {
        let mut prover = populate_tree(vec![(vec![1u8], 10u64.to_be_bytes().to_vec())]);
        let initial_digest = ADDigest::scorex_parse_bytes(&prover.digest().unwrap()).unwrap();

        let key1 = Bytes::from(vec![1u8]);
        let op1 = Operation::Remove(key1);
        prover.perform_one_operation(&op1).unwrap();
        let final_digest = ADDigest::scorex_parse_bytes(&prover.digest().unwrap()).unwrap();
        let proof: Constant = prover
            .generate_proof()
            .into_iter()
            .collect::<Vec<_>>()
            .into();

        let tree_flags = AvlTreeFlags::new(false, false, true);
        let obj = Expr::Const(
            AvlTreeData {
                digest: initial_digest.0.to_vec(),
                tree_flags,
                key_length: 1,
                value_length_opt: None,
            }
            .into(),
        );

        let key1 = Literal::from(vec![1u8]);
        let keys = Constant {
            tpe: SType::SColl(Arc::new(SType::SColl(Arc::new(SType::SByte)))),
            v: Literal::Coll(CollKind::WrappedColl {
                items: Arc::new([key1]),
                elem_tpe: SType::SColl(Arc::new(SType::SByte)),
            }),
        };
        let expr: Expr = MethodCall::new(
            obj,
            savltree::REMOVE_METHOD.clone(),
            vec![keys.into(), proof.into()],
        )
        .unwrap()
        .into();

        let res = eval_out_wo_ctx::<Value>(&expr);
        if let Value::Opt(opt) = res {
            if let Some(Value::AvlTree(avl)) = opt.as_deref() {
                assert_eq!(
                    Value::Coll(avl.digest.clone()),
                    Value::from(final_digest.0.to_vec())
                );
            } else {
                unreachable!();
            }
        } else {
            unreachable!();
        }
    }

    #[test]
    fn eval_avl_update() {
        let mut prover = populate_tree(vec![
            (vec![1u8], 10u64.to_be_bytes().to_vec()),
            (vec![2u8], 20u64.to_be_bytes().to_vec()),
            (vec![3u8], 30u64.to_be_bytes().to_vec()),
        ]);
        let initial_digest = ADDigest::scorex_parse_bytes(&prover.digest().unwrap()).unwrap();

        let op1 = Operation::Update(KeyValue {
            key: Bytes::from(vec![2u8]),
            value: Bytes::from(40u64.to_be_bytes().to_vec()),
        });
        let op2 = Operation::Update(KeyValue {
            key: Bytes::from(vec![3u8]),
            value: Bytes::from(50u64.to_be_bytes().to_vec()),
        });
        prover.perform_one_operation(&op1).unwrap();
        prover.perform_one_operation(&op2).unwrap();

        let final_digest = ADDigest::scorex_parse_bytes(&prover.digest().unwrap()).unwrap();
        let proof: Constant = prover
            .generate_proof()
            .into_iter()
            .collect::<Vec<_>>()
            .into();

        let tree_flags = AvlTreeFlags::new(false, true, false);
        let obj = Expr::Const(
            AvlTreeData {
                digest: initial_digest.0.to_vec(),
                tree_flags,
                key_length: 1,
                value_length_opt: None,
            }
            .into(),
        );

        let pair1 = Literal::Tup(mk_pair(2u8, 40u64).into());
        let pair2 = Literal::Tup(mk_pair(3u8, 50u64).into());
        let entries = Constant {
            tpe: SType::SColl(Arc::new(SType::STuple(STuple::pair(
                SType::SColl(Arc::new(SType::SByte)),
                SType::SColl(Arc::new(SType::SByte)),
            )))),
            v: Literal::Coll(CollKind::WrappedColl {
                items: Arc::new([pair1, pair2]),
                elem_tpe: SType::STuple(STuple::pair(
                    SType::SColl(Arc::new(SType::SByte)),
                    SType::SColl(Arc::new(SType::SByte)),
                )),
            }),
        };
        let expr: Expr = MethodCall::new(
            obj,
            savltree::UPDATE_METHOD.clone(),
            vec![entries.into(), proof.into()],
        )
        .unwrap()
        .into();

        let res = eval_out_wo_ctx::<Value>(&expr);
        if let Value::Opt(opt) = res {
            if let Some(Value::AvlTree(avl)) = opt.as_deref() {
                assert_eq!(
                    Value::Coll(avl.digest.clone()),
                    Value::from(final_digest.0.to_vec())
                );
            } else {
                unreachable!();
            }
        } else {
            unreachable!();
        }
    }

    /// A structurally-valid proof generated from a DIFFERENT tree, paired with
    /// the digest of a non-empty tree: the construction-level digest mismatch
    /// of the wrong-tree-proof family. The reference impl's verifier
    /// construction never throws — every op fails on the no-tree verifier and
    /// each method maps that per its own semantics (santa-eval
    /// `AvlTree.wrong_tree_proof`).
    fn wrong_tree_proof_fixture(tree_flags: AvlTreeFlags) -> (Expr, Constant) {
        // tree A: non-empty (one committed insert), so its digest cannot match
        // the empty-tree digest the wrong proof starts from
        let mut prover_a = BatchAVLProver::new(
            AVLTree::new(
                |digest| Node::LabelOnly(NodeHeader::new(Some(*digest), None)),
                1,
                None,
            ),
            true,
        );
        prover_a
            .perform_one_operation(&Operation::Insert(KeyValue {
                key: Bytes::from(vec![5u8]),
                value: Bytes::from(50u64.to_be_bytes().to_vec()),
            }))
            .unwrap();
        prover_a.generate_proof();
        let tree_a_digest = ADDigest::scorex_parse_bytes(&prover_a.digest().unwrap()).unwrap();

        // proof from tree B (empty start) inserting key 1
        let mut prover_b = BatchAVLProver::new(
            AVLTree::new(
                |digest| Node::LabelOnly(NodeHeader::new(Some(*digest), None)),
                1,
                None,
            ),
            true,
        );
        prover_b
            .perform_one_operation(&Operation::Insert(KeyValue {
                key: Bytes::from(vec![1u8]),
                value: Bytes::from(10u64.to_be_bytes().to_vec()),
            }))
            .unwrap();
        let wrong_proof: Constant = prover_b
            .generate_proof()
            .into_iter()
            .collect::<Vec<_>>()
            .into();

        let obj = Expr::Const(
            AvlTreeData {
                digest: tree_a_digest.0.to_vec(),
                tree_flags,
                key_length: 1,
                value_length_opt: None,
            }
            .into(),
        );
        (obj, wrong_proof)
    }

    fn pairs_coll(items: Arc<[Literal]>) -> Constant {
        Constant {
            tpe: SType::SColl(Arc::new(SType::STuple(STuple::pair(
                SType::SColl(Arc::new(SType::SByte)),
                SType::SColl(Arc::new(SType::SByte)),
            )))),
            v: Literal::Coll(CollKind::WrappedColl {
                items,
                elem_tpe: SType::STuple(STuple::pair(
                    SType::SColl(Arc::new(SType::SByte)),
                    SType::SColl(Arc::new(SType::SByte)),
                )),
            }),
        }
    }

    fn keys_coll(items: Arc<[Literal]>) -> Constant {
        Constant {
            tpe: SType::SColl(Arc::new(SType::SColl(Arc::new(SType::SByte)))),
            v: Literal::Coll(CollKind::WrappedColl {
                items,
                elem_tpe: SType::SColl(Arc::new(SType::SByte)),
            }),
        }
    }

    #[test]
    fn eval_avl_contains_bad_proof() {
        let (obj, wrong_proof) = wrong_tree_proof_fixture(AvlTreeFlags::new(false, false, false));
        let key: Constant = vec![1u8].into();
        let expr: Expr = MethodCall::new(
            obj,
            savltree::CONTAINS_METHOD.clone(),
            vec![key.into(), wrong_proof.into()],
        )
        .unwrap()
        .into();
        // the failed lookup maps to false (`contains_eval`)
        assert!(!eval_out_wo_ctx::<bool>(&expr));
    }

    #[test]
    fn eval_avl_get_bad_proof() {
        let (obj, wrong_proof) = wrong_tree_proof_fixture(AvlTreeFlags::new(false, false, false));
        let key: Constant = vec![1u8].into();
        let expr: Expr = MethodCall::new(
            obj,
            savltree::GET_METHOD.clone(),
            vec![key.into(), wrong_proof.into()],
        )
        .unwrap()
        .into();
        // the failed lookup raises "Tree proof is incorrect" (`get_eval`)
        assert!(try_eval_out_wo_ctx::<Value<'_>>(&expr).is_err());
    }

    // --- F4 degenerate-input family (ergo_avltree_rust#14) ---
    // Additional construction-failure modes the reference verifier handles
    // gracefully but the crate used to panic on: unparseable/garbage proof bytes
    // and an out-of-range (wrapped-negative) key length. With the crate returning
    // Err instead of panicking (PR #14), they route exactly like the wrong-tree
    // digest mismatch above. Requires that crate fix; validated locally via
    // [patch.crates-io]. (The op-level wrong-value-length mode — santa-eval
    // `AvlTree.per_op_failure` — is covered crate-side in #14 and routes like
    // `eval_avl_insert_bad_proof` once the op returns Err.)

    /// A valid non-empty-tree digest paired with garbage proof bytes: the proof
    /// does not parse at all, a construction failure distinct from the wrong-tree
    /// digest mismatch. santa-eval `AvlTree.bad_proof_bytes`.
    fn bad_proof_bytes_fixture(tree_flags: AvlTreeFlags) -> (Expr, Constant) {
        let mut prover = BatchAVLProver::new(
            AVLTree::new(
                |digest| Node::LabelOnly(NodeHeader::new(Some(*digest), None)),
                1,
                None,
            ),
            true,
        );
        prover
            .perform_one_operation(&Operation::Insert(KeyValue {
                key: Bytes::from(vec![5u8]),
                value: Bytes::from(50u64.to_be_bytes().to_vec()),
            }))
            .unwrap();
        prover.generate_proof();
        let digest = ADDigest::scorex_parse_bytes(&prover.digest().unwrap())
            .unwrap()
            .0
            .to_vec();
        let garbage_proof: Constant = vec![0u8].into();
        let obj = Expr::Const(
            AvlTreeData {
                digest,
                tree_flags,
                key_length: 1,
                value_length_opt: None,
            }
            .into(),
        );
        (obj, garbage_proof)
    }

    /// An AvlTreeData whose key length is the wrapped-negative `0x8000_0000`: the
    /// reference verifier reads keyLength as a signed Int, fails its
    /// `keyLength > 0` check, and runs ops on a no-tree verifier. santa-eval
    /// `AvlTree.negative_keylength_tree`.
    fn negative_keylength_fixture(tree_flags: AvlTreeFlags) -> (Expr, Constant) {
        let mut prover = BatchAVLProver::new(
            AVLTree::new(
                |digest| Node::LabelOnly(NodeHeader::new(Some(*digest), None)),
                1,
                None,
            ),
            true,
        );
        prover
            .perform_one_operation(&Operation::Insert(KeyValue {
                key: Bytes::from(vec![1u8]),
                value: Bytes::from(10u64.to_be_bytes().to_vec()),
            }))
            .unwrap();
        let proof: Constant = prover
            .generate_proof()
            .into_iter()
            .collect::<Vec<_>>()
            .into();
        let digest = ADDigest::scorex_parse_bytes(&prover.digest().unwrap())
            .unwrap()
            .0
            .to_vec();
        let obj = Expr::Const(
            AvlTreeData {
                digest,
                tree_flags,
                key_length: 0x8000_0000,
                value_length_opt: None,
            }
            .into(),
        );
        (obj, proof)
    }

    #[test]
    fn eval_avl_contains_garbage_proof_bytes() {
        let (obj, proof) = bad_proof_bytes_fixture(AvlTreeFlags::new(false, false, false));
        let key: Constant = vec![1u8].into();
        let expr: Expr = MethodCall::new(
            obj,
            savltree::CONTAINS_METHOD.clone(),
            vec![key.into(), proof.into()],
        )
        .unwrap()
        .into();
        assert!(!eval_out_wo_ctx::<bool>(&expr));
    }

    #[test]
    fn eval_avl_get_garbage_proof_bytes() {
        let (obj, proof) = bad_proof_bytes_fixture(AvlTreeFlags::new(false, false, false));
        let key: Constant = vec![1u8].into();
        let expr: Expr = MethodCall::new(
            obj,
            savltree::GET_METHOD.clone(),
            vec![key.into(), proof.into()],
        )
        .unwrap()
        .into();
        assert!(try_eval_out_wo_ctx::<Value<'_>>(&expr).is_err());
    }

    #[test]
    fn eval_avl_contains_negative_key_length() {
        let (obj, proof) = negative_keylength_fixture(AvlTreeFlags::new(false, false, false));
        let key: Constant = vec![1u8].into();
        let expr: Expr = MethodCall::new(
            obj,
            savltree::CONTAINS_METHOD.clone(),
            vec![key.into(), proof.into()],
        )
        .unwrap()
        .into();
        assert!(!eval_out_wo_ctx::<bool>(&expr));
    }

    #[test]
    fn eval_avl_get_many_bad_proof() {
        let (obj, wrong_proof) = wrong_tree_proof_fixture(AvlTreeFlags::new(false, false, false));
        let key1 = Literal::from(vec![1u8]);
        let expr: Expr = MethodCall::new(
            obj.clone(),
            savltree::GET_MANY_METHOD.clone(),
            vec![
                keys_coll(Arc::new([key1])).into(),
                wrong_proof.clone().into(),
            ],
        )
        .unwrap()
        .into();
        // the first failed lookup raises "Tree proof is incorrect" (`getMany_eval`)
        assert!(try_eval_out_wo_ctx::<Value<'_>>(&expr).is_err());

        // with no keys no lookup runs at all: the empty collection, not an error
        let expr: Expr = MethodCall::new(
            obj,
            savltree::GET_MANY_METHOD.clone(),
            vec![keys_coll(Arc::new([])).into(), wrong_proof.into()],
        )
        .unwrap()
        .into();
        let res = try_eval_out_wo_ctx::<Value<'_>>(&expr).unwrap();
        if let Value::Coll(CollValue::Generic { items, .. }) = res {
            assert!(items.is_empty());
        } else {
            unreachable!();
        }
    }

    #[test]
    fn eval_avl_insert_bad_proof() {
        let (obj, wrong_proof) = wrong_tree_proof_fixture(AvlTreeFlags::new(true, false, false));
        let pair1 = Literal::Tup(mk_pair(1u8, 10u64).into());
        let expr: Expr = MethodCall::new(
            obj.clone(),
            savltree::INSERT_METHOD.clone(),
            vec![
                pairs_coll(Arc::new([pair1])).into(),
                wrong_proof.clone().into(),
            ],
        )
        .unwrap()
        .into();
        // pre-v3 the first failed insert raises; from v3 it fast-breaks and the
        // None digest makes the method evaluate to None (issue #908)
        assert!(try_eval_out_with_version::<Value<'_>>(&expr, &force_any_val(), 0, 3).is_err());
        assert_eq!(
            try_eval_out_with_version::<Value<'_>>(&expr, &force_any_val(), 3, 3).unwrap(),
            Value::Opt(None)
        );

        // with no entries no op runs and the None digest yields None at every
        // version — no raise even pre-v3
        let expr: Expr = MethodCall::new(
            obj,
            savltree::INSERT_METHOD.clone(),
            vec![pairs_coll(Arc::new([])).into(), wrong_proof.into()],
        )
        .unwrap()
        .into();
        assert_eq!(
            try_eval_out_with_version::<Value<'_>>(&expr, &force_any_val(), 0, 3).unwrap(),
            Value::Opt(None)
        );
    }

    #[test]
    fn eval_avl_update_bad_proof() {
        let (obj, wrong_proof) = wrong_tree_proof_fixture(AvlTreeFlags::new(false, true, false));
        let pair1 = Literal::Tup(mk_pair(1u8, 10u64).into());
        let expr: Expr = MethodCall::new(
            obj,
            savltree::UPDATE_METHOD.clone(),
            vec![pairs_coll(Arc::new([pair1])).into(), wrong_proof.into()],
        )
        .unwrap()
        .into();
        // the failed update fast-breaks and the None digest yields None
        let res = eval_out_wo_ctx::<Value>(&expr);
        assert!(matches!(res, Value::Opt(None)));
    }

    #[test]
    fn eval_avl_remove_bad_proof() {
        let (obj, wrong_proof) = wrong_tree_proof_fixture(AvlTreeFlags::new(false, false, true));
        let key1 = Literal::from(vec![1u8]);
        let expr: Expr = MethodCall::new(
            obj,
            savltree::REMOVE_METHOD.clone(),
            vec![keys_coll(Arc::new([key1])).into(), wrong_proof.into()],
        )
        .unwrap()
        .into();
        // failed removes are ignored and the None digest yields None
        let res = eval_out_wo_ctx::<Value>(&expr);
        assert!(matches!(res, Value::Opt(None)));
    }

    #[test]
    fn eval_avl_remove_mismatched_op() {
        // a VALID proof (construction succeeds: the digest matches) committing
        // remove(1), but the script removes key 9 — the op fails against the
        // proof, invalidating the verifier, and the None digest yields None
        // where the reference impl never raises (`remove_eval` ignores op
        // results — cfor, no break)
        let mut prover = populate_tree(vec![(vec![1u8], 10u64.to_be_bytes().to_vec())]);
        let initial_digest = ADDigest::scorex_parse_bytes(&prover.digest().unwrap()).unwrap();
        prover
            .perform_one_operation(&Operation::Remove(Bytes::from(vec![1u8])))
            .unwrap();
        let proof: Constant = prover
            .generate_proof()
            .into_iter()
            .collect::<Vec<_>>()
            .into();

        let obj = Expr::Const(
            AvlTreeData {
                digest: initial_digest.0.to_vec(),
                tree_flags: AvlTreeFlags::new(false, false, true),
                key_length: 1,
                value_length_opt: None,
            }
            .into(),
        );
        let key9 = Literal::from(vec![9u8]);
        let expr: Expr = MethodCall::new(
            obj,
            savltree::REMOVE_METHOD.clone(),
            vec![keys_coll(Arc::new([key9])).into(), proof.into()],
        )
        .unwrap()
        .into();
        let res = eval_out_wo_ctx::<Value>(&expr);
        assert!(matches!(res, Value::Opt(None)));
    }

    // JVM-blessed byte vectors (santa-eval `AvlTree.wrong_tree_proof` /
    // `AvlTree.insert_wrong_tree`): a valid proof from a committed n=4 tree
    // against the n=8 digest, carried with the args as segregated constants.
    // The blessed sized header (`1a`/`1b` + size VLQ) is rewritten to the
    // non-sized `12`/`13` (size-bit cleared, size dropped) because the sized
    // parse path rejects non-SigmaProp roots — the same lenient deserialize
    // the conformance runner applies to expression-rooted corpus trees; body
    // bytes verbatim.
    #[test]
    fn eval_avl_contains_bad_proof_blessed_bytes() {
        let tree_bytes = base16::decode("120364fb2b77372d81da43ce2d72714aec79ae5fcac20a9aff426fe6afb476a6fbc02c04072001080e204466b8f1af03542b3c35de30426ec12b04fc34fa0a8c48289ec868e1e00aec550e8f0103a4950597f640451a7f628ea42ce4525890593c75ab337afd6315122093bc6474024466b8f1af03542b3c35de30426ec12b04fc34fa0a8c48289ec868e1e00aec558f23c5ee24a49084f24812ee9dbcdc9e11974b05cad682c21e88fca866bc85ce000000005a17a002ff03b44482bbfc4a39a5f1bd1a35c8699de6c3c51cc71376cea9fe4b85a283801397ff0401dc640973000273017302").unwrap();
        let tree = ErgoTree::sigma_parse_bytes_lenient_from_unsized(&tree_bytes).unwrap();
        let expr = tree.proposition().unwrap();
        let ctx = force_any_val::<Context>();
        let res = try_eval_out_with_version::<Value>(&expr, &ctx, 2, 2).unwrap();
        assert!(matches!(res, Value::Boolean(false)));
    }

    #[test]
    fn eval_avl_update_bad_proof_blessed_bytes() {
        let tree_bytes = base16::decode("120464fb2b77372d81da43ce2d72714aec79ae5fcac20a9aff426fe6afb476a6fbc02c04072001080e200b65ce5d0a76265e6734964bf1a7620da7c4fb2659a9047a7b61869a007fb1110e08000000005a17a04d0eb10103f9859bd9b3050c599dc5b506caa4f2367f286c60a9af489e69d797ab16b9cff5020b65ce5d0a76265e6734964bf1a7620da7c4fb2659a9047a7b61869a007fb1114466b8f1af03542b3c35de30426ec12b04fc34fa0a8c48289ec868e1e00aec55000000005a17a003000335abe8a6e6c7b70addacbec9267be07ec3ac1c5a196b4438261d4a84648739c6ff03b44482bbfc4a39a5f1bd1a35c8699de6c3c51cc71376cea9fe4b85a283801397ff0403dc640d73000283013c0e0e8602730173027303").unwrap();
        let tree = ErgoTree::sigma_parse_bytes_lenient_from_unsized(&tree_bytes).unwrap();
        let expr = tree.proposition().unwrap();
        let ctx = force_any_val::<Context>();
        let res = try_eval_out_with_version::<Value>(&expr, &ctx, 2, 2).unwrap();
        assert!(matches!(res, Value::Opt(None)));
    }

    #[test]
    fn eval_avl_remove_bad_proof_blessed_bytes() {
        let tree_bytes = base16::decode("120364fb2b77372d81da43ce2d72714aec79ae5fcac20a9aff426fe6afb476a6fbc02c04072001080e208f23c5ee24a49084f24812ee9dbcdc9e11974b05cad682c21e88fca866bc85ce0eb90103a4950597f640451a7f628ea42ce4525890593c75ab337afd6315122093bc6474024466b8f1af03542b3c35de30426ec12b04fc34fa0a8c48289ec868e1e00aec558f23c5ee24a49084f24812ee9dbcdc9e11974b05cad682c21e88fca866bc85ce000000005a17a002ff02c4e67333f14133fce4ed7bbf147642b3e7b2b8324268ea6ded1be0caa2da237f000000005a17a000039594489172346c6b22ac52210d14be7fdb02f6ec52b9e525b3a9c77114374cd900ff0402dc640e73000283010e73017302").unwrap();
        let tree = ErgoTree::sigma_parse_bytes_lenient_from_unsized(&tree_bytes).unwrap();
        let expr = tree.proposition().unwrap();
        let ctx = force_any_val::<Context>();
        let res = try_eval_out_with_version::<Value>(&expr, &ctx, 2, 2).unwrap();
        assert!(matches!(res, Value::Opt(None)));
    }

    #[test]
    fn eval_avl_insert_bad_proof_v3_blessed_bytes() {
        let tree_bytes = base16::decode("130464fb2b77372d81da43ce2d72714aec79ae5fcac20a9aff426fe6afb476a6fbc02c04072001080e209a39c57d13039b50bfe4aa21b2c8238be31d133db1fbe4549b16d9b94338b4e20e08000000005a17a0320e8f0103fcbfd9e0c4781263bb161625674719024acef64654799a00c41cc148bdc4a891028f23c5ee24a49084f24812ee9dbcdc9e11974b05cad682c21e88fca866bc85cec4e67333f14133fce4ed7bbf147642b3e7b2b8324268ea6ded1be0caa2da237f000000005a17a000039594489172346c6b22ac52210d14be7fdb02f6ec52b9e525b3a9c77114374cd900ff0402dc640c73000283013c0e0e8602730173027303").unwrap();
        let tree = ErgoTree::sigma_parse_bytes_lenient_from_unsized(&tree_bytes).unwrap();
        let expr = tree.proposition().unwrap();
        let ctx = force_any_val::<Context>();
        let res = try_eval_out_with_version::<Value>(&expr, &ctx, 3, 3).unwrap();
        assert!(matches!(res, Value::Opt(None)));
    }

    fn populate_tree(entries: Vec<(Vec<u8>, Vec<u8>)>) -> BatchAVLProver {
        let mut prover = BatchAVLProver::new(
            AVLTree::new(
                |digest| Node::LabelOnly(NodeHeader::new(Some(*digest), None)),
                1,
                None,
            ),
            true,
        );

        for (key, value) in entries {
            let op = Operation::Insert(KeyValue {
                key: Bytes::from(key),
                value: Bytes::from(value),
            });
            prover.perform_one_operation(&op).unwrap();
        }

        prover.generate_proof();
        prover
    }

    fn mk_pair(x: u8, y: u64) -> [Literal; 2] {
        [
            Literal::from(vec![x]),
            Literal::from(y.to_be_bytes().to_vec()),
        ]
    }

    /// SANTA tx-tier regression (captured testnet txs at heights 2666 / 28474):
    /// the verifier-backed AvlTree ops must charge the JVM's verifier costs
    /// (Scala `CErgoTreeEvaluator.createVerifier` + per-op height-scaled costs).
    /// Pre-fix all seven ops charged nothing, under-counting the aggregate tx
    /// cost of any AVL-using script — the consensus-risky direction near
    /// MaxBlockCost.
    ///
    /// Expectations are formula-derived (no magic totals), mirroring Scala:
    ///   createVerifier = 110 + 20 * chunks(proof.len, 64)
    ///   lookup         = 40 + 10 * h          (contains/get; per key in getMany)
    ///   insert         = 40 + 10 * max(h, 1)  per entry
    ///   remove         = 100 + 15 * max(h, 1) per key, then digest read (15)
    ///   updateDigest   = 40 on successful mutation
    /// where h = the digest's height byte (what `BatchAVLVerifier.rootNodeHeight`
    /// reads) and chunks(n, sz) = (n - 1)/sz + 1.
    #[test]
    fn avl_verifier_op_costs_match_jvm() {
        use crate::eval::test_util::try_eval_out;
        use ergotree_ir::chain::context::Context;

        let chunks = |n: u64, sz: u64| (n - 1) / sz + 1;
        // Expr overhead outside the AvlTree method eval itself:
        // Const(tree) 5 + MethodCall 4 + Const(arg1) 5 + Const(arg2) 5.
        const EXPR_OVERHEAD: u64 = 19;

        let cost_of = |expr: &Expr| -> u64 {
            let ctx = force_any_val::<Context>();
            let before = ctx.jit_cost_value();
            let _ = try_eval_out::<Value>(expr, &ctx).unwrap();
            ctx.jit_cost_value() - before
        };

        let keys_const = |keys: &[Vec<u8>]| Constant {
            tpe: SType::SColl(Arc::new(SType::SColl(Arc::new(SType::SByte)))),
            v: Literal::Coll(CollKind::WrappedColl {
                items: keys.iter().map(|k| Literal::from(k.clone())).collect(),
                elem_tpe: SType::SColl(Arc::new(SType::SByte)),
            }),
        };
        let tree_const = |digest: &ADDigest, flags: AvlTreeFlags| {
            Expr::Const(
                AvlTreeData {
                    digest: digest.0.to_vec(),
                    tree_flags: flags,
                    key_length: 1,
                    value_length_opt: None,
                }
                .into(),
            )
        };

        // --- get: createVerifier + one height-scaled lookup ---
        let mut prover = populate_tree(vec![(vec![1u8], 10u64.to_be_bytes().to_vec())]);
        let digest = ADDigest::scorex_parse_bytes(&prover.digest().unwrap()).unwrap();
        let h = u64::from(*digest.0.last().unwrap());
        prover
            .perform_one_operation(&Operation::Lookup(Bytes::from(vec![1u8])))
            .unwrap();
        let proof_bytes: Vec<u8> = prover.generate_proof().into_iter().collect();
        let plen = proof_bytes.len() as u64;
        let expr: Expr = MethodCall::new(
            tree_const(&digest, AvlTreeFlags::new(false, false, false)),
            savltree::GET_METHOD.clone(),
            vec![
                Constant::from(vec![1u8]).into(),
                Constant::from(proof_bytes).into(),
            ],
        )
        .unwrap()
        .into();
        assert_eq!(
            cost_of(&expr),
            EXPR_OVERHEAD + (110 + 20 * chunks(plen, 64)) + (40 + 10 * h),
            "get = expr overhead + createVerifier(proof) + lookup(h)"
        );

        // --- getMany: createVerifier + per-key height-scaled lookups ---
        let mut prover = populate_tree(vec![
            (vec![1u8], 10u64.to_be_bytes().to_vec()),
            (vec![2u8], 20u64.to_be_bytes().to_vec()),
        ]);
        let digest = ADDigest::scorex_parse_bytes(&prover.digest().unwrap()).unwrap();
        let h = u64::from(*digest.0.last().unwrap());
        for k in [vec![1u8], vec![2u8], vec![3u8]] {
            prover
                .perform_one_operation(&Operation::Lookup(Bytes::from(k)))
                .unwrap();
        }
        let proof_bytes: Vec<u8> = prover.generate_proof().into_iter().collect();
        let plen = proof_bytes.len() as u64;
        let expr: Expr = MethodCall::new(
            tree_const(&digest, AvlTreeFlags::new(false, false, false)),
            savltree::GET_MANY_METHOD.clone(),
            vec![
                keys_const(&[vec![1u8], vec![2u8], vec![3u8]]).into(),
                Constant::from(proof_bytes).into(),
            ],
        )
        .unwrap()
        .into();
        assert_eq!(
            cost_of(&expr),
            EXPR_OVERHEAD + (110 + 20 * chunks(plen, 64)) + 3 * (40 + 10 * h),
            "getMany = expr overhead + createVerifier(proof) + 3 lookups"
        );

        // --- remove: flag check + createVerifier + per-key remove + digest
        //     read + updateDigest (remove is the one mutation that also
        //     charges the unconditional digest read, mirroring Scala) ---
        let mut prover = populate_tree(vec![(vec![1u8], 10u64.to_be_bytes().to_vec())]);
        let digest = ADDigest::scorex_parse_bytes(&prover.digest().unwrap()).unwrap();
        let h = u64::from(*digest.0.last().unwrap());
        prover
            .perform_one_operation(&Operation::Remove(Bytes::from(vec![1u8])))
            .unwrap();
        let proof_bytes: Vec<u8> = prover.generate_proof().into_iter().collect();
        let plen = proof_bytes.len() as u64;
        let expr: Expr = MethodCall::new(
            tree_const(&digest, AvlTreeFlags::new(false, false, true)),
            savltree::REMOVE_METHOD.clone(),
            vec![
                keys_const(&[vec![1u8]]).into(),
                Constant::from(proof_bytes).into(),
            ],
        )
        .unwrap()
        .into();
        assert_eq!(
            cost_of(&expr),
            EXPR_OVERHEAD + 15 + (110 + 20 * chunks(plen, 64)) + (100 + 15 * h.max(1)) + 15 + 40,
            "remove = expr overhead + isRemoveAllowed + createVerifier(proof) \
             + remove(max(h,1)) + digest + updateDigest"
        );

        // --- insert on an empty tree: max(h, 1) keeps per-entry cost nonzero ---
        let prover = BatchAVLProver::new(
            AVLTree::new(
                |digest| Node::LabelOnly(NodeHeader::new(Some(*digest), None)),
                1,
                None,
            ),
            true,
        );
        let digest = ADDigest::scorex_parse_bytes(&prover.digest().unwrap()).unwrap();
        let h = u64::from(*digest.0.last().unwrap());
        assert_eq!(h, 0, "fresh tree digest height byte");
        let mut prover = prover;
        prover
            .perform_one_operation(&Operation::Insert(KeyValue {
                key: Bytes::from(vec![1u8]),
                value: Bytes::from(10u64.to_be_bytes().to_vec()),
            }))
            .unwrap();
        let proof_bytes: Vec<u8> = prover.generate_proof().into_iter().collect();
        let plen = proof_bytes.len() as u64;
        let entries = Constant {
            tpe: SType::SColl(Arc::new(SType::STuple(STuple::pair(
                SType::SColl(Arc::new(SType::SByte)),
                SType::SColl(Arc::new(SType::SByte)),
            )))),
            v: Literal::Coll(CollKind::WrappedColl {
                items: Arc::new([Literal::Tup(mk_pair(1u8, 10u64).into())]),
                elem_tpe: SType::STuple(STuple::pair(
                    SType::SColl(Arc::new(SType::SByte)),
                    SType::SColl(Arc::new(SType::SByte)),
                )),
            }),
        };
        let insert_expr = |flags: AvlTreeFlags, proof: Vec<u8>| -> Expr {
            MethodCall::new(
                tree_const(&digest, flags),
                savltree::INSERT_METHOD.clone(),
                vec![entries.clone().into(), Constant::from(proof).into()],
            )
            .unwrap()
            .into()
        };
        assert_eq!(
            cost_of(&insert_expr(
                AvlTreeFlags::new(true, false, false),
                proof_bytes.clone()
            )),
            EXPR_OVERHEAD + 15 + (110 + 20 * chunks(plen, 64)) + (40 + 10 * h.max(1)) + 40,
            "insert = expr overhead + isInsertAllowed + createVerifier(proof) \
             + insert(max(h,1)) + updateDigest"
        );

        // --- disallowed mutation: only the flag check is charged ---
        assert_eq!(
            cost_of(&insert_expr(
                AvlTreeFlags::new(false, false, false),
                proof_bytes
            )),
            EXPR_OVERHEAD + 15,
            "insert on a non-insert-allowed tree charges only the flag check"
        );
    }

    /// JVM parity: `BatchAVLVerifier.rootNodeHeight` is assigned from the
    /// digest's trailing byte only after the `keyLength > 0` and digest-length
    /// requires; a non-positive keyLength (signed `Int` on the JVM) or a
    /// non-33-byte digest fails reconstruction before the assignment, so the
    /// op-cost height is 0 (`CAvlTreeVerifier.treeHeight`).
    #[test]
    fn tree_height_zero_when_reconstruction_cannot_start() {
        let mut digest_bytes = vec![0u8; 33];
        digest_bytes[32] = 4;
        let tree = |digest: Vec<u8>, key_length: u32| AvlTreeData {
            digest,
            tree_flags: AvlTreeFlags::new(false, false, false),
            key_length,
            value_length_opt: None,
        };
        // a valid-shaped tree keeps its digest-derived height
        assert_eq!(tree_height(&tree(digest_bytes.clone(), 32)), 4);
        // keyLength 0 or negative-on-the-JVM (high bit set) => height 0
        assert_eq!(tree_height(&tree(digest_bytes.clone(), 0)), 0);
        assert_eq!(tree_height(&tree(digest_bytes.clone(), u32::MAX)), 0); // JVM Int -1
        assert_eq!(tree_height(&tree(digest_bytes, 0x8000_0000)), 0); // JVM Int.MinValue

        // a non-33-byte in-memory digest (via updateDigest) => height 0
        assert_eq!(tree_height(&tree(vec![1, 2, 3], 32)), 0);
        assert_eq!(tree_height(&tree(vec![0u8; 32], 32)), 0);
        assert_eq!(tree_height(&tree(vec![0u8; 34], 32)), 0);
    }
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
#[cfg(feature = "arbitrary")]
mod charge_order {
    use crate::eval::test_util::{eval_body, eval_body_err, innermost};
    use crate::eval::EvalError;
    use ergotree_ir::ergo_tree::ErgoTreeVersion;
    use ergotree_ir::mir::value::Value;

    /// A tree constant: an all-zero 33-byte digest, every operation allowed, 32-byte keys,
    /// no fixed value length
    const TREE: &str = "64000000000000000000000000000000000000000000000000000000000000000000072000";

    // A proof that is a collection of another item type passes the call's class check.
    // sigmastate's `createVerifier` adds its per-item cost over the proof's LENGTH and builds
    // the verifier inside that block, where the proof's array is cast
    // (`CErgoTreeEvaluator.scala:71-75`); a modifying method has added its flag costs before.
    // Each row: an empty key, keys or entries, and an empty `Coll[Int]` as the proof.
    // Measured on the sigma-state 6.0.7 jars.
    #[test]
    fn a_proof_of_another_item_type_pays_for_the_verifier() {
        for (method_id, args, cost) in [
            ("09", "020e001000", 149),       // contains
            ("0a", "020e001000", 149),       // get
            ("0b", "021a001000", 149),       // getMany
            ("0c", "020c3c0e0e001000", 164), // insert
            ("0d", "020c3c0e0e001000", 164), // update
            ("0e", "021a001000", 164),       // remove
            ("10", "020c3c0e0e001000", 179), // insertOrUpdate
        ] {
            let body = alloc::format!("dc64{method_id}{TREE}{args}");
            let (e, c) = eval_body_err(&body);
            assert!(matches!(e, EvalError::TryExtractFrom(_)), "{body}: {e:?}");
            assert_eq!(c, cost, "{body}");
        }
    }

    // `insert` (from tree version 3), `update` and `insertOrUpdate` stop at the first entry
    // whose operation fails, so an entry after it is never read: a second entry of another
    // item type, or one that is no pair, does not make the call fail. Each row's first entry
    // is right in kind and fails its operation under a VALID proof (an insert of a key the
    // tree holds, an update of a key it lacks); the JVM answers `None`. The last row is the
    // control with that entry alone.
    #[test]
    fn an_entry_after_a_failed_one_is_never_read() {
        #[rustfmt::skip]
        let rows = [
            // built-insert-first-fails-second-int
            ("dc640c6415cc1ba1054bf74431a2ccfc57e6051b9d0e7561f70586e5512c0ae3f3bac4fd0207010002483c0e0e3c100e04010108000000000000000a01040105000e550385ab460a6564d1e5ded17716cd5866650d7fbb9cada7fecabd5fe21e2f80e43a039d4d9434aa05d4dc888d939d6771fe05418df5bd91101dbe63384514c1869e140202ff00000008000000000000001400010400", 244),
            // built-insert-first-fails-second-nopair
            ("dc640c6415cc1ba1054bf74431a2ccfc57e6051b9d0e7561f70586e5512c0ae3f3bac4fd0207010002483c0e0e0404010108000000000000000a0a000e550385ab460a6564d1e5ded17716cd5866650d7fbb9cada7fecabd5fe21e2f80e43a039d4d9434aa05d4dc888d939d6771fe05418df5bd91101dbe63384514c1869e140202ff00000008000000000000001400010400", 244),
            // built-update-first-fails-second-int
            ("dc640d6415cc1ba1054bf74431a2ccfc57e6051b9d0e7561f70586e5512c0ae3f3bac4fd0207010002483c0e0e3c100e04010308000000000000001e01040105000e550385ab460a6564d1e5ded17716cd5866650d7fbb9cada7fecabd5fe21e2f80e43a02010200000008000000000000000a03fc29fc93353c639f7cf275ea69851dfa41d720ed53bcc9c10bda17166966d5ed00010402", 344),
            // built-update-first-fails-second-nopair
            ("dc640d6415cc1ba1054bf74431a2ccfc57e6051b9d0e7561f70586e5512c0ae3f3bac4fd0207010002483c0e0e0404010308000000000000001e0a000e550385ab460a6564d1e5ded17716cd5866650d7fbb9cada7fecabd5fe21e2f80e43a02010200000008000000000000000a03fc29fc93353c639f7cf275ea69851dfa41d720ed53bcc9c10bda17166966d5ed00010402", 344),
            // built-iou-first-fails-second-int
            ("dc64106415cc1ba1054bf74431a2ccfc57e6051b9d0e7561f70586e5512c0ae3f3bac4fd0207010002483c0e0e3c100e04010108000000000000000a01040105000e550385ab460a6564d1e5ded17716cd5866650d7fbb9cada7fecabd5fe21e2f80e43a039d4d9434aa05d4dc888d939d6771fe05418df5bd91101dbe63384514c1869e140202ff00000008000000000000001400010400", 359),
            // built-insert-first-fails-control
            ("dc640c6415cc1ba1054bf74431a2ccfc57e6051b9d0e7561f70586e5512c0ae3f3bac4fd02070100020c3c0e0e01010108000000000000000a0e550385ab460a6564d1e5ded17716cd5866650d7fbb9cada7fecabd5fe21e2f80e43a039d4d9434aa05d4dc888d939d6771fe05418df5bd91101dbe63384514c1869e140202ff00000008000000000000001400010400", 244),
        ];
        for (body, cost) in rows {
            let (res, c) = eval_body(body, ErgoTreeVersion::V3);
            assert!(matches!(res, Ok(Value::Opt(None))), "{body}: {res:?}");
            assert_eq!(c, cost, "{body}");
        }
    }

    #[derive(Debug)]
    enum Refusal {
        /// an array that is not of bytes
        Cast,
        /// an entry that is no pair
        NoPair,
    }

    // sigmastate reads a key or an entry where it uses it: inside the operation's cost block,
    // so after that item's cost (`CErgoTreeEvaluator.scala:78-241`: `key.toArray`,
    // `performInsert(key.toArray, value.toArray)`), while an entry that is no pair fails the
    // `case (key, value)` match before its cost. Each tree below holds {1: 10, 2: 20} under
    // 1-byte keys with a VALID proof for its operation; a key is a `Coll[Int]`, and the keys
    // and entries are 3-tuple constants (a `Coll[Any]` value) holding an item of another
    // type first or second. Measured on the sigma-state 6.0.7 jars.
    #[test]
    fn a_key_or_entry_is_read_where_the_jvm_casts_it() {
        use Refusal::*;
        #[rustfmt::skip]
        let refused = [
            // contains|key-Coll[Int]
            ("dc64096415cc1ba1054bf74431a2ccfc57e6051b9d0e7561f70586e5512c0ae3f3bac4fd02070100021001020e550385ab460a6564d1e5ded17716cd5866650d7fbb9cada7fecabd5fe21e2f80e43a02010200000008000000000000000a03fc29fc93353c639f7cf275ea69851dfa41d720ed53bcc9c10bda17166966d5ed00010402", Cast, 229),
            // get|key-Coll[Int]
            ("dc640a6415cc1ba1054bf74431a2ccfc57e6051b9d0e7561f70586e5512c0ae3f3bac4fd02070100021001020e550385ab460a6564d1e5ded17716cd5866650d7fbb9cada7fecabd5fe21e2f80e43a02010200000008000000000000000a03fc29fc93353c639f7cf275ea69851dfa41d720ed53bcc9c10bda17166966d5ed00010402", Cast, 229),
            // getMany|keys-second-Coll[Int]
            ("dc640b6415cc1ba1054bf74431a2ccfc57e6051b9d0e7561f70586e5512c0ae3f3bac4fd0207010002480e100401010104000e420385ab460a6564d1e5ded17716cd5866650d7fbb9cada7fecabd5fe21e2f80e43a02010200000008000000000000000a02ff00000008000000000000001400010402", Cast, 289),
            // getMany|keys-first-Coll[Int]
            ("dc640b6415cc1ba1054bf74431a2ccfc57e6051b9d0e7561f70586e5512c0ae3f3bac4fd020701000248100e0401040101000e420385ab460a6564d1e5ded17716cd5866650d7fbb9cada7fecabd5fe21e2f80e43a02010200000008000000000000000a02ff00000008000000000000001400010402", Cast, 229),
            // insert|entries-second-key-Coll[Int]
            ("dc640c6415cc1ba1054bf74431a2ccfc57e6051b9d0e7561f70586e5512c0ae3f3bac4fd0207010002483c0e0e3c100e04010308000000000000001e01040105000e550385ab460a6564d1e5ded17716cd5866650d7fbb9cada7fecabd5fe21e2f80e43a039d4d9434aa05d4dc888d939d6771fe05418df5bd91101dbe63384514c1869e140202ff00000008000000000000001400010400", Cast, 304),
            // insert|entries-second-value-Coll[Int]
            ("dc640c6415cc1ba1054bf74431a2ccfc57e6051b9d0e7561f70586e5512c0ae3f3bac4fd0207010002483c0e0e3c0e1004010308000000000000001e01050104000e550385ab460a6564d1e5ded17716cd5866650d7fbb9cada7fecabd5fe21e2f80e43a039d4d9434aa05d4dc888d939d6771fe05418df5bd91101dbe63384514c1869e140202ff00000008000000000000001400010400", Cast, 304),
            // insert|entries-second-no-pair
            ("dc640c6415cc1ba1054bf74431a2ccfc57e6051b9d0e7561f70586e5512c0ae3f3bac4fd0207010002483c0e0e0404010308000000000000001e0a000e550385ab460a6564d1e5ded17716cd5866650d7fbb9cada7fecabd5fe21e2f80e43a039d4d9434aa05d4dc888d939d6771fe05418df5bd91101dbe63384514c1869e140202ff00000008000000000000001400010400", NoPair, 244),
            // insert|entries-first-no-pair
            ("dc640c6415cc1ba1054bf74431a2ccfc57e6051b9d0e7561f70586e5512c0ae3f3bac4fd020701000248043c0e0e040a010308000000000000001e000e550385ab460a6564d1e5ded17716cd5866650d7fbb9cada7fecabd5fe21e2f80e43a039d4d9434aa05d4dc888d939d6771fe05418df5bd91101dbe63384514c1869e140202ff00000008000000000000001400010400", NoPair, 184),
            // update|entries-second-key-Coll[Int]
            ("dc640d6415cc1ba1054bf74431a2ccfc57e6051b9d0e7561f70586e5512c0ae3f3bac4fd0207010002483c0e0e3c100e04010108000000000000000b01040105000e550385ab460a6564d1e5ded17716cd5866650d7fbb9cada7fecabd5fe21e2f80e43a02010200000008000000000000000a03fc29fc93353c639f7cf275ea69851dfa41d720ed53bcc9c10bda17166966d5ed00010402", Cast, 504),
            // update|entries-second-value-Coll[Int]
            ("dc640d6415cc1ba1054bf74431a2ccfc57e6051b9d0e7561f70586e5512c0ae3f3bac4fd0207010002483c0e0e3c0e1004010108000000000000000b01050104000e550385ab460a6564d1e5ded17716cd5866650d7fbb9cada7fecabd5fe21e2f80e43a02010200000008000000000000000a03fc29fc93353c639f7cf275ea69851dfa41d720ed53bcc9c10bda17166966d5ed00010402", Cast, 504),
            // update|entries-second-no-pair
            ("dc640d6415cc1ba1054bf74431a2ccfc57e6051b9d0e7561f70586e5512c0ae3f3bac4fd0207010002483c0e0e0404010108000000000000000b0a000e550385ab460a6564d1e5ded17716cd5866650d7fbb9cada7fecabd5fe21e2f80e43a02010200000008000000000000000a03fc29fc93353c639f7cf275ea69851dfa41d720ed53bcc9c10bda17166966d5ed00010402", NoPair, 344),
            // update|entries-first-no-pair
            ("dc640d6415cc1ba1054bf74431a2ccfc57e6051b9d0e7561f70586e5512c0ae3f3bac4fd020701000248043c0e0e040a010108000000000000000b000e550385ab460a6564d1e5ded17716cd5866650d7fbb9cada7fecabd5fe21e2f80e43a02010200000008000000000000000a03fc29fc93353c639f7cf275ea69851dfa41d720ed53bcc9c10bda17166966d5ed00010402", NoPair, 184),
            // insertOrUpdate|entries-second-key-Coll[Int]
            ("dc64106415cc1ba1054bf74431a2ccfc57e6051b9d0e7561f70586e5512c0ae3f3bac4fd0207010002483c0e0e3c100e04010308000000000000001e01040105000e550385ab460a6564d1e5ded17716cd5866650d7fbb9cada7fecabd5fe21e2f80e43a039d4d9434aa05d4dc888d939d6771fe05418df5bd91101dbe63384514c1869e140202ff00000008000000000000001400010400", Cast, 519),
            // insertOrUpdate|entries-second-value-Coll[Int]
            ("dc64106415cc1ba1054bf74431a2ccfc57e6051b9d0e7561f70586e5512c0ae3f3bac4fd0207010002483c0e0e3c0e1004010308000000000000001e01050104000e550385ab460a6564d1e5ded17716cd5866650d7fbb9cada7fecabd5fe21e2f80e43a039d4d9434aa05d4dc888d939d6771fe05418df5bd91101dbe63384514c1869e140202ff00000008000000000000001400010400", Cast, 519),
            // insertOrUpdate|entries-second-no-pair
            ("dc64106415cc1ba1054bf74431a2ccfc57e6051b9d0e7561f70586e5512c0ae3f3bac4fd0207010002483c0e0e0404010308000000000000001e0a000e550385ab460a6564d1e5ded17716cd5866650d7fbb9cada7fecabd5fe21e2f80e43a039d4d9434aa05d4dc888d939d6771fe05418df5bd91101dbe63384514c1869e140202ff00000008000000000000001400010400", NoPair, 359),
            // insertOrUpdate|entries-first-no-pair
            ("dc64106415cc1ba1054bf74431a2ccfc57e6051b9d0e7561f70586e5512c0ae3f3bac4fd020701000248043c0e0e040a010308000000000000001e000e550385ab460a6564d1e5ded17716cd5866650d7fbb9cada7fecabd5fe21e2f80e43a039d4d9434aa05d4dc888d939d6771fe05418df5bd91101dbe63384514c1869e140202ff00000008000000000000001400010400", NoPair, 199),
            // remove|keys-second-Coll[Int]
            ("dc640e6415cc1ba1054bf74431a2ccfc57e6051b9d0e7561f70586e5512c0ae3f3bac4fd0207010002480e100401010104000e3a02000100000000020200000008000000000000000a03fc29fc93353c639f7cf275ea69851dfa41d720ed53bcc9c10bda17166966d5ed00010402", Cast, 424),
            // remove|keys-first-Coll[Int]
            ("dc640e6415cc1ba1054bf74431a2ccfc57e6051b9d0e7561f70586e5512c0ae3f3bac4fd020701000248100e0401040101000e3a02000100000000020200000008000000000000000a03fc29fc93353c639f7cf275ea69851dfa41d720ed53bcc9c10bda17166966d5ed00010402", Cast, 294),
        ];
        for (body, refusal, cost) in refused {
            let (e, c) = eval_body_err(body);
            let named = match refusal {
                Cast => matches!(e, EvalError::TryExtractFrom(_)),
                NoPair => matches!(e, EvalError::UnexpectedValue(_)),
            };
            assert!(named, "{body}: expected {refusal:?}, got {e:?}");
            assert_eq!(c, cost, "{body}");
        }
        // the controls, and the same wrong entries or keys under a tree whose flag is off: the
        // JVM answers `None` before it reads anything
        #[rustfmt::skip]
        let evaluated = [
            // contains|control
            ("dc64096415cc1ba1054bf74431a2ccfc57e6051b9d0e7561f70586e5512c0ae3f3bac4fd02070100020e01010e550385ab460a6564d1e5ded17716cd5866650d7fbb9cada7fecabd5fe21e2f80e43a02010200000008000000000000000a03fc29fc93353c639f7cf275ea69851dfa41d720ed53bcc9c10bda17166966d5ed00010402", false, 229),
            // get|control
            ("dc640a6415cc1ba1054bf74431a2ccfc57e6051b9d0e7561f70586e5512c0ae3f3bac4fd02070100020e01010e550385ab460a6564d1e5ded17716cd5866650d7fbb9cada7fecabd5fe21e2f80e43a02010200000008000000000000000a03fc29fc93353c639f7cf275ea69851dfa41d720ed53bcc9c10bda17166966d5ed00010402", false, 229),
            // getMany|control
            ("dc640b6415cc1ba1054bf74431a2ccfc57e6051b9d0e7561f70586e5512c0ae3f3bac4fd02070100021a02010101020e420385ab460a6564d1e5ded17716cd5866650d7fbb9cada7fecabd5fe21e2f80e43a02010200000008000000000000000a02ff00000008000000000000001400010402", false, 289),
            // insert|control
            ("dc640c6415cc1ba1054bf74431a2ccfc57e6051b9d0e7561f70586e5512c0ae3f3bac4fd02070100020c3c0e0e01010308000000000000001e0e550385ab460a6564d1e5ded17716cd5866650d7fbb9cada7fecabd5fe21e2f80e43a039d4d9434aa05d4dc888d939d6771fe05418df5bd91101dbe63384514c1869e140202ff00000008000000000000001400010400", false, 284),
            // insert|flag-off|entries-second-key-Coll[Int]
            ("dc640c6415cc1ba1054bf74431a2ccfc57e6051b9d0e7561f70586e5512c0ae3f3bac4fd0200010002483c0e0e3c100e04010308000000000000001e01040105000e550385ab460a6564d1e5ded17716cd5866650d7fbb9cada7fecabd5fe21e2f80e43a039d4d9434aa05d4dc888d939d6771fe05418df5bd91101dbe63384514c1869e140202ff00000008000000000000001400010400", true, 34),
            // update|control
            ("dc640d6415cc1ba1054bf74431a2ccfc57e6051b9d0e7561f70586e5512c0ae3f3bac4fd02070100020c3c0e0e01010108000000000000000b0e550385ab460a6564d1e5ded17716cd5866650d7fbb9cada7fecabd5fe21e2f80e43a02010200000008000000000000000a03fc29fc93353c639f7cf275ea69851dfa41d720ed53bcc9c10bda17166966d5ed00010402", false, 384),
            // update|flag-off|entries-second-key-Coll[Int]
            ("dc640d6415cc1ba1054bf74431a2ccfc57e6051b9d0e7561f70586e5512c0ae3f3bac4fd0200010002483c0e0e3c100e04010108000000000000000b01040105000e550385ab460a6564d1e5ded17716cd5866650d7fbb9cada7fecabd5fe21e2f80e43a02010200000008000000000000000a03fc29fc93353c639f7cf275ea69851dfa41d720ed53bcc9c10bda17166966d5ed00010402", true, 34),
            // insertOrUpdate|control
            ("dc64106415cc1ba1054bf74431a2ccfc57e6051b9d0e7561f70586e5512c0ae3f3bac4fd02070100020c3c0e0e01010308000000000000001e0e550385ab460a6564d1e5ded17716cd5866650d7fbb9cada7fecabd5fe21e2f80e43a039d4d9434aa05d4dc888d939d6771fe05418df5bd91101dbe63384514c1869e140202ff00000008000000000000001400010400", false, 399),
            // insertOrUpdate|flag-off|entries-second-key-Coll[Int]
            ("dc64106415cc1ba1054bf74431a2ccfc57e6051b9d0e7561f70586e5512c0ae3f3bac4fd0200010002483c0e0e3c100e04010308000000000000001e01040105000e550385ab460a6564d1e5ded17716cd5866650d7fbb9cada7fecabd5fe21e2f80e43a039d4d9434aa05d4dc888d939d6771fe05418df5bd91101dbe63384514c1869e140202ff00000008000000000000001400010400", true, 49),
            // remove|control
            ("dc640e6415cc1ba1054bf74431a2ccfc57e6051b9d0e7561f70586e5512c0ae3f3bac4fd02070100021a0101010e3a02000100000000020200000008000000000000000a03fc29fc93353c639f7cf275ea69851dfa41d720ed53bcc9c10bda17166966d5ed00010402", false, 349),
            // remove|flag-off|keys-second-Coll[Int]
            ("dc640e6415cc1ba1054bf74431a2ccfc57e6051b9d0e7561f70586e5512c0ae3f3bac4fd0200010002480e100401010104000e3a02000100000000020200000008000000000000000a03fc29fc93353c639f7cf275ea69851dfa41d720ed53bcc9c10bda17166966d5ed00010402", true, 34),
        ];
        for (body, is_none, cost) in evaluated {
            let (res, c) = eval_body(body, ErgoTreeVersion::V3);
            let v = res.unwrap();
            assert_eq!(matches!(v, Value::Opt(None)), is_none, "{body}: {v:?}");
            assert_eq!(c, cost, "{body}");
        }
    }

    // The same under a tree the proofs do not belong to ({5: 50}): sigmastate builds a verifier
    // with no tree, every operation fails, and the order of each item's cost and its casts is
    // the same. `insert`, `update` and `insertOrUpdate` stop at the first failed entry, so a
    // wrong SECOND entry is never read; `remove` goes on. Measured on the jars.
    #[test]
    fn a_key_or_entry_is_read_where_the_jvm_casts_it_also_without_a_tree() {
        #[rustfmt::skip]
        let refused = [
            // contains|other-tree|key-Coll[Int]
            ("dc6409647ca99024833a7c538e35710c94acf164ca93693117acee3a5bfdd3035810ee1101070100021001020e550385ab460a6564d1e5ded17716cd5866650d7fbb9cada7fecabd5fe21e2f80e43a02010200000008000000000000000a03fc29fc93353c639f7cf275ea69851dfa41d720ed53bcc9c10bda17166966d5ed00010402", "Cast", 219),
            // get|other-tree|control
            ("dc640a647ca99024833a7c538e35710c94acf164ca93693117acee3a5bfdd3035810ee1101070100020e01010e550385ab460a6564d1e5ded17716cd5866650d7fbb9cada7fecabd5fe21e2f80e43a02010200000008000000000000000a03fc29fc93353c639f7cf275ea69851dfa41d720ed53bcc9c10bda17166966d5ed00010402", "Proof", 219),
            // get|other-tree|key-Coll[Int]
            ("dc640a647ca99024833a7c538e35710c94acf164ca93693117acee3a5bfdd3035810ee1101070100021001020e550385ab460a6564d1e5ded17716cd5866650d7fbb9cada7fecabd5fe21e2f80e43a02010200000008000000000000000a03fc29fc93353c639f7cf275ea69851dfa41d720ed53bcc9c10bda17166966d5ed00010402", "Cast", 219),
            // getMany|other-tree|control
            ("dc640b647ca99024833a7c538e35710c94acf164ca93693117acee3a5bfdd3035810ee1101070100021a02010101020e420385ab460a6564d1e5ded17716cd5866650d7fbb9cada7fecabd5fe21e2f80e43a02010200000008000000000000000a02ff00000008000000000000001400010402", "Proof", 219),
            // getMany|other-tree|keys-first-Coll[Int]
            ("dc640b647ca99024833a7c538e35710c94acf164ca93693117acee3a5bfdd3035810ee11010701000248100e0401040101000e420385ab460a6564d1e5ded17716cd5866650d7fbb9cada7fecabd5fe21e2f80e43a02010200000008000000000000000a02ff00000008000000000000001400010402", "Cast", 219),
            // getMany|other-tree|keys-second-Coll[Int]
            ("dc640b647ca99024833a7c538e35710c94acf164ca93693117acee3a5bfdd3035810ee110107010002480e100401010104000e420385ab460a6564d1e5ded17716cd5866650d7fbb9cada7fecabd5fe21e2f80e43a02010200000008000000000000000a02ff00000008000000000000001400010402", "Proof", 219),
            // insert|other-tree|entries-first-key-Coll[Int]
            ("dc640c647ca99024833a7c538e35710c94acf164ca93693117acee3a5bfdd3035810ee110107010002483c100e3c0e0e0401040105010308000000000000001e000e550385ab460a6564d1e5ded17716cd5866650d7fbb9cada7fecabd5fe21e2f80e43a039d4d9434aa05d4dc888d939d6771fe05418df5bd91101dbe63384514c1869e140202ff00000008000000000000001400010400", "Cast", 234),
            // insert|other-tree|entries-first-no-pair
            ("dc640c647ca99024833a7c538e35710c94acf164ca93693117acee3a5bfdd3035810ee11010701000248043c0e0e040a010308000000000000001e000e550385ab460a6564d1e5ded17716cd5866650d7fbb9cada7fecabd5fe21e2f80e43a039d4d9434aa05d4dc888d939d6771fe05418df5bd91101dbe63384514c1869e140202ff00000008000000000000001400010400", "NoPair", 184),
            // update|other-tree|entries-first-key-Coll[Int]
            ("dc640d647ca99024833a7c538e35710c94acf164ca93693117acee3a5bfdd3035810ee110107010002483c100e3c0e0e0401040105010108000000000000000b000e550385ab460a6564d1e5ded17716cd5866650d7fbb9cada7fecabd5fe21e2f80e43a02010200000008000000000000000a03fc29fc93353c639f7cf275ea69851dfa41d720ed53bcc9c10bda17166966d5ed00010402", "Cast", 324),
            // update|other-tree|entries-first-no-pair
            ("dc640d647ca99024833a7c538e35710c94acf164ca93693117acee3a5bfdd3035810ee11010701000248043c0e0e040a010108000000000000000b000e550385ab460a6564d1e5ded17716cd5866650d7fbb9cada7fecabd5fe21e2f80e43a02010200000008000000000000000a03fc29fc93353c639f7cf275ea69851dfa41d720ed53bcc9c10bda17166966d5ed00010402", "NoPair", 184),
            // insertOrUpdate|other-tree|entries-first-key-Coll[Int]
            ("dc6410647ca99024833a7c538e35710c94acf164ca93693117acee3a5bfdd3035810ee110107010002483c100e3c0e0e0401040105010308000000000000001e000e550385ab460a6564d1e5ded17716cd5866650d7fbb9cada7fecabd5fe21e2f80e43a039d4d9434aa05d4dc888d939d6771fe05418df5bd91101dbe63384514c1869e140202ff00000008000000000000001400010400", "Cast", 339),
            // insertOrUpdate|other-tree|entries-first-no-pair
            ("dc6410647ca99024833a7c538e35710c94acf164ca93693117acee3a5bfdd3035810ee11010701000248043c0e0e040a010308000000000000001e000e550385ab460a6564d1e5ded17716cd5866650d7fbb9cada7fecabd5fe21e2f80e43a039d4d9434aa05d4dc888d939d6771fe05418df5bd91101dbe63384514c1869e140202ff00000008000000000000001400010400", "NoPair", 199),
            // remove|other-tree|keys-first-Coll[Int]
            ("dc640e647ca99024833a7c538e35710c94acf164ca93693117acee3a5bfdd3035810ee11010701000248100e0401040101000e3a02000100000000020200000008000000000000000a03fc29fc93353c639f7cf275ea69851dfa41d720ed53bcc9c10bda17166966d5ed00010402", "Cast", 279),
            // remove|other-tree|keys-second-Coll[Int]
            ("dc640e647ca99024833a7c538e35710c94acf164ca93693117acee3a5bfdd3035810ee110107010002480e100401010104000e3a02000100000000020200000008000000000000000a03fc29fc93353c639f7cf275ea69851dfa41d720ed53bcc9c10bda17166966d5ed00010402", "Cast", 394),
        ];
        for (body, refusal, cost) in refused {
            let (e, c) = eval_body_err(body);
            let named = match refusal {
                "Cast" => matches!(e, EvalError::TryExtractFrom(_)),
                "NoPair" => matches!(e, EvalError::UnexpectedValue(_)),
                // "Tree proof is incorrect"
                _ => matches!(e, EvalError::AvlTree(_)),
            };
            assert!(named, "{body}: expected {refusal}, got {e:?}");
            assert_eq!(c, cost, "{body}");
        }
        #[rustfmt::skip]
        let evaluated = [
            // contains|other-tree|control
            ("dc6409647ca99024833a7c538e35710c94acf164ca93693117acee3a5bfdd3035810ee1101070100020e01010e550385ab460a6564d1e5ded17716cd5866650d7fbb9cada7fecabd5fe21e2f80e43a02010200000008000000000000000a03fc29fc93353c639f7cf275ea69851dfa41d720ed53bcc9c10bda17166966d5ed00010402", false, 219),
            // insert|other-tree|control
            ("dc640c647ca99024833a7c538e35710c94acf164ca93693117acee3a5bfdd3035810ee1101070100020c3c0e0e01010308000000000000001e0e550385ab460a6564d1e5ded17716cd5866650d7fbb9cada7fecabd5fe21e2f80e43a039d4d9434aa05d4dc888d939d6771fe05418df5bd91101dbe63384514c1869e140202ff00000008000000000000001400010400", true, 234),
            // insert|other-tree|entries-second-key-Coll[Int]
            ("dc640c647ca99024833a7c538e35710c94acf164ca93693117acee3a5bfdd3035810ee110107010002483c0e0e3c100e04010308000000000000001e01040105000e550385ab460a6564d1e5ded17716cd5866650d7fbb9cada7fecabd5fe21e2f80e43a039d4d9434aa05d4dc888d939d6771fe05418df5bd91101dbe63384514c1869e140202ff00000008000000000000001400010400", true, 234),
            // insert|other-tree|entries-second-no-pair
            ("dc640c647ca99024833a7c538e35710c94acf164ca93693117acee3a5bfdd3035810ee110107010002483c0e0e0404010308000000000000001e0a000e550385ab460a6564d1e5ded17716cd5866650d7fbb9cada7fecabd5fe21e2f80e43a039d4d9434aa05d4dc888d939d6771fe05418df5bd91101dbe63384514c1869e140202ff00000008000000000000001400010400", true, 234),
            // update|other-tree|control
            ("dc640d647ca99024833a7c538e35710c94acf164ca93693117acee3a5bfdd3035810ee1101070100020c3c0e0e01010108000000000000000b0e550385ab460a6564d1e5ded17716cd5866650d7fbb9cada7fecabd5fe21e2f80e43a02010200000008000000000000000a03fc29fc93353c639f7cf275ea69851dfa41d720ed53bcc9c10bda17166966d5ed00010402", true, 324),
            // update|other-tree|entries-second-key-Coll[Int]
            ("dc640d647ca99024833a7c538e35710c94acf164ca93693117acee3a5bfdd3035810ee110107010002483c0e0e3c100e04010108000000000000000b01040105000e550385ab460a6564d1e5ded17716cd5866650d7fbb9cada7fecabd5fe21e2f80e43a02010200000008000000000000000a03fc29fc93353c639f7cf275ea69851dfa41d720ed53bcc9c10bda17166966d5ed00010402", true, 324),
            // update|other-tree|entries-second-no-pair
            ("dc640d647ca99024833a7c538e35710c94acf164ca93693117acee3a5bfdd3035810ee110107010002483c0e0e0404010108000000000000000b0a000e550385ab460a6564d1e5ded17716cd5866650d7fbb9cada7fecabd5fe21e2f80e43a02010200000008000000000000000a03fc29fc93353c639f7cf275ea69851dfa41d720ed53bcc9c10bda17166966d5ed00010402", true, 324),
            // insertOrUpdate|other-tree|control
            ("dc6410647ca99024833a7c538e35710c94acf164ca93693117acee3a5bfdd3035810ee1101070100020c3c0e0e01010308000000000000001e0e550385ab460a6564d1e5ded17716cd5866650d7fbb9cada7fecabd5fe21e2f80e43a039d4d9434aa05d4dc888d939d6771fe05418df5bd91101dbe63384514c1869e140202ff00000008000000000000001400010400", true, 339),
            // insertOrUpdate|other-tree|entries-second-key-Coll[Int]
            ("dc6410647ca99024833a7c538e35710c94acf164ca93693117acee3a5bfdd3035810ee110107010002483c0e0e3c100e04010308000000000000001e01040105000e550385ab460a6564d1e5ded17716cd5866650d7fbb9cada7fecabd5fe21e2f80e43a039d4d9434aa05d4dc888d939d6771fe05418df5bd91101dbe63384514c1869e140202ff00000008000000000000001400010400", true, 339),
            // insertOrUpdate|other-tree|entries-second-no-pair
            ("dc6410647ca99024833a7c538e35710c94acf164ca93693117acee3a5bfdd3035810ee110107010002483c0e0e0404010308000000000000001e0a000e550385ab460a6564d1e5ded17716cd5866650d7fbb9cada7fecabd5fe21e2f80e43a039d4d9434aa05d4dc888d939d6771fe05418df5bd91101dbe63384514c1869e140202ff00000008000000000000001400010400", true, 339),
            // remove|other-tree|control
            ("dc640e647ca99024833a7c538e35710c94acf164ca93693117acee3a5bfdd3035810ee1101070100021a0101010e3a02000100000000020200000008000000000000000a03fc29fc93353c639f7cf275ea69851dfa41d720ed53bcc9c10bda17166966d5ed00010402", true, 294),
        ];
        for (body, is_none, cost) in evaluated {
            let (res, c) = eval_body(body, ErgoTreeVersion::V3);
            let v = res.unwrap();
            assert_eq!(matches!(v, Value::Opt(None)), is_none, "{body}: {v:?}");
            assert_eq!(c, cost, "{body}");
        }
    }

    // A degenerate proof. sigmastate's verifier construction never throws: a proof that cannot
    // be read gives a verifier with no tree, the operation fails, and each method maps that
    // failure as it does for a proof of another tree. The one-byte proof rows are each method
    // on an all-zero digest; the last two are `contains` and `get` with a `Coll[Int]` key and
    // an EMPTY proof, where the key is cast after the verifier's and the lookup's cost.
    // `ergo_avltree_rust` 0.1.1 panics on such a proof; the workspace takes the crate from the
    // fork that returns an error. Measured on the sigma-state 6.0.7 jars.
    #[test]
    fn a_degenerate_proof_answers_or_errors_as_the_jvm() {
        #[rustfmt::skip]
        let rows: [(&str, Result<&str, &str>, u64); 9] = [
            // AvlTree.contains#9|control|-
            ("dc640964000000000000000000000000000000000000000000000000000000000000000000072000020e01010e0102", Ok("false"), 189),
            // AvlTree.get#10|control|-
            ("dc640a64000000000000000000000000000000000000000000000000000000000000000000072000020e01010e0102", Err("AvlTree"), 189),
            // AvlTree.getMany#11|control|-
            ("dc640b64000000000000000000000000000000000000000000000000000000000000000000072000021a0101010e0102", Err("AvlTree"), 189),
            // AvlTree.insert#12|control|-
            ("dc640c64000000000000000000000000000000000000000000000000000000000000000000072000020c3c0e0e000e0102", Ok("None"), 164),
            // AvlTree.update#13|control|-
            ("dc640d64000000000000000000000000000000000000000000000000000000000000000000072000020c3c0e0e000e0102", Ok("None"), 164),
            // AvlTree.remove#14|control|-
            ("dc640e64000000000000000000000000000000000000000000000000000000000000000000072000021a0101010e0102", Ok("None"), 294),
            // AvlTree.insertOrUpdate#16|control|-
            ("dc641064000000000000000000000000000000000000000000000000000000000000000000072000020c3c0e0e000e0102", Ok("None"), 179),
            // AvlTree.contains|arg|EmptyCollInt|direct
            ("dc6409640000000000000000000000000000000000000000000000000000000000000000000720000210000e00", Err("TryExtractFrom"), 189),
            // AvlTree.get|arg|EmptyCollInt|direct
            ("dc640a640000000000000000000000000000000000000000000000000000000000000000000720000210000e00", Err("TryExtractFrom"), 189),
        ];
        for (body, expected, cost) in rows {
            let (res, c) = eval_body(body, ErgoTreeVersion::V3);
            let as_expected = match (expected, &res) {
                (Ok("false"), Ok(Value::Boolean(false))) | (Ok("None"), Ok(Value::Opt(None))) => {
                    true
                }
                (Err("AvlTree"), Err(e)) => matches!(innermost(e), EvalError::AvlTree(_)),
                (Err("TryExtractFrom"), Err(e)) => {
                    matches!(innermost(e), EvalError::TryExtractFrom(_))
                }
                _ => false,
            };
            assert!(as_expected, "{body}: expected {expected:?}, got {res:?}");
            assert_eq!(c, cost, "{body}");
        }
    }
}
