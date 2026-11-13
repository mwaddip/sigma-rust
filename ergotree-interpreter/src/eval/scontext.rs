use alloc::boxed::Box;
use alloc::string::ToString;
use alloc::sync::Arc;
use alloc::vec::Vec;

use ergotree_ir::mir::coll_value::CollValue;
use ergotree_ir::mir::constant::TryExtractInto;
use ergotree_ir::mir::value::Value;
use ergotree_ir::reference::Ref;
use ergotree_ir::serialization::SigmaSerializable;
use ergotree_ir::types::stype::SType;

use super::Context;
use super::EvalError;
use super::EvalFn;

// sigmastate's context has one accessor per value (`CContext.scala:36-48`). The value's node
// reads it after adding its own cost (`values.scala:1465-1542`: `addCost(costKind)`, then
// `E.context.<accessor>`), and so does the `Context` method declared by that node
// (`property(name, type, id, <node>)`, `methods.scala:222-229`, `:1742-1748`), which invokes
// the accessor on its receiver. Both forms read the six functions below, so they cannot
// differ; none adds a cost: the node adds its own, the call adds the method's.

/// `Context.INPUTS`: the boxes the transaction spends
pub(crate) fn inputs<'ctx>(ctx: &Context<'ctx>) -> Value<'ctx> {
    ctx.inputs
        .iter()
        .map(|&i| Ref::Borrowed(i))
        .collect::<Vec<_>>()
        .into()
}

/// `Context.OUTPUTS`: the boxes the transaction creates
pub(crate) fn outputs<'ctx>(ctx: &Context<'ctx>) -> Value<'ctx> {
    ctx.outputs
        .iter()
        .map(Ref::Borrowed)
        .collect::<Vec<_>>()
        .into()
}

/// `Context.HEIGHT`
pub(crate) fn height<'ctx>(ctx: &Context<'ctx>) -> Value<'ctx> {
    (ctx.height as i32).into()
}

/// `Context.SELF`: the box whose script is evaluated
pub(crate) fn self_box<'ctx>(ctx: &Context<'ctx>) -> Value<'ctx> {
    Value::CBox(Ref::from(ctx.self_box))
}

/// `Context.LastBlockUtxoRootHash`. The root is a standalone context input, as in the JVM
/// (`CContext` returns `lastBlockUtxoRootHash` directly, never deriving it from `headers(0)`)
pub(crate) fn last_block_utxo_root_hash<'ctx>(ctx: &Context<'ctx>) -> Value<'ctx> {
    Value::AvlTree(Box::from(ctx.last_block_utxo_root.clone()))
}

/// `Context.minerPubKey`: the pre-header's miner key, as its bytes
pub(crate) fn miner_pub_key<'ctx>(ctx: &Context<'ctx>) -> Result<Value<'ctx>, EvalError> {
    Ok(ctx.pre_header.miner_pk.sigma_serialize_bytes()?.into())
}

pub(crate) static DATA_INPUTS_EVAL_FN: EvalFn = |_mc, _env, ctx, _obj, _args| {
    Ok(Value::Coll(CollValue::Generic {
        items: ctx.data_inputs.clone().map_or(Arc::new([]), |d| {
            d.iter().map(|&di| Ref::from(di)).map(Value::CBox).collect()
        }),
        elem_tpe: SType::SBox,
    }))
};

pub(crate) static SELF_BOX_INDEX_EVAL_FN: EvalFn = |_mc, _env, ctx, _obj, _args| {
    // JVM bug compatibility: selfBoxIndex always returned -1 before JIT
    // activation (v5.0). The bug was `eq` (reference equality) instead of
    // `==` (value equality) in CostingDataContext.scala — a global impl
    // bug, not per-script semantics. Fixed in v5.x for ALL scripts.
    // Gate: activated_script_version (block level), NOT tree_version.
    // See: https://github.com/ScorexFoundation/sigmastate-interpreter/issues/603
    if ctx.activated_script_version() < ergotree_ir::ergo_tree::ErgoTreeVersion::V2 {
        return Ok(Value::Int(-1));
    }
    let box_index = ctx
        .inputs
        .iter()
        .position(|it| *it == ctx.self_box)
        .ok_or_else(|| EvalError::NotFound("Context.selfBoxIndex: box not found".to_string()))?;
    Ok(Value::Int(box_index as i32))
};

pub(crate) static HEADERS_EVAL_FN: EvalFn = |_mc, _env, ctx, _obj, _args| {
    Ok(Value::Coll(CollValue::Generic {
        items: ctx
            .headers
            .iter()
            .map(|h| Value::Header(Box::new(h.clone())))
            .collect(),
        elem_tpe: SType::SHeader,
    }))
};

pub(crate) static PRE_HEADER_EVAL_FN: EvalFn =
    |_mc, _env, ctx, _obj, _args| Ok(Box::from(ctx.pre_header.clone()).into());

// The six methods sigmastate declares by a node (`methods.scala:1742-1748`): each returns
// its node's accessor and adds no cost (the call adds the method's fixed cost, which is the
// node's). A new `property(name, type, id, <node>)` method takes its node's accessor the same
// way. SANTA `Context.property_call_forms`, `Context.property_call_forms_v6`
pub(crate) static INPUTS_EVAL_FN: EvalFn = |_mc, _env, ctx, _obj, _args| Ok(inputs(ctx));

pub(crate) static OUTPUTS_EVAL_FN: EvalFn = |_mc, _env, ctx, _obj, _args| Ok(outputs(ctx));

pub(crate) static HEIGHT_EVAL_FN: EvalFn = |_mc, _env, ctx, _obj, _args| Ok(height(ctx));

pub(crate) static SELF_EVAL_FN: EvalFn = |_mc, _env, ctx, _obj, _args| Ok(self_box(ctx));

pub(crate) static LAST_BLOCK_UTXO_ROOT_HASH_EVAL_FN: EvalFn =
    |_mc, _env, ctx, _obj, _args| Ok(last_block_utxo_root_hash(ctx));

pub(crate) static MINER_PUBKEY_EVAL_FN: EvalFn = |_mc, _env, ctx, _obj, _args| miner_pub_key(ctx);

pub(crate) static GET_VAR_FROM_INPUT_EVAL_FN: EvalFn = |mc, _env, ctx, _obj, args| {
    #[allow(clippy::unreachable)] // getVarFromInput output type is always SOption[T]
    let SType::SOption(output_tpe) = &*mc.tpe().t_range
    else {
        unreachable!()
    };
    let arg = |i: usize| {
        args.get(i)
            .cloned()
            .ok_or_else(|| EvalError::NotFound(format!("getVarFromInput: missing argument {i}")))
    };
    let input_idx = arg(0)?.try_extract_into::<i16>()? as usize;
    let var_id = arg(1)?.try_extract_into::<i8>()? as u8;
    // `CContext.getVarFromInput` (v6.0.6 `CContext.scala:76-82`): the variable's type decides
    // first, `stypeToRType(v.tpe) == tT`, and its data is read only then; a variable of another
    // type is `None` whatever its data
    let var = ctx
        .extension_provider
        .context_extension(input_idx)
        .and_then(|extension| extension.values.get(&var_id));
    let var = match var {
        Some(var) if var.script_type()? == **output_tpe => var,
        _ => return Ok(Value::Opt(None)),
    };
    Ok(Value::Opt(Some(Box::new(var.to_script_value()?.v))))
};

#[cfg(test)]
#[cfg(feature = "arbitrary")]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use crate::eval::test_util::{eval_out, try_eval_out_with_version};
    use core::cell::Cell;
    use ergo_chain_types::{Header, PreHeader};
    use ergotree_ir::chain::context::{Context, ContextHeaders};
    use ergotree_ir::chain::ergo_box::ErgoBox;
    use ergotree_ir::ergo_tree::{ErgoTree, ErgoTreeVersion};
    use ergotree_ir::mir::avl_tree_data::{AvlTreeData, AvlTreeFlags};
    use ergotree_ir::mir::constant::TryExtractFrom;
    use ergotree_ir::mir::expr::Expr;
    use ergotree_ir::mir::method_call::MethodCall;
    use ergotree_ir::mir::property_call::PropertyCall;
    use ergotree_ir::mir::value::Value;
    use ergotree_ir::serialization::SigmaSerializable;
    use ergotree_ir::types::scontext::{self, GET_VAR_FROM_INPUT_METHOD};
    use ergotree_ir::types::stype::LiftIntoSType;
    use ergotree_ir::types::stype_param::STypeVar;
    use sigma_test_util::force_any_val;

    fn make_ctx_inputs_includes_self_box(
        tree_version: ErgoTreeVersion,
        activated_script_version_byte: i8,
    ) -> Context<'static> {
        let ctx = force_any_val::<Context>();
        let self_box = &*Box::leak(Box::new(force_any_val::<ErgoBox>()));
        let inputs = vec![&*Box::leak(Box::new(force_any_val::<ErgoBox>())), self_box]
            .try_into()
            .unwrap();
        Context {
            height: 0u32,
            self_box,
            inputs,
            activated_script_version_byte,
            tree_version: Cell::new(tree_version),
            ..ctx
        }
    }

    #[test]
    fn eval_self_box_index_v2_tree() {
        let expr: Expr =
            PropertyCall::new(Expr::Context, scontext::SELF_BOX_INDEX_PROPERTY.clone())
                .unwrap()
                .into();
        // V2 tree in v5+ block (activated 2): real index.
        let context = make_ctx_inputs_includes_self_box(ErgoTreeVersion::V2, 2);
        assert_eq!(eval_out::<i32>(&expr, &context), 1);
    }

    #[test]
    fn eval_self_box_index_v0_tree_pre_v5() {
        let expr: Expr =
            PropertyCall::new(Expr::Context, scontext::SELF_BOX_INDEX_PROPERTY.clone())
                .unwrap()
                .into();
        // V0 tree in pre-v5 block (activated 0): -1.
        let context = make_ctx_inputs_includes_self_box(ErgoTreeVersion::V0, 0);
        assert_eq!(eval_out::<i32>(&expr, &context), -1);
    }

    #[test]
    fn eval_self_box_index_v1_tree_pre_v5() {
        let expr: Expr =
            PropertyCall::new(Expr::Context, scontext::SELF_BOX_INDEX_PROPERTY.clone())
                .unwrap()
                .into();
        // V1 tree in pre-v5 block (activated 0): -1.
        let context = make_ctx_inputs_includes_self_box(ErgoTreeVersion::V1, 0);
        assert_eq!(eval_out::<i32>(&expr, &context), -1);
    }

    #[test]
    fn eval_self_box_index_v0_tree_v5_context() {
        let expr: Expr =
            PropertyCall::new(Expr::Context, scontext::SELF_BOX_INDEX_PROPERTY.clone())
                .unwrap()
                .into();
        // V0 tree in v5+ block (activated 2): real index.
        // JVM bug #603 was a global impl bug fixed in v5.x for ALL scripts.
        let context = make_ctx_inputs_includes_self_box(ErgoTreeVersion::V0, 2);
        assert_eq!(eval_out::<i32>(&expr, &context), 1);
    }

    #[test]
    fn eval_headers() {
        let expr: Expr = PropertyCall::new(Expr::Context, scontext::HEADERS_PROPERTY.clone())
            .expect("internal error: `headers` method has parameters length != 1")
            .into();
        let ctx = force_any_val::<Context>();
        assert_eq!(eval_out::<Vec<Header>>(&expr, &ctx), *ctx.headers.as_vec());
    }

    #[test]
    fn eval_preheader() {
        let expr: Expr = PropertyCall::new(Expr::Context, scontext::PRE_HEADER_PROPERTY.clone())
            .unwrap()
            .into();
        let ctx = force_any_val::<Context>();
        assert_eq!(eval_out::<PreHeader>(&expr, &ctx), ctx.pre_header);
    }

    #[test]
    fn eval_miner_pubkey() {
        let expr: Expr = PropertyCall::new(Expr::Context, scontext::MINER_PUBKEY_PROPERTY.clone())
            .unwrap()
            .into();
        let ctx = force_any_val::<Context>();
        assert_eq!(
            eval_out::<Vec<u8>>(&expr, &ctx),
            ctx.pre_header.miner_pk.sigma_serialize_bytes().unwrap()
        );
    }

    #[test]
    fn eval_last_block_utxo_root_hash() {
        let expr: Expr = PropertyCall::new(
            Expr::Context,
            scontext::LAST_BLOCK_UTXO_ROOT_HASH_PROPERTY.clone(),
        )
        .unwrap()
        .into();
        let mut ctx = force_any_val::<Context>();
        // Pin the field to a digest distinct from headers[0].state_root: the eval
        // must return the standalone context input (JVM `CContext` semantics),
        // not a value derived from the headers.
        ctx.last_block_utxo_root.digest = [7u8; 33].into();
        assert_ne!(
            ctx.last_block_utxo_root.digest,
            ctx.headers.first().unwrap().state_root.0.to_vec()
        );
        assert_eq!(
            eval_out::<AvlTreeData>(&expr, &ctx),
            ctx.last_block_utxo_root
        );
    }

    /// Canonical synthetic eval context essentials for the blessed vectors below:
    /// EMPTY headers — the honest value for a contextless eval, which the JVM
    /// expresses (`headers.isEmpty` is legal, `ErgoLikeContext.scala:85`) — and
    /// the dummy root (all-zero 33-byte digest, all operations allowed).
    fn empty_headers_ctx() -> Context<'static> {
        let mut ctx = force_any_val::<Context>();
        ctx.headers = ContextHeaders::from_vec(vec![]).unwrap();
        ctx.last_block_utxo_root = AvlTreeData {
            digest: [0u8; 33].into(),
            tree_flags: AvlTreeFlags::new(true, true, true),
            key_length: 32,
            value_length_opt: None,
        };
        ctx
    }

    /// JVM-blessed byte vectors (santa-eval `Context.properties`, eval/v5/authored):
    /// closed v2 trees. The blessed sized header (`1a` + size VLQ) is rewritten to
    /// the non-sized `12` (size bit cleared, size byte dropped) because the sized
    /// parse path rejects non-SigmaProp roots — the same lenient deserialize the
    /// conformance runner applies to expression-rooted corpus trees; body verbatim.
    fn eval_blessed_context_tree<T: TryExtractFrom<Value<'static>> + 'static>(
        tree_hex: &str,
        ctx: &Context<'static>,
    ) -> T {
        let tree_bytes = base16::decode(tree_hex).unwrap();
        // Unsized (size-bit-cleared) fixture: restore the size slot + skip the root check.
        let tree = ErgoTree::sigma_parse_bytes_lenient_from_unsized(&tree_bytes).unwrap();
        let expr = tree.proposition().unwrap();
        try_eval_out_with_version::<T>(&expr, ctx, 2, 2).unwrap()
    }

    #[test]
    fn eval_headers_empty_context_blessed_bytes() {
        // `{ CONTEXT.headers }` (`CONTEXT.headers#dummy`): the JVM yields the
        // context's actual — here empty — header collection.
        let ctx = empty_headers_ctx();
        assert_eq!(
            eval_blessed_context_tree::<Vec<Header>>("1200db6502fe", &ctx),
            Vec::<Header>::new()
        );
    }

    #[test]
    fn eval_last_block_utxo_root_hash_empty_context_blessed_bytes() {
        // `{ CONTEXT.LastBlockUtxoRootHash }` (`CONTEXT.LastBlockUtxoRootHash#dummy`):
        // with no headers the standalone field is the only source of the root —
        // the JVM returns it; a `headers(0)`-derived value cannot express this.
        let ctx = empty_headers_ctx();
        assert_eq!(
            eval_blessed_context_tree::<AvlTreeData>("1200db6509fe", &ctx),
            ctx.last_block_utxo_root
        );
    }

    #[test]
    fn eval_get_var_from_input() {
        fn get_var_from_input<T: LiftIntoSType + TryExtractFrom<Value<'static>> + 'static>(
            ctx: &Context<'static>,
            input_index: i16,
            var_id: i8,
        ) -> Option<T> {
            let mc = MethodCall::with_type_args(
                Expr::Context,
                GET_VAR_FROM_INPUT_METHOD.clone(),
                vec![input_index.into(), var_id.into()],
                [(STypeVar::t(), T::stype())].into_iter().collect(),
            )
            .unwrap()
            .into();
            eval_out(&mc, ctx)
        }
        let context = crate::eval::get_var::tests::prepare_context();
        assert_eq!(get_var_from_input::<i32>(&context, 0, 3), Some(123));
        assert_eq!(get_var_from_input::<i64>(&context, 0, 3), None); // wrong type
        assert_eq!(get_var_from_input::<i32>(&context, 0, 4), None); // context extension var doesn't exist
        assert_eq!(
            get_var_from_input::<i32>(&context, context.inputs.len() as i16 + 1, 4),
            None
        ); // input out of bounds
    }
}

#[cfg(test)]
#[cfg(feature = "arbitrary")]
#[allow(clippy::unwrap_used)]
mod get_var_from_input_tests {
    use crate::eval::env::Env;
    use crate::eval::Evaluable;
    use ergotree_ir::chain::context::arbitrary::DummyContextExtensionProvider;
    use ergotree_ir::chain::context::Context;
    use ergotree_ir::chain::context_extension::ContextExtension;
    use ergotree_ir::chain::evaluated_value::EvaluatedValue;
    use ergotree_ir::mir::coll_value::CollValue;
    use ergotree_ir::mir::expr::Expr;
    use ergotree_ir::mir::method_call::MethodCall;
    use ergotree_ir::mir::value::Value;
    use ergotree_ir::serialization::SigmaSerializable;
    use ergotree_ir::types::scontext::GET_VAR_FROM_INPUT_METHOD;
    use ergotree_ir::types::stuple::STuple;
    use ergotree_ir::types::stype::SType;
    use ergotree_ir::types::stype_param::STypeVar;
    use sigma_test_util::force_any_val;

    #[test]
    fn get_var_from_input_checks_the_type_before_it_reads_the_data() {
        // `CContext.getVarFromInput` (v6.0.6 `CContext.scala:76-82`) compares
        // `stypeToRType(v.tpe)` with the asked type and reads the data only on a match (from
        // source): input 1's var 0, `Tuple(1, HEIGHT)`, read as an Int, is `None`; its var 1, C1's
        // `Coll[(Int, Int)]` holding a tuple expression, read at its own type, fails
        let value =
            |hex: &str| EvaluatedValue::sigma_parse_bytes(&base16::decode(hex).unwrap()).unwrap();
        let other = ContextExtension {
            values: [
                (0u8, value("86020402a3")),
                (1u8, value("830158860204020404")),
                (2u8, value("48040404020406")),
            ]
            .into_iter()
            .collect(),
        };
        let mut ctx = force_any_val::<Context>();
        ctx.extension = Box::leak(Box::new(ContextExtension::empty()));
        ctx.extension_provider = Box::leak(Box::new(DummyContextExtensionProvider(vec![
            ContextExtension::empty(),
            other,
        ])));
        let read = |var_id: i8, tpe: SType| {
            let expr: Expr = MethodCall::with_type_args(
                Expr::Context,
                GET_VAR_FROM_INPUT_METHOD.clone(),
                vec![1i16.into(), var_id.into()],
                [(STypeVar::t(), tpe)].into_iter().collect(),
            )
            .unwrap()
            .into();
            expr.eval(&mut Env::empty(), &ctx).map(|v| v.to_static())
        };
        assert_eq!(read(0, SType::SInt).unwrap(), Value::Opt(None));
        let pairs = SType::SColl(SType::STuple(STuple::pair(SType::SInt, SType::SInt)).into());
        assert!(read(1, pairs).is_err());
        // its var 2, the constant `(1, 2, 3)`: read at its own type it is the tuple's
        // `Coll[Any]` (SANTA's tuple-values reply: `getVarFromInput[(Int, Int, Int)](0, 1).get.size`
        // is 3), read as a `Coll[Int]` it is `None`, not a throw
        let triple = SType::STuple(STuple::triple(SType::SInt, SType::SInt, SType::SInt));
        let ints = alloc::vec![Value::Int(1), Value::Int(2), Value::Int(3)];
        assert_eq!(
            read(2, triple).unwrap(),
            Value::Opt(Some(Box::new(Value::Coll(CollValue::generic(
                SType::SAny,
                ints
            )))))
        );
        assert_eq!(
            read(2, SType::SColl(SType::SInt.into())).unwrap(),
            Value::Opt(None)
        );
    }
}

#[cfg(test)]
#[cfg(feature = "arbitrary")]
#[allow(clippy::unwrap_used)]
mod node_property_tests {
    use crate::eval::test_util::eval_body_err;
    use crate::eval::{Env, EvalError, Evaluable};
    use ergotree_ir::chain::context::Context;
    use ergotree_ir::chain::ergo_box::box_value::BoxValue;
    use ergotree_ir::chain::ergo_box::{ErgoBox, NonMandatoryRegisters};
    use ergotree_ir::chain::tx_id::TxId;
    use ergotree_ir::ergo_tree::{ErgoTree, ErgoTreeVersion};
    use ergotree_ir::mir::expr::Expr;
    use ergotree_ir::mir::global_vars::GlobalVars;
    use ergotree_ir::mir::property_call::PropertyCall;
    use ergotree_ir::mir::value::Value;
    use ergotree_ir::reference::Ref;
    use ergotree_ir::serialization::sigma_byte_reader::{from_bytes, SigmaByteRead};
    use ergotree_ir::serialization::SigmaSerializable;
    use ergotree_ir::types::smethod::MethodId;
    use ergotree_ir::types::stype_companion::STypeCompanion;
    use sigma_test_util::force_any_val;

    const V0: ErgoTreeVersion = ErgoTreeVersion::V0;
    const V3: ErgoTreeVersion = ErgoTreeVersion::V3;
    /// (tree version, activated version): the three pairs of SANTA's rows
    const VERSIONS: [(ErgoTreeVersion, i8); 3] = [(V0, 2), (V0, 3), (V3, 3)];

    /// A box of the given value; `index` tells two boxes of one value apart
    fn a_box(value: u64, index: u16) -> ErgoBox {
        // `sigmaProp(true)`
        let tree = ErgoTree::sigma_parse_bytes(&base16::decode("0b03d10101").unwrap()).unwrap();
        ErgoBox::new(
            BoxValue::try_from(value).unwrap(),
            tree,
            None,
            NonMandatoryRegisters::empty(),
            0,
            TxId::zero(),
            index,
        )
        .unwrap()
    }

    fn leak<T>(v: T) -> &'static T {
        Box::leak(Box::new(v))
    }

    /// SANTA's eval context (its contract, section 2): INPUTS = [SELF], OUTPUTS = [],
    /// HEIGHT 0, SELF.value 1000000
    fn santa_ctx() -> Context<'static> {
        let self_box = leak(a_box(1_000_000, 0));
        Context {
            height: 0,
            self_box,
            inputs: vec![self_box].try_into().unwrap(),
            outputs: &[],
            ..force_any_val::<Context>()
        }
    }

    /// The context of SANTA's reply, section 2: three inputs, SELF the SECOND (value
    /// 1234567890), two outputs, HEIGHT 777777
    fn distinct_ctx() -> Context<'static> {
        let self_box = leak(a_box(1_234_567_890, 1));
        Context {
            height: 777_777,
            self_box,
            inputs: vec![
                leak(a_box(1_000_000_000, 0)),
                self_box,
                leak(a_box(2_000_000, 2)),
            ]
            .try_into()
            .unwrap(),
            outputs: leak(vec![a_box(3_000_000, 0), a_box(4_000_000, 1)]),
            ..force_any_val::<Context>()
        }
    }

    fn eval_in(
        ctx: &Context<'static>,
        expr: &Expr,
        tree: ErgoTreeVersion,
        activated: i8,
    ) -> (Result<Value<'static>, EvalError>, u64) {
        let mut ctx = ctx.clone();
        ctx.activated_script_version_byte = activated;
        ctx.tree_version.set(tree);
        ctx.jit_cost.set(0);
        let res = expr.eval(&mut Env::empty(), &ctx).map(|v| v.to_static());
        (res, ctx.jit_cost_value())
    }

    fn body(body_hex: &str, tree: ErgoTreeVersion) -> Expr {
        let bytes = base16::decode(body_hex.as_bytes()).unwrap();
        from_bytes(&bytes)
            .with_tree_version(tree, Expr::sigma_parse)
            .unwrap()
    }

    fn assert_rows(ctx: &Context<'static>, rows: &[(&str, Value<'static>, u64)]) {
        for (tree, activated) in VERSIONS {
            for (body_hex, value, cost) in rows {
                let (res, c) = eval_in(ctx, &body(body_hex, tree), tree, activated);
                let at = format!("{body_hex} at tree {tree:?}, activated {activated}");
                assert_eq!(res.as_ref().ok(), Some(value), "{at}: {res:?}");
                assert_eq!(c, *cost, "{at}");
            }
        }
    }

    fn boxes(items: &[&'static ErgoBox]) -> Value<'static> {
        items
            .iter()
            .map(|&b| Ref::Borrowed(b))
            .collect::<Vec<_>>()
            .into()
    }

    // SANTA `Context.property_call_forms` and `Context.property_call_forms_v6` (`8bb666d`,
    // sigma-state 6.0.7): each property call beside its node. The eight bare rows that return
    // the SELF box at tree version 0 are measured (its reply, section 1) and not vectored
    #[test]
    fn santa_property_call_rows() {
        let ctx = santa_ctx();
        let me = ctx.self_box;
        assert_rows(
            &ctx,
            &[
                ("b1db6504fe", Value::Int(1), 29),
                ("b1a4", Value::Int(1), 24),
                ("b1db6505fe", Value::Int(0), 29),
                ("b1a5", Value::Int(0), 24),
                ("db6506fe", Value::Int(0), 31),
                ("a3", Value::Int(0), 26),
                ("c1db6507fe", Value::Long(1_000_000), 23),
                ("c1a7", Value::Long(1_000_000), 18),
                ("db6504fe", boxes(&[me]), 15),
                ("a4", boxes(&[me]), 10),
                ("db6505fe", boxes(&[]), 15),
                ("a5", boxes(&[]), 10),
                ("db6507fe", Value::CBox(Ref::Borrowed(me)), 15),
                ("a7", Value::CBox(Ref::Borrowed(me)), 10),
                ("93db6504fea4", Value::Boolean(true), 46),
                ("93db6505fea5", Value::Boolean(true), 41),
                ("93db6506fea3", Value::Boolean(true), 60),
                ("93db6507fea7", Value::Boolean(true), 31),
            ],
        );
    }

    // SANTA's reply, section 2 (measured, not vectored): a context whose values differ, so
    // HEIGHT is not a zero and SELF is not `INPUTS(0)`. The two bare rows are by reading
    #[test]
    fn property_calls_in_a_context_of_distinct_values() {
        let ctx = distinct_ctx();
        let me = ctx.self_box;
        let inputs: Vec<&'static ErgoBox> = ctx.inputs.iter().copied().collect();
        assert_rows(
            &ctx,
            &[
                ("b1db6504fe", Value::Int(3), 29),
                ("b1db6505fe", Value::Int(2), 29),
                ("db6506fe", Value::Int(777_777), 31),
                ("c1db6507fe", Value::Long(1_234_567_890), 23),
                ("93db6504fea4", Value::Boolean(true), 56),
                ("93db6505fea5", Value::Boolean(true), 51),
                ("93db6506fea3", Value::Boolean(true), 60),
                ("93db6507fea7", Value::Boolean(true), 31),
                // by reading
                ("db6507fe", Value::CBox(Ref::Borrowed(me)), 15),
                ("db6504fe", boxes(&inputs), 15),
            ],
        );
    }

    // sigmastate declares six `Context` methods by their NODE (`property(name, type, id,
    // <node>)`, `methods.scala:222-229`, `:1742-1748`): the node and the method read one
    // accessor of the context, and the method's fixed cost is the node's. A call costs
    // `Context`'s 1 and the call's 4 more
    #[test]
    fn a_node_property_reads_what_its_node_reads() {
        let nodes = [
            (GlobalVars::Inputs, 4, 10),
            (GlobalVars::Outputs, 5, 10),
            (GlobalVars::Height, 6, 26),
            (GlobalVars::SelfBox, 7, 10),
            (GlobalVars::LastBlockUtxoRootHash, 9, 15),
            (GlobalVars::MinerPubKey, 10, 20),
        ];
        for ctx in [force_any_val::<Context>(), distinct_ctx()] {
            for (node, id, node_cost) in &nodes {
                let method = STypeCompanion::Context
                    .method_by_id(&MethodId(*id))
                    .unwrap();
                let call: Expr = PropertyCall::new(Expr::Context, method).unwrap().into();
                let node: Expr = node.clone().into();
                for (tree, activated) in VERSIONS {
                    let (of_node, c) = eval_in(&ctx, &node, tree, activated);
                    assert_eq!(c, *node_cost, "{node:?}");
                    let (of_call, c) = eval_in(&ctx, &call, tree, activated);
                    assert_eq!(of_call.unwrap(), of_node.unwrap(), "{node:?}");
                    assert_eq!(c, node_cost + 5, "{node:?}");
                }
            }
        }
    }

    // Below tree version 3 a method call with no arguments parses
    // (`MethodCallSerializer.scala:52-55` asserts arguments from version 3 only). Its
    // companion is `PropertyCall` (`values.scala:1351`), and the parser gives the property
    // call itself (`ergotree-ir`, `serialization/expr.rs`): the two wire forms are ONE
    // expression here. By reading (no SANTA row): it evaluates to its node's value at the
    // node's cost + 5
    #[test]
    fn a_method_call_without_arguments_is_the_property_call_below_v3() {
        let ctx = distinct_ctx();
        let nodes = [
            (4, GlobalVars::Inputs, 10),
            (5, GlobalVars::Outputs, 10),
            (6, GlobalVars::Height, 26),
            (7, GlobalVars::SelfBox, 10),
        ];
        for (id, node, node_cost) in nodes {
            let method_form = body(&format!("dc650{id}fe00"), V0);
            assert_eq!(method_form, body(&format!("db650{id}fe"), V0), "{id}");
            let node: Expr = node.into();
            for activated in [2, 3] {
                let (of_node, _) = eval_in(&ctx, &node, V0, activated);
                let (of_method, c) = eval_in(&ctx, &method_form, V0, activated);
                assert_eq!(of_method.unwrap(), of_node.unwrap(), "{id}");
                assert_eq!(c, node_cost + 5, "{id}");
            }
        }
    }

    // Measured on the sigma-state 6.0.7 jars (the method-call charge order round's rows): a
    // receiver of another kind and an argument too many are refused where they were, the
    // method's fixed cost paid
    #[test]
    fn a_node_property_call_is_refused_as_any_other() {
        for (id, cost) in [(4, 19), (5, 19), (6, 35), (7, 19)] {
            for receiver in ["0101", "0402"] {
                let (err, c) = eval_body_err(&format!("db650{id}{receiver}"));
                assert!(
                    matches!(err, EvalError::UnexpectedValue(_)),
                    "{id}: {err:?}"
                );
                assert_eq!(c, cost, "{id}");
            }
            let (err, c) = eval_body_err(&format!("dc650{id}0101010402"));
            assert!(matches!(err, EvalError::Misc(_)), "{id}: {err:?}");
            assert_eq!(c, cost + 5, "{id}");
        }
    }
}
