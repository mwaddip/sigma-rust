use crate::eval::EvalError;

use alloc::boxed::Box;
use alloc::string::ToString;
use alloc::vec::Vec;
use ergotree_ir::chain::ergo_box::ErgoBox;
use ergotree_ir::chain::ergo_box::RegisterId;
use ergotree_ir::ergo_tree::ErgoTreeVersion;
use ergotree_ir::mir::constant::TryExtractInto;
use ergotree_ir::mir::value::Value;
use ergotree_ir::reference::Ref;
use ergotree_ir::types::stype::SType;

use super::EvalFn;

pub(crate) static VALUE_EVAL_FN: EvalFn = |_mc, _env, ctx, obj, _args| {
    ctx.add_jit_cost(8)?;
    Ok(Value::Long(
        obj.try_extract_into::<Ref<'_, ErgoBox>>()?.value.as_i64(),
    ))
};

pub(crate) static GET_REG_EVAL_FN: EvalFn = |mc, _env, ctx, obj, args| {
    ctx.add_jit_cost(50)?;
    if ctx.tree_version() < ErgoTreeVersion::V3 {
        return Err(EvalError::ScriptVersionError {
            required_version: ErgoTreeVersion::V3,
            activated_version: ctx.tree_version(),
        });
    }
    let reg_idx = args
        .first()
        .cloned()
        .ok_or_else(|| EvalError::NotFound("register index is missing".to_string()))?
        .try_extract_into::<i32>()?;
    // Mirror JVM CBox.getReg: an out-of-range index (negative or >= maxRegisters)
    // yields None rather than an error; only a present register of the wrong type
    // errors below.
    let reg_id = i8::try_from(reg_idx)
        .ok()
        .and_then(|id| RegisterId::try_from(id).ok());
    let reg_val_opt = match reg_id {
        Some(reg_id) => obj
            .try_extract_into::<Ref<'_, ErgoBox>>()?
            .get_register(reg_id)
            .map_err(|e| {
                EvalError::NotFound(format!(
                    "Error getting the register id {reg_id} with error {e:?}"
                ))
            })?,
        None => None,
    };
    // Return type of getReg[T] is always Option[T]
    #[allow(clippy::unreachable)]
    let SType::SOption(expected_type) = &*mc.tpe().t_range
    else {
        unreachable!()
    };
    match reg_val_opt {
        Some(constant) if constant.tpe == **expected_type => {
            Ok(Value::Opt(Some(Box::new(constant.v.into()))))
        }
        Some(constant) => Err(EvalError::UnexpectedValue(format!(
            "Expected register {reg_idx} to be of type {}, got {}",
            expected_type, constant.tpe
        ))),
        None => Ok(Value::Opt(None)),
    }
};

pub(crate) static TOKENS_EVAL_FN: EvalFn = |_mc, _env, ctx, obj, _args| {
    ctx.add_jit_cost(15)?;
    let res: Value = obj
        .try_extract_into::<Ref<'_, ErgoBox>>()?
        .tokens_raw()
        .into();
    Ok(res)
};

// The accessor method-forms (PropertyCall 99:2..6) dispatch through the same box
// accessors as their op-form twins (ExtractScriptBytes/ExtractBytes/ExtractBytesWithNoRef/
// ExtractId/ExtractCreationInfo) and charge the op-form's `costKind`; the MethodCall/
// PropertyCall envelope (Fixed(4)) is charged by the call site, so the full-tree cost is
// `receiver(ConstantPlaceholder 1) + envelope(4) + op-form costKind` (JVM
// `MethodCall.eval` -> `invokeFixed`, methods.scala SBoxMethods).

pub(crate) static PROPOSITION_BYTES_EVAL_FN: EvalFn = |_mc, _env, ctx, obj, _args| {
    ctx.add_jit_cost(10)?; // ExtractScriptBytes = Fixed(10)
    Ok(obj
        .try_extract_into::<Ref<'_, ErgoBox>>()?
        .script_bytes()?
        .into())
};

pub(crate) static BYTES_EVAL_FN: EvalFn = |_mc, _env, ctx, obj, _args| {
    ctx.add_jit_cost(12)?; // ExtractBytes = Fixed(12)
                           // `Box.bytes` returns the parse-retained wire slice (the reference impl's
                           // `ErgoBox.bytes`), so a non-canonically-encoded box keeps its on-the-wire
                           // image — unlike `bytesWithoutRef`, which re-serializes canonically.
    Ok(obj.try_extract_into::<Ref<'_, ErgoBox>>()?.bytes()?.into())
};

pub(crate) static BYTES_WITHOUT_REF_EVAL_FN: EvalFn = |_mc, _env, ctx, obj, _args| {
    ctx.add_jit_cost(12)?; // ExtractBytesWithNoRef = Fixed(12)
    Ok(obj
        .try_extract_into::<Ref<'_, ErgoBox>>()?
        .bytes_without_ref()?
        .into())
};

pub(crate) static ID_EVAL_FN: EvalFn = |_mc, _env, ctx, obj, _args| {
    ctx.add_jit_cost(12)?; // ExtractId = Fixed(12)
    let bytes: Vec<i8> = obj.try_extract_into::<Ref<'_, ErgoBox>>()?.box_id().into();
    Ok(bytes.into())
};

pub(crate) static CREATION_INFO_EVAL_FN: EvalFn = |_mc, _env, ctx, obj, _args| {
    ctx.add_jit_cost(16)?; // ExtractCreationInfo = Fixed(16)
    Ok(obj
        .try_extract_into::<Ref<'_, ErgoBox>>()?
        .creation_info()
        .into())
};

#[allow(clippy::unwrap_used)]
#[allow(clippy::panic)]
#[cfg(test)]
#[cfg(feature = "arbitrary")]
mod tests {
    use alloc::boxed::Box;

    use ergotree_ir::chain::context_extension::ContextExtension;
    use ergotree_ir::chain::ergo_box::ErgoBox;
    use ergotree_ir::ergo_tree::{ErgoTree, ErgoTreeVersion};
    use ergotree_ir::mir::constant::Constant;
    use ergotree_ir::mir::expr::Expr;
    use ergotree_ir::mir::extract_amount::ExtractAmount;
    use ergotree_ir::mir::extract_reg_as::ExtractRegisterAs;
    use ergotree_ir::mir::global_vars::GlobalVars;
    use ergotree_ir::mir::method_call::MethodCall;
    use ergotree_ir::mir::option_get::OptionGet;
    use ergotree_ir::mir::property_call::PropertyCall;
    use ergotree_ir::mir::unary_op::OneArgOpTryBuild;
    use ergotree_ir::mir::value::Value;
    use ergotree_ir::serialization::SigmaSerializable;
    use ergotree_ir::types::sbox;
    use ergotree_ir::types::stype::SType;
    use ergotree_ir::types::stype_param::STypeVar;
    use sigma_test_util::force_any_val;

    use crate::eval::test_util::{eval_out, try_eval_out_with_version};
    use crate::eval::EvalError;
    use ergotree_ir::chain::context::Context;

    // The vector trees carry arbitrary-typed roots (eval-tier corpus); the sized
    // parse path rejects non-SigmaProp roots, so clear the size bit and drop the
    // size byte to route through the non-sized path — the same leniency the
    // conformance runner applies.
    fn parse_tree_lenient(hex: &str) -> Expr {
        let bytes: Vec<u8> = (0..hex.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
            .collect();
        // Arbitrary-typed root (eval-tier corpus): the lenient entry skips only the
        // SigmaProp-root check; the real header (size bit set) keeps Rule-1012 happy.
        let tree = ErgoTree::sigma_parse_bytes_lenient(&bytes).unwrap();
        tree.proposition().unwrap()
    }

    // SELF with exactly R4 = Long(7) and ContextExtension var 1 = `idx` — the
    // setup of the blessed `Box.getReg_dynamic_index` vectors.
    fn ctx_with_r4_long7_and_var1(idx: i32) -> Context<'static> {
        let b = force_any_val::<ErgoBox>()
            .with_additional_registers(vec![Constant::from(7i64)].try_into().unwrap());
        let mut ext = ContextExtension::empty();
        ext.values.insert(1u8, Constant::from(idx));
        Context {
            self_box: Box::leak(Box::new(b)),
            extension: Box::leak(Box::new(ext)),
            ..force_any_val::<Context>()
        }
    }

    #[test]
    fn eval_box_value() {
        let expr: Expr = PropertyCall::new(GlobalVars::SelfBox.into(), sbox::VALUE_METHOD.clone())
            .unwrap()
            .into();
        let ctx = force_any_val::<Context>();
        assert_eq!(eval_out::<i64>(&expr, &ctx), ctx.self_box.value.as_i64());
    }

    // JVM-blessed vectors (santa-eval `Box.signed_view_u64`): box value and token
    // amounts are unbounded u64 on the wire (reference impl reads `getULong()` with
    // no range check); values in `[2^63, 2^64)` hydrate and surface as their signed
    // (negative) view at eval, like the JVM's Long.
    #[test]
    fn eval_box_signed_view_u64() {
        fn ctx_with_self_box(bytes_hex: &str) -> Context<'static> {
            let b = ErgoBox::sigma_parse_bytes(&base16::decode(bytes_hex).unwrap()).unwrap();
            let ctx = force_any_val::<Context>();
            Context {
                self_box: Box::leak(Box::new(b)),
                ..ctx
            }
        }
        let value_expr: Expr = ExtractAmount {
            input: Box::new(GlobalVars::SelfBox.into()),
        }
        .into();
        let r0_expr: Expr = OptionGet::try_build(
            ExtractRegisterAs::new(
                GlobalVars::SelfBox.into(),
                0,
                SType::SOption(SType::SLong.into()),
            )
            .unwrap()
            .into(),
        )
        .unwrap()
        .into();
        let tokens_expr: Expr =
            PropertyCall::new(GlobalVars::SelfBox.into(), sbox::TOKENS_METHOD.clone())
                .unwrap()
                .into();

        // box value = 2^63 → SELF.value / SELF.R0[Long].get = Long(-2^63)
        let ctx = ctx_with_self_box("808080808080808080010008cd0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798000000000000000000000000000000000000000000000000000000000000000000000000");
        assert_eq!(eval_out::<i64>(&value_expr, &ctx), i64::MIN);
        assert_eq!(eval_out::<i64>(&r0_expr, &ctx), i64::MIN);

        // box value = u64::MAX → Long(-1)
        let ctx = ctx_with_self_box("ffffffffffffffffff010008cd0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798000000000000000000000000000000000000000000000000000000000000000000000000");
        assert_eq!(eval_out::<i64>(&value_expr, &ctx), -1i64);
        assert_eq!(eval_out::<i64>(&r0_expr, &ctx), -1i64);

        // token amount = 2^63 → SELF.tokens(0)._2 = Long(-2^63)
        let ctx = ctx_with_self_box("c0843d0008cd0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798000107070707070707070707070707070707070707070707070707070707070707078080808080808080800100000000000000000000000000000000000000000000000000000000000000000000");
        assert_eq!(
            eval_out::<Vec<(Vec<i8>, i64)>>(&tokens_expr, &ctx)[0].1,
            i64::MIN
        );

        // token amount = u64::MAX → SELF.tokens(0)._2 = Long(-1)
        let ctx = ctx_with_self_box("c0843d0008cd0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f8179800010707070707070707070707070707070707070707070707070707070707070707ffffffffffffffffff0100000000000000000000000000000000000000000000000000000000000000000000");
        assert_eq!(
            eval_out::<Vec<(Vec<i8>, i64)>>(&tokens_expr, &ctx)[0].1,
            -1i64
        );
    }

    #[test]
    fn eval_box_tokens() {
        let expr: Expr = PropertyCall::new(GlobalVars::SelfBox.into(), sbox::TOKENS_METHOD.clone())
            .unwrap()
            .into();
        let ctx = force_any_val::<Context>();
        assert_eq!(
            eval_out::<Vec<(Vec<i8>, i64)>>(&expr, &ctx),
            ctx.self_box.tokens_raw()
        );
    }

    #[test]
    fn eval_reg_out() {
        let type_args = std::iter::once((STypeVar::t(), SType::SLong)).collect();
        let expr: Expr = MethodCall::with_type_args(
            GlobalVars::SelfBox.into(),
            sbox::GET_REG_METHOD.clone().with_concrete_types(&type_args),
            vec![Constant::from(0i32).into()],
            type_args,
        )
        .unwrap()
        .into();
        let ctx = force_any_val::<Context>();
        (0..ErgoTreeVersion::V3.into()).for_each(|version| {
            assert!(try_eval_out_with_version::<i64>(&expr, &ctx, version, version).is_err())
        });
        (ErgoTreeVersion::V3.into()..=ErgoTreeVersion::MAX_SCRIPT_VERSION.into()).for_each(
            |version| {
                assert_eq!(
                    try_eval_out_with_version::<Option<i64>>(&expr, &ctx, version, version)
                        .unwrap()
                        .unwrap(),
                    ctx.self_box.value.as_i64()
                )
            },
        );
    }

    // Attempt to extract SigmaProp from register of type SLong
    #[test]
    fn eval_reg_out_wrong_type() {
        let type_args = std::iter::once((STypeVar::t(), SType::SSigmaProp)).collect();
        let expr: Expr = MethodCall::with_type_args(
            GlobalVars::SelfBox.into(),
            sbox::GET_REG_METHOD.clone().with_concrete_types(&type_args),
            vec![Constant::from(0i32).into()],
            type_args,
        )
        .unwrap()
        .into();
        let ctx = force_any_val::<Context>();
        (0..ErgoTreeVersion::V3.into()).for_each(|version| {
            // Pre-V3 the method id is not in the method table at all (min_version),
            // so the versioned roundtrip fails at deserialization — mirroring JVM
            // v5Methods, which has no getReg (id 19) entry.
            let res = try_eval_out_with_version::<Option<i64>>(&expr, &ctx, version, version);
            match res {
                Err(EvalError::SigmaParsingError(_)) => {}
                _ => panic!("Expected method-id parsing rejection, got {:?}", res),
            }
        });
        (ErgoTreeVersion::V3.into()..=ErgoTreeVersion::MAX_SCRIPT_VERSION.into()).for_each(
            |version| {
                assert!(
                    try_eval_out_with_version::<Option<i64>>(&expr, &ctx, version, version)
                        .is_err()
                )
            },
        );
    }

    // End-to-end over the blessed `Box.getReg_dynamic_index` vector trees
    // (sigma-state 6.0.3): `{ SELF.getReg[Long](getVar[Int](1).get) }` with
    // SELF carrying only R4 = Long(7). JVM CBox.getReg yields None for an
    // absent or out-of-range index; only a present register of the wrong
    // type errors.
    #[test]
    fn eval_reg_dynamic_index_absent_or_out_of_range_is_none() {
        const GET_REG_LONG: &str = "1b0b00dc6313a701e4e3010405";
        let run = |idx: i32| -> Option<i64> {
            let expr = parse_tree_lenient(GET_REG_LONG);
            let ctx = ctx_with_r4_long7_and_var1(idx);
            try_eval_out_with_version::<Option<i64>>(&expr, &ctx, 3, 3).unwrap()
        };
        assert_eq!(run(4), Some(7), "present register of matching type");
        assert_eq!(run(5), None, "absent register R5");
        assert_eq!(run(10), None, "index beyond R9");
        assert_eq!(run(-1), None, "negative index");
        assert_eq!(run(1_000_000), None, "index beyond i8 range");

        // The wrong-type boundary stays an error: getReg[Int] over the Long R4.
        const GET_REG_INT: &str = "1b0b00dc6313a701e4e3010404";
        let expr = parse_tree_lenient(GET_REG_INT);
        let ctx = ctx_with_r4_long7_and_var1(4);
        assert!(
            try_eval_out_with_version::<Option<i32>>(&expr, &ctx, 3, 3).is_err(),
            "present register of the wrong type must error"
        );
    }

    // End-to-end over the blessed `Box.accessor_method_form` vectors (sigma-state 6.0.3,
    // santa:authored-sbox-method-form). Each tree is a self-contained ErgoTree whose box
    // receiver is `ConstantPlaceholder(0)`; evaluated through the lazy-constants path
    // (`root_expr` + `with_constants`, the same path the conformance runner uses) so the
    // reported cost is the JVM's full-tree `ConstantPlaceholder(1) + MethodCall/PropertyCall
    // envelope(4) + op-form costKind`. The byte-basis trio (bytes/bytesWithoutRef/id) reuses
    // the `Box.bytes_byte_basis` op-form twins' box (non-canonical `00‖aa×32` R4 GE): `.bytes`
    // and `.id` are the parse-RETAINED wire slice, `.bytesWithoutRef` is the CANONICAL
    // re-serialization (the asymmetry). Version pin {activated 2, ergoTree 0}.
    #[test]
    fn eval_accessor_method_forms_blessed_vectors() {
        use crate::eval::test_util::try_eval_with_deserialize;
        use ergotree_ir::mir::constant::TryExtractFrom;

        fn run<T: TryExtractFrom<Value<'static>> + 'static>(tree_hex: &str) -> (T, u64) {
            let bytes = base16::decode(tree_hex).unwrap();
            let tree = ErgoTree::sigma_parse_bytes_lenient(&bytes).unwrap();
            let root = tree.root_expr().unwrap();
            let constants = tree.constants().unwrap();
            let mut ctx = force_any_val::<Context>();
            ctx.pre_header.version = 3; // activated 2 (+1 block-version convention)
            ctx.tree_version.set(ErgoTreeVersion::V0); // ergoTree 0
            let eval_ctx = ctx.with_constants(constants);
            let before = eval_ctx.jit_cost_value();
            let v = try_eval_with_deserialize::<T>(root, &eval_ctx).unwrap();
            let cost = eval_ctx.jit_cost_value() - before;
            (v, cost)
        }

        // value (99:1) — the already-registered control; must stay green at 13.
        let (value, cost) = run::<i64>("100163c0843d10010101d1730000000107000000000000000000000000000000000000000000000000000000000000000000111111111111111111111111111111111111111111111111111111111111111100db63017300");
        assert_eq!(value, 1_000_000);
        assert_eq!(cost, 13);

        // propositionBytes (99:2) = CP(1)+envelope(4)+ExtractScriptBytes(10)
        let (pb, cost) = run::<Vec<i8>>("100163c0843d10010101d1730000000107000000000000000000000000000000000000000000000000000000000000000000111111111111111111111111111111111111111111111111111111111111111100db63027300");
        assert_eq!(pb, vec![16, 1, 1, 1, -47, 115, 0]);
        assert_eq!(cost, 15);

        // bytes (99:3) — the parse-RETAINED slice (0x00‖aa×32 R4 survives); byte-matches
        // the op-form Box.bytes_byte_basis retained twin. CP(1)+envelope(4)+ExtractBytes(12)
        let (bytes, cost) = run::<Vec<i8>>("100163c0843d10010101d173000000010700aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa111111111111111111111111111111111111111111111111111111111111111100db63037300");
        assert_eq!(
            bytes,
            vec![
                -64, -124, 61, 16, 1, 1, 1, -47, 115, 0, 0, 0, 1, 7, 0, -86, -86, -86, -86, -86,
                -86, -86, -86, -86, -86, -86, -86, -86, -86, -86, -86, -86, -86, -86, -86, -86,
                -86, -86, -86, -86, -86, -86, -86, -86, -86, -86, -86, 17, 17, 17, 17, 17, 17, 17,
                17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17,
                17, 17, 17, 17, 0
            ]
        );
        assert_eq!(cost, 17);

        // bytesWithoutRef (99:4) — the CANONICAL re-serialization (R4 GE normalized to 00×33,
        // no txid/index ref); distinct from .bytes above. CP(1)+envelope(4)+ExtractBytesWithNoRef(12)
        let (bwr, cost) = run::<Vec<i8>>("100163c0843d10010101d173000000010700aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa111111111111111111111111111111111111111111111111111111111111111100db63047300");
        assert_eq!(
            bwr,
            vec![
                -64, -124, 61, 16, 1, 1, 1, -47, 115, 0, 0, 0, 1, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            ]
        );
        assert_eq!(cost, 17);

        // id (99:5) — blake2b256 of the RETAINED slice. CP(1)+envelope(4)+ExtractId(12)
        let (id, cost) = run::<Vec<i8>>("100163c0843d10010101d173000000010700aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa111111111111111111111111111111111111111111111111111111111111111100db63057300");
        assert_eq!(
            id,
            vec![
                -119, 61, -98, 90, 67, -3, -45, 97, 115, -94, -113, -118, -12, 3, 125, -121, -47,
                12, 125, 85, -68, 60, -13, -112, -22, 121, 65, 67, -86, -63, -11, -84
            ]
        );
        assert_eq!(cost, 17);

        // creationInfo (99:6) — (height, txid‖index bytes). CP(1)+envelope(4)+ExtractCreationInfo(16)
        let (ci, cost) = run::<(i32, Vec<i8>)>("100163c0843d10010101d1730000000107000000000000000000000000000000000000000000000000000000000000000000111111111111111111111111111111111111111111111111111111111111111100db63067300");
        assert_eq!(
            ci,
            (
                0i32,
                vec![
                    17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17,
                    17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 0, 0
                ]
            )
        );
        assert_eq!(cost, 21);
    }

    // getRegV5 (method id 7) deserializes but has no eval — mirroring the JVM,
    // where getRegMethodV5's reflective lookup fails. A live occurrence errors;
    // a dead-branch occurrence leaves the script evaluable. Trees are the
    // blessed `Box.getReg_adversarial` vectors.
    #[test]
    fn eval_getregv5_parses_but_does_not_eval() {
        // { SELF.getRegV5(getVar[Int](1).get) } — live, must error at eval.
        let expr = parse_tree_lenient("1b0a00dc6307a701e4e30104");
        let ctx = ctx_with_r4_long7_and_var1(4);
        assert!(try_eval_out_with_version::<Value>(&expr, &ctx, 3, 3).is_err());

        // { if (true) true else SELF.getRegV5(getVar[Int](1).get).isDefined }
        // — dead branch, the tree must parse and evaluate to true.
        let expr = parse_tree_lenient("1b1402010101019573007301e6dc6307a701e4e30104");
        let ctx = ctx_with_r4_long7_and_var1(4);
        assert!(try_eval_out_with_version::<bool>(&expr, &ctx, 3, 3).unwrap());
    }
}
