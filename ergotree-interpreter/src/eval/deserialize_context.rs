#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    use ergotree_ir::ergo_tree::{ErgoTree, ErgoTreeHeader, ErgoTreeVersion};
    use ergotree_ir::mir::constant::Constant;
    use ergotree_ir::mir::deserialize_context::DeserializeContext;
    use ergotree_ir::mir::expr::Expr;
    use ergotree_ir::mir::global_vars::GlobalVars;
    use ergotree_ir::mir::if_op::If;
    use ergotree_ir::mir::value::Value;
    use ergotree_ir::serialization::SigmaSerializable;
    use ergotree_ir::types::stype::SType;
    use ergotree_ir::unsignedbigint256::UnsignedBigInt;
    use num_traits::Zero;
    use sigma_test_util::force_any_val;

    use crate::eval::reduce_to_crypto;
    use crate::eval::test_util::try_eval_with_deserialize;
    use ergotree_ir::chain::context::Context;
    use ergotree_ir::chain::context_extension::ContextExtension;

    #[test]
    fn eval() {
        let expr: Expr = Expr::from(DeserializeContext {
            tpe: SType::SBoolean,
            id: 1,
        });
        let inner_expr: Expr = true.into();
        let ctx_ext = ContextExtension {
            values: [(1u8, inner_expr.sigma_serialize_bytes().unwrap().into())]
                .iter()
                .cloned()
                .collect(),
        };
        let ctx = force_any_val::<Context>().with_extension(&ctx_ext);
        assert!(try_eval_with_deserialize::<bool>(&expr, &ctx).unwrap());
    }

    // Verify that reduce_to_crypto performs deserialize substitution
    #[test]
    fn eval_reduction() {
        let expr: Expr = Expr::from(DeserializeContext {
            tpe: SType::SBoolean,
            id: 1,
        });
        let inner_expr: Expr = true.into();
        let ctx_ext = ContextExtension {
            values: [(1u8, inner_expr.sigma_serialize_bytes().unwrap().into())]
                .iter()
                .cloned()
                .collect(),
        };
        let ctx = force_any_val::<Context>().with_extension(&ctx_ext);
        assert_eq!(
            reduce_to_crypto(
                &ErgoTree::new(ErgoTreeHeader::v1(false), &expr).unwrap(),
                &ctx,
            )
            .unwrap()
            .sigma_prop,
            true.into()
        );
    }

    #[test]
    fn eval_id_not_found() {
        let expr: Expr = DeserializeContext {
            tpe: SType::SBoolean,
            id: 1,
        }
        .into();
        let extension = ContextExtension::empty();
        let ctx = force_any_val::<Context>().with_extension(&extension);
        assert!(try_eval_with_deserialize::<bool>(&expr, &ctx).is_err());
    }

    // SANTA tx-tier regression (captured testnet tx at height 111,927): trees
    // containing deserialize nodes must charge the JVM's substitution pass —
    // `ergoTree.bytes.length * CostPerTreeByte(2)` block cost, i.e. bytes × 20
    // JitCost (Scala `Interpreter.reductionWithDeserialize`). Since V6
    // activation the charge is part of the reported cost; pre-V6 the JVM
    // checks it against the cost limit but excludes it from the result.
    #[test]
    fn deserialize_substitution_cost_charged() {
        let expr: Expr = Expr::from(DeserializeContext {
            tpe: SType::SBoolean,
            id: 1,
        });
        let inner_expr: Expr = true.into();
        let ctx_ext = ContextExtension {
            values: [(1u8, inner_expr.sigma_serialize_bytes().unwrap().into())]
                .iter()
                .cloned()
                .collect(),
        };
        let tree = ErgoTree::new(ErgoTreeHeader::v1(false), &expr).unwrap();
        let tree_len = tree.sigma_serialize_bytes().unwrap().len() as u64;

        let run = |block_version: u8| -> (u64, u64) {
            let mut ctx = force_any_val::<Context>().with_extension(&ctx_ext);
            ctx.pre_header.version = block_version;
            ctx.jit_cost.set(0);
            let res = reduce_to_crypto(&tree, &ctx).unwrap();
            assert_eq!(res.sigma_prop, true.into());
            (ctx.jit_cost_value(), res.cost)
        };

        // V6 activated (block version 4 → activated script version 3): the
        // substitution charge is included; pre-V6 it is rolled back after the
        // limit check. The subtraction isolates exactly the per-byte charge.
        let (jit_v6, cost_v6) = run(4);
        let (jit_pre, cost_pre) = run(3);
        assert_eq!(
            jit_v6 - jit_pre,
            tree_len * 20,
            "V6 reduction must charge tree bytes ({tree_len}) × 20 JitCost over pre-V6"
        );
        assert_eq!(
            cost_v6 - cost_pre,
            tree_len * 2,
            "reported block cost must include bytes × CostPerTreeByte(2) since V6"
        );

        // The limit check fires in BOTH eras (Scala's `addCostChecked` runs
        // before the era branch): a limit below the substitution charge must
        // reject even pre-V6, where the charge is excluded from the result.
        let mut ctx = force_any_val::<Context>().with_extension(&ctx_ext);
        ctx.pre_header.version = 3;
        ctx.jit_cost.set(0);
        ctx.jit_cost_limit = Some(tree_len * 20 - 1);
        assert!(
            reduce_to_crypto(&tree, &ctx).is_err(),
            "substitution cost must be limit-checked even pre-V6"
        );
    }

    // Each actually-substituted var charges the JVM's deserialization
    // complexity — `scriptBytes.length × CostPerByteDeserialized(2)` block =
    // bytes × 20 JitCost (`Interpreter.deserializeMeasured`). On this branch
    // an absent var errors (no dead-branch leniency yet), so isolate the
    // charge by varying the substituted payload size: the deserialize node
    // sits on a dead `if` branch, keeping the evaluated path identical
    // between the two runs — only the substitution charge differs.
    #[test]
    fn deserialize_substituted_var_charges_per_byte() {
        let deser: Expr = DeserializeContext {
            tpe: SType::SBoolean,
            id: 0,
        }
        .into();
        // if (true) true else deserializeContext(0)
        let expr: Expr = If {
            condition: Expr::Const(true.into()).into(),
            true_branch: Expr::Const(true.into()).into(),
            false_branch: deser.into(),
        }
        .into();

        let run = |inner: &Expr| -> (u64, u64) {
            let bytes = inner.sigma_serialize_bytes().unwrap();
            let len = bytes.len() as u64;
            let ext = ContextExtension {
                values: [(0u8, Constant::from(bytes))].iter().cloned().collect(),
            };
            let ctx = force_any_val::<Context>().with_extension(&ext);
            let before = ctx.jit_cost_value();
            assert!(try_eval_with_deserialize::<bool>(&expr, &ctx).unwrap());
            (ctx.jit_cost_value() - before, len)
        };

        let small: Expr = false.into();
        let big: Expr = If {
            condition: Expr::Const(true.into()).into(),
            true_branch: Expr::Const(false.into()).into(),
            false_branch: Expr::Const(false.into()).into(),
        }
        .into();
        let (cost_small, len_small) = run(&small);
        let (cost_big, len_big) = run(&big);
        assert!(len_big > len_small);
        assert_eq!(
            cost_big - cost_small,
            (len_big - len_small) * 20,
            "substitution must charge the var's bytes × 20 JitCost"
        );
    }

    // Regression for testnet block 111,927: a `DeserializeContext` over an
    // absent context var sitting on a dead `if` branch must NOT sink reduction.
    // Substitution walks the whole tree but, mirroring the JVM
    // `Interpreter.substDeserialize` `else None`, leaves the absent-var node in
    // place; the live branch then reduces normally. A leftover node on the
    // *live* path still errors at eval (see `eval_id_not_found`).
    #[test]
    fn eval_absent_var_on_dead_branch() {
        let deser: Expr = DeserializeContext {
            tpe: SType::SBoolean,
            id: 0,
        }
        .into();
        // if (true) true else deserializeContext(0)
        let expr: Expr = If {
            condition: Expr::Const(true.into()).into(),
            true_branch: Expr::Const(true.into()).into(),
            false_branch: deser.into(),
        }
        .into();
        let extension = ContextExtension::empty();
        let ctx = force_any_val::<Context>().with_extension(&extension);
        assert!(try_eval_with_deserialize::<bool>(&expr, &ctx).unwrap());
    }

    // Parity with the JVM `Interpreter.substDeserialize` inner `case _ => None`:
    // a context var that is present but not a `Coll[Byte]` is not substituted,
    // so a dead-branch deserialize over a wrong-typed var does not sink
    // reduction either. (A wrong-typed var on the *live* path still errors at
    // eval — see `eval_context_extension_wrong_type`.)
    #[test]
    fn eval_wrong_type_var_on_dead_branch() {
        let deser: Expr = DeserializeContext {
            tpe: SType::SBoolean,
            id: 0,
        }
        .into();
        // if (true) true else deserializeContext(0)
        let expr: Expr = If {
            condition: Expr::Const(true.into()).into(),
            true_branch: Expr::Const(true.into()).into(),
            false_branch: deser.into(),
        }
        .into();
        // var 0 present but an Int, not a Coll[Byte]
        let ctx_ext_val: Constant = 1i32.into();
        let ctx_ext = ContextExtension {
            values: [(0u8, ctx_ext_val)].iter().cloned().collect(),
        };
        let ctx = force_any_val::<Context>().with_extension(&ctx_ext);
        assert!(try_eval_with_deserialize::<bool>(&expr, &ctx).unwrap());
    }

    #[test]
    fn eval_context_extension_wrong_type() {
        let expr: Expr = DeserializeContext {
            tpe: SType::SBoolean,
            id: 1,
        }
        .into();
        // should be byte array
        let ctx_ext_val: Constant = 1i32.into();
        let ctx_ext = ContextExtension {
            values: [(1u8, ctx_ext_val)].iter().cloned().collect(),
        };
        let ctx = force_any_val::<Context>().with_extension(&ctx_ext);
        assert!(try_eval_with_deserialize::<bool>(&expr, &ctx).is_err());
    }

    #[test]
    fn evaluated_expr_wrong_type() {
        let expr: Expr = DeserializeContext {
            tpe: SType::SBoolean,
            id: 1,
        }
        .into();
        // should be SBoolean
        let inner_expr: Expr = GlobalVars::Height.into();
        let ctx_ext = ContextExtension {
            values: [(1u8, inner_expr.sigma_serialize_bytes().unwrap().into())]
                .iter()
                .cloned()
                .collect(),
        };
        let ctx = force_any_val::<Context>().with_extension(&ctx_ext);
        assert!(try_eval_with_deserialize::<Value>(&expr, &ctx).is_err());
    }

    #[test]
    fn eval_recursive() {
        let expr: Expr = DeserializeContext {
            tpe: SType::SBoolean,
            id: 1,
        }
        .into();
        let ctx_ext = ContextExtension {
            values: [(1u8, expr.sigma_serialize_bytes().unwrap().into())]
                .iter()
                .cloned()
                .collect(),
        };
        let ctx = force_any_val::<Context>().with_extension(&ctx_ext);
        // Evaluating executeFromVar(1) with ctx[1] being executeFromVar(1) should fail during evaluation
        assert!(try_eval_with_deserialize::<bool>(&expr, &ctx).is_err());
    }
    #[test]
    fn deserialize_v6_type() {
        let expr: Expr = DeserializeContext {
            tpe: SType::SUnsignedBigInt,
            id: 1,
        }
        .into();
        let inner_expr: Expr = Constant::from(UnsignedBigInt::zero()).into();
        let ctx_ext = ContextExtension {
            values: [(1u8, inner_expr.sigma_serialize_bytes().unwrap().into())]
                .iter()
                .cloned()
                .collect(),
        };
        let ctx = force_any_val::<Context>().with_extension(&ctx_ext);
        ctx.tree_version.set(ErgoTreeVersion::V0);
        assert!(try_eval_with_deserialize::<Value>(&expr, &ctx).is_err());
        ctx.tree_version.set(ErgoTreeVersion::V3);
        assert_eq!(
            try_eval_with_deserialize::<UnsignedBigInt>(&expr, &ctx).unwrap(),
            UnsignedBigInt::zero()
        );
    }

    // `Interpreter.fullReduction` reduces a deserialize-bearing SEGREGATED tree
    // from its constants-substituted proposition, not the lazy placeholder form
    // it uses for ordinary trees; `try_eval_with_deserialize` mirrors that
    // conditionality. Values are identical either way — the distinction becomes
    // observable once JIT costing lands (Constant vs ConstantPlaceholder visit
    // costs; the costing lineage pins the blessed cost 20 on these trees). This
    // drives the substituted route end-to-end over the conformance vector trees
    // `{ if (true) true else deserializeContext[Boolean](0|1) }`, with both
    // vars bound to valid serialized scripts (this branch substitutes eagerly
    // over the whole tree, dead branches included).
    #[test]
    fn deserialize_bearing_segregated_tree_evals_substituted() {
        let inner_bytes: Constant = Expr::from(true).sigma_serialize_bytes().unwrap().into();
        for hex in [
            "1b0d02010101019573007301d40100", // deserializeContext(0) on the dead branch
            "1b0d02010101019573007301d40101", // deserializeContext(1) on the dead branch
        ] {
            let bytes: Vec<u8> = (0..hex.len())
                .step_by(2)
                .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
                .collect();
            // Clear the size bit and drop the size byte (non-SigmaProp root —
            // the conformance runner's lenient parse path).
            let mut lenient = Vec::with_capacity(bytes.len() - 1);
            lenient.push(bytes[0] & !0x08);
            lenient.extend_from_slice(&bytes[2..]);
            let tree = ErgoTree::sigma_parse_bytes(&lenient).unwrap();

            let ctx_ext = ContextExtension {
                values: [(0u8, inner_bytes.clone()), (1u8, inner_bytes.clone())]
                    .iter()
                    .cloned()
                    .collect(),
            };
            let mut ctx = force_any_val::<Context>().with_extension(&ctx_ext);
            ctx.pre_header.version = 4;
            ctx.tree_version.set(ErgoTreeVersion::V3);
            let constants = tree.constants().unwrap();
            let eval_ctx = ctx.with_constants(constants);
            assert!(
                try_eval_with_deserialize::<bool>(tree.root_expr().unwrap(), &eval_ctx).unwrap()
            );
        }
    }
}
