use alloc::boxed::Box;
use alloc::vec::Vec;
use bounded_vec::NonEmptyVec;
use ergotree_ir::mir::constant::TryExtractInto;
use ergotree_ir::mir::sigma_and::SigmaAnd;
use ergotree_ir::mir::value::Value;
use ergotree_ir::sigma_protocol::sigma_boolean::cand::Cand;
use ergotree_ir::sigma_protocol::sigma_boolean::SigmaProp;

use crate::eval::env::Env;
use crate::eval::Context;
use crate::eval::EvalError;
use crate::eval::Evaluable;

impl Evaluable for SigmaAnd {
    fn eval<'ctx>(
        &self,
        env: &mut Env<'ctx>,
        ctx: &Context<'ctx>,
    ) -> Result<Value<'ctx>, EvalError> {
        ctx.add_per_item_jit_cost(10, 2, 1, self.items.len() as u32)?;
        // sigmastate's `allZK` requires its items to be non-empty
        let items_v: NonEmptyVec<_> = self
            .items
            .iter()
            .map(|it| it.eval(env, ctx))
            .collect::<Result<Vec<_>, _>>()?
            .try_into()?;
        let items_sigmabool = items_v
            .try_mapped(|it| it.try_extract_into::<SigmaProp>())?
            .mapped(|it| it.value().clone());
        Ok(Value::SigmaProp(Box::new(SigmaProp::new(
            Cand::normalized(items_sigmabool),
        ))))
    }
}

#[allow(clippy::unwrap_used)]
#[allow(clippy::panic)]
#[cfg(test)]
#[cfg(feature = "arbitrary")]
mod tests {
    use ergotree_ir::sigma_protocol::sigma_boolean::SigmaBoolean;
    use ergotree_ir::sigma_protocol::sigma_boolean::SigmaConjecture;

    use crate::eval::test_util::eval_out;
    use ergotree_ir::chain::context::Context;

    use super::*;

    use ergotree_ir::mir::expr::Expr;
    use proptest::collection;
    use proptest::prelude::*;
    use sigma_test_util::force_any_val;

    proptest! {

        #![proptest_config(ProptestConfig::with_cases(8))]

        #[test]
        fn eval(sigmaprops in collection::vec(any::<SigmaProp>(), 2..10)) {
            let items = sigmaprops.clone().into_iter().map(|sp| Expr::Const(sp.into())).collect();
            let expr: Expr = SigmaAnd::new(items).unwrap().into();
            let ctx = force_any_val::<Context>();
            let res = eval_out::<SigmaProp>(&expr, &ctx);
            let expected_sb: Vec<SigmaBoolean> = sigmaprops.into_iter().map(|sp| sp.into()).collect();
            prop_assert!(matches!(res.clone().into(), SigmaBoolean::SigmaConjecture(SigmaConjecture::Cand(_))));
            if let SigmaBoolean::SigmaConjecture(SigmaConjecture::Cand(Cand {items: actual_sb})) = res.into() {
                prop_assert_eq!(actual_sb, expected_sb);
            }
        }
    }
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
#[cfg(feature = "arbitrary")]
mod bounds_tests {
    //! JVM parity: `SigmaAnd.eval` evaluates its items, charges `PerItemCost(10, 2, 1)` for their
    //! count and calls `allZK`, whose `CAND.normalized` requires a non-empty list
    //! (`trees.scala:127-141`, `CSigmaDslBuilder.scala:134-138`, `SigmaBoolean.scala:165`).
    use super::*;
    use crate::eval::reduce_to_crypto;
    use crate::eval::test_util::{eval_out, try_eval_out};
    use ergotree_ir::chain::context::Context;
    use ergotree_ir::ergo_tree::ErgoTree;
    use ergotree_ir::mir::expr::Expr;
    use ergotree_ir::serialization::SigmaSerializable;
    use ergotree_ir::sigma_protocol::sigma_boolean::SigmaBoolean;
    use sigma_test_util::force_any_val;

    #[test]
    fn a_node_without_items_is_an_error() {
        // SANTA `sized-tree-spend` #10
        let expr: Expr = SigmaAnd::new(vec![]).unwrap().into();
        let ctx = force_any_val::<Context>();
        assert!(try_eval_out::<SigmaProp>(&expr, &ctx).is_err());
    }

    #[test]
    fn items_256_of_the_neutral_proposition_reduce_to_it() {
        // SANTA `sized-tree-spend` #12: 256 × `sigmaProp(true)` is TrueProp.
        // The cost is 10 + 2 × 256 for the node and 5 for each constant.
        let neutral = SigmaBoolean::TrivialProp(true);
        let items = vec![Expr::Const(SigmaProp::new(neutral.clone()).into()); 256];
        let expr: Expr = SigmaAnd::new(items).unwrap().into();
        let ctx = force_any_val::<Context>();
        let before = ctx.jit_cost_value();
        let res = eval_out::<SigmaProp>(&expr, &ctx);
        assert_eq!(SigmaBoolean::from(res), neutral);
        assert_eq!(ctx.jit_cost_value() - before, 10 + 2 * 256 + 5 * 256);
    }

    #[test]
    fn a_tree_whose_root_has_no_items_does_not_reduce() {
        // A failed reduction prints the tree and evaluates what was printed, so this is the
        // printer's test too.
        let tree = ErgoTree::sigma_parse_bytes(&[0x00, 0xea, 0x00]).unwrap();
        let ctx = force_any_val::<Context>();
        assert!(reduce_to_crypto(&tree, &ctx).is_err());
    }
}
