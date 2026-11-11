use core::convert::TryInto;

use alloc::boxed::Box;
use alloc::vec::Vec;
use ergotree_ir::mir::atleast::Atleast;
use ergotree_ir::mir::coll_value::CollValue;
use ergotree_ir::mir::constant::TryExtractFromError;
use ergotree_ir::mir::constant::TryExtractInto;
use ergotree_ir::mir::value::Value;
use ergotree_ir::sigma_protocol::sigma_boolean::cthreshold::Cthreshold;
use ergotree_ir::sigma_protocol::sigma_boolean::SigmaBoolean;
use ergotree_ir::sigma_protocol::sigma_boolean::SigmaProp;
use ergotree_ir::types::stype::SType;

use crate::eval::env::Env;
use crate::eval::Context;
use crate::eval::EvalError;
use crate::eval::Evaluable;

/// The propositions of `AtLeast`'s input. sigmastate casts the collection's backing array to
/// `SigmaProp[]` (`toSigmaTrees(props.toArray)`, v6.0.6 `CSigmaDslBuilder.scala:105`), which only
/// a collection built at the item type `SigmaProp` has: a tuple of propositions, a `Coll[Any]`
/// over `Object[]`, throws there, empty or not (SANTA's tuple-values reply, A4)
fn sigma_props(coll: &CollValue<'_>) -> Result<Vec<SigmaBoolean>, EvalError> {
    if coll.elem_tpe() != SType::SSigmaProp {
        return Err(EvalError::UnexpectedValue(format!(
            "Atleast: a collection of {:?} is not an array of SigmaProp",
            coll.elem_tpe()
        )));
    }
    Ok(coll
        .as_vec()
        .into_iter()
        .map(|i| {
            i.try_extract_into::<SigmaProp>()
                .map(|sp| sp.value().clone())
        })
        .collect::<Result<Vec<SigmaBoolean>, TryExtractFromError>>()?)
}

impl Evaluable for Atleast {
    fn eval<'ctx>(
        &self,
        env: &mut Env<'ctx>,
        ctx: &Context<'ctx>,
    ) -> Result<Value<'ctx>, EvalError> {
        let bound_v = self.bound.eval(env, ctx)?;
        let input_v = self.input.eval(env, ctx)?;

        let coll = match input_v {
            Value::Coll(coll) => Ok(coll),
            _ => Err(EvalError::UnexpectedValue(format!(
                "Atleast: expected input to be Value::Coll, got: {0:?}",
                input_v
            ))),
        }?;
        ctx.add_per_item_jit_cost(20, 3, 5, coll.len() as u32)?;

        let bound = bound_v.try_extract_into::<i32>()?;
        let input = sigma_props(&coll)?;

        // Mirror Scala `CSigmaDslBuilder.atLeast` (sigma/data/CSigmaDslBuilder.scala): the
        // children-count cap (`MaxChildrenCountForAtLeastOp = 255`) is checked BEFORE
        // `AtLeast.reduce`, so it precedes the degenerate-bound short-circuits below. A
        // `ConcreteCollection` carries a u16 count, so >255 children are wire-constructible.
        if input.len() > 255 {
            return Err(EvalError::Misc(format!(
                "Atleast: expected input elements count ({}) should not exceed 255",
                input.len()
            )));
        }

        // Mirror Scala `AtLeast.reduce` (sigma/ast/trees.scala) degenerate-bound handling,
        // applied before constructing a CTHRESHOLD:
        //   bound <= 0            => TrueProp  (always satisfied)
        //   bound > children.size => FalseProp (unsatisfiable; NOT an error)
        if bound <= 0 {
            return Ok(Value::SigmaProp(Box::new(SigmaProp::new(
                SigmaBoolean::TrivialProp(true),
            ))));
        }
        if bound > input.len() as i32 {
            return Ok(Value::SigmaProp(Box::new(SigmaProp::new(
                SigmaBoolean::TrivialProp(false),
            ))));
        }
        // Here `1 <= bound <= input.len()`, and the check above holds `input.len()` to 255, so
        // `bound` fits in u8.
        let bound_u8: u8 = bound.try_into().map_err(|_| {
            EvalError::Misc(format!(
                "Atleast: bound ({}) too large for input size {}",
                bound,
                input.len()
            ))
        })?;
        Ok(Value::SigmaProp(Box::new(SigmaProp::new(
            Cthreshold::reduce(bound_u8, input.try_into()?),
        ))))
    }
}

#[allow(clippy::unwrap_used)]
#[allow(clippy::panic)]
#[cfg(test)]
mod tests {
    use alloc::sync::Arc;

    use crate::eval::test_util::try_eval_out_wo_ctx;
    use ergotree_ir::mir::constant::Constant;
    use ergotree_ir::mir::constant::Literal;
    use ergotree_ir::mir::value::CollKind;
    use ergotree_ir::sigma_protocol::sigma_boolean::SigmaBoolean;
    use ergotree_ir::sigma_protocol::sigma_boolean::SigmaConjecture;
    use ergotree_ir::types::stype::SType;

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
        fn eval(sigmaprops in collection::vec(any::<SigmaProp>(), 4..8)) {
            let items = Literal::Coll(CollKind::from_collection(SType::SSigmaProp,
                sigmaprops.into_iter().map(|s| s.into()).collect::<Arc<[Literal]>>()).unwrap());
            let expr: Expr = Atleast::new(2i32.into(),
                Constant {tpe: SType::SColl(SType::SSigmaProp.into()), v: items}.into()).unwrap().into();
            let ctx = force_any_val::<Context>();
            let res = eval_out::<SigmaProp>(&expr, &ctx);
            prop_assert!(matches!(res.into(),
                SigmaBoolean::SigmaConjecture(SigmaConjecture::Cthreshold(_))));
        }
    }

    fn two_sigmaprops_coll() -> Literal {
        let sigmaprops = vec![force_any_val::<SigmaProp>(), force_any_val::<SigmaProp>()];
        Literal::Coll(
            CollKind::from_collection(
                SType::SSigmaProp,
                sigmaprops
                    .into_iter()
                    .map(|s| s.into())
                    .collect::<Arc<[Literal]>>(),
            )
            .unwrap(),
        )
    }

    fn n_sigmaprops_coll(n: usize) -> Literal {
        let one: Literal = force_any_val::<SigmaProp>().into();
        Literal::Coll(
            CollKind::from_collection(
                SType::SSigmaProp,
                core::iter::repeat_n(one, n).collect::<Arc<[Literal]>>(),
            )
            .unwrap(),
        )
    }

    fn atleast_expr(bound: i32, items: Literal) -> Expr {
        Atleast::new(
            bound.into(),
            Constant {
                tpe: SType::SColl(SType::SSigmaProp.into()),
                v: items,
            }
            .into(),
        )
        .unwrap()
        .into()
    }

    #[test]
    fn needs_an_array_of_sigma_props() {
        let props = || -> Vec<Value<'static>> {
            vec![
                Value::SigmaProp(Box::new(force_any_val::<SigmaProp>())),
                Value::SigmaProp(Box::new(force_any_val::<SigmaProp>())),
            ]
        };
        let typed = CollValue::generic(SType::SSigmaProp, props());
        assert_eq!(sigma_props(&typed).unwrap().len(), 2);
        // a tuple of propositions is a `Coll[Any]`: the cast of its array throws, empty or not
        let tuple = CollValue::generic(SType::SAny, props());
        assert!(sigma_props(&tuple).is_err());
        let empty_tuple = CollValue::generic(SType::SAny, Vec::<Value<'static>>::new());
        assert!(sigma_props(&empty_tuple).is_err());
    }

    #[test]
    fn a_concrete_collection_is_an_array_of_sigma_props() {
        use ergotree_ir::mir::collection::Collection;
        let prop = || Expr::Const(force_any_val::<SigmaProp>().into());
        let coll: Expr = Collection::new(SType::SSigmaProp, vec![prop(), prop()])
            .unwrap()
            .into();
        let expr: Expr = Atleast::new(2i32.into(), coll).unwrap().into();
        assert!(try_eval_out_wo_ctx::<SigmaProp>(&expr).is_ok());
    }

    // Scala `AtLeast.reduce` (sigma/ast/trees.scala): `bound > children.size` reduces to
    // FalseProp (unsatisfiable), not an error — for any bound, including `bound > 255`.
    #[test]
    fn bound_exceeds_input_size_reduces_to_false() {
        let items = two_sigmaprops_coll();
        for bound in [3i32, 256] {
            let expr = atleast_expr(bound, items.clone());
            let res: SigmaBoolean = try_eval_out_wo_ctx::<SigmaProp>(&expr).unwrap().into();
            assert!(matches!(res, SigmaBoolean::TrivialProp(false)));
        }
    }

    // Scala `CSigmaDslBuilder.atLeast`: the 255-children cap is checked BEFORE
    // `AtLeast.reduce`, so it precedes (overrides) the degenerate-bound short-circuits.
    // `atLeast(0, 256 props)` errors even though `bound <= 0` would otherwise be TrueProp.
    #[test]
    fn children_count_cap_precedes_degenerate_bound() {
        let too_many = n_sigmaprops_coll(256);
        for bound in [0i32, 2] {
            let expr = atleast_expr(bound, too_many.clone());
            assert!(
                try_eval_out_wo_ctx::<SigmaProp>(&expr).is_err(),
                "256 children must error before the degenerate bound={bound} reduction"
            );
        }
        // Cap is exclusive at 255: 255 children with `bound <= 0` still reduces to TrueProp.
        let at_cap = n_sigmaprops_coll(255);
        let expr = atleast_expr(0, at_cap);
        let res: SigmaBoolean = try_eval_out_wo_ctx::<SigmaProp>(&expr).unwrap().into();
        assert!(matches!(res, SigmaBoolean::TrivialProp(true)));
    }

    // testnet block 184,137: `atLeast(1, Coll[SigmaProp]())` — bound 1 > size 0 => FalseProp.
    #[test]
    fn empty_input_reduces_to_false() {
        let items = Literal::Coll(
            CollKind::from_collection(
                SType::SSigmaProp,
                core::iter::empty::<Literal>().collect::<Arc<[Literal]>>(),
            )
            .unwrap(),
        );
        let expr = atleast_expr(1, items);
        let res: SigmaBoolean = try_eval_out_wo_ctx::<SigmaProp>(&expr).unwrap().into();
        assert!(matches!(res, SigmaBoolean::TrivialProp(false)));
    }

    // Scala `AtLeast.reduce`: `bound <= 0` => TrueProp (handled before the size/255 checks).
    #[test]
    fn nonpositive_bound_reduces_to_true() {
        let items = two_sigmaprops_coll();
        for bound in [0i32, -1] {
            let expr = atleast_expr(bound, items.clone());
            let res: SigmaBoolean = try_eval_out_wo_ctx::<SigmaProp>(&expr).unwrap().into();
            assert!(matches!(res, SigmaBoolean::TrivialProp(true)));
        }
    }
}
