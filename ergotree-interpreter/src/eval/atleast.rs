use core::convert::TryInto;

use alloc::boxed::Box;
use alloc::vec::Vec;
use ergotree_ir::mir::atleast::Atleast;
use ergotree_ir::mir::constant::TryExtractFromError;
use ergotree_ir::mir::constant::TryExtractInto;
use ergotree_ir::mir::value::Value;
use ergotree_ir::sigma_protocol::sigma_boolean::cthreshold::Cthreshold;
use ergotree_ir::sigma_protocol::sigma_boolean::SigmaBoolean;
use ergotree_ir::sigma_protocol::sigma_boolean::SigmaProp;

use crate::eval::env::Env;
use crate::eval::Context;
use crate::eval::EvalError;
use crate::eval::Evaluable;

impl Evaluable for Atleast {
    fn eval<'ctx>(
        &self,
        env: &mut Env<'ctx>,
        ctx: &Context<'ctx>,
    ) -> Result<Value<'ctx>, EvalError> {
        let bound_v = self.bound.eval(env, ctx)?;
        let input_v = self.input.eval(env, ctx)?;

        let normalized_input_vals: Vec<Value> = match input_v {
            Value::Coll(coll) => Ok(coll.as_vec()),
            _ => Err(EvalError::UnexpectedValue(format!(
                "Atleast: expected input to be Value::Coll, got: {0:?}",
                input_v
            ))),
        }?;
        ctx.add_per_item_jit_cost(20, 3, 5, normalized_input_vals.len() as u32)?;

        let bound = bound_v.try_extract_into::<i32>()?;
        let input = normalized_input_vals
            .into_iter()
            .map(|i| {
                i.try_extract_into::<SigmaProp>()
                    .map(|sp| sp.value().clone())
            })
            .collect::<Result<Vec<SigmaBoolean>, TryExtractFromError>>()?;

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
        // Here `1 <= bound <= input.len()`. CTHRESHOLD permits at most 255 children
        // (`input.try_into()` enforces the `BoundedVec<_, 1, 255>` bound), so `bound` fits in u8.
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
