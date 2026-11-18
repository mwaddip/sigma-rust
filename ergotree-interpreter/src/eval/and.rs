use alloc::format;
use ergotree_ir::mir::and::And;
use ergotree_ir::mir::value::debug_start;
use ergotree_ir::mir::value::Value;

use crate::eval::env::Env;
use crate::eval::Context;
use crate::eval::EvalError;
use crate::eval::Evaluable;

impl Evaluable for And {
    fn eval<'ctx>(
        &self,
        env: &mut Env<'ctx>,
        ctx: &Context<'ctx>,
    ) -> Result<Value<'ctx>, EvalError> {
        let input_v = self.input.eval(env, ctx)?;
        // the loop stops at the first `false`, and the cost is of the items it read
        // (`trees.scala:269-283`)
        let (decided, read) = booleans_read_until(&input_v, false)?;
        ctx.add_per_item_jit_cost(10, 5, 32, read)?;
        Ok((!decided).into())
    }
}

/// The items of a collection read as Booleans, one at a time and in order, up to and including the
/// first that is `deciding`: whether one was, and how many items were read. This is the loop of
/// sigmastate's `AND.eval` and `OR.eval` (`while (i < len && <undecided>) { res = res op inputV(i); i += 1 }`):
/// an item is unboxed to a Boolean where it is read, so an item of another kind fails only if the
/// loop reaches it, and an item after the deciding one is never read. The input is cast to a collection
/// first (`input.evalTo[Coll[Boolean]]`: the class only, so a pair is none and an empty collection of
/// any item type is one); nothing of the node's cost is added before the loop ends
pub(crate) fn booleans_read_until(
    input: &Value<'_>,
    deciding: bool,
) -> Result<(bool, u32), EvalError> {
    let coll = match input {
        Value::Coll(coll) => coll,
        _ => {
            return Err(EvalError::UnexpectedValue(format!(
                "expected a collection, found {}",
                debug_start(input)
            )))
        }
    };
    let mut read: u32 = 0;
    while let Some(item) = coll.get_val(read as usize) {
        let Value::Boolean(b) = item else {
            return Err(EvalError::UnexpectedValue(format!(
                "expected a Boolean item, found {}",
                debug_start(&item)
            )));
        };
        read += 1;
        if b == deciding {
            return Ok((true, read));
        }
    }
    Ok((false, read))
}

#[allow(clippy::panic, clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    use crate::eval::test_util::eval_out;
    use alloc::vec::Vec;
    use ergotree_ir::chain::context::Context;

    use super::*;

    use ergotree_ir::mir::expr::Expr;
    use proptest::collection;
    use proptest::prelude::*;
    use sigma_test_util::force_any_val;

    proptest! {

        #[test]
        fn eval(bools in collection::vec(any::<bool>(), 0..10)) {
            let expr: Expr = And {input: Expr::Const(bools.clone().into()).into()}.into();
            let ctx = force_any_val::<Context>();
            let res = eval_out::<bool>(&expr, &ctx);
            prop_assert_eq!(res, bools.iter().all(|b| *b));
        }
    }

    #[test]
    fn and_empty_jit_cost() {
        // santa eval fixture `and_empty` (tree 00960d00): And over an empty
        // Coll[Boolean] constant. Cost = Const(empty coll) 5 + And per-item at
        // n=0 (base 10 + one chunk * per_chunk 5 = 15, post n=0 chunks fix) = 20.
        // Matches the JVM under activated=3 (was 15 before the n=0 fix).
        let expr: Expr = And {
            input: Expr::Const(Vec::<bool>::new().into()).into(),
        }
        .into();
        let ctx = force_any_val::<Context>();
        let before = ctx.jit_cost_value();
        let res = eval_out::<bool>(&expr, &ctx);
        assert!(res); // all() over empty = true
        assert_eq!(ctx.jit_cost_value() - before, 20);
    }

    // An input that is no collection and an item that is no Boolean are named by their start: a
    // value built at evaluation can be large (`debug_start`)
    #[test]
    fn a_refused_value_is_named_by_its_start() {
        use ergotree_ir::mir::coll_value::CollValue;
        use ergotree_ir::types::stype::SType;
        let bytes = || Value::Coll(CollValue::Bytes(alloc::vec![0i8; 1000].try_into().unwrap()));
        let no_collection = booleans_read_until(&Value::pair(bytes(), bytes()), false);
        let items = Value::Coll(CollValue::generic(SType::SAny, alloc::vec![bytes()]).unwrap());
        for refused in [no_collection, booleans_read_until(&items, false)] {
            assert!(
                matches!(&refused, Err(EvalError::UnexpectedValue(text)) if text.chars().count() < 300),
                "the refusal's text is not the value's start"
            );
        }
    }
}
