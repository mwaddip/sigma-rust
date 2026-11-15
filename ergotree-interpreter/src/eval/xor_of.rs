use alloc::format;
use alloc::vec::Vec;
use ergotree_ir::ergo_tree::ErgoTreeVersion;
use ergotree_ir::mir::constant::TryExtractInto;
use ergotree_ir::mir::value::Value;
use ergotree_ir::types::stype::SType;

use crate::eval::env::Env;
use crate::eval::operand::{eval_to, OperandClass};
use crate::eval::Context;
use crate::eval::EvalError;
use crate::eval::Evaluable;
use ergotree_ir::mir::xor_of::XorOf;

impl Evaluable for XorOf {
    fn eval<'ctx>(
        &self,
        env: &mut Env<'ctx>,
        ctx: &Context<'ctx>,
    ) -> Result<Value<'ctx>, EvalError> {
        let input_v = eval_to(&self.input, OperandClass::Coll, env, ctx)?;
        // the cost is over the collection's length, whatever its items; the booleans are the body's
        // (`trees.scala:238-243`)
        // sigmastate's `xorOf` (`sigma/data/CSigmaDslBuilder.scala:116-132`) takes its 4.x reading
        // while `VersionContext.isJitActivated` is false: below an ACTIVATED version of 2,
        // whatever the tree's version, it is `conditions.toArray.distinct.length == 2`, true when
        // the collection holds both a `true` and a `false`. From the activation on it is the
        // exclusive or of the items, also for a tree of version 0 or 1
        let below_the_activation = ctx.activated_script_version() < ErgoTreeVersion::V2;
        if let Value::Coll(coll) = &input_v {
            ctx.add_per_item_jit_cost(20, 5, 32, coll.len() as u32)?;
            // the 4.x reading takes the collection's ARRAY, which is cast to `boolean[]`
            // (`:130`): a collection of another item type is refused whatever it holds, an
            // empty one and a tuple of booleans included. The reading from the activation on
            // takes the items one by one and casts each
            if below_the_activation && coll.elem_tpe() != SType::SBoolean {
                return Err(EvalError::UnexpectedValue(format!(
                    "xorOf: the array of a collection of {:?} is not an array of Boolean",
                    coll.elem_tpe()
                )));
            }
        }
        let input_v_bools = input_v.try_extract_into::<Vec<bool>>()?;
        if below_the_activation {
            let mut has_true = false;
            let mut has_false = false;
            for b in input_v_bools {
                if b {
                    has_true = true;
                } else {
                    has_false = true;
                }
                if has_true && has_false {
                    break;
                }
            }
            return Ok((has_true && has_false).into());
        }
        Ok(input_v_bools.into_iter().fold(false, |a, b| a ^ b).into())
    }
}

#[allow(clippy::panic)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::test_util::eval_out;
    use core::cell::Cell;
    use ergotree_ir::chain::context::Context;
    use ergotree_ir::mir::expr::Expr;
    use proptest::collection;
    use proptest::prelude::*;
    use sigma_test_util::force_any_val;

    fn ctx_at(activated: i8, tree: ErgoTreeVersion) -> Context<'static> {
        let ctx = force_any_val::<Context>();
        Context {
            activated_script_version_byte: activated,
            tree_version: Cell::new(tree),
            ..ctx
        }
    }

    proptest! {

        #[test]
        fn eval_from_the_activation_left_fold(bools in collection::vec(any::<bool>(), 0..=10)) {
            // from the activation on: the exclusive or (true iff an odd number of trues), also
            // for a tree of version 0
            let expr: Expr = XorOf {input: Expr::Const(bools.clone().into()).into()}.into();
            let ctx = ctx_at(3, ErgoTreeVersion::V0);
            let res = eval_out::<bool>(&expr, &ctx);
            let expected = bools.into_iter().filter(|x| *x).count() & 1 == 1;
            prop_assert_eq!(res, expected);
        }

        #[test]
        fn eval_below_the_activation_distinct_length(bools in collection::vec(any::<bool>(), 0..=10)) {
            // below the activation: the 4.x reading, true iff the collection holds BOTH true
            // and false (count and order independent)
            let expr: Expr = XorOf {input: Expr::Const(bools.clone().into()).into()}.into();
            let ctx = ctx_at(1, ErgoTreeVersion::V0);
            let res = eval_out::<bool>(&expr, &ctx);
            let has_true = bools.iter().any(|x| *x);
            let has_false = bools.iter().any(|x| !*x);
            prop_assert_eq!(res, has_true && has_false);
        }
    }

    /// `[true, true, false]` is true under the 4.x reading (both values present) and false as
    /// an exclusive or (an even number of trues): the activated version decides, not the tree's
    #[test]
    fn eval_below_the_activation_two_trues_one_false_is_true() {
        let bools = vec![true, true, false];
        let expr: Expr = XorOf {
            input: Expr::Const(bools.into()).into(),
        }
        .into();
        let ctx = ctx_at(1, ErgoTreeVersion::V1);
        assert!(eval_out::<bool>(&expr, &ctx));
    }

    #[test]
    fn eval_from_the_activation_two_trues_one_false_is_false() {
        let bools = vec![true, true, false];
        let expr: Expr = XorOf {
            input: Expr::Const(bools.into()).into(),
        }
        .into();
        let ctx = ctx_at(2, ErgoTreeVersion::V0);
        assert!(!eval_out::<bool>(&expr, &ctx));
    }
}

#[cfg(test)]
#[cfg(feature = "arbitrary")]
#[allow(clippy::unwrap_used)]
mod activation_rows {
    use crate::eval::test_util::{eval_body_at, shown};
    use ergotree_ir::ergo_tree::ErgoTreeVersion;

    /// A row: its name in the round's row set, the expression's bytes, the tree version, the activated
    /// version, the value as `shown` prints it or `ERR`, and the cost counter
    type Row = (&'static str, &'static str, u8, i8, &'static str, u64);

    /// `xorOf` on sigma-state 6.0.7 at every pair of activated and tree version sigmastate can hold: below an
    /// activated version of 2 it is "the collection holds both values", from 2 on the exclusive or, whatever
    /// the tree's version (the round's `measure/jvm-act*.txt`, rows `gate|XorOf|*`)
    #[rustfmt::skip]
    const ROWS: &[Row] = &[
        ("gate|XorOf|tft|v0", "ff0d0305", 0, 0, "true", 30),
        ("arr|XorOf|lie-empty-CollInt|v0", "ff9501000d03051000", 0, 0, "ERR", 45),
        ("arr|XorOf|lie-empty-CollInt|v1", "ff9501000d03051000", 1, 0, "ERR", 45),
        ("arr|XorOf|lie-empty-CollInt|v2", "ff9501000d03051000", 2, 0, "ERR", 45),
        ("arr|XorOf|lie-empty-CollInt|v3", "ff9501000d03051000", 3, 0, "ERR", 45),
        ("arr|XorOf|lie-empty-CollByte|v0", "ff9501000d03050e00", 0, 0, "ERR", 45),
        ("arr|XorOf|lie-empty-CollByte|v1", "ff9501000d03050e00", 1, 0, "ERR", 45),
        ("arr|XorOf|lie-empty-CollByte|v2", "ff9501000d03050e00", 2, 0, "ERR", 45),
        ("arr|XorOf|lie-empty-CollByte|v3", "ff9501000d03050e00", 3, 0, "ERR", 45),
        ("arr|XorOf|lie-empty-CollLong|v0", "ff9501000d03051100", 0, 0, "ERR", 45),
        ("arr|XorOf|lie-empty-CollLong|v1", "ff9501000d03051100", 1, 0, "ERR", 45),
        ("arr|XorOf|lie-empty-CollLong|v2", "ff9501000d03051100", 2, 0, "ERR", 45),
        ("arr|XorOf|lie-empty-CollLong|v3", "ff9501000d03051100", 3, 0, "ERR", 45),
        ("arr|XorOf|empty-CollBool|v0", "ff0d00", 0, 0, "false", 30),
        ("arr|XorOf|empty-CollBool|v1", "ff0d00", 1, 0, "false", 30),
        ("arr|XorOf|empty-CollBool|v2", "ff0d00", 2, 0, "false", 30),
        ("arr|XorOf|empty-CollBool|v3", "ff0d00", 3, 0, "false", 30),
        ("arr|XorOf|Triple-tft|v0", "ff48010101010001", 0, 0, "ERR", 30),
        ("arr|XorOf|Triple-tft|v1", "ff48010101010001", 1, 0, "ERR", 30),
        ("arr|XorOf|Triple-tft|v2", "ff48010101010001", 2, 0, "ERR", 30),
        ("arr|XorOf|Triple-tft|v3", "ff48010101010001", 3, 0, "ERR", 30),
        ("arr|XorOf|Triple-ttt|v0", "ff48010101010101", 0, 0, "ERR", 30),
        ("arr|XorOf|Triple-ttt|v1", "ff48010101010101", 1, 0, "ERR", 30),
        ("arr|XorOf|Triple-ttt|v2", "ff48010101010101", 2, 0, "ERR", 30),
        ("arr|XorOf|Triple-ttt|v3", "ff48010101010101", 3, 0, "ERR", 30),
        ("arr|XorOf|Triple-tff|v0", "ff48010101010000", 0, 0, "ERR", 30),
        ("arr|XorOf|Triple-tff|v1", "ff48010101010000", 1, 0, "ERR", 30),
        ("arr|XorOf|Triple-tff|v2", "ff48010101010000", 2, 0, "ERR", 30),
        ("arr|XorOf|Triple-tff|v3", "ff48010101010000", 3, 0, "ERR", 30),
        ("arr|XorOf|Triple-fff|v0", "ff48010101000000", 0, 0, "ERR", 30),
        ("arr|XorOf|Triple-fff|v1", "ff48010101000000", 1, 0, "ERR", 30),
        ("arr|XorOf|Triple-fff|v2", "ff48010101000000", 2, 0, "ERR", 30),
        ("arr|XorOf|Triple-fff|v3", "ff48010101000000", 3, 0, "ERR", 30),
        ("arr|XorOf|Triple-tfInt|v0", "ff48010104010002", 0, 0, "ERR", 30),
        ("arr|XorOf|Triple-tfInt|v1", "ff48010104010002", 1, 0, "ERR", 30),
        ("arr|XorOf|Triple-tfInt|v2", "ff48010104010002", 2, 0, "ERR", 30),
        ("arr|XorOf|Triple-tfInt|v3", "ff48010104010002", 3, 0, "ERR", 30),
        ("arr|XorOf|lie-CollInt|v0", "ff9501000d03051003020406", 0, 0, "ERR", 45),
        ("arr|XorOf|lie-CollInt|v1", "ff9501000d03051003020406", 1, 0, "ERR", 45),
        ("arr|XorOf|lie-CollInt|v2", "ff9501000d03051003020406", 2, 0, "ERR", 45),
        ("arr|XorOf|lie-CollInt|v3", "ff9501000d03051003020406", 3, 0, "ERR", 45),
        ("arr|XorOf|Coll-1-true|v0", "ff0d0101", 0, 0, "false", 30),
        ("arr|XorOf|Coll-1-true|v1", "ff0d0101", 1, 0, "false", 30),
        ("arr|XorOf|Coll-1-true|v2", "ff0d0101", 2, 0, "false", 30),
        ("arr|XorOf|Coll-1-true|v3", "ff0d0101", 3, 0, "false", 30),
        ("arr|XorOf|Coll-1-false|v0", "ff0d0100", 0, 0, "false", 30),
        ("arr|XorOf|Coll-1-false|v1", "ff0d0100", 1, 0, "false", 30),
        ("arr|XorOf|Coll-1-false|v2", "ff0d0100", 2, 0, "false", 30),
        ("arr|XorOf|Coll-1-false|v3", "ff0d0100", 3, 0, "false", 30),
        ("arr|XorOf|lie-empty-CollInt|v0", "ff9501000d03051000", 0, 1, "ERR", 45),
        ("arr|XorOf|lie-empty-CollInt|v1", "ff9501000d03051000", 1, 1, "ERR", 45),
        ("arr|XorOf|lie-empty-CollInt|v2", "ff9501000d03051000", 2, 1, "ERR", 45),
        ("arr|XorOf|lie-empty-CollInt|v3", "ff9501000d03051000", 3, 1, "ERR", 45),
        ("arr|XorOf|lie-empty-CollByte|v0", "ff9501000d03050e00", 0, 1, "ERR", 45),
        ("arr|XorOf|lie-empty-CollByte|v1", "ff9501000d03050e00", 1, 1, "ERR", 45),
        ("arr|XorOf|lie-empty-CollByte|v2", "ff9501000d03050e00", 2, 1, "ERR", 45),
        ("arr|XorOf|lie-empty-CollByte|v3", "ff9501000d03050e00", 3, 1, "ERR", 45),
        ("arr|XorOf|lie-empty-CollLong|v0", "ff9501000d03051100", 0, 1, "ERR", 45),
        ("arr|XorOf|lie-empty-CollLong|v1", "ff9501000d03051100", 1, 1, "ERR", 45),
        ("arr|XorOf|lie-empty-CollLong|v2", "ff9501000d03051100", 2, 1, "ERR", 45),
        ("arr|XorOf|lie-empty-CollLong|v3", "ff9501000d03051100", 3, 1, "ERR", 45),
        ("arr|XorOf|empty-CollBool|v0", "ff0d00", 0, 1, "false", 30),
        ("arr|XorOf|empty-CollBool|v1", "ff0d00", 1, 1, "false", 30),
        ("arr|XorOf|empty-CollBool|v2", "ff0d00", 2, 1, "false", 30),
        ("arr|XorOf|empty-CollBool|v3", "ff0d00", 3, 1, "false", 30),
        ("arr|XorOf|Triple-tft|v0", "ff48010101010001", 0, 1, "ERR", 30),
        ("arr|XorOf|Triple-tft|v1", "ff48010101010001", 1, 1, "ERR", 30),
        ("arr|XorOf|Triple-tft|v2", "ff48010101010001", 2, 1, "ERR", 30),
        ("arr|XorOf|Triple-tft|v3", "ff48010101010001", 3, 1, "ERR", 30),
        ("arr|XorOf|Triple-ttt|v0", "ff48010101010101", 0, 1, "ERR", 30),
        ("arr|XorOf|Triple-ttt|v1", "ff48010101010101", 1, 1, "ERR", 30),
        ("arr|XorOf|Triple-ttt|v2", "ff48010101010101", 2, 1, "ERR", 30),
        ("arr|XorOf|Triple-ttt|v3", "ff48010101010101", 3, 1, "ERR", 30),
        ("arr|XorOf|Triple-tff|v0", "ff48010101010000", 0, 1, "ERR", 30),
        ("arr|XorOf|Triple-tff|v1", "ff48010101010000", 1, 1, "ERR", 30),
        ("arr|XorOf|Triple-tff|v2", "ff48010101010000", 2, 1, "ERR", 30),
        ("arr|XorOf|Triple-tff|v3", "ff48010101010000", 3, 1, "ERR", 30),
        ("arr|XorOf|Triple-fff|v0", "ff48010101000000", 0, 1, "ERR", 30),
        ("arr|XorOf|Triple-fff|v1", "ff48010101000000", 1, 1, "ERR", 30),
        ("arr|XorOf|Triple-fff|v2", "ff48010101000000", 2, 1, "ERR", 30),
        ("arr|XorOf|Triple-fff|v3", "ff48010101000000", 3, 1, "ERR", 30),
        ("arr|XorOf|Triple-tfInt|v0", "ff48010104010002", 0, 1, "ERR", 30),
        ("arr|XorOf|Triple-tfInt|v1", "ff48010104010002", 1, 1, "ERR", 30),
        ("arr|XorOf|Triple-tfInt|v2", "ff48010104010002", 2, 1, "ERR", 30),
        ("arr|XorOf|Triple-tfInt|v3", "ff48010104010002", 3, 1, "ERR", 30),
        ("arr|XorOf|lie-CollInt|v0", "ff9501000d03051003020406", 0, 1, "ERR", 45),
        ("arr|XorOf|lie-CollInt|v1", "ff9501000d03051003020406", 1, 1, "ERR", 45),
        ("arr|XorOf|lie-CollInt|v2", "ff9501000d03051003020406", 2, 1, "ERR", 45),
        ("arr|XorOf|lie-CollInt|v3", "ff9501000d03051003020406", 3, 1, "ERR", 45),
        ("arr|XorOf|Coll-1-true|v0", "ff0d0101", 0, 1, "false", 30),
        ("arr|XorOf|Coll-1-true|v1", "ff0d0101", 1, 1, "false", 30),
        ("arr|XorOf|Coll-1-true|v2", "ff0d0101", 2, 1, "false", 30),
        ("arr|XorOf|Coll-1-true|v3", "ff0d0101", 3, 1, "false", 30),
        ("arr|XorOf|Coll-1-false|v0", "ff0d0100", 0, 1, "false", 30),
        ("arr|XorOf|Coll-1-false|v1", "ff0d0100", 1, 1, "false", 30),
        ("arr|XorOf|Coll-1-false|v2", "ff0d0100", 2, 1, "false", 30),
        ("arr|XorOf|Coll-1-false|v3", "ff0d0100", 3, 1, "false", 30),
        ("arr|XorOf|lie-empty-CollInt|v0", "ff9501000d03051000", 0, 2, "false", 45),
        ("arr|XorOf|lie-empty-CollInt|v1", "ff9501000d03051000", 1, 2, "false", 45),
        ("arr|XorOf|lie-empty-CollInt|v2", "ff9501000d03051000", 2, 2, "false", 45),
        ("arr|XorOf|lie-empty-CollByte|v0", "ff9501000d03050e00", 0, 2, "false", 45),
        ("arr|XorOf|lie-empty-CollByte|v1", "ff9501000d03050e00", 1, 2, "false", 45),
        ("arr|XorOf|lie-empty-CollByte|v2", "ff9501000d03050e00", 2, 2, "false", 45),
        ("arr|XorOf|lie-empty-CollLong|v0", "ff9501000d03051100", 0, 2, "false", 45),
        ("arr|XorOf|lie-empty-CollLong|v1", "ff9501000d03051100", 1, 2, "false", 45),
        ("arr|XorOf|lie-empty-CollLong|v2", "ff9501000d03051100", 2, 2, "false", 45),
        ("arr|XorOf|empty-CollBool|v0", "ff0d00", 0, 2, "false", 30),
        ("arr|XorOf|empty-CollBool|v1", "ff0d00", 1, 2, "false", 30),
        ("arr|XorOf|empty-CollBool|v2", "ff0d00", 2, 2, "false", 30),
        ("arr|XorOf|Triple-tft|v0", "ff48010101010001", 0, 2, "false", 30),
        ("arr|XorOf|Triple-tft|v1", "ff48010101010001", 1, 2, "false", 30),
        ("arr|XorOf|Triple-tft|v2", "ff48010101010001", 2, 2, "false", 30),
        ("arr|XorOf|Triple-ttt|v0", "ff48010101010101", 0, 2, "true", 30),
        ("arr|XorOf|Triple-ttt|v1", "ff48010101010101", 1, 2, "true", 30),
        ("arr|XorOf|Triple-ttt|v2", "ff48010101010101", 2, 2, "true", 30),
        ("arr|XorOf|Triple-tff|v0", "ff48010101010000", 0, 2, "true", 30),
        ("arr|XorOf|Triple-tff|v1", "ff48010101010000", 1, 2, "true", 30),
        ("arr|XorOf|Triple-tff|v2", "ff48010101010000", 2, 2, "true", 30),
        ("arr|XorOf|Triple-fff|v0", "ff48010101000000", 0, 2, "false", 30),
        ("arr|XorOf|Triple-fff|v1", "ff48010101000000", 1, 2, "false", 30),
        ("arr|XorOf|Triple-fff|v2", "ff48010101000000", 2, 2, "false", 30),
        ("arr|XorOf|Triple-tfInt|v0", "ff48010104010002", 0, 2, "ERR", 30),
        ("arr|XorOf|Triple-tfInt|v1", "ff48010104010002", 1, 2, "ERR", 30),
        ("arr|XorOf|Triple-tfInt|v2", "ff48010104010002", 2, 2, "ERR", 30),
        ("arr|XorOf|lie-CollInt|v0", "ff9501000d03051003020406", 0, 2, "ERR", 45),
        ("arr|XorOf|lie-CollInt|v1", "ff9501000d03051003020406", 1, 2, "ERR", 45),
        ("arr|XorOf|lie-CollInt|v2", "ff9501000d03051003020406", 2, 2, "ERR", 45),
        ("arr|XorOf|Coll-1-true|v0", "ff0d0101", 0, 2, "true", 30),
        ("arr|XorOf|Coll-1-true|v1", "ff0d0101", 1, 2, "true", 30),
        ("arr|XorOf|Coll-1-true|v2", "ff0d0101", 2, 2, "true", 30),
        ("arr|XorOf|Coll-1-false|v0", "ff0d0100", 0, 2, "false", 30),
        ("arr|XorOf|Coll-1-false|v1", "ff0d0100", 1, 2, "false", 30),
        ("arr|XorOf|Coll-1-false|v2", "ff0d0100", 2, 2, "false", 30),
        ("arr|XorOf|lie-empty-CollInt|v0", "ff9501000d03051000", 0, 3, "false", 45),
        ("arr|XorOf|lie-empty-CollInt|v1", "ff9501000d03051000", 1, 3, "false", 45),
        ("arr|XorOf|lie-empty-CollInt|v2", "ff9501000d03051000", 2, 3, "false", 45),
        ("arr|XorOf|lie-empty-CollInt|v3", "ff9501000d03051000", 3, 3, "false", 45),
        ("arr|XorOf|lie-empty-CollByte|v0", "ff9501000d03050e00", 0, 3, "false", 45),
        ("arr|XorOf|lie-empty-CollByte|v1", "ff9501000d03050e00", 1, 3, "false", 45),
        ("arr|XorOf|lie-empty-CollByte|v2", "ff9501000d03050e00", 2, 3, "false", 45),
        ("arr|XorOf|lie-empty-CollByte|v3", "ff9501000d03050e00", 3, 3, "false", 45),
        ("arr|XorOf|lie-empty-CollLong|v0", "ff9501000d03051100", 0, 3, "false", 45),
        ("arr|XorOf|lie-empty-CollLong|v1", "ff9501000d03051100", 1, 3, "false", 45),
        ("arr|XorOf|lie-empty-CollLong|v2", "ff9501000d03051100", 2, 3, "false", 45),
        ("arr|XorOf|lie-empty-CollLong|v3", "ff9501000d03051100", 3, 3, "false", 45),
        ("arr|XorOf|empty-CollBool|v0", "ff0d00", 0, 3, "false", 30),
        ("arr|XorOf|empty-CollBool|v1", "ff0d00", 1, 3, "false", 30),
        ("arr|XorOf|empty-CollBool|v2", "ff0d00", 2, 3, "false", 30),
        ("arr|XorOf|empty-CollBool|v3", "ff0d00", 3, 3, "false", 30),
        ("arr|XorOf|Triple-tft|v0", "ff48010101010001", 0, 3, "false", 30),
        ("arr|XorOf|Triple-tft|v1", "ff48010101010001", 1, 3, "false", 30),
        ("arr|XorOf|Triple-tft|v2", "ff48010101010001", 2, 3, "false", 30),
        ("arr|XorOf|Triple-tft|v3", "ff48010101010001", 3, 3, "false", 30),
        ("arr|XorOf|Triple-ttt|v0", "ff48010101010101", 0, 3, "true", 30),
        ("arr|XorOf|Triple-ttt|v1", "ff48010101010101", 1, 3, "true", 30),
        ("arr|XorOf|Triple-ttt|v2", "ff48010101010101", 2, 3, "true", 30),
        ("arr|XorOf|Triple-ttt|v3", "ff48010101010101", 3, 3, "true", 30),
        ("arr|XorOf|Triple-tff|v0", "ff48010101010000", 0, 3, "true", 30),
        ("arr|XorOf|Triple-tff|v1", "ff48010101010000", 1, 3, "true", 30),
        ("arr|XorOf|Triple-tff|v2", "ff48010101010000", 2, 3, "true", 30),
        ("arr|XorOf|Triple-tff|v3", "ff48010101010000", 3, 3, "true", 30),
        ("arr|XorOf|Triple-fff|v0", "ff48010101000000", 0, 3, "false", 30),
        ("arr|XorOf|Triple-fff|v1", "ff48010101000000", 1, 3, "false", 30),
        ("arr|XorOf|Triple-fff|v2", "ff48010101000000", 2, 3, "false", 30),
        ("arr|XorOf|Triple-fff|v3", "ff48010101000000", 3, 3, "false", 30),
        ("arr|XorOf|Triple-tfInt|v0", "ff48010104010002", 0, 3, "ERR", 30),
        ("arr|XorOf|Triple-tfInt|v1", "ff48010104010002", 1, 3, "ERR", 30),
        ("arr|XorOf|Triple-tfInt|v2", "ff48010104010002", 2, 3, "ERR", 30),
        ("arr|XorOf|Triple-tfInt|v3", "ff48010104010002", 3, 3, "ERR", 30),
        ("arr|XorOf|lie-CollInt|v0", "ff9501000d03051003020406", 0, 3, "ERR", 45),
        ("arr|XorOf|lie-CollInt|v1", "ff9501000d03051003020406", 1, 3, "ERR", 45),
        ("arr|XorOf|lie-CollInt|v2", "ff9501000d03051003020406", 2, 3, "ERR", 45),
        ("arr|XorOf|lie-CollInt|v3", "ff9501000d03051003020406", 3, 3, "ERR", 45),
        ("arr|XorOf|Coll-1-true|v0", "ff0d0101", 0, 3, "true", 30),
        ("arr|XorOf|Coll-1-true|v1", "ff0d0101", 1, 3, "true", 30),
        ("arr|XorOf|Coll-1-true|v2", "ff0d0101", 2, 3, "true", 30),
        ("arr|XorOf|Coll-1-true|v3", "ff0d0101", 3, 3, "true", 30),
        ("arr|XorOf|Coll-1-false|v0", "ff0d0100", 0, 3, "false", 30),
        ("arr|XorOf|Coll-1-false|v1", "ff0d0100", 1, 3, "false", 30),
        ("arr|XorOf|Coll-1-false|v2", "ff0d0100", 2, 3, "false", 30),
        ("arr|XorOf|Coll-1-false|v3", "ff0d0100", 3, 3, "false", 30),
        ("gate|XorOf|tft|v1", "ff0d0305", 1, 0, "true", 30),
        ("gate|XorOf|tt|v0", "ff0d0203", 0, 0, "false", 30),
        ("gate|XorOf|tt|v1", "ff0d0203", 1, 0, "false", 30),
        ("gate|XorOf|tf|v0", "ff0d0201", 0, 0, "true", 30),
        ("gate|XorOf|tf|v1", "ff0d0201", 1, 0, "true", 30),
        ("gate|XorOf|ttf|v0", "ff0d0303", 0, 0, "true", 30),
        ("gate|XorOf|ttf|v1", "ff0d0303", 1, 0, "true", 30),
        ("gate|XorOf|ttt|v0", "ff0d0307", 0, 0, "false", 30),
        ("gate|XorOf|ttt|v1", "ff0d0307", 1, 0, "false", 30),
        ("gate|XorOf|ff|v0", "ff0d0200", 0, 0, "false", 30),
        ("gate|XorOf|ff|v1", "ff0d0200", 1, 0, "false", 30),
        ("gate|XorOf|t|v0", "ff0d0101", 0, 0, "false", 30),
        ("gate|XorOf|t|v1", "ff0d0101", 1, 0, "false", 30),
        ("gate|XorOf|f|v0", "ff0d0100", 0, 0, "false", 30),
        ("gate|XorOf|f|v1", "ff0d0100", 1, 0, "false", 30),
        ("gate|XorOf|empty|v0", "ff0d00", 0, 0, "false", 30),
        ("gate|XorOf|empty|v1", "ff0d00", 1, 0, "false", 30),
        ("gate|XorOf|tft|v0", "ff0d0305", 0, 1, "true", 30),
        ("gate|XorOf|tft|v1", "ff0d0305", 1, 1, "true", 30),
        ("gate|XorOf|tt|v0", "ff0d0203", 0, 1, "false", 30),
        ("gate|XorOf|tt|v1", "ff0d0203", 1, 1, "false", 30),
        ("gate|XorOf|tf|v0", "ff0d0201", 0, 1, "true", 30),
        ("gate|XorOf|tf|v1", "ff0d0201", 1, 1, "true", 30),
        ("gate|XorOf|ttf|v0", "ff0d0303", 0, 1, "true", 30),
        ("gate|XorOf|ttf|v1", "ff0d0303", 1, 1, "true", 30),
        ("gate|XorOf|ttt|v0", "ff0d0307", 0, 1, "false", 30),
        ("gate|XorOf|ttt|v1", "ff0d0307", 1, 1, "false", 30),
        ("gate|XorOf|ff|v0", "ff0d0200", 0, 1, "false", 30),
        ("gate|XorOf|ff|v1", "ff0d0200", 1, 1, "false", 30),
        ("gate|XorOf|t|v0", "ff0d0101", 0, 1, "false", 30),
        ("gate|XorOf|t|v1", "ff0d0101", 1, 1, "false", 30),
        ("gate|XorOf|f|v0", "ff0d0100", 0, 1, "false", 30),
        ("gate|XorOf|f|v1", "ff0d0100", 1, 1, "false", 30),
        ("gate|XorOf|empty|v0", "ff0d00", 0, 1, "false", 30),
        ("gate|XorOf|empty|v1", "ff0d00", 1, 1, "false", 30),
        ("gate|XorOf|tft|v0", "ff0d0305", 0, 2, "false", 30),
        ("gate|XorOf|tft|v1", "ff0d0305", 1, 2, "false", 30),
        ("gate|XorOf|tft|v2", "ff0d0305", 2, 2, "false", 30),
        ("gate|XorOf|tt|v0", "ff0d0203", 0, 2, "false", 30),
        ("gate|XorOf|tt|v1", "ff0d0203", 1, 2, "false", 30),
        ("gate|XorOf|tt|v2", "ff0d0203", 2, 2, "false", 30),
        ("gate|XorOf|tf|v0", "ff0d0201", 0, 2, "true", 30),
        ("gate|XorOf|tf|v1", "ff0d0201", 1, 2, "true", 30),
        ("gate|XorOf|tf|v2", "ff0d0201", 2, 2, "true", 30),
        ("gate|XorOf|ttf|v0", "ff0d0303", 0, 2, "false", 30),
        ("gate|XorOf|ttf|v1", "ff0d0303", 1, 2, "false", 30),
        ("gate|XorOf|ttf|v2", "ff0d0303", 2, 2, "false", 30),
        ("gate|XorOf|ttt|v0", "ff0d0307", 0, 2, "true", 30),
        ("gate|XorOf|ttt|v1", "ff0d0307", 1, 2, "true", 30),
        ("gate|XorOf|ttt|v2", "ff0d0307", 2, 2, "true", 30),
        ("gate|XorOf|ff|v0", "ff0d0200", 0, 2, "false", 30),
        ("gate|XorOf|ff|v1", "ff0d0200", 1, 2, "false", 30),
        ("gate|XorOf|ff|v2", "ff0d0200", 2, 2, "false", 30),
        ("gate|XorOf|t|v0", "ff0d0101", 0, 2, "true", 30),
        ("gate|XorOf|t|v1", "ff0d0101", 1, 2, "true", 30),
        ("gate|XorOf|t|v2", "ff0d0101", 2, 2, "true", 30),
        ("gate|XorOf|f|v0", "ff0d0100", 0, 2, "false", 30),
        ("gate|XorOf|f|v1", "ff0d0100", 1, 2, "false", 30),
        ("gate|XorOf|f|v2", "ff0d0100", 2, 2, "false", 30),
        ("gate|XorOf|empty|v0", "ff0d00", 0, 2, "false", 30),
        ("gate|XorOf|empty|v1", "ff0d00", 1, 2, "false", 30),
        ("gate|XorOf|empty|v2", "ff0d00", 2, 2, "false", 30),
        ("gate|XorOf|tft|v0", "ff0d0305", 0, 3, "false", 30),
        ("gate|XorOf|tft|v1", "ff0d0305", 1, 3, "false", 30),
        ("gate|XorOf|tft|v2", "ff0d0305", 2, 3, "false", 30),
        ("gate|XorOf|tft|v3", "ff0d0305", 3, 3, "false", 30),
        ("gate|XorOf|tt|v0", "ff0d0203", 0, 3, "false", 30),
        ("gate|XorOf|tt|v1", "ff0d0203", 1, 3, "false", 30),
        ("gate|XorOf|tt|v2", "ff0d0203", 2, 3, "false", 30),
        ("gate|XorOf|tt|v3", "ff0d0203", 3, 3, "false", 30),
        ("gate|XorOf|tf|v0", "ff0d0201", 0, 3, "true", 30),
        ("gate|XorOf|tf|v1", "ff0d0201", 1, 3, "true", 30),
        ("gate|XorOf|tf|v2", "ff0d0201", 2, 3, "true", 30),
        ("gate|XorOf|tf|v3", "ff0d0201", 3, 3, "true", 30),
        ("gate|XorOf|ttf|v0", "ff0d0303", 0, 3, "false", 30),
        ("gate|XorOf|ttf|v1", "ff0d0303", 1, 3, "false", 30),
        ("gate|XorOf|ttf|v2", "ff0d0303", 2, 3, "false", 30),
        ("gate|XorOf|ttf|v3", "ff0d0303", 3, 3, "false", 30),
        ("gate|XorOf|ttt|v0", "ff0d0307", 0, 3, "true", 30),
        ("gate|XorOf|ttt|v1", "ff0d0307", 1, 3, "true", 30),
        ("gate|XorOf|ttt|v2", "ff0d0307", 2, 3, "true", 30),
        ("gate|XorOf|ttt|v3", "ff0d0307", 3, 3, "true", 30),
        ("gate|XorOf|ff|v0", "ff0d0200", 0, 3, "false", 30),
        ("gate|XorOf|ff|v1", "ff0d0200", 1, 3, "false", 30),
        ("gate|XorOf|ff|v2", "ff0d0200", 2, 3, "false", 30),
        ("gate|XorOf|ff|v3", "ff0d0200", 3, 3, "false", 30),
        ("gate|XorOf|t|v0", "ff0d0101", 0, 3, "true", 30),
        ("gate|XorOf|t|v1", "ff0d0101", 1, 3, "true", 30),
        ("gate|XorOf|t|v2", "ff0d0101", 2, 3, "true", 30),
        ("gate|XorOf|t|v3", "ff0d0101", 3, 3, "true", 30),
        ("gate|XorOf|f|v0", "ff0d0100", 0, 3, "false", 30),
        ("gate|XorOf|f|v1", "ff0d0100", 1, 3, "false", 30),
        ("gate|XorOf|f|v2", "ff0d0100", 2, 3, "false", 30),
        ("gate|XorOf|f|v3", "ff0d0100", 3, 3, "false", 30),
        ("gate|XorOf|empty|v0", "ff0d00", 0, 3, "false", 30),
        ("gate|XorOf|empty|v1", "ff0d00", 1, 3, "false", 30),
        ("gate|XorOf|empty|v2", "ff0d00", 2, 3, "false", 30),
        ("gate|XorOf|empty|v3", "ff0d00", 3, 3, "false", 30),
    ];

    #[test]
    fn each_row_is_sigmastates() {
        let mut wrong = alloc::vec::Vec::new();
        for (name, body, version, activated, expected, cost) in ROWS {
            let (res, stopped_at) = eval_body_at(body, ErgoTreeVersion::from(*version), *activated);
            let got = match &res {
                Ok(v) => shown(v),
                Err(_) => "ERR".into(),
            };
            if got != *expected || stopped_at != *cost {
                wrong.push(alloc::format!(
                    "{name} at activated {activated}: expected {expected} at {cost}, got {got} at {stopped_at} ({res:?})"
                ));
            }
        }
        assert!(
            wrong.is_empty(),
            "{} of {} rows:\n{}",
            wrong.len(),
            ROWS.len(),
            wrong.join("\n")
        );
    }
}
