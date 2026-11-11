use alloc::sync::Arc;

use ergotree_ir::mir::coll_value::CollValue;
use ergotree_ir::mir::value::Value;
use ergotree_ir::mir::xor::Xor;

use crate::eval::env::Env;
use crate::eval::Context;
use crate::eval::EvalError;
use crate::eval::Evaluable;

/// sigmastate's `Colls.xor`: `left.zip(right).map { case (l, r) => (l ^ r).toByte }` (v6.0.6
/// `CollsOverArrays.scala:261-262`). It reads both operands through the `Coll` interface, so any
/// collection passes, a tuple of bytes (a `Coll[Any]`) included, where the hashes need a
/// `byte[]`; `zip` cuts both to the shorter length, and each paired item is unboxed as a `Byte`,
/// so another item among those is an error (SANTA `Tuple.non_pair_value_ops` #36-#39)
fn xor_colls(left: &CollValue<'_>, right: &CollValue<'_>) -> Result<Arc<[i8]>, EvalError> {
    let byte = |coll: &CollValue<'_>, i: usize| match coll.get_val(i) {
        Some(Value::Byte(b)) => Ok(b),
        other => Err(EvalError::UnexpectedValue(format!(
            "Xor: expected a Byte item, got: {other:?}"
        ))),
    };
    (0..left.len().min(right.len()))
        .map(|i| Ok(byte(left, i)? ^ byte(right, i)?))
        .collect()
}

/// sigmastate's `Xor.xorWithCosting` (v6.0.6 `trees.scala:1021-1025`), which the `Xor` node and
/// the `Global.xor` method share (`xorMethod`'s cost kind is `Xor.costKind`,
/// `methods.scala:1876-1877`, and `xor_eval` calls it, `:1917-1920`): the per-item cost
/// `PerItemCost(10, 2, 128)` on the SHORTER operand's length, then [`xor_colls`]. SANTA
/// `Xor.per_item_cost_base`: 129 bytes against 1 costs what 1 against 129 does
pub(crate) fn xor_with_costing<'ctx>(
    ctx: &Context<'ctx>,
    left: &Value<'ctx>,
    right: &Value<'ctx>,
) -> Result<Value<'ctx>, EvalError> {
    let (Value::Coll(left), Value::Coll(right)) = (left, right) else {
        return Err(EvalError::UnexpectedValue(format!(
            "Xor: expected two collections, got: {left:?} and {right:?}"
        )));
    };
    ctx.add_per_item_jit_cost(10, 2, 128, left.len().min(right.len()) as u32)?;
    Ok(CollValue::Bytes(xor_colls(left, right)?).into())
}

impl Evaluable for Xor {
    fn eval<'ctx>(
        &self,
        env: &mut Env<'ctx>,
        ctx: &Context<'ctx>,
    ) -> Result<Value<'ctx>, EvalError> {
        let left_v = self.left.eval(env, ctx)?;
        let right_v = self.right.eval(env, ctx)?;
        xor_with_costing(ctx, &left_v, &right_v)
    }
}

/// SANTA `Tuple.non_pair_value_ops` #36-#39 and the tuple-values reply: `Xor` over a tuple of
/// bytes, a `Coll[Any]`, in either operand position
#[cfg(test)]
#[cfg(feature = "arbitrary")]
#[allow(clippy::unwrap_used)]
mod interface_rows {
    use ergotree_ir::mir::coll_value::CollValue;
    use ergotree_ir::mir::value::Value;

    use crate::eval::test_util::{assert_accepts, eval_v3_body};

    fn bytes(b: &[i8]) -> Value<'static> {
        Value::Coll(CollValue::Bytes(b.into()))
    }

    #[test]
    fn xor_reads_a_tuple_of_bytes() {
        // Xor((1b, 2b, 3b), Coll[Byte](9, 9, 9)) and the mirror: the plain collection's value
        // and cost
        assert_accepts("0b0d9b480202020102030e03090909", bytes(&[8, 11, 10]), 22);
        assert_accepts("0b0d9b0e0309090948020202010203", bytes(&[8, 11, 10]), 22);
        // Xor((1b, 2b, 3b), (9b)), both tuples: `zip` cuts to the shorter (by reading)
        let (_, res, _) = eval_v3_body("0b0c9b4802020201020360010209");
        assert_eq!(res.unwrap(), bytes(&[8]));
        // Xor((1, 2, 3), Coll[Byte](9, 9, 9)): an `Int` item is not a `Byte`
        assert!(eval_v3_body("0b0d9b480404040204060e03090909").1.is_err());
    }

    #[test]
    fn global_xor_reads_a_tuple_of_bytes() {
        // Global.xor over the same operands, in both argument positions, at the plain
        // collection's cost (SANTA's sweep: 31)
        for tree_hex in [
            "0b11dc6a02dd02480202020102030e03090909",
            "0b11dc6a02dd020e0309090948020202010203",
        ] {
            assert_accepts(tree_hex, bytes(&[8, 11, 10]), 31);
        }
    }

    /// A v3 tree whose root is `prefix` followed by two `Coll[Byte]` constants: `left` bytes of
    /// 1 and `right` bytes of 2
    fn xor_tree(prefix: &str, left: usize, right: usize) -> alloc::string::String {
        let coll = |len: usize, byte: &str| {
            let size = if len < 0x80 {
                alloc::format!("{len:02x}")
            } else {
                alloc::format!("{:02x}{:02x}", (len & 0x7f) | 0x80, len >> 7)
            };
            alloc::format!("0e{size}{}", byte.repeat(len))
        };
        let body = alloc::format!("{prefix}{}{}", coll(left, "01"), coll(right, "02"));
        let len = body.len() / 2;
        let size = if len < 0x80 {
            alloc::format!("{len:02x}")
        } else {
            alloc::format!("{:02x}{:02x}", (len & 0x7f) | 0x80, len >> 7)
        };
        alloc::format!("0b{size}{body}")
    }

    #[test]
    fn the_per_item_cost_is_on_the_shorter_operand() {
        // SANTA `Xor.per_item_cost_base` (sigma-state 6.0.7): one chunk of 128 bytes; the node
        // 22 up to 128 and 24 from 129, the method 9 above it; 129 against 1 costs what 1
        // against 129 does, in both forms
        assert_eq!(xor_tree("9b", 1, 1), "0b079b0e01010e0102");
        assert_eq!(xor_tree("dc6a02dd02", 1, 1), "0b0bdc6a02dd020e01010e0102");
        for (left, right, node, method) in [
            (1, 1, 22, 31),
            (128, 128, 22, 31),
            (129, 129, 24, 33),
            (129, 1, 22, 31),
            (1, 129, 22, 31),
        ] {
            let threes = alloc::vec![3i8; left.min(right)];
            assert_accepts(&xor_tree("9b", left, right), bytes(&threes), node);
            assert_accepts(&xor_tree("dc6a02dd02", left, right), bytes(&threes), method);
        }
    }
}

#[allow(clippy::panic)]
#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::test_util::{eval_out, eval_out_wo_ctx};
    use alloc::boxed::Box;
    use alloc::vec::Vec;
    use ergotree_ir::chain::context::Context;
    use ergotree_ir::mir::expr::Expr;
    use proptest::prelude::*;
    use sigma_test_util::force_any_val;

    #[test]
    fn eval_1_xor_0() {
        let left = vec![1_i8];
        let right = vec![0_i8];
        let expected_xor = vec![1_i8];

        let expr: Expr = Xor {
            left: Box::new(Expr::Const(left.into())),
            right: Box::new(Expr::Const(right.into())),
        }
        .into();

        assert_eq!(eval_out_wo_ctx::<Vec<i8>>(&expr), expected_xor);
    }

    #[test]
    fn eval_0_xor_1() {
        let left = vec![0_i8];
        let right = vec![1_i8];
        let expected_xor = vec![1_i8];

        let expr: Expr = Xor {
            left: Box::new(Expr::Const(left.into())),
            right: Box::new(Expr::Const(right.into())),
        }
        .into();

        let ctx = force_any_val::<Context>();
        assert_eq!(eval_out::<Vec<i8>>(&expr, &ctx), expected_xor);
    }

    #[test]
    fn eval_1_xor_1() {
        let left = vec![1_i8];
        let right = vec![1_i8];
        let expected_xor = vec![0_i8];

        let expr: Expr = Xor {
            left: Box::new(Expr::Const(left.into())),
            right: Box::new(Expr::Const(right.into())),
        }
        .into();

        assert_eq!(eval_out_wo_ctx::<Vec<i8>>(&expr), expected_xor);
    }

    #[test]
    fn eval_0_xor_0() {
        let left = vec![0_i8];
        let right = vec![0_i8];
        let expected_xor = vec![0_i8];

        let expr: Expr = Xor {
            left: Box::new(Expr::Const(left.into())),
            right: Box::new(Expr::Const(right.into())),
        }
        .into();

        assert_eq!(eval_out_wo_ctx::<Vec<i8>>(&expr), expected_xor);
    }

    #[test]
    fn eval_1100_xor_0101() {
        let left = vec![1_i8, 1, 0, 0];
        let right = vec![0_i8, 1, 0, 1];
        let expected_xor = vec![1_i8, 0, 0, 1];

        let expr: Expr = Xor {
            left: Box::new(Expr::Const(left.into())),
            right: Box::new(Expr::Const(right.into())),
        }
        .into();

        assert_eq!(eval_out_wo_ctx::<Vec<i8>>(&expr), expected_xor);
    }

    proptest! {

        #[test]
        fn eval_any(left_bytes in any::<Vec<i8>>(), right_bytes in any::<Vec<i8>>()) {

            let expected_xor: Vec<i8> = left_bytes
                .iter()
                .zip(right_bytes.iter())
                .map(|(l, r)| l ^ r)
                .collect();

            let expr: Expr = Xor {
                left: Box::new(Expr::Const(left_bytes.into())),
                right: Box::new(Expr::Const(right_bytes.into())),
            }
            .into();

            assert_eq!(&eval_out_wo_ctx::<Vec<i8>>(&expr)[..], &expected_xor[..]);
        }
    }
}
