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
pub(crate) fn xor_colls(left: &Value<'_>, right: &Value<'_>) -> Result<Arc<[i8]>, EvalError> {
    let (Value::Coll(left), Value::Coll(right)) = (left, right) else {
        return Err(EvalError::UnexpectedValue(format!(
            "Xor: expected two collections, got: {left:?} and {right:?}"
        )));
    };
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

impl Evaluable for Xor {
    fn eval<'ctx>(
        &self,
        env: &mut Env<'ctx>,
        ctx: &Context<'ctx>,
    ) -> Result<Value<'ctx>, EvalError> {
        let left_v = self.left.eval(env, ctx)?;
        let right_v = self.right.eval(env, ctx)?;
        let left_len = match &left_v {
            Value::Coll(coll) => coll.len(),
            _ => 0,
        };
        let xor = xor_colls(&left_v, &right_v)?;
        ctx.add_per_item_jit_cost(10, 2, 128, left_len as u32)?;
        Ok(CollValue::Bytes(xor).into())
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
        // Global.xor over the same operands, in both argument positions: the value. Its cost
        // is not asserted here: SANTA's sweep measured 31, and the method's cost base is its
        // own change
        for tree_hex in [
            "0b11dc6a02dd02480202020102030e03090909",
            "0b11dc6a02dd020e0309090948020202010203",
        ] {
            let (_, res, _) = eval_v3_body(tree_hex);
            assert_eq!(res.unwrap(), bytes(&[8, 11, 10]), "{tree_hex}");
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
