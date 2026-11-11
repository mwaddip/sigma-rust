use ergotree_ir::mir::extract_script_bytes::ExtractScriptBytes;
use ergotree_ir::mir::value::Value;

use crate::eval::env::Env;
use crate::eval::Context;
use crate::eval::EvalError;
use crate::eval::Evaluable;

impl Evaluable for ExtractScriptBytes {
    fn eval<'ctx>(
        &self,
        env: &mut Env<'ctx>,
        ctx: &Context<'ctx>,
    ) -> Result<Value<'ctx>, EvalError> {
        ctx.add_jit_cost(10)?; // ExtractScriptBytes = Fixed(10)
        let input_v = self.input.eval(env, ctx)?;
        match input_v {
            Value::CBox(b) => Ok(b.script_bytes()?.into()),
            _ => Err(EvalError::UnexpectedValue(format!(
                "Expected ExtractScriptBytes input to be Value::CBox, got {0:?}",
                input_v
            ))),
        }
    }
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
#[cfg(feature = "arbitrary")]
mod tests {
    use super::*;
    use crate::eval::test_util::eval_out;
    use ergotree_ir::chain::context::Context;
    use ergotree_ir::chain::ergo_box::ErgoBox;
    use ergotree_ir::ergo_tree::ErgoTree;
    use ergotree_ir::mir::expr::Expr;
    use ergotree_ir::mir::global_vars::GlobalVars;
    use ergotree_ir::serialization::SigmaSerializable;
    use sigma_test_util::force_any_val;
    use sigma_util::AsVecI8;

    #[test]
    fn eval() {
        let e: Expr = ExtractScriptBytes {
            input: Box::new(GlobalVars::SelfBox.into()),
        }
        .into();
        let ctx = force_any_val::<Context>();
        assert_eq!(
            eval_out::<Vec<i8>>(&e, &ctx),
            ctx.self_box.script_bytes().unwrap()
        );
    }

    #[test]
    fn kept_bytes_of_a_non_canonical_tree() {
        // sigmastate returns the bytes the box's tree was read from (`ErgoBoxCandidate.scala:50`,
        // `ErgoTree.scala:123-130`): a constants count written as the VLQ `82 00` stays two
        // bytes, and the legacy leaf `7f` stays one, where the writer gives `02` and `01 01`
        for read in ["10820004020402d19373007301", "00d1ed93040204027f"] {
            let bytes = base16::decode(read).unwrap();
            let mut b = force_any_val::<ErgoBox>();
            b.ergo_tree = ErgoTree::sigma_parse_bytes(&bytes).unwrap();
            let ctx = Context {
                self_box: Box::leak(Box::new(b)),
                ..force_any_val::<Context>()
            };
            let e: Expr = ExtractScriptBytes {
                input: Box::new(GlobalVars::SelfBox.into()),
            }
            .into();
            assert_eq!(eval_out::<Vec<i8>>(&e, &ctx), bytes.as_vec_i8(), "{read}");
        }
    }
}
