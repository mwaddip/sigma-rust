use ergotree_ir::mir::extract_bytes_with_no_ref::ExtractBytesWithNoRef;
use ergotree_ir::mir::value::Value;

use crate::eval::env::Env;
use crate::eval::Context;
use crate::eval::EvalError;
use crate::eval::Evaluable;

impl Evaluable for ExtractBytesWithNoRef {
    fn eval<'ctx>(
        &self,
        env: &mut Env<'ctx>,
        ctx: &Context<'ctx>,
    ) -> Result<Value<'ctx>, EvalError> {
        ctx.add_jit_cost(12)?; // ExtractBytesWithNoRef = Fixed(12)
        let input_v = self.input.eval(env, ctx)?;
        match input_v {
            Value::CBox(b) => Ok(b.bytes_without_ref(ctx.tree_version())?.into()),
            _ => Err(EvalError::UnexpectedValue(format!(
                "Expected ExtractBytesWithNoRef input to be Value::CBox, got {0:?}",
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
    use crate::eval::test_util::{eval_out, try_eval_out};
    use ergotree_ir::chain::context::Context;
    use ergotree_ir::chain::ergo_box::box_value::BoxValue;
    use ergotree_ir::chain::ergo_box::{ErgoBox, NonMandatoryRegisters, RegisterValue};
    use ergotree_ir::chain::tx_id::TxId;
    use ergotree_ir::ergo_tree::{ErgoTree, ErgoTreeVersion};
    use ergotree_ir::mir::coll_by_index::ByIndex;
    use ergotree_ir::mir::expr::Expr;
    use ergotree_ir::mir::global_vars::GlobalVars;
    use ergotree_ir::serialization::SigmaSerializable;
    use sigma_test_util::force_any_val;

    #[test]
    fn the_first_reader_writes_an_output_s_bytes_without_ref() {
        // ergo's `bytesWithNoRef` is a lazy val (sigma-state 6.0.6 `ErgoBoxCandidate.scala:54`) on
        // the transaction's one set of outputs (`ErgoLikeTransaction.scala:46`), which each
        // input's context wraps (`ErgoLikeContext.scala:157`). The first script to read it writes
        // it, under its own tree version, and later readers get those bytes (SANTA
        // `evaluated-values-spend` entries 51-62). Below version 3, X15's `Upcast` in R4 is
        // written as the constant, 2 bytes shorter.
        let x15 =
            RegisterValue::sigma_parse_bytes(&[0x86, 0x02, 0x04, 0x02, 0x7e, 0x04, 0x02, 0x05]);
        let output = ErgoBox::new(
            BoxValue::SAFE_USER_MIN,
            ErgoTree::sigma_parse_bytes(&[0x00, 0x08, 0xd3]).unwrap(),
            None,
            NonMandatoryRegisters::try_from(vec![x15]).unwrap(),
            1,
            TxId::zero(),
            0,
        )
        .unwrap();
        let e: Expr = ExtractBytesWithNoRef {
            input: Box::new(
                ByIndex::new(GlobalVars::Outputs.into(), Expr::Const(0i32.into()), None)
                    .unwrap()
                    .into(),
            ),
        }
        .into();
        /// `e`'s length, read by a script of tree version `version` in a context over `outputs`
        fn size(e: &Expr, outputs: &[ErgoBox], version: ErgoTreeVersion) -> usize {
            let ctx = Context {
                outputs,
                ..force_any_val::<Context>()
            };
            ctx.tree_version.set(version);
            try_eval_out::<Vec<i8>>(e, &ctx).unwrap().len()
        }
        // read first below version 3, then at 3
        let outputs = [output.clone()];
        let below_3 = size(&e, &outputs, ErgoTreeVersion::V0);
        assert_eq!(size(&e, &outputs, ErgoTreeVersion::V3), below_3);
        // read first at 3, then below 3
        let outputs = [output];
        let at_3 = size(&e, &outputs, ErgoTreeVersion::V3);
        assert_eq!(at_3, below_3 + 2);
        assert_eq!(size(&e, &outputs, ErgoTreeVersion::V0), at_3);
    }

    #[test]
    fn eval() {
        let e: Expr = ExtractBytesWithNoRef {
            input: Box::new(GlobalVars::SelfBox.into()),
        }
        .into();
        let ctx = force_any_val::<Context>();
        assert_eq!(
            eval_out::<Vec<i8>>(&e, &ctx),
            ctx.self_box.bytes_without_ref(ctx.tree_version()).unwrap()
        );
    }
}
