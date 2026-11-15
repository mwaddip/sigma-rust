use ergotree_ir::ergo_tree::ErgoTreeVersion;
use ergotree_ir::mir::bool_to_sigma::BoolToSigmaProp;
use ergotree_ir::mir::constant::TryExtractInto;
use ergotree_ir::mir::value::Value;
use ergotree_ir::sigma_protocol::sigma_boolean::SigmaBoolean;
use ergotree_ir::sigma_protocol::sigma_boolean::SigmaProp;

use crate::eval::env::Env;
use crate::eval::Context;
use crate::eval::EvalError;
use crate::eval::Evaluable;

impl Evaluable for BoolToSigmaProp {
    fn eval<'ctx>(
        &self,
        env: &mut Env<'ctx>,
        ctx: &Context<'ctx>,
    ) -> Result<Value<'ctx>, EvalError> {
        // the operand is not cast where it is evaluated (`evalTo[Any]`): the cost comes between its
        // evaluation and the cast in the body (`trees.scala:37-40`)
        let input_v = self.input.eval(env, ctx)?;
        ctx.add_jit_cost(15)?; // BoolToSigmaProp = Fixed(15)

        // sigmastate passes a `SigmaProp` operand through while `VersionContext.isJitActivated`
        // is false (`trees.scala:39-49`): below an ACTIVATED version of 2, whatever the tree's
        // version. Mainnet holds such scripts from the 4.x years (`sigmaProp(sigmaProp(...))`,
        // transaction 5fe235558... in block 680,692, address Fo6oijFP2JM87ac7w). From the
        // activation on the operand is cast to `Boolean`, also in a tree of version 0 or 1
        if ctx.activated_script_version() < ErgoTreeVersion::V2 {
            if let Value::SigmaProp(sp) = input_v {
                return Ok(Value::SigmaProp(sp));
            }
        }
        let input_v_bool = input_v.try_extract_into::<bool>()?;
        Ok((SigmaProp::new(SigmaBoolean::TrivialProp(input_v_bool))).into())
    }
}

#[cfg(test)]
#[allow(clippy::panic)]
mod tests {
    use super::*;
    use crate::eval::test_util::{eval_out, innermost, try_eval_out};
    use core::cell::Cell;
    use ergotree_ir::chain::context::Context;
    use ergotree_ir::mir::expr::Expr;
    use proptest::prelude::*;
    use sigma_test_util::force_any_val;

    proptest! {

        #[test]
        fn eval(b in any::<bool>()) {
            let expr: Expr = BoolToSigmaProp {input: Expr::Const(b.into()).into()}.into();
            let res = eval_out::<SigmaProp>(&expr, &ctx_at(3, ErgoTreeVersion::V3));
            prop_assert_eq!(res, SigmaProp::new(SigmaBoolean::TrivialProp(b)));
        }
    }

    fn ctx_at(activated: i8, tree: ErgoTreeVersion) -> Context<'static> {
        let ctx = force_any_val::<Context>();
        Context {
            activated_script_version_byte: activated,
            tree_version: Cell::new(tree),
            ..ctx
        }
    }

    /// Below the activation `BoolToSigmaProp(SigmaProp)` passes the proposition through, also in
    /// a tree of version 1: sigmastate's 4.x behaviour, which historical mainnet scripts rely on
    #[test]
    fn eval_below_the_activation_passes_sigmaprop_through() {
        let inner: Expr = BoolToSigmaProp {
            input: Expr::Const(true.into()).into(),
        }
        .into();
        let outer: Expr = BoolToSigmaProp {
            input: inner.into(),
        }
        .into();

        let res = eval_out::<SigmaProp>(&outer, &ctx_at(1, ErgoTreeVersion::V1));
        assert_eq!(res, SigmaProp::new(SigmaBoolean::TrivialProp(true)));
    }

    /// From the activation on `BoolToSigmaProp(SigmaProp)` is refused, also in a tree of
    /// version 0: sigmastate casts the operand to `Boolean`
    #[test]
    fn eval_from_the_activation_rejects_sigmaprop() {
        let inner: Expr = BoolToSigmaProp {
            input: Expr::Const(true.into()).into(),
        }
        .into();
        let outer: Expr = BoolToSigmaProp {
            input: inner.into(),
        }
        .into();

        let ctx = ctx_at(2, ErgoTreeVersion::V0);
        let res = try_eval_out::<SigmaProp>(&outer, &ctx);
        assert!(
            matches!(
                res.as_ref().map_err(innermost),
                Err(EvalError::TryExtractFrom(_))
            ),
            "{res:?}"
        );
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

    /// `BoolToSigmaProp` on sigma-state 6.0.7 at every pair of activated and tree version sigmastate can
    /// hold: a `SigmaProp` operand passes through below an activated version of 2 and is refused from 2 on,
    /// whatever the tree's version (the round's `measure/jvm-act*.txt`, rows `gate|BoolToSigmaProp|*`)
    #[rustfmt::skip]
    const ROWS: &[Row] = &[
        ("gate|BoolToSigmaProp|true|v0", "d10101", 0, 0, "<sigmaprop>", 20),
        ("gate|BoolToSigmaProp|true|v1", "d10101", 1, 0, "<sigmaprop>", 20),
        ("gate|BoolToSigmaProp|SP-const|v0", "d108cd0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798", 0, 0, "<sigmaprop>", 20),
        ("gate|BoolToSigmaProp|SP-const|v1", "d108cd0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798", 1, 0, "<sigmaprop>", 20),
        ("gate|BoolToSigmaProp|ProveDlog|v0", "d1cd070279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798", 0, 0, "<sigmaprop>", 30),
        ("gate|BoolToSigmaProp|ProveDlog|v1", "d1cd070279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798", 1, 0, "<sigmaprop>", 30),
        ("gate|BoolToSigmaProp|nested|v0", "d1d108cd0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798", 0, 0, "<sigmaprop>", 35),
        ("gate|BoolToSigmaProp|nested|v1", "d1d108cd0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798", 1, 0, "<sigmaprop>", 35),
        ("gate|BoolToSigmaProp|Int|v0", "d10402", 0, 0, "ERR", 20),
        ("gate|BoolToSigmaProp|Int|v1", "d10402", 1, 0, "ERR", 20),
        ("gate|BoolToSigmaProp|SP-throws|v0", "d1b2830008040000", 0, 0, "ERR", 55),
        ("gate|BoolToSigmaProp|SP-throws|v1", "d1b2830008040000", 1, 0, "ERR", 55),
        ("gate|BoolToSigmaProp|true|v0", "d10101", 0, 1, "<sigmaprop>", 20),
        ("gate|BoolToSigmaProp|true|v1", "d10101", 1, 1, "<sigmaprop>", 20),
        ("gate|BoolToSigmaProp|SP-const|v0", "d108cd0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798", 0, 1, "<sigmaprop>", 20),
        ("gate|BoolToSigmaProp|SP-const|v1", "d108cd0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798", 1, 1, "<sigmaprop>", 20),
        ("gate|BoolToSigmaProp|ProveDlog|v0", "d1cd070279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798", 0, 1, "<sigmaprop>", 30),
        ("gate|BoolToSigmaProp|ProveDlog|v1", "d1cd070279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798", 1, 1, "<sigmaprop>", 30),
        ("gate|BoolToSigmaProp|nested|v0", "d1d108cd0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798", 0, 1, "<sigmaprop>", 35),
        ("gate|BoolToSigmaProp|nested|v1", "d1d108cd0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798", 1, 1, "<sigmaprop>", 35),
        ("gate|BoolToSigmaProp|Int|v0", "d10402", 0, 1, "ERR", 20),
        ("gate|BoolToSigmaProp|Int|v1", "d10402", 1, 1, "ERR", 20),
        ("gate|BoolToSigmaProp|SP-throws|v0", "d1b2830008040000", 0, 1, "ERR", 55),
        ("gate|BoolToSigmaProp|SP-throws|v1", "d1b2830008040000", 1, 1, "ERR", 55),
        ("gate|BoolToSigmaProp|true|v0", "d10101", 0, 2, "<sigmaprop>", 20),
        ("gate|BoolToSigmaProp|true|v1", "d10101", 1, 2, "<sigmaprop>", 20),
        ("gate|BoolToSigmaProp|true|v2", "d10101", 2, 2, "<sigmaprop>", 20),
        ("gate|BoolToSigmaProp|SP-const|v0", "d108cd0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798", 0, 2, "ERR", 20),
        ("gate|BoolToSigmaProp|SP-const|v1", "d108cd0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798", 1, 2, "ERR", 20),
        ("gate|BoolToSigmaProp|SP-const|v2", "d108cd0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798", 2, 2, "ERR", 20),
        ("gate|BoolToSigmaProp|ProveDlog|v0", "d1cd070279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798", 0, 2, "ERR", 30),
        ("gate|BoolToSigmaProp|ProveDlog|v1", "d1cd070279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798", 1, 2, "ERR", 30),
        ("gate|BoolToSigmaProp|ProveDlog|v2", "d1cd070279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798", 2, 2, "ERR", 30),
        ("gate|BoolToSigmaProp|nested|v0", "d1d108cd0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798", 0, 2, "ERR", 20),
        ("gate|BoolToSigmaProp|nested|v1", "d1d108cd0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798", 1, 2, "ERR", 20),
        ("gate|BoolToSigmaProp|nested|v2", "d1d108cd0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798", 2, 2, "ERR", 20),
        ("gate|BoolToSigmaProp|Int|v0", "d10402", 0, 2, "ERR", 20),
        ("gate|BoolToSigmaProp|Int|v1", "d10402", 1, 2, "ERR", 20),
        ("gate|BoolToSigmaProp|Int|v2", "d10402", 2, 2, "ERR", 20),
        ("gate|BoolToSigmaProp|SP-throws|v0", "d1b2830008040000", 0, 2, "ERR", 55),
        ("gate|BoolToSigmaProp|SP-throws|v1", "d1b2830008040000", 1, 2, "ERR", 55),
        ("gate|BoolToSigmaProp|SP-throws|v2", "d1b2830008040000", 2, 2, "ERR", 55),
        ("gate|BoolToSigmaProp|true|v0", "d10101", 0, 3, "<sigmaprop>", 20),
        ("gate|BoolToSigmaProp|true|v1", "d10101", 1, 3, "<sigmaprop>", 20),
        ("gate|BoolToSigmaProp|true|v2", "d10101", 2, 3, "<sigmaprop>", 20),
        ("gate|BoolToSigmaProp|true|v3", "d10101", 3, 3, "<sigmaprop>", 20),
        ("gate|BoolToSigmaProp|SP-const|v0", "d108cd0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798", 0, 3, "ERR", 20),
        ("gate|BoolToSigmaProp|SP-const|v1", "d108cd0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798", 1, 3, "ERR", 20),
        ("gate|BoolToSigmaProp|SP-const|v2", "d108cd0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798", 2, 3, "ERR", 20),
        ("gate|BoolToSigmaProp|SP-const|v3", "d108cd0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798", 3, 3, "ERR", 20),
        ("gate|BoolToSigmaProp|ProveDlog|v0", "d1cd070279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798", 0, 3, "ERR", 30),
        ("gate|BoolToSigmaProp|ProveDlog|v1", "d1cd070279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798", 1, 3, "ERR", 30),
        ("gate|BoolToSigmaProp|ProveDlog|v2", "d1cd070279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798", 2, 3, "ERR", 30),
        ("gate|BoolToSigmaProp|ProveDlog|v3", "d1cd070279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798", 3, 3, "ERR", 30),
        ("gate|BoolToSigmaProp|nested|v0", "d1d108cd0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798", 0, 3, "ERR", 20),
        ("gate|BoolToSigmaProp|nested|v1", "d1d108cd0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798", 1, 3, "ERR", 20),
        ("gate|BoolToSigmaProp|nested|v2", "d1d108cd0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798", 2, 3, "ERR", 20),
        ("gate|BoolToSigmaProp|nested|v3", "d1d108cd0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798", 3, 3, "ERR", 20),
        ("gate|BoolToSigmaProp|Int|v0", "d10402", 0, 3, "ERR", 20),
        ("gate|BoolToSigmaProp|Int|v1", "d10402", 1, 3, "ERR", 20),
        ("gate|BoolToSigmaProp|Int|v2", "d10402", 2, 3, "ERR", 20),
        ("gate|BoolToSigmaProp|Int|v3", "d10402", 3, 3, "ERR", 20),
        ("gate|BoolToSigmaProp|SP-throws|v0", "d1b2830008040000", 0, 3, "ERR", 55),
        ("gate|BoolToSigmaProp|SP-throws|v1", "d1b2830008040000", 1, 3, "ERR", 55),
        ("gate|BoolToSigmaProp|SP-throws|v2", "d1b2830008040000", 2, 3, "ERR", 55),
        ("gate|BoolToSigmaProp|SP-throws|v3", "d1b2830008040000", 3, 3, "ERR", 55),
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
