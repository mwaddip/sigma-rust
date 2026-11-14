use crate::eval::env::Env;
use crate::eval::fits_256_bits;
use crate::eval::Context;
use crate::eval::EvalError;
use crate::eval::Evaluable;
use alloc::boxed::Box;
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::convert::TryFrom;
use ergotree_ir::ergo_tree::ErgoTree;
use ergotree_ir::ergo_tree::ErgoTreeVersion;
use ergotree_ir::mir::avl_tree_data::AvlTreeData;
use ergotree_ir::mir::coll_value::CollValue;
use ergotree_ir::mir::constant::Constant;
use ergotree_ir::mir::constant::TryExtractInto;
use ergotree_ir::mir::subst_const::SubstConstants;
use ergotree_ir::mir::value::Value;
use ergotree_ir::types::stype::SType;
use sigma_util::AsVecI8;
use sigma_util::AsVecU8;

impl Evaluable for SubstConstants {
    fn eval<'ctx>(
        &self,
        env: &mut Env<'ctx>,
        ctx: &Context<'ctx>,
    ) -> Result<Value<'ctx>, EvalError> {
        let script_bytes_v = self.script_bytes.eval(env, ctx)?;
        let positions_v = self.positions.eval(env, ctx)?;
        let new_values_v = self.new_values.eval(env, ctx)?;

        // sigmastate hands `positionsV.toArray` to a parameter of type `Array[Int]` (v6.0.6
        // `trees.scala:641-644`, `:674`): the cast needs an `int[]`, which only a collection
        // built at the item type `Int` has. A tuple of `Int`s is a `Coll[Any]` over `Object[]`
        // and throws there (SANTA's tuple-values-3 reply: `[Ljava.lang.Object;` to `[I`)
        let is_int_array =
            matches!(&positions_v, Value::Coll(coll) if coll.elem_tpe() == SType::SInt);
        if !is_int_array {
            return Err(EvalError::UnexpectedValue(format!(
                "SubstConstants: expected `positions` to be an array of Int, got {positions_v:?}"
            )));
        }
        let positions: Vec<usize> = positions_v
            .try_extract_into::<Vec<i32>>()?
            .into_iter()
            .map(|i| i as usize)
            .collect();

        // every item of the new values becomes a constant, whatever the collection's form
        // (sigmastate's `toArrayOfConstants`, v6.0.7 `Extensions.scala:53-64`). The lift checks
        // no `BigInt` range and reads no tree's digest (`Platform.liftToConstant`): a `BigInt`
        // outside 256 bits, which a tree below version 3 can hold, and an `AvlTree` whose digest
        // is not of bytes are refused later, by the constant WRITER
        // (`CoreDataSerializer.scala:35`, `AvlTreeData.scala:75`), and so only if they are written
        let new_values = if let Value::Coll(coll) = new_values_v {
            let mut new_values = vec![];
            for v in &coll.as_vec() {
                let v = v.to_static();
                // sigmastate's lift is a table of classes with no arm for a pair, an option
                // or `Unit` (`Platform.liftToConstant`, v6.0.6 `Platform.scala:16-63`): such
                // a new value throws here, whatever its position (`Extensions.scala:60`).
                // Inside a collection they are lifted with it (the `Coll` arm)
                if matches!(v, Value::Pair(_) | Value::Opt(_) | Value::Unit) {
                    return Err(EvalError::Misc(format!(
                        "SubstConstants: a new value of this kind is not lifted to a constant: \
                         {v:?}"
                    )));
                }
                new_values.push(match Constant::try_from(v.clone()) {
                    Ok(c) => NewValue::Constant(c),
                    Err(e) => {
                        // not a constant even with what only the writer refuses replaced:
                        // the lift throws
                        if Constant::try_from(with_writable_parts(v)).is_err() {
                            return Err(EvalError::Misc(e));
                        }
                        NewValue::Unwritable(e)
                    }
                });
            }
            new_values
        } else {
            return Err(EvalError::Misc(format!(
                "SubstConstants: expected evaluation of `new_values` be of type `Coll[_]`, got \
                    {:?} instead",
                new_values_v
            )));
        };

        if new_values.len() != positions.len() {
            return Err(EvalError::Misc(format!(
                "SubstConstants: `positions.len()` (== {}) and `new_values.len()` (== {}) differ",
                positions.len(),
                new_values.len()
            )));
        }

        // the bytes of any `Coll[Byte]`, specialised or generic (sigmastate reads `toArray`)
        if let Ok(b) = script_bytes_v.clone().try_extract_into::<Vec<i8>>() {
            let (positions, new_constants) = written_values(
                b.as_vec_u8(),
                positions,
                new_values,
                ctx.activated_script_version(),
                ctx.tree_version(),
            )?;
            // Byte-level substitution mirroring sigma-state's
            // `ErgoTreeSerializer.substituteConstants`: the tree body is never
            // parsed and out-of-range positions are a no-op, so a malformed
            // body or an OOB position returns the original bytes (JVM parity)
            // instead of erroring.
            let (new_bytes, num_constants) = ErgoTree::substitute_constants(
                b.as_vec_u8(),
                &positions,
                &new_constants,
                ctx.activated_script_version(),
                ctx.tree_version(),
            )
            .map_err(to_misc_err)?;
            // Charge based on the template's constants count, matching Scala's
            // `ErgoTreeSerializer.substituteConstants` — the walk visits every
            // constant in the template regardless of how many positions the
            // caller asks to replace.
            ctx.add_per_item_jit_cost(100, 100, 1, num_constants as u32)?;
            Ok(Value::Coll(CollValue::Bytes(new_bytes.as_vec_i8().into())))
        } else {
            Err(EvalError::Misc(format!(
                "SubstConstants: expected evaluation of `script_bytes` to be of type `Coll[SBytes]`, \
                 got {:?} instead",
                script_bytes_v
            )))
        }
    }
}

fn to_misc_err<T: core::fmt::Debug>(e: T) -> EvalError {
    EvalError::Misc(format!("{:?}", e))
}

/// A new value as sigmastate holds it once lifted to a constant
enum NewValue {
    /// A constant
    Constant(Constant),
    /// A constant on the JVM that its writer refuses: it holds a `BigInt` outside 256 bits or
    /// an `AvlTree` whose digest is not of bytes. The text is the refusal
    Unwritable(String),
}

/// The positions and constants to substitute, without the new values sigmastate's writer never
/// reaches; an error if it would reach one it refuses. The writer serializes a new value only
/// when its position is inside the tree's constants and is the first to name it
/// (`getPositionsBackref`, v6.0.6 `ErgoTreeSerializer.scala:286-300`, and the loop at
/// `:345-361`), so dropping a value it does not reach changes nothing for the others
fn written_values(
    script_bytes: Vec<u8>,
    positions: Vec<usize>,
    new_values: Vec<NewValue>,
    activated_version: ErgoTreeVersion,
    tree_version: ErgoTreeVersion,
) -> Result<(Vec<usize>, Vec<Constant>), EvalError> {
    let mut unwritable = false;
    let mut constants = Vec::with_capacity(new_values.len());
    for v in &new_values {
        match v {
            NewValue::Constant(c) => constants.push(c.clone()),
            NewValue::Unwritable(_) => unwritable = true,
        }
    }
    if !unwritable {
        return Ok((positions, constants));
    }
    // the tree's number of constants, by the read the substitution itself makes
    let (_, num_constants) =
        ErgoTree::substitute_constants(script_bytes, &[], &[], activated_version, tree_version)
            .map_err(to_misc_err)?;
    let mut named = vec![false; num_constants];
    let mut kept_positions = Vec::with_capacity(positions.len());
    let mut kept_constants = Vec::with_capacity(positions.len());
    for (pos, v) in positions.into_iter().zip(new_values) {
        let written = named.get(pos) == Some(&false);
        if let Some(slot) = named.get_mut(pos) {
            *slot = true;
        }
        match v {
            NewValue::Constant(c) => {
                kept_positions.push(pos);
                kept_constants.push(c);
            }
            NewValue::Unwritable(refusal) if written => return Err(EvalError::Misc(refusal)),
            NewValue::Unwritable(_) => {}
        }
    }
    Ok((kept_positions, kept_constants))
}

/// The value with every part that only sigmastate's WRITER refuses replaced by a stand-in the
/// lift accepts: a `BigInt` outside the 256-bit range by zero, an `AvlTree` whose digest is not
/// of bytes by the same tree with an empty digest of bytes. What the conversion to a constant
/// still refuses then, the writer was not the one to refuse. The stand-in is never written:
/// such a value is [`NewValue::Unwritable`] either way
fn with_writable_parts(v: Value<'static>) -> Value<'static> {
    match v {
        Value::BigInt(b) if !fits_256_bits(&b) => Value::BigInt(num_bigint::BigInt::from(0)),
        Value::AvlTree(t) if AvlTreeData::try_from((*t).clone()).is_err() => {
            let mut t = *t;
            t.digest = CollValue::Bytes(Arc::new([]));
            Value::AvlTree(Box::new(t))
        }
        Value::Pair(pair) => {
            let (l, r) = *pair;
            Value::Pair(Box::new((with_writable_parts(l), with_writable_parts(r))))
        }
        Value::Opt(opt) => Value::Opt(opt.map(|v| Box::new(with_writable_parts(*v)))),
        Value::Coll(coll) => Value::Coll(coll_with_writable_parts(coll)),
        other => other,
    }
}

fn coll_with_writable_parts(coll: CollValue<'static>) -> CollValue<'static> {
    match coll {
        CollValue::Generic { elem_tpe, items } => CollValue::Generic {
            elem_tpe,
            items: items.iter().cloned().map(with_writable_parts).collect(),
        },
        CollValue::Pairs { ls, rs } => CollValue::Pairs {
            ls: Arc::new(coll_with_writable_parts((*ls).clone())),
            rs: Arc::new(coll_with_writable_parts((*rs).clone())),
        },
        other => other,
    }
}

/// What `substConstants` lifts. sigmastate turns each new value into a constant with
/// `liftToConstant`, a table of classes (v6.0.6 `Platform.scala:16-63`), and throws for a value
/// the table has no arm for (`Extensions.scala:53-64`). Each row gives one new value for
/// position 7 of a script whose only constant is at position 0, so nothing is written.
/// Measured on the sigma-state 6.0.7 jars, cost included, also at a throw
#[cfg(test)]
#[cfg(feature = "arbitrary")]
#[allow(clippy::unwrap_used)]
mod lift_rows {
    use crate::eval::test_util::{eval_body, innermost};
    use crate::eval::EvalError;
    use alloc::format;
    use alloc::string::String;
    use ergotree_ir::ergo_tree::ErgoTreeVersion;
    use ergotree_ir::mir::value::Value;

    const V0: ErgoTreeVersion = ErgoTreeVersion::V0;
    const V3: ErgoTreeVersion = ErgoTreeVersion::V3;

    /// A tree constant: 1-byte keys, every operation allowed
    const TREE: &str = "6415cc1ba1054bf74431a2ccfc57e6051b9d0e7561f70586e5512c0ae3f3bac4fd02070100";
    /// The group's generator
    const G: &str = "0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798";

    /// A tree with constant segregation whose one constant is [`TREE`] and whose root is the
    /// placeholder
    fn script() -> String {
        format!("1001{TREE}7300")
    }
    /// `substConstants(script, Coll(7), Coll[item_type](item))`
    fn subst(item_type: &str, item: &str) -> String {
        let script = script();
        format!(
            "740e{:02x}{script}10010e8301{item_type}{item}",
            script.len() / 2
        )
    }
    /// `TREE.updateDigest(Coll[Int](1, 2))`: a tree only the writer refuses
    fn tree_with_ints() -> String {
        format!("dc640f{TREE}0110020204")
    }

    // a pair, an option and `Unit` have no arm: the lift throws before anything is
    // substituted, also when the value holds a tree the writer alone would refuse
    #[test]
    fn a_value_the_jvms_table_has_no_arm_for_is_refused() {
        let t = tree_with_ints();
        let both: &[ErgoTreeVersion] = &[V0, V3];
        // an option constant does not parse below tree version 3, and `Global.some` is a
        // method of version 3
        let v3: &[ErgoTreeVersion] = &[V3];
        let rows: [(&str, String, &[ErgoTreeVersion], u64); 8] = [
            ("62", "62".into(), both, 35),                   // ()
            ("58", "580204".into(), both, 35),               // the constant (1, 2)
            ("58", "860204020404".into(), both, 55),         // (1, 2) built by a node
            ("3c6404", format!("8602{t}0402"), both, 104),   // (that tree, 1)
            ("28", "280102".into(), v3, 35),                 // the constant Some(1)
            ("28", "dc6a09dd01040204".into(), v3, 49),       // Global.some(1)
            ("2464", format!("dc6a09dd01{TREE}64"), v3, 49), // Global.some(the tree)
            ("2464", format!("dc6a09dd01{t}64"), v3, 98),    // Global.some(that tree)
        ];
        for (item_type, item, versions, cost) in rows {
            for version in versions {
                let body = subst(item_type, &item);
                let (res, c) = eval_body(&body, *version);
                assert!(res.is_err(), "{body}: {res:?}");
                let e = res.unwrap_err();
                assert!(matches!(innermost(&e), EvalError::Misc(_)), "{body}: {e:?}");
                assert_eq!(c, cost, "{body}");
            }
        }
    }

    // every other kind is lifted and, at a position that holds no constant, leaves the script
    // as it was; a pair or an option INSIDE a collection is lifted with the collection
    #[test]
    fn a_value_the_jvms_table_has_an_arm_for_is_lifted() {
        let unchanged = Value::from(base16::decode(script().as_bytes()).unwrap());
        let t = tree_with_ints();
        let rows: [(&str, String, u64); 11] = [
            ("01", "0101".into(), 235),      // true
            ("02", "0201".into(), 235),      // 1.toByte
            ("03", "0302".into(), 235),      // 1.toShort
            ("04", "0402".into(), 235),      // 1
            ("05", "0502".into(), 235),      // 1L
            ("06", "060101".into(), 235),    // 1.toBigInt
            ("07", format!("07{G}"), 235),   // a group element
            ("08", format!("08cd{G}"), 235), // a sigma proposition
            ("64", TREE.into(), 235),        // a tree
            ("10", "100102".into(), 235),    // Coll[Int](1)
            ("64", t.clone(), 284),          // the tree only the writer refuses
        ];
        for (item_type, item, cost) in rows {
            for version in [V0, V3] {
                let (res, c) = eval_body(&subst(item_type, &item), version);
                assert_eq!(res.ok().as_ref(), Some(&unchanged), "{item}");
                assert_eq!(c, cost, "{item}");
            }
        }
        // Coll(Coll((that tree, 1))) at both versions, Coll(Coll(Some(that tree))) at version 3
        let pairs = format!("83014c648602{t}0402");
        for version in [V0, V3] {
            let (res, c) = eval_body(&subst("0c4c64", &pairs), version);
            assert_eq!(res.ok().as_ref(), Some(&unchanged), "{pairs}");
            assert_eq!(c, 324, "{pairs}");
        }
        let options = format!("83012464dc6a09dd01{t}64");
        let (res, c) = eval_body(&subst("0c2464", &options), V3);
        assert_eq!(res.ok().as_ref(), Some(&unchanged), "{options}");
        assert_eq!(c, 318, "{options}");
    }
}

/// SANTA `Tuple.non_pair_value_type_directed` #5 and #6 (sigma-state 6.0.7), and the positions'
/// array
#[cfg(test)]
#[cfg(feature = "arbitrary")]
#[allow(clippy::unwrap_used)]
mod tuple_operand_rows {
    use ergotree_ir::mir::coll_value::CollValue;
    use ergotree_ir::mir::value::Value;

    use crate::eval::test_util::{assert_accepts, eval_v3_body};

    #[test]
    fn new_values_and_positions_need_their_types() {
        // the control: substConstants(script with two Int constants, Coll(0), Coll(9))
        let substituted: [i8; 13] = [27, 11, 2, 4, 18, 4, 14, -47, -109, 115, 0, 115, 1];
        assert_accepts(
            "0b16740e0d1b0b02040e040ed19373007301100100100112",
            Value::Coll(CollValue::Bytes(substituted.into())),
            315,
        );
        // new values `Coll((1, 2, 3))` for a constant of type (Int, Int, Int): each lifts to a
        // `Coll[Any]` constant, which is not the type of the constant it replaces
        let tuple_values =
            "0b22740e131b1102480404040204060406d193b1730073011001000c4804040401020406";
        assert!(eval_v3_body(tuple_values).1.is_err());
        // positions as the tuple `(0)`, a `Coll[Any]`, is not an `int[]` (SANTA's
        // tuple-values-3 reply, section 2)
        let tuple_positions = "0b17740e0d1b0b02040e040ed1937300730160010400100112";
        assert!(eval_v3_body(tuple_positions).1.is_err());
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
#[allow(clippy::panic)]
#[allow(clippy::expect_used)]
#[allow(clippy::unreachable)]
mod tests {
    use ergotree_ir::{
        ergo_tree::{ErgoTree, ErgoTreeHeader},
        mir::{
            bin_op::{ArithOp, BinOp, BinOpKind},
            expr::Expr,
        },
        serialization::SigmaSerializable,
    };
    use proptest::prelude::*;

    use crate::eval::test_util::try_eval_out_with_version;
    use crate::eval::test_util::try_eval_out_wo_ctx;
    use sigma_test_util::force_any_val;

    use super::*;
    proptest! {

        // a collection, which sigmastate's lift has an arm for; a pair it refuses (`lift_rows`)
        #[test]
        fn eval_single_substitution(original in any::<Vec<i64>>(), new in any::<Vec<i64>>()) {
            test_single_substitution(original, new);
        }

        #[test]
        fn eval_3_substitutions(original in any::<(i32, i32, i32)>(), new in any::<(i32, i32, i32)>()) {
            test_3_substitutions(original, new);
        }
    }

    #[test]
    fn a_template_s_constants_are_read_under_the_activated_version() {
        // `SubstConstants` substitutes inside the spend's `VersionContext` (sigmastate v6.0.6
        // `Interpreter.scala:366`), and `substituteConstants` reads every constant of the
        // template (`ErgoTreeSerializer.scala:320-326`). The template is the tree of SANTA's
        // `Box.tree_version_above_activated` #9: its constant 0 is a Box with a v4 tree.
        // #10's has a v3 one. SANTA `tree-version-above-activated-eval` #6 and #7 are the
        // spends (replayed in ergo-lib's `tx_context`).
        let template = |header: u8| {
            let hex = format!(
                "1b310263c0843d{header:02x}0208d3010000\
                 0000000000000000000000000000000000000000000000000000000000000000\
                 0008d37301"
            );
            base16::decode(hex.as_bytes()).unwrap()
        };
        let subst = |header: u8, activated: u8| {
            let node = Expr::SubstConstants(
                SubstConstants {
                    script_bytes: Expr::Const(Constant::from(template(header))).into(),
                    positions: Expr::Const(Constant::from(Vec::<i32>::new())).into(),
                    new_values: Expr::Const(Constant::from(Vec::<i32>::new())).into(),
                }
                .into(),
            );
            let ctx = force_any_val::<Context>();
            try_eval_out_with_version::<Vec<u8>>(&node, &ctx, 3, activated)
        };
        // the evaluator reports a substitution's failure as a `Misc` error, wrapped with the
        // node's span
        let refused = format!("{:?}", subst(0x0c, 3).unwrap_err());
        assert!(
            refused.contains("TreeVersionAboveActivated(4, 3)"),
            "{refused}"
        );
        assert_eq!(subst(0x0b, 3).unwrap(), template(0x0b));
        assert_eq!(subst(0x0c, 1).unwrap(), template(0x0c));
    }

    #[test]
    fn new_values_of_every_collection_form() {
        // sigmastate lifts each item of the new values to a constant whatever the collection
        // (`toArrayOfConstants`, v6.0.7 `Extensions.scala:53-64`): a `Coll[Boolean]` and a
        // `Coll[Byte]`, both specialised, substitute as a generic collection does
        test_single_substitution(true, false);
        test_single_substitution(5i8, 7i8);
    }

    fn test_single_substitution<T: Clone + Into<Constant>>(original: T, new: T)
    where
        Vec<T>: Into<Constant>,
    {
        let expr = Expr::Const(original.clone().into());
        let ergo_tree = ErgoTree::new(ErgoTreeHeader::v0(true), &expr).unwrap();
        assert_eq!(ergo_tree.constants_len().unwrap(), 1);
        assert_eq!(ergo_tree.get_constant(0).unwrap().unwrap(), original.into());

        let script_bytes =
            Expr::Const(Constant::from(ergo_tree.sigma_serialize_bytes().unwrap())).into();
        let positions = Expr::Const(Constant::from(vec![0])).into();
        let new_values = Expr::Const(vec![new.clone()].into()).into();

        let subst_const = Expr::SubstConstants(
            SubstConstants {
                script_bytes,
                positions,
                new_values,
            }
            .into(),
        );

        let x: Value = try_eval_out_wo_ctx(&subst_const).unwrap();
        if let Value::Coll(CollValue::Bytes(b)) = x {
            // The template's root is not a `SigmaProp`, which a tree parse rejects (rule
            // 1001) but SubstConstants never parses: compare the bytes with the tree built
            // from the new constant.
            let expected = ErgoTree::new(ErgoTreeHeader::v0(true), &Expr::Const(new.into()))
                .unwrap()
                .sigma_serialize_bytes()
                .unwrap();
            assert_eq!(b.as_vec_u8(), expected);
        } else {
            unreachable!();
        }
    }

    /// `a + b * c`, three `Int` constants
    fn plus_times(a: i32, b: i32, c: i32) -> Expr {
        Expr::BinOp(
            BinOp {
                kind: BinOpKind::Arith(ArithOp::Plus),
                left: Box::new(Expr::Const(a.into())),
                right: Box::new(Expr::BinOp(
                    BinOp {
                        kind: BinOpKind::Arith(ArithOp::Multiply),
                        left: Box::new(Expr::Const(b.into())),
                        right: Box::new(Expr::Const(c.into())),
                    }
                    .into(),
                )),
            }
            .into(),
        )
    }

    fn test_3_substitutions(original: (i32, i32, i32), new: (i32, i32, i32)) {
        let (o0, o1, o2) = original;
        let (n0, n1, n2) = new;
        let expr = plus_times(o0, o1, o2);
        let ergo_tree = ErgoTree::new(ErgoTreeHeader::v0(true), &expr).unwrap();
        assert_eq!(ergo_tree.constants_len().unwrap(), 3);
        assert_eq!(ergo_tree.get_constant(0).unwrap().unwrap(), o0.into());
        assert_eq!(ergo_tree.get_constant(1).unwrap().unwrap(), o1.into());
        assert_eq!(ergo_tree.get_constant(2).unwrap().unwrap(), o2.into());

        let script_bytes =
            Expr::Const(Constant::from(ergo_tree.sigma_serialize_bytes().unwrap())).into();

        let positions = Expr::Const(Constant::from(vec![1, 2, 0])).into();

        let new_values = Expr::Const(Constant::from(vec![n0, n1, n2])).into();

        let subst_const = Expr::SubstConstants(
            SubstConstants {
                script_bytes,
                positions,
                new_values,
            }
            .into(),
        );

        let x: Value = try_eval_out_wo_ctx(&subst_const).unwrap();
        if let Value::Coll(CollValue::Bytes(b)) = x {
            // As in `test_single_substitution`. Positions [1, 2, 0] take n0, n1 and n2,
            // so the constants become [n2, n0, n1].
            let expected = ErgoTree::new(ErgoTreeHeader::v0(true), &plus_times(n2, n0, n1))
                .unwrap()
                .sigma_serialize_bytes()
                .unwrap();
            assert_eq!(b.as_vec_u8(), expected);
        } else {
            unreachable!();
        }
    }

    // Bug 3 regression: SubstConstants cost must be charged based on the
    // template tree's `constants_len()`, not on the caller-supplied positions
    // count. Build one 3-constant template and run substitution twice with
    // different position counts — after the fix, total JIT cost must be
    // identical since the template walk is the same. Pre-fix, the two runs
    // diverged because the charge scaled with positions.len().
    #[test]
    #[allow(clippy::identity_op)]
    fn subst_const_cost_uses_template_count_not_positions_len() {
        use crate::eval::test_util::try_eval_out;
        use ergotree_ir::chain::context::Context;
        use sigma_test_util::force_any_val;

        // Template tree with 3 constants (a + b * c).
        let expr = Expr::BinOp(
            BinOp {
                kind: BinOpKind::Arith(ArithOp::Plus),
                left: Box::new(Expr::Const(1i32.into())),
                right: Box::new(Expr::BinOp(
                    BinOp {
                        kind: BinOpKind::Arith(ArithOp::Multiply),
                        left: Box::new(Expr::Const(2i32.into())),
                        right: Box::new(Expr::Const(3i32.into())),
                    }
                    .into(),
                )),
            }
            .into(),
        );
        let ergo_tree = ErgoTree::new(ErgoTreeHeader::v0(true), &expr).unwrap();
        assert_eq!(ergo_tree.constants_len().unwrap(), 3);
        let tree_bytes = ergo_tree.sigma_serialize_bytes().unwrap();
        let script_bytes_expr: Expr = Expr::Const(Constant::from(tree_bytes));

        let run = |positions: Vec<i32>, new_vals: Vec<i32>| -> u64 {
            let ctx = force_any_val::<Context>();
            let before = ctx.jit_cost_value();
            let script_bytes: Box<Expr> = Box::new(script_bytes_expr.clone());
            let positions_expr: Box<Expr> = Box::new(Expr::Const(Constant::from(positions)));
            let new_values_expr: Box<Expr> = Box::new(Expr::Const(Constant::from(new_vals)));
            let subst = Expr::SubstConstants(
                SubstConstants {
                    script_bytes,
                    positions: positions_expr,
                    new_values: new_values_expr,
                }
                .into(),
            );
            let _: Value = try_eval_out(&subst, &ctx).unwrap();
            ctx.jit_cost_value() - before
        };

        let delta_1_position = run(vec![0], vec![999]);
        let delta_3_positions = run(vec![0, 1, 2], vec![999, 888, 777]);

        // Both runs walk the same 3-constant template, so SubstConstants-specific
        // cost must be identical. `Expr::Const` eval is fixed (5 JIT) regardless
        // of payload size, so the Const arms used to feed positions/new_values
        // don't contribute differently. Any non-zero difference indicates the
        // old positions.len()-based charge leaked back in.
        assert_eq!(
            delta_1_position, delta_3_positions,
            "SubstConstants cost must depend on template's constants_len (3), \
             not on positions.len() (1 vs 3). Deltas diverged: {} vs {}",
            delta_1_position, delta_3_positions,
        );
    }
}

#[cfg(test)]
#[cfg(feature = "arbitrary")]
#[allow(clippy::unwrap_used)]
mod bigint_range_tests {
    use ergotree_ir::ergo_tree::ErgoTreeVersion;
    use ergotree_ir::mir::value::Value;

    use crate::eval::test_util::{bigint_x, bigint_y, eval_body};

    #[test]
    fn santa_subst_constants_refuses_a_bigint_outside_256_bits() {
        // SANTA's reply `sigma-rust-bigint-range-reply.md` (sigma-state 6.0.7), measured:
        // `substConstants(t, Coll(0), Coll(BigInt.MIN / -1))` over a tree `t` with one
        // segregated `BigInt` constant throws at the constant writer below tree version 3
        // ("doesn't fit into 256 bits", v6.0.6 `CoreDataSerializer.scala:35`): the one consumer
        // there that range-checks a value it is handed. Its in-range twin returns the tree
        let t = "1001060101d1939f82730082";
        let subst = |value: &str| format!("740e0c{t}100100830106{value}");
        let (res, _) = eval_body(&subst(&bigint_x()), ErgoTreeVersion::V0);
        assert!(res.is_err(), "{res:?}");

        let (res, cost) = eval_body(&subst(&bigint_y()), ErgoTreeVersion::V0);
        let bytes = base16::decode("10010601fad1939f82730082").unwrap();
        assert_eq!(res.ok(), Some(Value::from(bytes)));
        assert_eq!(cost, 265);
    }

    #[test]
    fn a_bigint_outside_256_bits_is_refused_only_where_it_would_be_written() {
        // By reading, not measured: sigmastate lifts every new value to a constant whatever a
        // `BigInt`'s width (`toArrayOfConstants`, v6.0.6 `Extensions.scala:53-64`;
        // `Platform.liftToConstant` checks no range), and only the constant WRITER requires 256
        // bits. The writer reaches a new value only when its position is inside the tree's
        // constants and is the first to name it (`ErgoTreeSerializer.scala:286-300`,
        // `:345-361`). The costs are those of the in-range twins, sigma-rust's own
        let t = "1001060101d1939f82730082"; // one constant
        let (x, y) = (bigint_x(), bigint_y());
        let unchanged = Some(Value::from(base16::decode(t).unwrap()));
        let with_minus_six = Some(Value::from(
            base16::decode("10010601fad1939f82730082").unwrap(),
        ));
        let subst = |positions: &str, values: &str| {
            eval_body(
                &format!("740e0c{t}{positions}{values}"),
                ErgoTreeVersion::V0,
            )
        };

        // `Coll(5)`: the position is outside the tree's constants
        let (res, cost) = subst("10010a", &format!("830106{x}"));
        assert_eq!((res.ok(), cost), (unchanged.clone(), 265));
        // `Coll(-1)`
        let (res, cost) = subst("100101", &format!("830106{x}"));
        assert_eq!((res.ok(), cost), (unchanged, 265));
        // `Coll(0, 0)` with `Coll(Y, X)`: the first to name a position wins, `X` is not written
        let (res, cost) = subst("10020000", &format!("830206{y}{x}"));
        assert_eq!((res.ok(), cost), (with_minus_six, 300));
        // `Coll(0, 0)` with `Coll(X, Y)`: `X` is the one written
        let (res, _) = subst("10020000", &format!("830206{x}{y}"));
        assert!(res.is_err(), "{res:?}");
    }
}
