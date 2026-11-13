use ergotree_ir::mir::expr::Expr;
use ergotree_ir::mir::tuple::Tuple;
use ergotree_ir::mir::value::Value;

use crate::eval::env::Env;
use crate::eval::Context;
use crate::eval::EvalError;
use crate::eval::Evaluable;

impl Evaluable for Tuple {
    fn eval<'ctx>(
        &self,
        env: &mut Env<'ctx>,
        ctx: &Context<'ctx>,
    ) -> Result<Value<'ctx>, EvalError> {
        // sigma-state restricts tuples to exactly 2 elements: `Tuple.eval`
        // rejects `items.length != 2` with "Invalid tuple" before doing any work
        // (sigma `values.scala`: "in v5.0 version we support only tuples of 2
        // elements to be equivalent with v4.x"). The check is unconditional —
        // no version gate, and sigma-state 6.0.3 evaluates every height through
        // it. sigma-rust models tuples as flat N-ary (`TupleItems`, 0 to 255
        // items), so without this guard it accepts tuples of other arities
        // the JVM rejects — a consensus accept/reject divergence (sigma-rust
        // would accept a spend the JVM rejects). Mirror the JVM: reject non-pairs.
        let [first, second] = self.items.as_slice() else {
            return Err(EvalError::Misc(format!(
                "Tuple: sigma-state allows only 2-element tuples, got {}",
                self.items.len()
            )));
        };
        // Mirror the JVM Tuple eval, which `checkType`s each item's value after
        // evaluating it (`values.scala:828-833`): an item whose type is an
        // unsupported tuple (arity != 2) is rejected ("Unsupported tuple
        // type", SType.scala:200), which catches an arity-3 tuple constant
        // carried as a pair item; so is a pair item whose value is not a pair,
        // the data of a pair tuple expression.
        let mut item_v = |i: &Expr| -> Result<Value<'ctx>, EvalError> {
            let v = i.eval(env, ctx)?;
            crate::eval::check_value_of_type(&i.tpe(), &v, ctx.tree_version())?;
            Ok(v)
        };
        let first_v = item_v(first)?;
        let second_v = item_v(second)?;
        // Tuple = Fixed(15), added once both items are evaluated and checked
        // (`values.scala:824-836`): an item that throws leaves the node uncharged
        ctx.add_jit_cost(15)?;
        Ok(Value::pair(first_v, second_v))
    }
}

/// A tuple value of other than two items at evaluation: sigmastate holds it as a generic
/// `Coll[Any]` (`Tuple.value`, v6.0.6 `values.scala:818-822`; `CoreDataSerializer.scala:134-139`),
/// so every consumer that needs only the `Coll` interface answers as on a plain collection of the
/// same items, value and cost, and one that needs the backing array of a specific class throws.
/// Rows and costs: SANTA's reply `sigma-rust-tuple-values-reply.md` and its
/// `eval/v6/authored/Tuple.non_pair_value_ops.json` (sigma-state 6.0.7), each tuple row beside
/// its plain-collection control. `t` = `(1, 2, 3)` = `48040404020406`
#[cfg(test)]
#[cfg(feature = "arbitrary")]
#[allow(clippy::unwrap_used)]
mod non_pair_value_rows {
    use alloc::vec::Vec;
    use ergotree_ir::bigint256::BigInt256;
    use ergotree_ir::ergo_tree::ErgoTree;
    use ergotree_ir::ergo_tree::ErgoTreeVersion;
    use ergotree_ir::mir::coll_value::CollValue;
    use ergotree_ir::mir::value::Value;
    use ergotree_ir::serialization::sigma_byte_reader;
    use ergotree_ir::serialization::sigma_byte_reader::SigmaByteRead;
    use ergotree_ir::serialization::SigmaSerializable;
    use ergotree_ir::sigma_protocol::sigma_boolean::{SigmaBoolean, SigmaProp};
    use ergotree_ir::types::stype::SType;

    use crate::eval::test_util::{assert_accepts, eval_v3_body};

    fn int(i: i32) -> Value<'static> {
        Value::Int(i)
    }

    fn truth() -> Value<'static> {
        Value::Boolean(true)
    }

    fn coll(elem_tpe: SType, items: &[i32]) -> Value<'static> {
        let items: Vec<Value> = items.iter().map(|i| Value::Int(*i)).collect();
        Value::Coll(CollValue::generic(elem_tpe, items))
    }

    fn bytes(b: &[i8]) -> Value<'static> {
        Value::Coll(CollValue::Bytes(b.into()))
    }

    fn true_prop() -> Value<'static> {
        Value::SigmaProp(alloc::boxed::Box::new(SigmaProp::new(
            SigmaBoolean::TrivialProp(true),
        )))
    }

    fn assert_throws(tree_hex: &str) {
        assert!(eval_v3_body(tree_hex).1.is_err(), "{tree_hex}");
    }

    #[test]
    fn the_collection_nodes_read_a_tuple_as_a_collection() {
        // SizeOf, ByIndex, ByIndex with a default: the tuple, then the Coll[Int] control
        assert_accepts("0b08b148040404020406", int(3), 19);
        assert_accepts("0b06b11003020406", int(3), 19);
        assert_accepts("0b0bb248040404020406040000", int(1), 40);
        assert_accepts("0b09b21003020406040000", int(1), 40);
        assert_accepts("0b0db248040404020406040a010412", int(9), 45);
        assert_accepts("0b0bb21003020406040a010412", int(9), 45);
        // Exists, ForAll, MapCollection, Fold
        assert_accepts("0b11ae48040404020406d90101049372010402", truth(), 32);
        assert_accepts("0b0fae1003020406d90101049372010402", truth(), 32);
        assert_accepts("0b11af48040404020406d90101049172010400", truth(), 119);
        assert_accepts("0b0faf1003020406d90101049172010400", truth(), 119);
        let mapped = coll(SType::SInt, &[2, 3, 4]);
        assert_accepts(
            "0b11ad48040404020406d90101049a72010402",
            mapped.clone(),
            121,
        );
        assert_accepts("0b0fad1003020406d90101049a72010402", mapped, 121);
        assert_accepts(
            "0b17b0480404040204060400d90101589a8c7201018c720102",
            int(6),
            169,
        );
        assert_accepts(
            "0b15b010030204060400d90101589a8c7201018c720102",
            int(6),
            169,
        );
        // a tuple of none and of one item
        assert_accepts("0b03b16000", int(0), 19);
        assert_accepts("0b05b160010402", int(1), 19);
    }

    #[test]
    fn a_result_that_keeps_the_receiver_keeps_the_item_type_any() {
        // Slice, Append, Filter: the same items and cost as the control, the item type `Any`
        assert_accepts("0b0db1b44804040402040604000404", int(2), 41);
        assert_accepts("0b0bb1b4100302040604000404", int(2), 41);
        assert_accepts(
            "0b0cb44804040402040604000404",
            coll(SType::SAny, &[1, 2]),
            27,
        );
        assert_accepts("0b0ab4100302040604000404", coll(SType::SInt, &[1, 2]), 27);
        assert_accepts("0b10b1b34804040402040648040404020406", int(6), 46);
        assert_accepts("0b0cb1b310030204061003020406", int(6), 46);
        assert_accepts(
            "0b0fb34804040402040648040404020406",
            coll(SType::SAny, &[1, 2, 3, 1, 2, 3]),
            32,
        );
        assert_accepts("0b12b1b548040404020406d90101049172010402", int(2), 150);
        assert_accepts("0b10b1b51003020406d90101049172010402", int(2), 150);
        assert_accepts(
            "0b11b548040404020406d90101049172010402",
            coll(SType::SAny, &[2, 3]),
            136,
        );
    }

    #[test]
    fn the_coll_methods_read_a_tuple_as_a_collection() {
        // SANTA's tuple-values reply, section 3: the `Coll` methods on `(1, 2, 3)` at the plain
        // collection's costs (the trees are ours; the costs are the reply's)
        // size, apply(0), get(0), getOrElse(5, 9)
        assert_accepts("0b0adb0c0148040404020406", int(3), 23);
        assert_accepts("0b0ddc0c0a48040404020406010400", int(1), 44);
        assert_accepts(
            "0b0ddc0c2148040404020406010400",
            Value::Opt(Some(alloc::boxed::Box::new(int(1)))),
            44,
        );
        assert_accepts("0b0fdc0c024804040402040602040a0412", int(9), 49);
        // indexOf(1, 1): -1, searching from index 1; indices
        assert_accepts("0b0fdc0c1a480404040204060204020402", int(-1), 55);
        assert_accepts(
            "0b0adb0c0e48040404020406",
            coll(SType::SInt, &[0, 1, 2]),
            31,
        );
        // startsWith and endsWith, the tuple as receiver and as argument
        assert_accepts("0b12dc0c1f480404040204060148040404020406", truth(), 25);
        assert_accepts("0b12dc0c20480404040204060148040404020406", truth(), 25);
        assert_accepts("0b10dc0c1f10030204060148040404020406", truth(), 25);
        // reverse, updated(0, 7), patch(0, t, 1): the item type stays `Any`
        assert_accepts(
            "0b0adb0c1e48040404020406",
            coll(SType::SAny, &[3, 2, 1]),
            31,
        );
        assert_accepts(
            "0b0fdc0c1448040404020406020400040e",
            coll(SType::SAny, &[7, 2, 3]),
            40,
        );
        assert_accepts(
            "0b16dc0c1348040404020406030400480404040204060402",
            coll(SType::SAny, &[1, 2, 3, 2, 3]),
            56,
        );
        // zip: the pair's item type has `Any` on the tuple's side
        let zipped = |other: SType| {
            let ints = |xs: &[i32]| xs.iter().map(|i| Value::Int(*i)).collect::<Vec<_>>();
            Value::Coll(CollValue::pairs(
                CollValue::generic(SType::SAny, ints(&[1, 2, 3])),
                CollValue::generic(other, ints(&[1, 2, 3])),
            ))
        };
        assert_accepts(
            "0b12dc0c1d480404040204060148040404020406",
            zipped(SType::SAny),
            25,
        );
        assert_accepts(
            "0b10dc0c1d48040404020406011003020406",
            zipped(SType::SInt),
            25,
        );
    }

    #[test]
    fn a_primitive_array_and_a_tuple_do_not_append() {
        // `ArrayStoreException`, in both directions; the Coll[Int] controls append
        assert_throws("0b0cb1b34804040402040610010e");
        assert_accepts("0b0ab1b3100302040610010e", int(4), 46);
        assert_throws("0b0cb1b310010e48040404020406");
        assert_accepts("0b0ab1b310010e1003020406", int(4), 46);
        assert_throws("0b0bb34804040402040610010e");
    }

    #[test]
    fn the_boolean_reductions_read_a_tuple_as_a_collection() {
        // AND, OR, XorOf over (true, true, true), then the Coll[Boolean] controls
        assert_accepts("0b089648010101010101", truth(), 20);
        assert_accepts("0b04960d0307", truth(), 20);
        assert_accepts("0b089748010101010101", truth(), 15);
        assert_accepts("0b04970d0307", truth(), 15);
        assert_accepts("0b08ff48010101010101", truth(), 30);
        assert_accepts("0b04ff0d0307", truth(), 30);
    }

    #[test]
    fn a_consumer_of_a_typed_array_refuses_a_tuple() {
        // CalcBlake2b256, CalcSha256, ByteArrayToBigInt over (1b, 2b, 3b): `Object[]` is not a
        // `byte[]`; AtLeast over three propositions: not a `SigmaProp[]`. Each control answers
        assert_throws("0b09b1cb48020202010203");
        assert_accepts("0b07b1cb0e03010203", int(32), 46);
        assert_throws("0b08cb48020202010203");
        assert_throws("0b09b1cc48020202010203");
        assert_accepts("0b07b1cc0e03010203", int(32), 107);
        assert_throws("0b087b48020202010203");
        assert_accepts(
            "0b067b0e03010203",
            Value::from(BigInt256::from(66051i64)),
            35,
        );
        assert_throws("0b0a98040448080808d3d3d3");
        assert_accepts("0b0c98040483030808d308d308d3", true_prop(), 63);
        // the Xor controls (the tuple rows: `xor.rs`)
        assert_accepts("0b0b9b0e030102030e03090909", bytes(&[8, 11, 10]), 22);
        assert_accepts("0b0b9b0e030909090e03010203", bytes(&[8, 11, 10]), 22);
    }

    #[test]
    fn a_pair_is_not_a_collection() {
        // SizeOf((1, 2)): a `Tuple2` is not a `Coll`
        assert_throws("0b04b1580204");
    }

    #[test]
    fn a_tuple_arrives_through_every_data_route() {
        // SizeOf of: SelectField(((1,2,3), 4), 1); ByIndex(Coll((1,2,3)), 0);
        // OptionGet(Some((1,2,3))); Global.deserializeTo[(Int,Int,Int)](020406)
        assert_accepts("0b0cb18c4c480404040204060801", int(3), 29);
        assert_accepts("0b0eb1b20c4804040401020406040000", int(3), 54);
        assert_accepts("0b0bb1e4244804040401020406", int(3), 34);
        assert_accepts("0b0fb1dc6a04dd010e0302040648040404", int(3), 160);
        // sigmaProp(SizeOf((1,2,3)) == 3): the script of a box anyone can create
        assert_accepts("0b0cd193b1480404040204060406", true_prop(), 42);
    }

    #[test]
    fn a_nested_tuple_compares_as_a_collection() {
        // Coll((1,2,3)) == Coll((1,2,3)), Some((1,2,3)) == Some((1,2,3)),
        // ((1,2,3), 4) == ((1,2,3), 4): `checkType` is shallow, and the comparer meets a
        // `Coll[Any]` (MatchType 1 + 10 + 2 x 3 + three Ints 9 = 26)
        assert_accepts("0b13930c48040404010204060c4804040401020406", truth(), 49);
        assert_accepts("0b1393244804040401020406244804040401020406", truth(), 40);
        assert_accepts("0b13934c48040404020406084c4804040402040608", truth(), 43);
    }

    #[test]
    fn a_pair_collection_s_side_keeps_the_tuple_item_type() {
        // sigmaProp(Coll[((Int,Int,Int), Int)](((1,2,3), 4)) ==
        // ConcreteCollection(((1,2,3), 4))). The constant's left side is built at the tuple
        // type (`fromBoxedPairs`), as the in-script collection's is, so the item types agree
        // and the items compare (SANTA's tuple-values-3 reply, V1: true @91)
        assert_accepts(
            "0b1dd1930c4c48040404010204060883014c480404044c4804040402040608",
            true_prop(),
            91,
        );
    }

    #[test]
    fn an_any_typed_item_meets_the_three_gates() {
        // ByIndex((1,2,3), i) is typed `Any`. EQ against an Int and a comparison are refused
        // at parse, with and without the size bit (`ConstraintFailed` is no validation error)
        for tree_hex in [
            "0b0fd193b2480404040204060400000402",
            "00d193b2480404040204060400000402",
            "0b0fd191b2480404040204060400000402",
            "00d191b2480404040204060400000402",
        ] {
            let bytes = base16::decode(tree_hex.as_bytes()).unwrap();
            // read as a box reads it: `sigma_parse_bytes` is not the strict reader for an
            // unsized tree
            let res = sigma_byte_reader::from_bytes(&bytes)
                .with_tree_version(ErgoTreeVersion::MAX_SCRIPT_VERSION, ErgoTree::sigma_parse);
            assert!(res.is_err(), "{tree_hex}");
        }
        // EQ of two `Any` operands and an arithmetic operation parse and throw at evaluation
        // ("Unknown type SAny"; no `Any` in `ArithOp.impls`): SANTA
        // `Tuple.non_pair_value_gates` G1-G3, and the equality's controls over a Coll[Int]
        assert_throws("0b1793b248040404020406040000b248040404020406040200");
        assert_throws("0b18d193b248040404020406040000b248040404020406040200");
        assert_throws("0b0e9ab2480404040204060400000402");
        assert_accepts(
            "0b1393b21003020406040000b21003020406040200",
            Value::Boolean(false),
            83,
        );
        let false_prop = Value::SigmaProp(alloc::boxed::Box::new(SigmaProp::new(
            SigmaBoolean::TrivialProp(false),
        )));
        assert_accepts(
            "0b14d193b21003020406040000b21003020406040200",
            false_prop,
            98,
        );
    }

    #[test]
    fn a_tuple_does_not_pass_a_type_check_site() {
        // SizeOf(ConcreteCollection[(Int,Int,Int)](t, t)): each item is checked against its
        // type ("Unsupported tuple type"), while the same collection as data answers (above)
        assert_throws("0b15b18302480404044804040402040648040404020406");
        // the same collection of PAIRS answers: SizeOf(Coll((1, 2), (3, 4))) (SANTA gates C6)
        assert_accepts("0b10b1830258860204020404860204060408", int(2), 84);
        // Fold(Coll[Int](1), t, (p: ((Int,Int,Int), Int)) => p._1): the zero is checked; and
        // SANTA gates G9, the same over Coll[Int](1, 2, 3) inside SizeOf
        assert_throws("0b17b010010248040404020406d901014c480404048c720101");
        assert_throws("0b1ab1b0100302040648040404020406d901014c480404048c720101");
        // (1,2,3).map((x: Int) => x + 1), the METHOD: the receiver's item type is `Any`
        assert_throws("0b14dc0c034804040402040601d90101049a72010402");
    }
}

/// Where the node's cost lands. sigmastate evaluates and checks item 0, then item 1, and only
/// then adds the cost (v6.0.6 `values.scala:824-836`). Measured on the sigma-state 6.0.7 jars
/// (a profiling evaluator on the expression's bytes, its accumulator read also at a throw)
#[cfg(test)]
#[cfg(feature = "arbitrary")]
mod charge_order {
    use ergotree_ir::mir::value::Value;

    use crate::eval::test_util::{assert_trees, eval_tree_err};
    use crate::eval::EvalError;

    #[test]
    fn a_tuple_adds_its_cost_after_its_items() {
        // T0 `(1, 2)`: the two constants and the node
        assert_trees(&[(
            "0b06860204020404",
            Some((Value::pair(Value::Int(1), Value::Int(2)), 25)),
        )]);
        // T1 `(1 / 0, 2)`, T2 `(1, 1 / 0)`: an `ArithmeticException` on the JVM at 25 and at
        // 30. The counter holds what was evaluated and not the node
        for (tree_hex, cost) in [
            ("0b0986029d040204000404", 25),
            ("0b09860204029d04020400", 30),
        ] {
            let (e, stopped_at) = eval_tree_err(tree_hex);
            assert!(
                matches!(e, EvalError::ArithmeticException(_)),
                "{tree_hex}: {e:?}"
            );
            assert_eq!(stopped_at, cost, "{tree_hex}");
        }
    }
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::test_util::{eval_out_wo_ctx, try_eval_out};
    use ergotree_ir::chain::context::Context;
    use ergotree_ir::mir::expr::Expr;
    use ergotree_ir::mir::global_vars::GlobalVars;
    use sigma_test_util::force_any_val;

    #[test]
    fn eval() {
        let e1: Expr = 1i64.into();
        let e2: Expr = GlobalVars::Height.into();
        let exprs = vec![e1, e2];
        let tuple: Expr = Tuple::new(exprs).unwrap().into();
        let res = eval_out_wo_ctx::<Value>(&tuple);
        assert!(matches!(res, Value::Pair(_)));
    }

    #[test]
    fn eval_rejects_non_pair_tuple() {
        // sigma-state rejects tuples with length != 2 ("Invalid tuple", JVM
        // consensus). A flat arity-3 tuple is constructible here (TupleItems
        // allows 2..=255), but must be rejected at eval to match the JVM — else
        // sigma-rust would accept a spend the JVM rejects.
        let e1: Expr = 1i64.into();
        let e2: Expr = 2i64.into();
        let e3: Expr = 3i64.into();
        let tuple: Expr = Tuple::new(vec![e1, e2, e3]).unwrap().into();
        let ctx = force_any_val::<Context>();
        let res = try_eval_out::<Value>(&tuple, &ctx);
        assert!(
            res.is_err(),
            "arity-3 Tuple must be rejected at eval, got {res:?}"
        );
    }

    #[test]
    fn eval_rejects_arity3_tuple_item_constant() {
        use crate::eval::test_util::try_eval_out_wo_ctx;
        use ergotree_ir::mir::constant::{Constant, Literal};
        use ergotree_ir::types::stuple::{STuple, TupleItems};
        use ergotree_ir::types::stype::SType;

        // SANTA Tuple.checkType_unsupported (inline): a pair Tuple whose item0 is
        // an arity-3 (Bool,Bool,Bool) tuple CONSTANT. The JVM `checkType`s each
        // Tuple item (values.scala:801/804) → arity != 2 → "Unsupported tuple
        // type" (SType.scala:200-202). sigma-rust models flat N-ary tuples, so the
        // value is representable; it must be rejected at eval to match the JVM.
        let triple = Constant {
            tpe: SType::STuple(STuple::triple(
                SType::SBoolean,
                SType::SBoolean,
                SType::SBoolean,
            )),
            v: Literal::Tup(
                TupleItems::try_from(vec![
                    Literal::Boolean(true),
                    Literal::Boolean(true),
                    Literal::Boolean(true),
                ])
                .unwrap(),
            ),
        };
        let tuple: Expr = Tuple::new(vec![Expr::Const(triple), 1i32.into()])
            .unwrap()
            .into();
        assert!(
            try_eval_out_wo_ctx::<Value>(&tuple).is_err(),
            "arity-3 tuple item constant must be rejected at eval (Unsupported tuple type)"
        );
    }
}
