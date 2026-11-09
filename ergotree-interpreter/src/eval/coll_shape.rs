//! Collection operations as sigmastate runs them on each JVM collection class (v6.0.7
//! `CollsOverArrays.scala`): the class a collection's producer built decides how some operations
//! answer

#[cfg(test)]
#[cfg(feature = "arbitrary")]
#[allow(clippy::unwrap_used, clippy::panic)]
mod producers {
    use alloc::vec::Vec;

    use ergotree_ir::mir::byte_array_to_bigint::ByteArrayToBigInt;
    use ergotree_ir::mir::byte_array_to_long::ByteArrayToLong;
    use ergotree_ir::mir::calc_blake2b256::CalcBlake2b256;
    use ergotree_ir::mir::calc_sha256::CalcSha256;
    use ergotree_ir::mir::coll_value::CollValue;
    use ergotree_ir::mir::collection::Collection;
    use ergotree_ir::mir::constant::Constant;
    use ergotree_ir::mir::decode_point::DecodePoint;
    use ergotree_ir::mir::expr::Expr;
    use ergotree_ir::mir::subst_const::SubstConstants;
    use ergotree_ir::mir::tuple::Tuple;
    use ergotree_ir::mir::value::Value;
    use ergotree_ir::mir::xor::Xor;
    use ergotree_ir::types::stuple::STuple;
    use ergotree_ir::types::stype::SType;

    use crate::eval::test_util::try_eval_out_wo_ctx;

    fn eval(e: &Expr) -> Value<'static> {
        try_eval_out_wo_ctx::<Value>(e).unwrap()
    }

    /// `bytes` as a `ConcreteCollection` of byte constants (a generic collection on the JVM), and
    /// as one `Coll[Byte]` constant (a specialised one)
    fn both_forms(bytes: &[i8]) -> [Expr; 2] {
        let items = bytes.iter().map(|b| Expr::Const((*b).into())).collect();
        [
            Collection::new(SType::SByte, items).unwrap().into(),
            Expr::Const(bytes.to_vec().into()),
        ]
    }

    #[test]
    fn concrete_collections_are_generic() {
        // `ConcreteCollection.eval` builds its result with `fromArray` at an abstract type
        // (`values.scala:897`): generic for every item type, a `PairOfCols` for pairs
        let [bytes, _] = both_forms(&[1, 2]);
        let v = eval(&bytes);
        assert!(
            matches!(
                v,
                Value::Coll(CollValue::Generic {
                    elem_tpe: SType::SByte,
                    ..
                })
            ),
            "{v:?}"
        );
        let bools: Expr = Collection::new(
            SType::SBoolean,
            [true, false].map(|b| Expr::Const(b.into())).to_vec(),
        )
        .unwrap()
        .into();
        let v = eval(&bools);
        assert!(
            matches!(
                v,
                Value::Coll(CollValue::Generic {
                    elem_tpe: SType::SBoolean,
                    ..
                })
            ),
            "{v:?}"
        );
        let pair = |a: i32, b: i32| -> Expr {
            Tuple::new([a, b].map(|i| Expr::Const(i.into())).to_vec())
                .unwrap()
                .into()
        };
        let pairs: Expr = Collection::new(
            SType::STuple(STuple::pair(SType::SInt, SType::SInt)),
            vec![pair(1, 2), pair(3, 4)],
        )
        .unwrap()
        .into();
        let v = eval(&pairs);
        assert!(
            matches!(&v, Value::Coll(CollValue::Pairs { ls, rs })
                if matches!(**ls, CollValue::Generic { elem_tpe: SType::SInt, .. })
                    && matches!(**rs, CollValue::Generic { elem_tpe: SType::SInt, .. })),
            "{v:?}"
        );
    }

    #[test]
    fn generic_bytes_reach_byte_consumers() {
        // sigmastate's byte consumers read the bytes with `toArray`, whatever the class
        let bytes: Vec<i8> = (1..=8).collect();
        let consumers: [fn(Expr) -> Expr; 5] = [
            |input| {
                CalcBlake2b256 {
                    input: input.into(),
                }
                .into()
            },
            |input| {
                CalcSha256 {
                    input: input.into(),
                }
                .into()
            },
            |input| {
                Expr::ByteArrayToLong(
                    ByteArrayToLong {
                        input: input.into(),
                    }
                    .into(),
                )
            },
            |input| {
                Expr::ByteArrayToBigInt(
                    ByteArrayToBigInt {
                        input: input.into(),
                    }
                    .into(),
                )
            },
            |input| {
                Xor {
                    left: input.clone().into(),
                    right: input.into(),
                }
                .into()
            },
        ];
        for consumer in consumers {
            let [generic, specialised] = both_forms(&bytes);
            assert_eq!(eval(&consumer(generic)), eval(&consumer(specialised)));
        }
        // the generator's compressed encoding
        let generator: Vec<i8> = base16::decode(
            "0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798".as_bytes(),
        )
        .unwrap()
        .into_iter()
        .map(|b| b as i8)
        .collect();
        let decode = |input: Expr| -> Expr {
            DecodePoint {
                input: input.into(),
            }
            .into()
        };
        let [generic, specialised] = both_forms(&generator);
        assert_eq!(eval(&decode(generic)), eval(&decode(specialised)));
        // a template's bytes: a tree whose only constant is `Int 1`, as a script
        let template: Vec<i8> = base16::decode("100104027300".as_bytes())
            .unwrap()
            .into_iter()
            .map(|b| b as i8)
            .collect();
        let subst = |script_bytes: Expr| -> Expr {
            Expr::SubstConstants(
                SubstConstants {
                    script_bytes: script_bytes.into(),
                    positions: Expr::Const(Constant::from(vec![0i32])).into(),
                    new_values: Expr::Const(Constant::from(vec![7i32])).into(),
                }
                .into(),
            )
        };
        let [generic, specialised] = both_forms(&template);
        assert_eq!(eval(&subst(generic)), eval(&subst(specialised)));
    }
}

/// SANTA's eval-tier rows (corpus `6ca4ae2`)
#[cfg(test)]
#[cfg(feature = "arbitrary")]
#[allow(clippy::unwrap_used, clippy::panic)]
pub(crate) mod santa_rows {
    use ergotree_ir::chain::context::Context;
    use ergotree_ir::ergo_tree::ErgoTreeVersion;
    use ergotree_ir::mir::coll_value::CollValue;
    use ergotree_ir::mir::expr::Expr;
    use ergotree_ir::mir::value::Value;
    use ergotree_ir::serialization::sigma_byte_reader::{from_bytes, SigmaByteRead};
    use ergotree_ir::serialization::SigmaSerializable;
    use ergotree_ir::types::stype::SType;
    use sigma_test_util::force_any_val;

    use crate::eval::{Env, EvalError, Evaluable};

    /// A SANTA eval row: a tree of version 0 or 3 whose root is the expression, read without the
    /// tree's root-type check and evaluated at activated version 3; its value and JIT cost
    pub(crate) fn eval_row(tree_hex: &str) -> (Result<Value<'static>, EvalError>, u64) {
        let bytes = base16::decode(tree_hex.as_bytes()).unwrap();
        let (version, body) = match bytes[0] {
            0x00 => (ErgoTreeVersion::V0, &bytes[1..]),
            0x0b => {
                assert!(bytes[1] < 0x80, "{tree_hex}");
                (ErgoTreeVersion::V3, &bytes[2..])
            }
            header => panic!("header {header:#x}: {tree_hex}"),
        };
        let expr = from_bytes(body)
            .with_tree_version(version, Expr::sigma_parse)
            .unwrap();
        let mut ctx = force_any_val::<Context>();
        ctx.activated_script_version_byte = 3;
        ctx.tree_version.set(version);
        ctx.jit_cost.set(0);
        let mut env = Env::empty();
        let res = expr.eval(&mut env, &ctx).map(|v| v.to_static());
        (res, ctx.jit_cost_value())
    }

    /// A version-3 row of the expression `body`
    pub(crate) fn v3(body: &str) -> String {
        format!("0b{:02x}{body}", body.len() / 2)
    }

    pub(crate) fn assert_row(tree_hex: &str, value: Value<'static>, cost: u64) {
        let (res, c) = eval_row(tree_hex);
        assert_eq!(res.unwrap(), value, "{tree_hex}");
        assert_eq!(c, cost, "{tree_hex}");
    }

    /// The collection a row's expression evaluates to
    fn coll_of(body: &str) -> CollValue<'static> {
        match eval_row(&v3(body)).0.unwrap() {
            Value::Coll(c) => c,
            other => panic!("{body}: not a collection: {other:?}"),
        }
    }

    fn generic_bytes(c: &CollValue) -> bool {
        matches!(
            c,
            CollValue::Generic {
                elem_tpe: SType::SByte,
                ..
            }
        )
    }

    #[test]
    fn built_in_producers() {
        // SANTA `Coll.producer_specialisation_witness`: the four fixed-width `toBytes` are
        // generic (H3), `BigInt`/`UnsignedBigInt.toBytes` specialised (H4, H12), every `toBits`
        // a specialised `Coll[Boolean]` (H5)
        assert!(generic_bytes(&coll_of("db0406040a")), "Int.toBytes");
        assert!(generic_bytes(&coll_of("db0506050a")), "Long.toBytes");
        assert!(
            matches!(coll_of("db0606060105"), CollValue::Bytes(_)),
            "BigInt.toBytes"
        );
        assert!(
            matches!(coll_of("db0906090105"), CollValue::Bytes(_)),
            "UnsignedBigInt.toBytes"
        );
        assert!(
            matches!(coll_of("db0407040a"), CollValue::Booleans(_)),
            "Int.toBits"
        );
        // `SELF.creationInfo._2`, the node and the method: generic (H2), unlike `SELF.bytes`
        assert!(generic_bytes(&coll_of("8cc7a702")), "ExtractCreationInfo");
        assert!(generic_bytes(&coll_of("8cdb6306a702")), "Box.creationInfo");
        // `indices` is a specialised `Coll[Int]` from any receiver (`CollsOverArrays.scala:68`)
        assert!(
            matches!(coll_of("db0c0e83020202010202"), CollValue::Ints(_)),
            "indices"
        );
    }

    #[test]
    fn producer_rows() {
        // B3/B4: a `ConcreteCollection` of bytes returns the mistyped default
        assert_row(
            "0b0fdc0c028302020201020202040a0101",
            Value::Boolean(true),
            74,
        );
        assert_row("0b0fdc0c02830202020102020204000101", Value::Byte(1), 74);
        // H3/H7: so does a fixed-width `toBytes`
        assert_row("0b0edc0c02db0406040a0204c6010101", Value::Boolean(true), 58);
        assert_row("0b0edc0c02db0406040a0204c6010207", Value::Byte(7), 58);
    }

    #[test]
    fn item_store_primitive() {
        // SANTA `Coll.item_store_type_check` #0-#9: `updated`/`patch` store into the receiver's
        // array, and a generic collection over a primitive array refuses an item of another type,
        // as the specialised one does
        for row in [
            "0b0bdc0c148503050204000402",             // #0 Coll[Boolean] updated Int
            "0b10dc0c1385030503040083010404020402",   // #1 Coll[Boolean] patch Int
            "0b11dc0c148303020201020202030204000101", // #2 Coll[Byte] updated Boolean
            "0b14dc0c138303020201020202030304008501010402", // #3 Coll[Byte] patch Boolean
            "0b0fdc0c14830203030203040204000101",     // #4 Coll[Short]
            "0b12dc0c13830203030203040304008501010402", // #5
            "0b11dc0c148303040400040204040204000101", // #6 Coll[Int]
            "0b14dc0c138303040400040204040304008501010402", // #7
            "0b0fdc0c14830205050205040204000101",     // #8 Coll[Long]
            "0b12dc0c13830205050205040304008501010402", // #9
        ] {
            let (res, _) = eval_row(row);
            assert!(res.is_err(), "{row}: {res:?}");
        }
        // #10: a reference array takes a `Coll` of another item type
        let (res, cost) = eval_row("0b13dc0c1483011083010404020204008301050502");
        assert!(res.is_ok(), "{res:?}");
        assert_eq!(cost, 100);
    }

    #[test]
    fn receiver_class_results() {
        let is_pairs = |c: &CollValue| matches!(c, CollValue::Pairs { .. });
        let generic_pairs = |c: &CollValue| {
            matches!(
                c,
                CollValue::Generic {
                    elem_tpe: SType::STuple(_),
                    ..
                }
            )
        };
        // `map`/`flatMap` are generic, except INTO pairs from a non-pair receiver, which goes
        // through `builder.fromArray`'s pair arm (SANTA `Coll.pair_shape_restored` Y1, Y3); `map`
        // OVER a `PairOfCols` builds a generic collection directly (`PairOfCols.map`, M3)
        assert!(
            generic_bytes(&coll_of("dc0c030e02010201d90101027201")),
            "map of Bytes"
        );
        assert!(
            generic_pairs(&coll_of("dc0c030c58020204060801d90101587201")),
            "map over pairs"
        );
        assert!(
            is_pairs(&coll_of("dc0c0383030404000402040401d9010104860272017201")),
            "map into pairs"
        );
        assert!(
            is_pairs(&coll_of(
                "dc0c0f83030404000402040401d9010104830158860272017201"
            )),
            "flatMap into pairs"
        );
        // `reverse`, `slice`, `filter` keep the receiver's class
        assert!(
            matches!(coll_of("db0c1e0e020102"), CollValue::Bytes(_)),
            "reverse of Bytes"
        );
        assert!(
            generic_bytes(&coll_of("db0c1e83020202010202")),
            "reverse of generic"
        );
        assert!(
            matches!(coll_of("b40e02010204000402"), CollValue::Bytes(_)),
            "slice of Bytes"
        );
        assert!(
            generic_bytes(&coll_of("b48302020201020204000402")),
            "slice of generic"
        );
        assert!(
            matches!(coll_of("b50e020102d90101020101"), CollValue::Bytes(_)),
            "filter of Bytes"
        );
        assert!(
            generic_bytes(&coll_of("b583020202010202d90101020101")),
            "filter of generic"
        );
        // ... through `builder.fromArray`, so a generic collection of pairs comes back a
        // `PairOfCols` (Y4)
        assert!(
            is_pairs(&coll_of(
                "db0c1edc0c0383025886020402040486020406040801d90101587201"
            )),
            "reverse of mapped pairs"
        );
        // `zip` keeps both sides as they are
        assert!(
            matches!(coll_of("dc0c1d0e0301020301830304040004020404"), CollValue::Pairs { ls, rs }
                if matches!(*ls, CollValue::Bytes(_))
                    && matches!(*rs, CollValue::Generic { elem_tpe: SType::SInt, .. })),
            "zip"
        );
        // `updateMany` and the `Filter` node rebuild a `PairOfCols`'s sides generic (Y8, Y11)
        let generic_ls =
            |c: CollValue| matches!(c, CollValue::Pairs { ls, .. } if generic_bytes(&ls));
        assert!(
            generic_ls(coll_of(
                "dc0c15dc0c1d0e030102030183030404000402040402830104040083013e04860202090412"
            )),
            "updateMany of a zip"
        );
        assert!(
            generic_ls(coll_of(
                "b5dc0c1d0e0301020301830304040004020404d901013e040101"
            )),
            "Filter of a zip"
        );
        // `SELF.tokens` is a generic collection of pairs (M14-M17)
        assert!(generic_pairs(&coll_of("db6308a7")), "SELF.tokens");
    }

    #[test]
    fn zip_keeps_sides_and_takes_shorter_length() {
        // `pairColl` slices the longer side to the shorter's length (v5+, `CollsOverArrays.scala:183-196`),
        // keeping its class: `Coll[Byte](1,2,3)` constant `.zip(Coll(7))`
        let z = coll_of("dc0c1d0e0301020301830104040e");
        assert_eq!(z.len(), 1);
        assert!(
            matches!(&z, CollValue::Pairs { ls, .. } if matches!(&**ls, CollValue::Bytes(b) if b.len() == 1)),
            "{z:?}"
        );
    }

    #[test]
    #[ignore = "Task 5: getOrElse follows the collection class"]
    fn mapped_pairs_rows() {
        // M3/M4: `map` over a collection of pairs gives a generic collection of pairs
        let pair = |a, b| Value::Tup([Value::Int(a), Value::Int(b)].into());
        assert_row(
            "0b19dc0c02dc0c030c58020204060801d9010158720102040a0101",
            Value::Boolean(true),
            99,
        );
        assert_row(
            "0b19dc0c02dc0c030c58020204060801d901015872010204000101",
            pair(1, 2),
            99,
        );
    }
}
