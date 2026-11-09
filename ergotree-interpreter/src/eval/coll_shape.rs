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
