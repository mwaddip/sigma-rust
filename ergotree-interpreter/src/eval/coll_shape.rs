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
