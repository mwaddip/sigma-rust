//! Collection operations as sigmastate runs them on each JVM collection class (v6.0.7
//! `CollsOverArrays.scala`): the class a collection's producer built decides how some operations
//! answer

use alloc::format;

use ergotree_ir::mir::coll_value::CollValue;
use ergotree_ir::mir::value::Value;
use ergotree_ir::types::stype::SType;

use super::coll_by_index::item_at;
use crate::eval::EvalError;

/// `Coll.getOrElse` as the receiver's JVM class runs it: a specialised collection's bridge unboxes
/// the default to its own item class, and a `PairOfCols`'s casts it to a pair, both before the
/// index is looked at (`CollsOverArrays.scala:26`, `:309`); a generic collection returns the item
/// or the default, whatever the default holds. Reached by the `getOrElse` method and by `ByIndex`
/// with a default
pub(crate) fn get_or_else<'ctx>(
    coll: &CollValue<'ctx>,
    index: i32,
    default: Value<'ctx>,
) -> Result<Value<'ctx>, EvalError> {
    let takes_default = match coll {
        CollValue::Bytes(_) => matches!(default, Value::Byte(_)),
        CollValue::Booleans(_) => matches!(default, Value::Boolean(_)),
        CollValue::Ints(_) => matches!(default, Value::Int(_)),
        CollValue::Pairs { .. } => matches!(&default, Value::Pair(_)),
        CollValue::Generic { .. } => true,
    };
    if !takes_default {
        return Err(EvalError::UnexpectedValue(format!(
            "getOrElse: a collection of {:?} held as {} cannot take the default {default:?}",
            coll.elem_tpe(),
            match coll {
                CollValue::Pairs { .. } => "a PairOfCols",
                _ => "a specialised array",
            }
        )));
    }
    Ok(item_at(coll, index).unwrap_or(default))
}

/// Scala's `==` on two boxed items (`BoxesRunTime.equals`): two integral numbers (`Byte`,
/// `Short`, `Int`, `Long`) are equal when their values are, whatever their widths; two pairs
/// compare item by item the same way (`Tuple2.equals`), and two options their values
/// (`Some.equals`); anything else by its own equality, so a
/// `Boolean` never equals a number, a `BigInt` never equals a small integer, and two collections
/// compare their item types first (`CollOverArray.equals`). SANTA
/// `Coll.prefix_cross_type_equality`
pub(crate) fn scala_eq(a: &Value<'_>, b: &Value<'_>) -> bool {
    fn integral(v: &Value<'_>) -> Option<i64> {
        match v {
            Value::Byte(x) => Some(i64::from(*x)),
            Value::Short(x) => Some(i64::from(*x)),
            Value::Int(x) => Some(i64::from(*x)),
            Value::Long(x) => Some(*x),
            _ => None,
        }
    }
    match (a, b) {
        (Value::Pair(l), Value::Pair(r)) => scala_eq(&l.0, &r.0) && scala_eq(&l.1, &r.1),
        // `Some.equals` compares the values the same way
        (Value::Opt(l), Value::Opt(r)) => match (l, r) {
            (Some(x), Some(y)) => scala_eq(x, y),
            (None, None) => true,
            _ => false,
        },
        _ => match (integral(a), integral(b)) {
            (Some(x), Some(y)) => x == y,
            _ => a == b,
        },
    }
}

/// The primitive item type of a collection's backing array, if it is a primitive array: a
/// specialised class's, or a generic collection's of `Boolean`, `Byte`, `Short`, `Int` or `Long`
/// (sigmastate builds those over `boolean[]`, `byte[]`, `short[]`, `int[]`, `long[]`)
fn primitive_array(coll: &CollValue<'_>) -> Option<SType> {
    match coll {
        CollValue::Bytes(_) => Some(SType::SByte),
        CollValue::Booleans(_) => Some(SType::SBoolean),
        CollValue::Ints(_) => Some(SType::SInt),
        CollValue::Generic { elem_tpe, .. }
            if matches!(
                elem_tpe,
                SType::SBoolean | SType::SByte | SType::SShort | SType::SInt | SType::SLong
            ) =>
        {
            Some(elem_tpe.clone())
        }
        _ => None,
    }
}

/// Whether `other` is held over the same primitive array as `coll`
pub(crate) fn same_primitive_array(coll: &CollValue<'_>, other: &CollValue<'_>) -> bool {
    match primitive_array(coll) {
        Some(t) => primitive_array(other) == Some(t),
        None => false,
    }
}

/// `Coll.startsWith` as the receiver's JVM class runs it (`CollsOverArrays.scala:44`, `:398-401`)
pub(crate) fn starts_with<'ctx>(
    coll: &CollValue<'ctx>,
    prefix: &CollValue<'ctx>,
) -> Result<bool, EvalError> {
    matches_at(coll, prefix, false)
}

/// `Coll.endsWith` as the receiver's JVM class runs it (`CollsOverArrays.scala:46`, `:403-406`)
pub(crate) fn ends_with<'ctx>(
    coll: &CollValue<'ctx>,
    suffix: &CollValue<'ctx>,
) -> Result<bool, EvalError> {
    matches_at(coll, suffix, true)
}

/// Whether `coll` starts (or, `at_end`, ends) with `other`: a specialised receiver casts the
/// argument's array to its own primitive array first, whatever the lengths, so another array is
/// an error (SANTA S1, S7); a `PairOfCols` compares side by side with another `PairOfCols`, the
/// first side first, so a `false` there never reaches the second, and with any other argument
/// casts that argument's array to `Object[]`, so a primitive one is an error; anything else
/// compares the items with Scala's `==` ([`scala_eq`])
fn matches_at<'ctx>(
    coll: &CollValue<'ctx>,
    other: &CollValue<'ctx>,
    at_end: bool,
) -> Result<bool, EvalError> {
    match (coll, other) {
        (CollValue::Bytes(_) | CollValue::Booleans(_) | CollValue::Ints(_), _)
            if !same_primitive_array(coll, other) =>
        {
            Err(EvalError::UnexpectedValue(format!(
                "a specialised collection of {:?} cannot compare with a collection of {:?}",
                coll.elem_tpe(),
                other.elem_tpe()
            )))
        }
        (CollValue::Pairs { ls, rs }, CollValue::Pairs { ls: ols, rs: ors }) => {
            Ok(matches_at(ls, ols, at_end)? && matches_at(rs, ors, at_end)?)
        }
        // with any other argument it compares the arrays, casting the argument's to `Object[]`
        // (`CollsOverArrays.scala:400`, `:405`): a primitive array fails, even an empty one
        (CollValue::Pairs { .. }, _) if primitive_array(other).is_some() => {
            Err(EvalError::UnexpectedValue(format!(
                "a collection of pairs cannot compare with a primitive collection of {:?}",
                other.elem_tpe()
            )))
        }
        _ => {
            let items = coll.as_vec();
            let other_items = other.as_vec();
            let Some(offset) = items.len().checked_sub(other_items.len()) else {
                return Ok(false);
            };
            let start = if at_end { offset } else { 0 };
            Ok(items
                .iter()
                .skip(start)
                .zip(other_items.iter())
                .all(|(x, y)| scala_eq(x, y)))
        }
    }
}

/// `Append` as the left side's JVM class runs it (`CollsOverArrays.scala:48-58`, `:379-382`,
/// `safeConcatArrays_v5`): an empty left side, unless a `PairOfCols`, returns the right side as it
/// is; a specialised left side casts the right side's array to its own, empty or not; a generic
/// one copies both into an array of its own item type, where a primitive array takes only the
/// same primitive array, even an empty one, and a reference array takes any reference array's
/// items whose class fits it (none to check when empty; [`CollValue::from_array`] refuses the
/// others), and the result keeps the LEFT's item type. A `PairOfCols` unzips the right side and
/// appends side by side. SANTA `Coll.append_runtime_item_type`, `Append.reference_array_store`,
/// `Coll.append_node_array_class`
pub(crate) fn append<'ctx>(
    left: &CollValue<'ctx>,
    right: &CollValue<'ctx>,
) -> Result<CollValue<'ctx>, EvalError> {
    let refuse = || {
        EvalError::UnexpectedValue(format!(
            "Append: a collection of {:?} cannot take the items of a collection of {:?}",
            left.elem_tpe(),
            right.elem_tpe()
        ))
    };
    match left {
        CollValue::Pairs { ls, rs } => {
            let (rls, rrs) = unzip(right).ok_or_else(refuse)?;
            Ok(CollValue::pairs(append(ls, &rls)?, append(rs, &rrs)?))
        }
        _ if left.is_empty() => Ok(right.clone()),
        CollValue::Bytes(_) | CollValue::Booleans(_) | CollValue::Ints(_) => {
            if !same_primitive_array(left, right) {
                return Err(refuse());
            }
            Ok(left.rebuild_like(left.as_vec().into_iter().chain(right.as_vec()).collect())?)
        }
        CollValue::Generic { elem_tpe, items } => {
            // `arraycopy` looks at the two ARRAYS first: a primitive one takes only the same
            // primitive one, a reference one no primitive one, whatever the lengths
            let arrays_fit = match primitive_array(left) {
                Some(_) => same_primitive_array(left, right),
                None => primitive_array(right).is_none(),
            };
            if !arrays_fit {
                return Err(refuse());
            }
            // then each item of a reference array is stored into the left's class
            let all = items.iter().cloned().chain(right.as_vec()).collect();
            Ok(CollValue::from_array(elem_tpe.clone(), all)?)
        }
    }
}

/// A collection of pairs as its two sides: a `PairOfCols`'s own, or a generic collection's items
/// split as `builder.unzip` splits them (`CollsOverArrays.scala:244-258`); `None` for a collection
/// of anything else (`tItem.tFst` throws)
fn unzip<'ctx>(coll: &CollValue<'ctx>) -> Option<(CollValue<'ctx>, CollValue<'ctx>)> {
    match coll {
        CollValue::Pairs { ls, rs } => Some(((**ls).clone(), (**rs).clone())),
        _ => match CollValue::from_array(coll.elem_tpe(), coll.as_vec()) {
            Ok(CollValue::Pairs { ls, rs }) => Some(((*ls).clone(), (*rs).clone())),
            _ => None,
        },
    }
}

#[cfg(test)]
#[cfg(feature = "arbitrary")]
mod typed_array_store {
    use ergotree_ir::mir::value::Value;

    use crate::eval::test_util::assert_trees;

    /// An `Int` at a cost, as an `assert_trees` row expects it; every tree's root is a `SizeOf`
    fn v(n: i32, cost: u64) -> Option<(Value<'static>, u64)> {
        Some((Value::Int(n), cost))
    }

    /// A single store into a `Coll[Unit]` is checked like any other: `updated` casts the item to
    /// `BoxedUnit`, `updateMany` and a pair's side store into a `BoxedUnit[]`, `Append` copies
    /// arrays (SANTA 6.0.7, the third reply: R4, R5, R6, X7, X8)
    #[test]
    fn a_single_store_into_a_unit_array_is_checked() {
        assert_trees(&[
            ("0b0db1dc0c14830162620204000402", None), // R4 `ConcreteCollection[Unit](()).updated(0, 1)`
            ("0b0fb1dc0c158301626202100100100102", None), // R5 `ConcreteCollection[Unit](()).updateMany(Coll(0), Coll[Int] constant (1))`
            ("0b0ab1b383016262830163a7", None), // R6 `ConcreteCollection[Unit](()) ++ Coll[Box](SELF)`
            (
                "0b25b1dc0c15dc0c1d83010404020183016262028301040400dc0c1d830104040201830104040e",
                None,
            ), // X7 `Coll(1).zip(CC[Unit](())).updateMany(Coll(0), Coll(1).zip(Coll(7)))`
            (
                "0b1ab1dc0c14dc0c1d8301040402018301626202040086020402040e",
                None,
            ), // X8 `Coll(1).zip(CC[Unit](())).updated(0, (1, 7))`
        ]);
    }

    /// `map` stores each result into an array of the STATIC result's class (the mapper
    /// expression's range), from a plain receiver and from a `PairOfCols` alike (v6.0.6
    /// `CollsOverArrays.scala:28-31`, `:317-323`). A mapper that is an `If` of two lambdas gets a
    /// value past the lambdas' own result checks, and the array refuses it; a `Tuple2[]` takes
    /// any pair (SANTA's third reply: P1-P4, X5, X6, X11)
    #[test]
    fn map_stores_into_an_array_of_the_static_result_class() {
        assert_trees(&[
            (
                "0b19b1addc0c1d10010201100102950100d9010158a7d901015882",
                None,
            ), // P1 `Coll(1).zip(Coll(1)).map(If(false, {p => SELF}, {p => groupGenerator}))`
            ("0b12b1ad100102950100d9010104a7d901010482", None), // P2 `Coll(1).map(If(false, {x => SELF}, {x => groupGenerator}))`
            (
                "0b21b1addc0c1d10010201100102950100d901015886020402a7d90101588602040282",
                v(1, 115),
            ), // P3 the pair receiver, `{p => (1, SELF)}` and `{p => (1, groupGenerator)}`
            (
                "0b1db1addc0c1d10010201100102950100d901015886020402a7d901015882",
                None,
            ), // P4 the pair receiver, `{p => (1, SELF)}` and `{p => groupGenerator}`
            ("0b15b1ad8301040402950100d901010462d90101040402", None), // X5 `Coll(1).map(If(false, {x => ()}, {x => 1}))`
            (
                "0b1eb1addc0c1d8301040402018301040402950100d901015862d90101580402",
                None,
            ), // X6 the same over `Coll(1).zip(Coll(1))`
            ("0b0cb1ad8301040402d901010462", v(1, 75)), // X11 `Coll(1).map({x => ()})`
        ]);
    }

    /// No array can be built for a type variable (`stypeToRType` throws), so nothing is stored
    /// under one (SANTA's third reply: R8, A2)
    #[test]
    fn a_type_variable_has_no_array() {
        assert_trees(&[
            ("0b17b1dc0c0f10010201d90101049501008300670154100102", None), // R8 `Coll(1).flatMap({x => If(false, Coll[T](), Coll[Int] constant (1))})`
            (
                "0b1cb1b3dc0c0f10010201d90101049501008300670154100102830163a7",
                None,
            ), // A2 R8 `++ Coll[Box](SELF)`
        ]);
    }

    /// `updated` and `patch` store into the receiver's array, which takes only items of its class;
    /// nothing compares item types (SANTA 6.0.7: `Coll.reference_store_ops`, the first reply's A, B and X
    /// rows)
    #[test]
    fn the_store_goes_by_the_arrays_class() {
        assert_trees(&[
            // `updated`
            ("0b0cb1dc0c14830163a702040082", None), // A1 `Coll[Box](SELF).updated(0, groupGenerator)`
            ("0b17b1dc0c14dc0c1d10010201830163a70204008602040282", None), // A2 `Coll(1).zip(Coll[Box](SELF)).updated(0, (1, groupGenerator))`
            (
                "0b17b1dc0c14dc0c1d10010201830163a702040086020402a7",
                v(1, 124),
            ), // A3 `Coll(1).zip(Coll[Box](SELF)).updated(0, (1, SELF))`
            ("0b0fb1dc0c144804040402040602040082", v(3, 59)), // A4 `(1,2,3).updated(0, groupGenerator)`
            ("0b0db1dc0c14830163a70204000402", None),         // A5 `Coll[Box](SELF).updated(0, 1)`
            // `patch`: only the patch's ITEMS reach the array
            ("0b11b1dc0c13830163a7030400830107820400", None), // B1 `Coll[Box](SELF).patch(0, Coll[GroupElement](g), 0)`
            ("0b10b1dc0c13830163a70304008300070400", v(1, 110)), // B2 `Coll[Box](SELF).patch(0, Coll[GroupElement](), 0)`
            (
                "0b19b1dc0c13830110830104040203040083011183010505020400",
                v(2, 150),
            ), // B3 `Coll[Coll[Int]](Coll(1)).patch(0, Coll[Coll[Long]](Coll(1L)), 0)`
            ("0b10b1dc0c13830163a70304001001020400", None), // B4 `Coll[Box](SELF).patch(0, Coll[Int] constant (1), 0)`
            ("0b15b1dc0c131003020406030400480404040e10120400", v(6, 70)), // B5 `Coll[Int] constant (1,2,3).patch(0, (7,8,9), 0)`
            (
                "0b19b1dc0c13830304040204040406030400480404040e10120400",
                v(6, 100),
            ), // B6 `ConcreteCollection[Int](1,2,3).patch(0, (7,8,9), 0)`
            ("0b13b1dc0c131003020406030400830104040e0400", v(4, 90)), // B7 `Coll[Int] constant (1,2,3).patch(0, ConcreteCollection[Int](7), 0)`
            ("0b11b1dc0c1310030204060304008300050400", v(3, 85)), // B8 `Coll[Int] constant (1,2,3).patch(0, ConcreteCollection[Long](), 0)`
            // other item classes and receivers (SANTA X1-X5)
            ("0b0eb1dc0c1483010606010102040082", None), // X1 `Coll[BigInt].updated(0, g)`
            ("0b0eb1dc0c14830128e3070402040082", None), // X2 `Coll[Option[Int]].updated(0, g)`
            ("0b15b1dc0c1483017001040400d9010104720102040082", None), // X3 `Coll[Int => Int].updated(0, g)`
            ("0b0bb1dc0c140e010102040082", None), // X4 `Coll[Byte] const.updated(0, g)`
            ("0b09b1dc0c14a402040082", None),     // X5 `INPUTS.updated(0, g)`
        ]);
    }

    /// The array is the collection VALUE's, whatever the static type says (SANTA
    /// `Coll.store_runtime_array_class`). K2-K4: the JVM by reading; Rust measured with blitzen on
    /// `e46ed668` and on the trial
    #[test]
    fn the_store_goes_by_the_runtime_array() {
        assert_trees(&[
        // the collection VALUE's item type decides, never the static type (SANTA X9, X11, X20, X21)
        ("0b16b1dc0c1495010083010404028301050502020400040a", None), // X9 `If(false, Coll(1), Coll(1L)).updated(0, 5)`
        ("0b11b1db0c1e95010083010404028301050502", v(1, 80)), // X11 `If(false, Coll(1), Coll(1L)).reverse`
        ("0b09b1b383006383010782", v(1, 86)), // X20 `Coll[Box]() ++ Coll[GroupElement](g)`
        ("0b10b1dc0c14b383006383010782020400a7", None), // X21 `(Coll[Box]() ++ Coll[GroupElement](g)).updated(0, SELF)`
        // decoded data is over a typed array too (K2, K3: the JVM by reading)
        ("0b2bb1dc0c1413010279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798020400a7", None), // K2 `Coll[GroupElement] constant (g).updated(0, SELF)`
        ("0b2bb1dc0c1413010279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f8179802040082", v(1, 59)), // K3 `Coll[GroupElement] constant (g).updated(0, groupGenerator)`
        // an `Any` array takes everything (K4: the JVM by reading)
        ("0b14b1dc0c1348040404020406030400830163a70400", v(4, 95)), // K4 `(1,2,3).patch(0, Coll[Box](SELF), 0)`
        ]);
    }

    /// `map`, `flatMap` and a pair collection's sides fill arrays of the result's classes (SANTA's first
    /// reply D rows, second reply G rows). K1: the JVM by reading, the mechanism of G3
    #[test]
    fn a_producer_stores_into_a_typed_array() {
        assert_trees(&[
        // `map`: the closure's result check comes first (D1, D7), a class-compatible lie passes (D2)
        ("0b0eb1ad100102d9010104950100a782", None), // D1 `Coll(1).map({x => If(false, SELF, groupGenerator)})`
        ("0b16b1ad100102d901010495010083010404028301050502", v(1, 90)), // D2 `Coll(1).map({x => If(false, Coll(1), Coll(1L))})`
        ("0b15b1addc0c1d10010201100102d9010158950100a782", None), // D7 `Coll(1).zip(Coll(1)).map({p => If(false, SELF, groupGenerator)})`
        ("0b11b1addc0c1d10010201100102d9010158a7", v(1, 80)), // D8 `Coll(1).zip(Coll(1)).map({p => SELF})`
        // `flatMap` copies each part's items into the result's array
        ("0b17b1dc0c0f10010201d9010104950100830163a783010782", None), // D3 `Coll(1).flatMap({x => If(false, Coll[Box](SELF), Coll[GroupElement](g))})`
        ("0b1fb1dc0c0f10010201d901010495010083011083010404028301118301050502", v(1, 163)), // D4 `Coll(1).flatMap({x => If(false, Coll[Coll[Int]](Coll(1)), Coll[Coll[Long]](Coll(1L)))})`
        // a pair's component is stored into its side's array (`fromBoxedPairs`)
        ("0b16b1ad100102d901010495010086020402a78602040282", None), // G1 `Coll(1).map({x => If(false, (1, SELF), (1, g))})`
        ("0b1db1addc0c1d10010201100102d901015895010086020402a78602040282", v(1, 115)), // G2 `Coll(1).zip(Coll(1)).map({p => If(false, (1, SELF), (1, g))})`
        ("0b20b1db0c1eaddc0c1d10010201100102d901015895010086020402a78602040282", None), // G3 `G2.reverse`
        ("0b28b1dc0c14addc0c1d10010201100102d901015895010086020402a7860204028202040086020402a7", v(1, 175)), // G4 `G2.updated(0, (1, SELF))`
        // a rebuild of a collection that hides a misfit (K1: the JVM by reading, as G3)
        ("0b22b1b4addc0c1d10010201100102d901015895010086020402a7860204028204000402", None), // K1 `G2.slice(0, 1)`
        ]);
    }

    /// `Append` copies ARRAYS (`System.arraycopy`), so an empty array of another kind is refused where
    /// an empty patch is not (SANTA `Coll.append_node_array_class`); unchanged by this round
    #[test]
    fn append_copies_arrays() {
        assert_trees(&[
            ("0b0ab1b3830163a783010782", None), // X8 `Coll[Box](SELF) ++ Coll[GroupElement](g)`
            ("0b09b1b3830163a7830007", v(1, 86)), // X14 `Coll[Box](SELF) ++ Coll[GroupElement]()`
            ("0b0ab1b31003020406830005", None), // X17 `Coll[Int] const (1,2,3) ++ ConcreteCollection[Long]()`
            ("0b0fb1b310030204068603040e04100412", None), // X18 `Coll[Int] const (1,2,3) ++ (7,8,9)`
            ("0b0cb1b31003020406830104040e", v(4, 66)), // X19 `Coll[Int] const (1,2,3) ++ ConcreteCollection[Int](7)`
        ]);
    }
}

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

    pub(crate) fn assert_errs(tree_hex: &str) {
        let (res, _) = eval_row(tree_hex);
        assert!(res.is_err(), "{tree_hex}: {res:?}");
    }

    fn pair(a: i32, b: i32) -> Value<'static> {
        Value::pair(Value::Int(a), Value::Int(b))
    }

    #[test]
    fn get_or_else_specialised() {
        // SANTA `Coll.specialisation_getorelse`, `Coll.runtime_shapes_methodcall` B,
        // `Coll.producer_specialisation_witness`: a specialised receiver refuses a default of
        // another type, at any index
        for row in [
            "0b0bdc0c020d030502040a0402",           // Z1
            "0b0bdc0c020d03050204000402",           // Z2
            "0b10dc0c02db0c0e0e0301020302040a0101", // I1
            "0b10dc0c02db0c0e0e030102030204000101", // I2
            "0b0cdc0c020e02010202040a0101",         // B1
            "0b0cdc0c020e0201020204000101",         // B2
            "0b0edc0c02db0407040a0204c6010402",     // H5
            "0b0fdc0c02db06060601050204c6010101",   // H4
            "0b0bdc0c02c3a70204c6010101",           // H1
            "0b0bdc0c02c2a70204c6010101",           // H8
            "0b0cdc0c02cbc3a70204c6010101",         // H9
            "0b0fdc0c02db09060901050204c6010101",   // H12
        ] {
            assert_errs(row);
        }
        assert_row("0b0bdc0c0285030502040a0402", Value::Int(1), 79); // Z3
        assert_row("0b0bdc0c028503050204000402", Value::Boolean(true), 79); // Z4
        assert_row("0b0bdc0c020d030502040a0100", Value::Boolean(false), 49); // Z5
        assert_row(
            "0b11dc0c0283030404000402040402040a0101",
            Value::Boolean(true),
            79,
        ); // I3
        assert_row("0b11dc0c028303040400040204040204000101", Value::Int(0), 79); // I4
        assert_row("0b10dc0c02db0c0e0e0301020302040a0412", Value::Int(9), 75); // I5
        assert_row("0b0ddc0c028cc7a7020204c6010101", Value::Boolean(true), 80); // H2
        assert_row("0b0fdc0c02db09060901050204c6010207", Value::Byte(7), 58); // H13
    }

    #[test]
    fn get_or_else_pairs() {
        // a `PairOfCols` refuses a default that is not a pair, at any index; a generic collection
        // of pairs (after `map`) does not (SANTA `Coll.pair_shape_map`, `Coll.pair_shape_restored`)
        for row in [
            "0b0fdc0c020c58020204060802040a0101", // M1
            "0b0fdc0c020c5802020406080204000101", // M2
            "0b0edc0c0283015858020402040a0101",   // A1
            "0b0edc0c028301585802040204000101",   // A2
            "0b1fdc0c02dc0c0383030404000402040401d901010486027201720102040a0101", // Y1
            "0b22dc0c02dc0c0f83030404000402040401d901010483015886027201720102040a0101", // Y3
            "0b24dc0c02db0c1edc0c0383025886020402040486020406040801d9010158720102040a0101", // Y4
        ] {
            assert_errs(row);
        }
        assert_row("0b13dc0c020c58020204060802040a860204120412", pair(9, 9), 69); // M8
        assert_row(
            "0b1ddc0c02dc0c030c58020204060801d9010158720102040a860204120412",
            pair(9, 9),
            119,
        ); // M9
        assert_row(
            "0b21dc0c02dc0c0383025886020402040486020406040801d9010158720102040a0101",
            Value::Boolean(true),
            164,
        ); // M7
        assert_row(
            "0b23dc0c02dc0c0383030404000402040401d901010486027201720102040a860204120412",
            pair(9, 9),
            219,
        ); // Y2
    }

    #[test]
    fn by_index_default_four_way() {
        // SANTA `ByIndex.default_specialisation`: below tree version 3 the default reaches the
        // collection even in range; from 3 only out of range
        for row in [
            "00b20e030102030400010101",       // N1
            "00b20e030102030412010101",       // N2
            "0b0bb20e030102030412010101",     // N8
            "0b0db20c580202040608040a010101", // M5
        ] {
            assert_errs(row);
        }
        assert_row("00b28303020201020202030400010101", Value::Byte(1), 75); // N3
        assert_row("00b28303020201020202030412010101", Value::Boolean(true), 75); // N4
        assert_row("00b20e030102030400010207", Value::Byte(1), 45); // N5
        assert_row("00b28303020201020202030412010207", Value::Byte(7), 75); // N6
        assert_row("0b0bb20e030102030400010101", Value::Byte(1), 40); // N7
        assert_row("0b0fb28303020201020202030400010101", Value::Byte(1), 70); // N9
        assert_row(
            "0b0fb28303020201020202030412010101",
            Value::Boolean(true),
            75,
        ); // N10
        assert_row("0b0bb20e030102030400010207", Value::Byte(1), 40); // N11
        assert_row("0b0fb28303020201020202030412010207", Value::Byte(7), 75); // N12
        assert_row(
            "0b17b2dc0c030c58020204060801d90101587201040a010101",
            Value::Boolean(true),
            95,
        ); // M6
    }

    #[test]
    fn prefix_suffix_per_class() {
        // SANTA `Coll.specialisation_prefix_suffix`: a specialised receiver refuses an argument
        // over another array, even an empty one; a generic receiver compares with Scala's `==`
        for row in [
            "0b16dc0c1f0e030102030195010083010202018301050502", // S1
            "0b12dc0c1f0d0305019501008501018301050502",         // S3
            "0b19dc0c1fdb0c0e0e030102030195010083010404008301050502", // S5
            "0b12dc0c1f0e0301020301950100830002830005",         // S7
            "0b16dc0c200e030102030195010083010202018301050502", // E1
        ] {
            assert_errs(row);
        }
        let t = Value::Boolean(true);
        let f = Value::Boolean(false);
        assert_row(
            "0b1adc0c1f8303020201020202030195010083010202018301050502",
            t.clone(),
            90,
        ); // S2
        assert_row("0b12dc0c1f850305019501008501018301050502", f.clone(), 90); // S4
        assert_row(
            "0b1adc0c1f8303040400040204040195010083010404008301050502",
            f.clone(),
            90,
        ); // S6
        assert_row(
            "0b16dc0c1f83030202010202020301950100830002830005",
            t.clone(),
            85,
        ); // S8
        assert_row("0b0edc0c1f0e03010203018301020201", t.clone(), 45); // S9
        assert_row("0b12dc0c1f830302020102020203018301020201", t.clone(), 75); // S10
        assert_row(
            "0b1adc0c208303020201020202030195010083010202018301050502",
            f,
            90,
        ); // E2
        assert_row("0b0edc0c200e03010203018301020203", t, 45); // E3
    }

    #[test]
    fn prefix_suffix_pairs() {
        // SANTA `Coll.specialisation_pair_prefix`, `Coll.pair_shape_map`, `Coll.pair_shape_restored`:
        // a `PairOfCols` compares side by side with another, so its specialised side refuses
        for row in [
            "0b30dc0c1fdc0c1d0e030102030183030404000402040401dc0c1d9501008301020201830105050201830304040004020404", // P1
            "0b30dc0c20dc0c1d0e030102030183030404000402040401dc0c1d9501008301020201830105050201830304040004020404", // P5
        ] {
            assert_errs(row);
        }
        let t = Value::Boolean(true);
        let f = Value::Boolean(false);
        assert_row("0b34dc0c1fdc0c1d8303020201020202030183030404000402040401dc0c1d9501008301020201830105050201830304040004020404", t.clone(), 190); // P2
        assert_row("0b2cdc0c1fdc0c1d0e030102030183030404000402040401dc0c1d83030202010202020301830304040004020404", t.clone(), 155); // P3
        assert_row("0b30dc0c1fdc0c1d8303020201020202030183030404000402040401dc0c1d83030202010202020301830304040004020404", t.clone(), 185); // P4
        assert_row("0b34dc0c20dc0c1d8303020201020202030183030404000402040401dc0c1d9501008301020201830105050201830304040004020404", f.clone(), 190); // P6
        assert_row(
            "0b1edc0c1fdc0c030c58020204060801d9010158720101830158860204020404",
            t.clone(),
            115,
        ); // M12
        assert_row(
            "0b1edc0c1f0c58020204060801dc0c0383015886020402040401d90101587201",
            t.clone(),
            105,
        ); // M13
        assert_row("0b43dc0c1fdc0c15dc0c1d0e030102030183030404000402040402830104040083013e0486020209041201dc0c1d9501008301020201830105050201830304040004020404", f, 256); // Y8
        assert_row("0b38dc0c1fb5dc0c1d0e0301020301830304040004020404d901013e04010101dc0c1d9501008301020201830105050201830304040004020404", t, 216);
        // Y11
    }

    #[test]
    fn prefix_suffix_runtime_item_type() {
        // SANTA `Coll.prefix_suffix_runtime_item_type`: a mismatched item type is invisible to
        // a generic receiver
        let t = Value::Boolean(true);
        let f = Value::Boolean(false);
        assert_row(
            "0b1adc0c1f9501008302040402040483020505020504018301040402",
            t.clone(),
            85,
        );
        assert_row(
            "0b1adc0c209501008302040402040483020505020504018301040404",
            t.clone(),
            85,
        );
        assert_row(
            "0b18dc0c1f830204040204040195010083010404028301050502",
            t.clone(),
            85,
        );
        assert_row(
            "0b14dc0c1f9501008301040402830005018301040402",
            f.clone(),
            75,
        );
        assert_row("0b14dc0c209501008301040402830005018301040402", f, 75);
        assert_row(
            "0b18dc0c1f950100830204040204048302050502050401830004",
            t.clone(),
            80,
        );
        assert_row(
            "0b18dc0c20950100830204040204048302050502050401830004",
            t.clone(),
            80,
        );
        assert_row("0b12dc0c1f950100830104040283000501830004", t, 70);
    }

    #[test]
    fn prefix_cross_type_equality() {
        // SANTA `Coll.prefix_cross_type_equality`: Scala's `==` widens every integral box, never
        // a `BigInt`, and never inside a nested collection
        let t = Value::Boolean(true);
        let f = Value::Boolean(false);
        assert_row(
            "0b18dc0c1f830203030203040195010083010303028301040402",
            t.clone(),
            85,
        ); // J1
        assert_row("0b10dc0c1f83020303020304018301030302", t.clone(), 70); // J2
        assert_row(
            "0b18dc0c1f830106060101019501008301060601018301050502",
            f.clone(),
            80,
        ); // J3
        assert_row("0b10dc0c1f83010606010101830106060101", t.clone(), 65); // J4
        assert_row(
            "0b1fdc0c1f83011083010404020195010083011083010404028301118301050502",
            f.clone(),
            120,
        ); // J5
        assert_row("0b14dc0c1f8301108301040402018301108301040402", t, 105); // J6
        assert_row(
            "0b1fdc0c2083011083010404020195010083011083010404028301118301050502",
            f,
            120,
        ); // J7
    }

    #[test]
    fn append_rows() {
        // SANTA `Coll.append_runtime_item_type`, `Append.reference_array_store`
        for row in [
            "0b17b395010083020404020404830205050205048301040406", // #0, left lies
            "0b17b383010404069501008302040402040483020505020504", // #1, right lies
            "0b11b383010404069501008301040402830005", // #3, an empty right of another width
            "0b10b3950100830163a783010782830163a7",   // #5, a Box into GroupElements
            "0b1fb3950100830158860204020402830159860205020502830158860204040404", // #6, pairs
            "0b10b38301078295010083010782830163a7",   // K2, one Box
        ] {
            assert_errs(row);
        }
        let coll = |t: SType, items: Vec<Value<'static>>| Value::Coll(CollValue::generic(t, items));
        // #2: an empty left side returns the right side as it is
        assert_row(
            "0b11b395010083010404028300058301040406",
            coll(SType::SInt, vec![Value::Int(3)]),
            82,
        );
        // #4: `Coll` items into a `Coll` array, keeping the LEFT's runtime item type
        assert_row(
            "0b1cb3950100830110830104040283011183010505048301108301040406",
            coll(
                SType::SColl(SType::SLong.into()),
                vec![
                    coll(SType::SLong, vec![Value::Long(2)]),
                    coll(SType::SInt, vec![Value::Int(3)]),
                ],
            ),
            127,
        );
        let generators = |n: usize| {
            coll(
                SType::SGroupElement,
                vec![Value::from(ergo_chain_types::ec_point::generator()); n],
            )
        };
        // K1: an empty `Coll[Box]` into a `Coll[GroupElement]`: no item to refuse
        assert_row("0b0eb383010782950100830007830063", generators(1), 87);
        // K3: an empty `Coll[Box]` on the left returns the right side
        assert_row("0b0eb395010083000783006383010782", generators(1), 87);
        assert_row("0b09b38301078283010782", generators(2), 82); // K4
    }

    #[test]
    fn mapped_pairs_rows() {
        // M3/M4: `map` over a collection of pairs gives a generic collection of pairs
        let pair = |a, b| Value::pair(Value::Int(a), Value::Int(b));
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

#[cfg(test)]
mod scala_eq_tests {
    use alloc::vec;

    use ergotree_ir::bigint256::BigInt256;
    use ergotree_ir::mir::coll_value::CollValue;
    use ergotree_ir::mir::value::Value;
    use ergotree_ir::types::stype::SType;

    use super::scala_eq;

    #[test]
    fn integral_boxes_widen() {
        assert!(scala_eq(&Value::Int(1), &Value::Long(1)));
        assert!(scala_eq(&Value::Byte(1), &Value::Long(1)));
        assert!(scala_eq(&Value::Short(1), &Value::Int(1)));
        assert!(!scala_eq(&Value::Int(1), &Value::Long(2)));
    }

    #[test]
    fn other_kinds_do_not() {
        assert!(!scala_eq(&Value::Boolean(true), &Value::Int(1)));
        assert!(!scala_eq(
            &Value::from(BigInt256::from(1i64)),
            &Value::Long(1)
        ));
        let coll = |t: SType, v: Value<'static>| Value::Coll(CollValue::generic(t, vec![v]));
        assert!(!scala_eq(
            &coll(SType::SInt, Value::Int(1)),
            &coll(SType::SLong, Value::Long(1))
        ));
    }

    #[test]
    fn options_compare_their_values() {
        // `Some.equals` compares the two values with `BoxesRunTime.equals` (review I3)
        let some = |v: Value<'static>| Value::Opt(Some(alloc::boxed::Box::new(v)));
        assert!(scala_eq(&some(Value::Long(1)), &some(Value::Int(1))));
        assert!(!scala_eq(&some(Value::Long(1)), &some(Value::Int(2))));
        assert!(scala_eq(&Value::Opt(None), &Value::Opt(None)));
        assert!(!scala_eq(&some(Value::Int(1)), &Value::Opt(None)));
    }

    #[test]
    fn pairs_compare_component_wise() {
        let pair = |a: Value<'static>, b: Value<'static>| Value::pair(a, b);
        assert!(scala_eq(
            &pair(Value::Long(1), Value::Int(2)),
            &pair(Value::Int(1), Value::Int(2))
        ));
        assert!(!scala_eq(
            &pair(Value::Long(1), Value::Int(2)),
            &pair(Value::Int(1), Value::Int(3))
        ));
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod append_tests {
    use alloc::vec;

    use ergotree_ir::mir::coll_value::CollValue;
    use ergotree_ir::mir::value::Value;
    use ergotree_ir::types::stype::SType;

    use super::append;

    fn bytes(b: &[i8]) -> CollValue<'static> {
        CollValue::Bytes(b.into())
    }

    #[test]
    fn append_takes_every_reference_class() {
        // `Context`, `Global` and function values sit in reference arrays on the JVM
        // (`Evaluation.scala:30-31`, `:51-53`; `RType.scala:75-76`), so an append of two such
        // collections copies them (review I1)
        use alloc::sync::Arc;
        use ergotree_ir::mir::expr::Expr;
        use ergotree_ir::mir::value::Lambda;
        use ergotree_ir::types::sfunc::SFunc;
        let twice = |t: SType, v: Value<'static>| {
            let one = CollValue::generic(t, vec![v]);
            append(&one, &one).map(|c| c.len())
        };
        assert_eq!(twice(SType::SContext, Value::Context).unwrap(), 2);
        assert_eq!(twice(SType::SGlobal, Value::Global).unwrap(), 2);
        let lambda = Value::Lambda(Lambda {
            args: vec![],
            body: Arc::new(Expr::Const(true.into())),
            captured: Arc::default(),
        });
        let func = SType::SFunc(SFunc::new(vec![SType::SInt], SType::SInt));
        assert_eq!(twice(func, lambda).unwrap(), 2);
    }

    #[test]
    fn append_specialised_rules() {
        // a specialised left side keeps its class, and takes the same primitive array in either
        // class
        let r = append(
            &bytes(&[1]),
            &CollValue::generic(SType::SByte, vec![Value::Byte(2)]),
        )
        .unwrap();
        assert!(matches!(r, CollValue::Bytes(_)), "{r:?}");
        assert_eq!(r, bytes(&[1, 2]));
        // but not another primitive array, even an empty one: the array cast comes first
        assert!(append(&bytes(&[1]), &CollValue::generic(SType::SLong, vec![])).is_err());
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod prefix_tests {
    use alloc::vec;

    use ergotree_ir::mir::coll_value::CollValue;
    use ergotree_ir::mir::value::Value;
    use ergotree_ir::types::stype::SType;

    use super::{ends_with, starts_with};

    #[test]
    fn pairs_receiver_casts_a_primitive_argument() {
        // `PairOfCols.startsWith`/`endsWith` with any other argument run
        // `toArray.startsWith(ys.toArray)`, whose `ys.toArray()` is cast to `Object[]`
        // (`CollsOverArrays.scala:400`, `:405`): an argument over a primitive array throws, even
        // an empty one (review I2)
        let pairs = CollValue::pairs(
            CollValue::generic(SType::SInt, vec![Value::Int(1)]),
            CollValue::generic(SType::SInt, vec![Value::Int(2)]),
        );
        for arg in [
            CollValue::generic(SType::SInt, vec![]),
            CollValue::Bytes([1i8].as_slice().into()),
            CollValue::Ints([1].as_slice().into()),
        ] {
            assert!(starts_with(&pairs, &arg).is_err(), "{arg:?}");
            assert!(ends_with(&pairs, &arg).is_err(), "{arg:?}");
        }
        // an argument over a reference array compares
        let colls = |items| CollValue::generic(SType::SColl(SType::SInt.into()), items);
        assert!(starts_with(&pairs, &colls(vec![])).unwrap());
        let one = colls(vec![Value::Coll(CollValue::generic(SType::SInt, vec![]))]);
        assert!(!starts_with(&pairs, &one).unwrap());
    }
}
