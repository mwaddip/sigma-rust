//! A collection value as sigmastate's evaluator holds it

use alloc::format;
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;

use super::constant::Literal;
use super::constant::TryExtractFromError;
use super::constant::TryExtractInto;
use super::value::CollKind;
use super::value::NativeColl;
use super::value::Value;
use crate::types::stuple::STuple;
use crate::types::stype::SType;

/// A collection as sigmastate's evaluator holds it: one form per JVM collection class (v6.0.7
/// `CollsOverArrays.scala`). The class decides how some operations answer (`getOrElse`,
/// `startsWith`, `append`), so a value keeps the class its producer built. Equality looks
/// through the forms, as the JVM's comparer does: it compares the item types, then the items
#[derive(Debug, Clone)]
pub enum CollValue<'ctx> {
    /// `CollOverArray$mcB$sp`: a `Coll[Byte]` specialised over `byte[]`
    Bytes(Arc<[i8]>),
    /// `CollOverArray$mcZ$sp`: a `Coll[Boolean]` specialised over `boolean[]`
    Booleans(Arc<[bool]>),
    /// `CollOverArray$mcI$sp`: a `Coll[Int]` specialised over `int[]`
    Ints(Arc<[i32]>),
    /// A generic `CollOverArray`
    Generic {
        /// The item type (the JVM's `tItem`)
        elem_tpe: SType,
        /// The items
        items: Arc<[Value<'ctx>]>,
    },
    /// `PairOfCols`: a collection of pairs held as its two sides
    Pairs {
        /// The pairs' first items
        ls: Arc<CollValue<'ctx>>,
        /// The pairs' second items
        rs: Arc<CollValue<'ctx>>,
    },
}

impl<'ctx> CollValue<'ctx> {
    /// A generic collection (`CollOverArray`) of `items`, whose item type is `elem_tpe`
    pub fn generic(elem_tpe: SType, items: impl Into<Arc<[Value<'ctx>]>>) -> Self {
        CollValue::Generic {
            elem_tpe,
            items: items.into(),
        }
    }

    /// A collection of pairs held as its two sides (`PairOfCols`)
    pub fn pairs(ls: CollValue<'ctx>, rs: CollValue<'ctx>) -> Self {
        CollValue::Pairs {
            ls: Arc::new(ls),
            rs: Arc::new(rs),
        }
    }

    /// sigmastate's `Colls.fromArray` called at an abstract item type (v6.0.7
    /// `CollsOverArrays.scala:226-232`): never specialised; a pair item type gives a `PairOfCols`
    /// whose sides are built the same way from each component (`fromBoxedPairs`, `:205-215`), any
    /// other item type a generic collection. A pair item type with an item that is not a pair is an
    /// error, as the JVM's `asInstanceOf[(A, B)]` throws. So is an item whose class does not fit
    /// the array of the item type (`fits_array`): sigmastate hands `fromArray` an array of that
    /// class, and the JVM refuses the store that would put another class into it, in `updated`,
    /// `patch`, `updateMany`, `map`, `flatMap` and a pair collection's sides (SANTA
    /// `Coll.item_store_type_check`, `Coll.reference_store_ops`, `Coll.store_runtime_array_class`).
    /// One store is not checked on the JVM, and its caller leaves it out: a builder handed items
    /// for a `Coll[Unit]` with `++=` (the interpreter's `coll_shape::unit_builder_items`)
    pub fn from_array(
        elem_tpe: SType,
        items: Vec<Value<'ctx>>,
    ) -> Result<Self, TryExtractFromError> {
        match split_pairs(&elem_tpe, items) {
            Ok((lt, rt, ls, rs)) => Ok(CollValue::pairs(
                CollValue::from_array(lt, ls)?,
                CollValue::from_array(rt, rs)?,
            )),
            Err(_) if pair_item_types(&elem_tpe).is_some() => Err(TryExtractFromError(format!(
                "a collection of {elem_tpe:?} holds an item that is not a pair"
            ))),
            Err(items) => match items.iter().find(|v| !fits_array(&elem_tpe, v)) {
                Some(item) => Err(TryExtractFromError(format!(
                    "a collection of {elem_tpe:?} cannot hold {item:?}"
                ))),
                None => Ok(CollValue::generic(elem_tpe, items)),
            },
        }
    }

    /// The collection sigmastate's `ConcreteCollection.value` builds from a collection
    /// EXPRESSION held in a register or a context variable (v6.0.6 `values.scala:879-884`):
    /// `Colls.fromArray` at the item type `stypeToRType(elementType)`. So one of tuples of other
    /// than two items keeps the tuple type as its item type, where decoded data of the same
    /// declared type has `Coll[Any]` ([`From<CollKind<Literal>>`]); equality compares item types
    /// first, so the two differ. `items` are the items' values. The forms are the ones data
    /// takes: a `Coll[Byte]` and a `Coll[Boolean]` specialised, pairs as their two sides
    pub fn from_expression_items(
        elem_tpe: SType,
        items: Vec<Value<'static>>,
    ) -> Result<CollValue<'static>, TryExtractFromError> {
        let all_booleans = || items.iter().all(|v| matches!(v, Value::Boolean(_)));
        match elem_tpe {
            SType::SByte => items
                .into_iter()
                .map(|v| v.try_extract_into::<i8>())
                .collect::<Result<Arc<[i8]>, _>>()
                .map(CollValue::Bytes),
            SType::SBoolean if all_booleans() => items
                .into_iter()
                .map(|v| v.try_extract_into::<bool>())
                .collect::<Result<Arc<[bool]>, _>>()
                .map(CollValue::Booleans),
            elem_tpe => Ok(from_data(elem_tpe, items)),
        }
    }

    /// The result of an operation that keeps its receiver's class: sigmastate builds it with
    /// `builder.fromArray` inside the receiver (v6.0.7 `CollsOverArrays.scala:31-111`), so a
    /// specialised receiver gives the same specialised form (an item of another kind is an error, as
    /// the store into its primitive array throws), a generic one goes through
    /// [`CollValue::from_array`], and a `PairOfCols` rebuilds each side the same way
    pub fn rebuild_like(&self, items: Vec<Value<'ctx>>) -> Result<Self, TryExtractFromError> {
        match self {
            CollValue::Bytes(_) => items
                .into_iter()
                .map(|v| v.try_extract_into::<i8>())
                .collect::<Result<Arc<[_]>, _>>()
                .map(CollValue::Bytes),
            CollValue::Booleans(_) => items
                .into_iter()
                .map(|v| v.try_extract_into::<bool>())
                .collect::<Result<Arc<[_]>, _>>()
                .map(CollValue::Booleans),
            CollValue::Ints(_) => items
                .into_iter()
                .map(|v| v.try_extract_into::<i32>())
                .collect::<Result<Arc<[_]>, _>>()
                .map(CollValue::Ints),
            CollValue::Generic { elem_tpe, .. } => CollValue::from_array(elem_tpe.clone(), items),
            CollValue::Pairs { ls, rs } => match split_pairs(&self.elem_tpe(), items) {
                Ok((_, _, l, r)) => Ok(CollValue::pairs(ls.rebuild_like(l)?, rs.rebuild_like(r)?)),
                Err(_) => Err(TryExtractFromError(format!(
                    "a collection of {:?} holds an item that is not a pair",
                    self.elem_tpe()
                ))),
            },
        }
    }

    /// The number of items; a `PairOfCols` has as many as its shorter side
    /// (`CollsOverArrays.scala:300`)
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        match self {
            CollValue::Bytes(items) => items.len(),
            CollValue::Booleans(items) => items.len(),
            CollValue::Ints(items) => items.len(),
            CollValue::Generic { items, .. } => items.len(),
            CollValue::Pairs { ls, rs } => ls.len().min(rs.len()),
        }
    }

    /// Whether it has no items
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The item type (the JVM's `tItem`); a `PairOfCols`'s is the pair of its sides' item types
    pub fn elem_tpe(&self) -> SType {
        match self {
            CollValue::Bytes(_) => SType::SByte,
            CollValue::Booleans(_) => SType::SBoolean,
            CollValue::Ints(_) => SType::SInt,
            CollValue::Generic { elem_tpe, .. } => elem_tpe.clone(),
            CollValue::Pairs { ls, rs } => {
                SType::STuple(STuple::pair(ls.elem_tpe(), rs.elem_tpe()))
            }
        }
    }

    /// The item at `index`, `None` out of range; a `PairOfCols`'s is the pair of its sides' items
    pub fn get_val(&self, index: usize) -> Option<Value<'ctx>> {
        match self {
            CollValue::Bytes(items) => items.get(index).map(|&b| Value::Byte(b)),
            CollValue::Booleans(items) => items.get(index).map(|&b| Value::Boolean(b)),
            CollValue::Ints(items) => items.get(index).map(|&i| Value::Int(i)),
            CollValue::Generic { items, .. } => items.get(index).cloned(),
            CollValue::Pairs { ls, rs } => {
                Some(Value::pair(ls.get_val(index)?, rs.get_val(index)?))
            }
        }
    }

    /// All items, in order
    pub fn as_vec(&self) -> Vec<Value<'ctx>> {
        match self {
            CollValue::Bytes(items) => items.iter().map(|&b| Value::Byte(b)).collect(),
            CollValue::Booleans(items) => items.iter().map(|&b| Value::Boolean(b)).collect(),
            CollValue::Ints(items) => items.iter().map(|&i| Value::Int(i)).collect(),
            CollValue::Generic { items, .. } => items.to_vec(),
            CollValue::Pairs { ls, rs } => ls
                .as_vec()
                .into_iter()
                .zip(rs.as_vec())
                .map(|(l, r)| Value::pair(l, r))
                .collect(),
        }
    }
}

/// Whether `item` can be stored into the JVM array of `elem_tpe`'s items. A primitive item type
/// has a primitive array, whose store unboxes the item, so it takes only its own values; any other
/// has an array of the item type's class, which takes any item of that class and no other (an
/// `ArrayStoreException`): a `Coll` of any item type fits a `Coll` array, a `Box` does not fit a
/// `GroupElement` array. The classes are those of the item type's `RType` class tag (v6.0.6
/// `Evaluation.scala:22-55`, `RType.scala`). SANTA `Coll.item_store_type_check`,
/// `Coll.reference_store_ops`, `Coll.append_runtime_item_type` #4, #5
fn fits_array(elem_tpe: &SType, item: &Value<'_>) -> bool {
    match elem_tpe {
        SType::SBoolean => matches!(item, Value::Boolean(_)),
        SType::SByte => matches!(item, Value::Byte(_)),
        SType::SShort => matches!(item, Value::Short(_)),
        SType::SInt => matches!(item, Value::Int(_)),
        SType::SLong => matches!(item, Value::Long(_)),
        // an `Object[]`
        SType::SAny => true,
        // no JVM array exists for these (`stypeToRType` throws), so nothing can be stored
        SType::STypeVar(_) | SType::SNoType => false,
        SType::SColl(_) => matches!(item, Value::Coll(_)),
        SType::STuple(t) if t.items.len() == 2 => matches!(item, Value::Pair(_)),
        // a tuple of other than two items is a `Coll[Any]` on the JVM (`Tuple.value`)
        SType::STuple(_) => matches!(item, Value::Coll(_)),
        SType::SOption(_) => matches!(item, Value::Opt(_)),
        SType::SBox => matches!(item, Value::CBox(_)),
        SType::SGroupElement => matches!(item, Value::GroupElement(_)),
        SType::SBigInt => matches!(item, Value::BigInt(_)),
        SType::SUnsignedBigInt => matches!(item, Value::UnsignedBigInt(_)),
        SType::SSigmaProp => matches!(item, Value::SigmaProp(_)),
        SType::SAvlTree => matches!(item, Value::AvlTree(_)),
        SType::SHeader => matches!(item, Value::Header(_)),
        SType::SPreHeader => matches!(item, Value::PreHeader(_)),
        SType::SUnit => matches!(item, Value::Unit),
        SType::SContext => matches!(item, Value::Context),
        SType::SGlobal => matches!(item, Value::Global),
        SType::SString => matches!(item, Value::String(_)),
        SType::SFunc(_) => matches!(item, Value::Lambda(_)),
    }
}

/// The two item types of a pair type
fn pair_item_types(tpe: &SType) -> Option<(SType, SType)> {
    match tpe {
        SType::STuple(t) if t.items.len() == 2 => {
            Some((t.items.get(0)?.clone(), t.items.get(1)?.clone()))
        }
        _ => None,
    }
}

/// A collection of pairs split into its sides: the sides' item types and items
type Sides<'ctx> = (SType, SType, Vec<Value<'ctx>>, Vec<Value<'ctx>>);

/// `items` split into two sides when `elem_tpe` is a pair type and every item is a pair; the items
/// back, untouched, otherwise
fn split_pairs<'ctx>(
    elem_tpe: &SType,
    items: Vec<Value<'ctx>>,
) -> Result<Sides<'ctx>, Vec<Value<'ctx>>> {
    let Some((lt, rt)) = pair_item_types(elem_tpe) else {
        return Err(items);
    };
    if !items.iter().all(|v| matches!(v, Value::Pair(_))) {
        return Err(items);
    }
    let mut ls = Vec::with_capacity(items.len());
    let mut rs = Vec::with_capacity(items.len());
    for item in items {
        if let Value::Pair(pair) = item {
            let (l, r) = *pair;
            ls.push(l);
            rs.push(r);
        }
    }
    Ok((lt, rt, ls, rs))
}

/// Data items as the JVM's deserializer holds them, past its two specialised item types
/// (`CoreDataSerializer.deserializeColl`'s generic branch, v6.0.7 `:175-191`): `fromArray` at an
/// abstract type, as [`CollValue::from_array`]. Total: a pair-typed collection holding a non-pair
/// item, which the parser never builds, stays generic. A pair collection's sides are built at
/// their own item types (`fromBoxedPairs`, `CollsOverArrays.scala:205-215`), a tuple type
/// included
fn from_data(elem_tpe: SType, items: Vec<Value<'static>>) -> CollValue<'static> {
    match split_pairs(&elem_tpe, items) {
        Ok((lt, rt, ls, rs)) => CollValue::pairs(from_data(lt, ls), from_data(rt, rs)),
        Err(items) => CollValue::generic(elem_tpe, items),
    }
}

impl PartialEq for CollValue<'_> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (CollValue::Bytes(l), CollValue::Bytes(r)) => l == r,
            (CollValue::Booleans(l), CollValue::Booleans(r)) => l == r,
            (CollValue::Ints(l), CollValue::Ints(r)) => l == r,
            (
                CollValue::Generic {
                    elem_tpe: lt,
                    items: l,
                },
                CollValue::Generic {
                    elem_tpe: rt,
                    items: r,
                },
            ) => lt == rt && l == r,
            _ => {
                self.len() == other.len()
                    && self.elem_tpe() == other.elem_tpe()
                    && (0..self.len()).all(|i| self.get_val(i) == other.get_val(i))
            }
        }
    }
}

impl Eq for CollValue<'_> {}

/// Data (a constant, a register, a context variable) as a runtime collection, in the shape the
/// JVM's deserializer builds (v6.0.7 `CoreDataSerializer.deserializeColl`): `Coll[Byte]` and
/// `Coll[Boolean]` specialised (`:172`, `:174`), any other item type through `fromArray` at an
/// abstract type. An inner collection is converted the same way, so it is specialised too
impl From<CollKind<Literal>> for CollValue<'static> {
    fn from(coll: CollKind<Literal>) -> Self {
        match coll {
            CollKind::NativeColl(NativeColl::CollByte(bytes)) => CollValue::Bytes(bytes),
            CollKind::WrappedColl { elem_tpe, items } => {
                let bits = (elem_tpe == SType::SBoolean)
                    .then(|| {
                        items
                            .iter()
                            .map(|l| match l {
                                Literal::Boolean(b) => Some(*b),
                                _ => None,
                            })
                            .collect::<Option<Arc<[bool]>>>()
                    })
                    .flatten();
                let values = || items.iter().cloned().map(Value::from).collect::<Vec<_>>();
                match (bits, elem_tpe) {
                    (Some(bits), _) => CollValue::Booleans(bits),
                    // the collection the decoder reads: one of tuples of other than two items
                    // is built at the item type `Coll[Any]`, the class of such a tuple's data,
                    // not at the tuple type (v6.0.6 `CoreDataSerializer.scala:160-163`; SANTA
                    // `Tuple.non_pair_value_producers` #5). Only that collection: a pair
                    // collection's sides keep their own types ([`from_data`])
                    (None, SType::STuple(t)) if t.items.len() != 2 => {
                        CollValue::generic(SType::SColl(SType::SAny.into()), values())
                    }
                    (None, elem_tpe) => from_data(elem_tpe, values()),
                }
            }
        }
    }
}

/// A runtime collection as data, in the data representation whatever its form: a `Coll[Byte]` as
/// native bytes, anything else as a list of items, so that it serializes the same
impl<'ctx> TryFrom<CollValue<'ctx>> for CollKind<Literal> {
    type Error = String;

    fn try_from(coll: CollValue<'ctx>) -> Result<Self, Self::Error> {
        match coll {
            CollValue::Bytes(bytes) => Ok(CollKind::NativeColl(NativeColl::CollByte(bytes))),
            CollValue::Generic {
                elem_tpe: SType::SByte,
                items,
            } => items
                .iter()
                .map(|v| match v {
                    Value::Byte(b) => Ok(*b),
                    other => Err(format!("a Coll[Byte] holds {other:?}")),
                })
                .collect::<Result<Arc<[i8]>, String>>()
                .map(|bytes| CollKind::NativeColl(NativeColl::CollByte(bytes))),
            other => Ok(CollKind::WrappedColl {
                elem_tpe: other.elem_tpe(),
                items: other
                    .as_vec()
                    .into_iter()
                    .map(Literal::try_from)
                    .collect::<Result<Arc<[Literal]>, String>>()?,
            }),
        }
    }
}

#[cfg(test)]
#[cfg(feature = "arbitrary")]
#[allow(clippy::unwrap_used, clippy::panic)]
mod array_class {
    use alloc::boxed::Box;
    use alloc::sync::Arc;
    use alloc::vec;
    use alloc::vec::Vec;

    use ergo_chain_types::ec_point::generator;
    use ergo_chain_types::{Header, PreHeader};
    use sigma_test_util::force_any_val;

    use super::CollValue;
    use crate::chain::ergo_box::ErgoBox;
    use crate::mir::avl_tree_data::AvlTreeData;
    use crate::mir::constant::Constant;
    use crate::mir::expr::Expr;
    use crate::mir::func_value::FuncArg;
    use crate::mir::value::{Lambda, Value};
    use crate::sigma_protocol::sigma_boolean::SigmaProp;
    use crate::types::sfunc::SFunc;
    use crate::types::stuple::STuple;
    use crate::types::stype::SType;
    use crate::types::stype_param::STypeVar;
    use crate::unsignedbigint256::UnsignedBigInt;

    /// One item type per JVM array class, with a value of that class
    fn classes() -> Vec<(SType, Value<'static>)> {
        let lambda = Lambda {
            args: vec![FuncArg {
                idx: 1.into(),
                tpe: SType::SInt,
            }],
            body: Arc::new(Expr::Const(Constant::from(1i32))),
            captured: Default::default(),
        };
        vec![
            (SType::SBoolean, Value::Boolean(true)),
            (SType::SByte, Value::Byte(1)),
            (SType::SShort, Value::Short(1)),
            (SType::SInt, Value::Int(1)),
            (SType::SLong, Value::Long(1)),
            (SType::SBigInt, Value::BigInt(1.into())),
            (
                SType::SUnsignedBigInt,
                Value::UnsignedBigInt(force_any_val::<UnsignedBigInt>()),
            ),
            (SType::SGroupElement, Value::from(generator())),
            (
                SType::SSigmaProp,
                Value::SigmaProp(Box::new(force_any_val::<SigmaProp>())),
            ),
            (SType::SBox, Value::CBox(force_any_val::<ErgoBox>().into())),
            (SType::SAvlTree, Value::from(force_any_val::<AvlTreeData>())),
            (
                SType::SHeader,
                Value::Header(Box::new(force_any_val::<Header>())),
            ),
            (
                SType::SPreHeader,
                Value::PreHeader(Box::new(force_any_val::<PreHeader>())),
            ),
            (SType::SUnit, Value::Unit),
            (SType::SContext, Value::Context),
            (SType::SGlobal, Value::Global),
            (SType::SString, Value::String("a".into())),
            (
                SType::SOption(SType::SInt.into()),
                Value::Opt(Some(Box::new(Value::Int(1)))),
            ),
            (
                SType::SColl(SType::SInt.into()),
                Value::Coll(CollValue::Ints([1].into())),
            ),
            (
                SType::SFunc(SFunc::new(vec![SType::SInt], SType::SInt)),
                Value::Lambda(lambda),
            ),
        ]
    }

    /// A JVM array takes its own class and refuses every other: a primitive one unboxes the item
    /// (`ClassCastException`), a reference one checks its class (`ArrayStoreException`); SANTA
    /// `Coll.item_store_type_check`, `Coll.reference_store_ops`
    #[test]
    fn an_array_takes_only_its_own_class() {
        let rows = classes();
        for (i, (tpe, _)) in rows.iter().enumerate() {
            for (j, (_, value)) in rows.iter().enumerate() {
                let built = CollValue::from_array(tpe.clone(), vec![value.clone()]);
                assert_eq!(built.is_ok(), i == j, "{tpe:?} <- {value:?}");
            }
        }
    }

    /// The class is all the array checks: the item type inside a collection or an option is not
    /// looked at (SANTA U11), and a tuple of other than two items is a collection
    #[test]
    fn a_class_ignores_the_item_type_inside() {
        let coll_of_longs = Value::Coll(CollValue::generic(SType::SLong, vec![Value::Long(1)]));
        let option_of_long = Value::Opt(Some(Box::new(Value::Long(1))));
        let triple = SType::STuple(STuple::triple(SType::SInt, SType::SInt, SType::SInt));
        for (tpe, value) in [
            (SType::SColl(SType::SInt.into()), coll_of_longs.clone()),
            (SType::SOption(SType::SInt.into()), option_of_long),
            (triple.clone(), coll_of_longs),
        ] {
            assert!(
                CollValue::from_array(tpe.clone(), vec![value.clone()]).is_ok(),
                "{tpe:?} <- {value:?}"
            );
        }
        // and nothing else is a tuple of three
        assert!(CollValue::from_array(triple, vec![Value::Int(1)]).is_err());
    }

    /// `Any` is an `Object[]`: it takes every class
    #[test]
    fn any_takes_everything() {
        for (_, value) in classes() {
            assert!(
                CollValue::from_array(SType::SAny, vec![value.clone()]).is_ok(),
                "{value:?}"
            );
        }
    }

    /// A type variable and `NoType` have no JVM array at all (`stypeToRType` throws), so nothing
    /// can be stored under one; SANTA's third reply R8, A2
    #[test]
    fn a_type_without_a_runtime_type_takes_no_item() {
        for tpe in [SType::STypeVar(STypeVar::t()), SType::SNoType] {
            for (_, value) in classes() {
                assert!(
                    CollValue::from_array(tpe.clone(), vec![value.clone()]).is_err(),
                    "{tpe:?} <- {value:?}"
                );
            }
            assert!(CollValue::from_array(tpe, vec![]).is_ok());
        }
    }

    /// A pair collection stores each component into its side's array (`fromBoxedPairs`; SANTA G1)
    #[test]
    fn a_pair_collection_refuses_a_component_of_another_class() {
        let tpe = SType::STuple(STuple::pair(SType::SInt, SType::SBox));
        let a_box = Value::CBox(force_any_val::<ErgoBox>().into());
        let good = CollValue::from_array(tpe.clone(), vec![Value::pair(Value::Int(1), a_box)]);
        assert!(matches!(good, Ok(CollValue::Pairs { .. })), "{good:?}");
        let lie = Value::pair(Value::Int(1), Value::from(generator()));
        assert!(CollValue::from_array(tpe, vec![lie]).is_err());
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use super::CollValue;
    use crate::mir::constant::{Literal, TryExtractInto};
    use crate::mir::value::{CollKind, NativeColl, Value};
    use crate::types::stuple::STuple;
    use crate::types::stype::SType;

    fn bytes(b: &[i8]) -> CollValue<'static> {
        CollValue::Bytes(b.into())
    }

    fn generic(t: SType, items: Vec<Value<'static>>) -> CollValue<'static> {
        CollValue::generic(t, items)
    }

    fn ints(xs: &[i32]) -> Vec<Value<'static>> {
        xs.iter().map(|&i| Value::Int(i)).collect()
    }

    fn pair(a: i32, b: i32) -> Value<'static> {
        Value::pair(Value::Int(a), Value::Int(b))
    }

    #[test]
    fn forms_compare_equal() {
        assert_eq!(
            bytes(&[1, 2]),
            generic(SType::SByte, vec![Value::Byte(1), Value::Byte(2)])
        );
        assert_eq!(
            CollValue::Booleans([true].as_slice().into()),
            generic(SType::SBoolean, vec![Value::Boolean(true)])
        );
        assert_eq!(
            CollValue::Ints([0, 1].as_slice().into()),
            generic(SType::SInt, ints(&[0, 1]))
        );
        assert_eq!(
            CollValue::pairs(
                generic(SType::SInt, ints(&[1, 3])),
                generic(SType::SInt, ints(&[2, 4]))
            ),
            generic(
                SType::STuple(STuple::pair(SType::SInt, SType::SInt)),
                vec![pair(1, 2), pair(3, 4)]
            )
        );
        // the item type decides even when both are empty
        assert_ne!(bytes(&[]), generic(SType::SInt, vec![]));
        assert_ne!(bytes(&[1]), bytes(&[2]));
    }

    #[test]
    fn forms_compare_equal_nested() {
        let spec = Value::Coll(bytes(&[1]));
        let gen = Value::Coll(generic(SType::SByte, vec![Value::Byte(1)]));
        assert_eq!(
            Value::pair(spec.clone(), Value::Int(1)),
            Value::pair(gen.clone(), Value::Int(1))
        );
        assert_eq!(
            Value::Opt(Some(spec.clone().into())),
            Value::Opt(Some(gen.clone().into()))
        );
        let t = SType::SColl(SType::SByte.into());
        assert_eq!(generic(t.clone(), vec![spec]), generic(t, vec![gen]));
    }

    #[test]
    fn typed_extraction_from_every_form() {
        let v = Value::Coll(CollValue::Booleans([true, false].as_slice().into()));
        assert_eq!(
            v.try_extract_into::<Vec<bool>>().unwrap(),
            vec![true, false]
        );
        let v = Value::Coll(CollValue::Ints([0, 1].as_slice().into()));
        assert_eq!(v.try_extract_into::<Vec<i32>>().unwrap(), vec![0, 1]);
        let v = Value::Coll(generic(SType::SByte, vec![Value::Byte(1)]));
        assert_eq!(v.try_extract_into::<Vec<i8>>().unwrap(), vec![1i8]);
        let v = Value::Coll(CollValue::pairs(
            generic(SType::SInt, ints(&[1])),
            generic(SType::SInt, ints(&[2])),
        ));
        assert_eq!(
            v.try_extract_into::<Vec<Value>>().unwrap(),
            vec![pair(1, 2)]
        );
    }

    #[test]
    fn pairs_len_is_shorter_side() {
        let p = CollValue::pairs(
            CollValue::Ints([0, 1, 2].as_slice().into()),
            generic(SType::SInt, ints(&[5])),
        );
        assert_eq!(p.len(), 1);
        assert_eq!(p.get_val(0), Some(pair(0, 5)));
        assert_eq!(p.get_val(1), None);
    }

    #[test]
    fn a_data_collection_of_non_pair_tuples_holds_collections_of_any() {
        // by reading: `deserializeColl` builds a collection of non-pair tuples at the item type
        // `Coll[Any]` (v6.0.6 `CoreDataSerializer.scala:160-163`), and each item is the tuple's
        // own `Coll[Any]` (`:134-139`)
        let triple = SType::STuple(STuple::triple(SType::SInt, SType::SInt, SType::SInt));
        let item = Literal::Tup(
            vec![Literal::Int(1), Literal::Int(2), Literal::Int(3)]
                .try_into()
                .unwrap(),
        );
        let coll_of_any = SType::SColl(SType::SAny.into());
        for items in [vec![item], vec![]] {
            let len = items.len();
            let coll = CollValue::from(CollKind::WrappedColl {
                elem_tpe: triple.clone(),
                items: items.into(),
            });
            assert!(matches!(&coll, CollValue::Generic { .. }));
            assert_eq!(coll.elem_tpe(), coll_of_any);
            assert_eq!(coll.len(), len);
            if len == 1 {
                let ints = vec![Value::Int(1), Value::Int(2), Value::Int(3)];
                assert_eq!(
                    coll.get_val(0),
                    Some(Value::Coll(CollValue::generic(SType::SAny, ints)))
                );
            }
        }
        // a pair collection's side is built at its own item type, the tuple type
        // (`fromBoxedPairs`): data `Coll[((Int, Int, Int), Int)]`
        let triple_lit = || {
            Literal::Tup(
                vec![Literal::Int(1), Literal::Int(2), Literal::Int(3)]
                    .try_into()
                    .unwrap(),
            )
        };
        let sides = CollValue::from(CollKind::WrappedColl {
            elem_tpe: SType::STuple(STuple::pair(triple.clone(), SType::SInt)),
            items: vec![Literal::Tup([triple_lit(), Literal::Int(4)].into())].into(),
        });
        match &sides {
            CollValue::Pairs { ls, .. } => assert_eq!(ls.elem_tpe(), triple),
            other => panic!("{other:?}"),
        }
        // and a collection EXPRESSION's value is built at the element type
        // (`ConcreteCollection.value`), so it differs from the same collection as data
        let ints = vec![Value::Int(1), Value::Int(2), Value::Int(3)];
        let item = Value::Coll(CollValue::generic(SType::SAny, ints));
        let expr_coll = CollValue::from_expression_items(triple.clone(), vec![item]).unwrap();
        assert_eq!(expr_coll.elem_tpe(), triple);
        let data_coll = CollValue::from(CollKind::WrappedColl {
            elem_tpe: triple.clone(),
            items: vec![triple_lit()].into(),
        });
        assert_ne!(expr_coll, data_coll);
        // a collection of pairs is still a `PairOfCols`
        let pair = Literal::Tup([Literal::Int(1), Literal::Int(2)].into());
        let pairs = CollValue::from(CollKind::WrappedColl {
            elem_tpe: SType::STuple(STuple::pair(SType::SInt, SType::SInt)),
            items: vec![pair].into(),
        });
        assert!(matches!(pairs, CollValue::Pairs { .. }));
    }

    /// Data collections, one per shape the JVM's deserializer builds, with their types
    fn data_collections() -> Vec<(SType, Literal)> {
        let int_pair = STuple::pair(SType::SInt, SType::SInt);
        let tup = |l: Literal, r: Literal| Literal::Tup([l, r].into());
        let wrapped = |elem_tpe: SType, items: Vec<Literal>| {
            Literal::Coll(CollKind::WrappedColl {
                elem_tpe,
                items: items.into(),
            })
        };
        let bytes = |b: &[i8]| Literal::Coll(CollKind::NativeColl(NativeColl::CollByte(b.into())));
        let coll = |t: SType| SType::SColl(t.into());
        vec![
            (coll(SType::SByte), bytes(&[1, 2])),
            (
                coll(SType::SBoolean),
                wrapped(
                    SType::SBoolean,
                    vec![Literal::Boolean(true), Literal::Boolean(false)],
                ),
            ),
            (
                coll(SType::SShort),
                wrapped(SType::SShort, vec![Literal::Short(1)]),
            ),
            (
                coll(SType::SInt),
                wrapped(SType::SInt, vec![Literal::Int(1), Literal::Int(2)]),
            ),
            (
                coll(SType::SLong),
                wrapped(SType::SLong, vec![Literal::Long(1)]),
            ),
            (
                coll(SType::STuple(int_pair.clone())),
                wrapped(
                    SType::STuple(int_pair.clone()),
                    vec![
                        tup(Literal::Int(1), Literal::Int(2)),
                        tup(Literal::Int(3), Literal::Int(4)),
                    ],
                ),
            ),
            (
                coll(SType::STuple(STuple::pair(SType::SByte, SType::SByte))),
                wrapped(
                    SType::STuple(STuple::pair(SType::SByte, SType::SByte)),
                    vec![tup(Literal::Byte(1), Literal::Byte(2))],
                ),
            ),
            (
                coll(coll(SType::SByte)),
                wrapped(coll(SType::SByte), vec![bytes(&[1]), bytes(&[])]),
            ),
            (
                coll(SType::STuple(STuple::pair(
                    SType::STuple(int_pair.clone()),
                    SType::SInt,
                ))),
                wrapped(
                    SType::STuple(STuple::pair(SType::STuple(int_pair), SType::SInt)),
                    vec![tup(tup(Literal::Int(1), Literal::Int(2)), Literal::Int(3))],
                ),
            ),
        ]
    }

    #[test]
    fn data_takes_deserializer_shapes() {
        use CollValue::*;
        let is_generic =
            |c: &CollValue, t: SType| matches!(c, Generic { elem_tpe, .. } if *elem_tpe == t);
        let shapes: Vec<CollValue<'static>> = data_collections()
            .into_iter()
            .map(|(_, lit)| match Value::from(lit) {
                Value::Coll(c) => c,
                other => panic!("not a collection: {other:?}"),
            })
            .collect();
        assert!(matches!(shapes[0], Bytes(_)), "{:?}", shapes[0]);
        assert!(matches!(shapes[1], Booleans(_)), "{:?}", shapes[1]);
        assert!(is_generic(&shapes[2], SType::SShort), "{:?}", shapes[2]);
        assert!(is_generic(&shapes[3], SType::SInt), "{:?}", shapes[3]);
        assert!(is_generic(&shapes[4], SType::SLong), "{:?}", shapes[4]);
        assert!(
            matches!(&shapes[5], Pairs { ls, rs } if is_generic(ls, SType::SInt) && is_generic(rs, SType::SInt)),
            "{:?}",
            shapes[5]
        );
        // a Byte SIDE is generic: `fromBoxedPairs` builds it at an abstract type
        assert!(
            matches!(&shapes[6], Pairs { ls, rs } if is_generic(ls, SType::SByte) && is_generic(rs, SType::SByte)),
            "{:?}",
            shapes[6]
        );
        // the inner collections of a `Coll[Coll[Byte]]` are specialised
        assert!(
            matches!(&shapes[7], Generic { items, .. }
                if items.iter().all(|v| matches!(v, Value::Coll(Bytes(_))))),
            "{:?}",
            shapes[7]
        );
        assert!(
            matches!(&shapes[8], Pairs { ls, rs }
                if matches!(**ls, Pairs { .. }) && is_generic(rs, SType::SInt)),
            "{:?}",
            shapes[8]
        );
    }

    #[test]
    fn runtime_to_data_normalises() {
        use crate::mir::constant::Constant;
        use crate::serialization::SigmaSerializable;
        for (tpe, lit) in data_collections() {
            let Value::Coll(coll) = Value::from(lit.clone()) else {
                panic!("not a collection")
            };
            let back = Literal::Coll(CollKind::try_from(coll).unwrap());
            assert_eq!(back, lit);
            let bytes = |v: Literal| {
                Constant {
                    tpe: tpe.clone(),
                    v,
                }
                .sigma_serialize_bytes()
                .unwrap()
            };
            assert_eq!(bytes(back), bytes(lit));
        }
        // a generic `Coll[Byte]` writes back as the native bytes it holds
        let generic_bytes = generic(SType::SByte, vec![Value::Byte(1), Value::Byte(2)]);
        assert_eq!(
            CollKind::try_from(generic_bytes).unwrap(),
            CollKind::NativeColl(NativeColl::CollByte([1i8, 2].as_slice().into()))
        );
    }

    #[test]
    fn data_round_trip_today() {
        let pair_lit = Literal::Tup([Literal::Int(1), Literal::Int(2)].into());
        let inner = Literal::Coll(CollKind::NativeColl(NativeColl::CollByte(
            [1i8].as_slice().into(),
        )));
        for lit in [
            Literal::Coll(CollKind::NativeColl(NativeColl::CollByte(
                [1i8, 2].as_slice().into(),
            ))),
            Literal::Coll(CollKind::WrappedColl {
                elem_tpe: SType::SBoolean,
                items: [Literal::Boolean(true)].as_slice().into(),
            }),
            Literal::Coll(CollKind::WrappedColl {
                elem_tpe: SType::STuple(STuple::pair(SType::SInt, SType::SInt)),
                items: [pair_lit].as_slice().into(),
            }),
            Literal::Coll(CollKind::WrappedColl {
                elem_tpe: SType::SColl(SType::SByte.into()),
                items: [inner].as_slice().into(),
            }),
        ] {
            assert_eq!(Literal::try_from(Value::from(lit.clone())), Ok(lit));
        }
    }
}
