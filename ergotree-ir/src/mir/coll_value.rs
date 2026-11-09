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

    /// A `Coll[Byte]` of `items` as native bytes, any other item type generic
    pub fn from_collection(
        elem_tpe: SType,
        items: impl Into<Arc<[Value<'ctx>]>>,
    ) -> Result<Self, TryExtractFromError> {
        let items = items.into();
        match elem_tpe {
            SType::SByte => items
                .iter()
                .cloned()
                .map(|v| v.try_extract_into::<i8>())
                .collect::<Result<Arc<[_]>, _>>()
                .map(CollValue::Bytes),
            _ => Ok(CollValue::Generic { elem_tpe, items }),
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
                Some(Value::Tup([ls.get_val(index)?, rs.get_val(index)?].into()))
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
                .map(|(l, r)| Value::Tup([l, r].into()))
                .collect(),
        }
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

/// Data (a constant, a register, a context variable) as a runtime collection
impl From<CollKind<Literal>> for CollValue<'static> {
    fn from(coll: CollKind<Literal>) -> Self {
        match coll {
            CollKind::NativeColl(NativeColl::CollByte(bytes)) => CollValue::Bytes(bytes),
            CollKind::WrappedColl { elem_tpe, items } => CollValue::Generic {
                elem_tpe,
                items: items.iter().cloned().map(Value::from).collect(),
            },
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
#[allow(clippy::unwrap_used)]
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
        Value::Tup([Value::Int(a), Value::Int(b)].into())
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
            Value::Tup([spec.clone(), Value::Int(1)].into()),
            Value::Tup([gen.clone(), Value::Int(1)].into())
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
