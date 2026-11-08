use core::convert::TryFrom;
use core::convert::TryInto;
use hashbrown::HashMap;

use alloc::vec;
use alloc::vec::Vec;
use bounded_vec::BoundedVecOutOfBounds;

use super::stype::SType;
use super::stype_param::STypeVar;

/// A tuple's items, 0 to [`TupleItems::MAX_LEN`] of them. sigmastate reads a tuple type or a
/// tuple expression of any size up to its count byte (v6.0.7 `TypeSerializer.scala:192-198`,
/// `TupleSerializer.scala:27-36`); it only refuses to write a tuple type of fewer than 2 items
/// (`TypeSerializer.scala:94-95`), and to evaluate a tuple expression of other than 2
/// (`values.scala:826-827`).
#[derive(PartialEq, Eq, Debug, Clone)]
pub struct TupleItems<T>(Vec<T>);

impl<T> TupleItems<T> {
    /// The most items a tuple holds: a tuple type's item count is one unsigned byte
    /// (sigmastate 6.0.7 `TypeSerializer.scala:192-198`)
    pub const MAX_LEN: usize = 255;

    /// The number of items
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether there are no items
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// The items, in order
    pub fn iter(&self) -> core::slice::Iter<'_, T> {
        self.0.iter()
    }

    /// The items, in order, mutably
    pub fn iter_mut(&mut self) -> core::slice::IterMut<'_, T> {
        self.0.iter_mut()
    }

    /// The items as a slice
    pub fn as_slice(&self) -> &[T] {
        &self.0
    }

    /// The item at `index`, counted from 0
    pub fn get(&self, index: usize) -> Option<&T> {
        self.0.get(index)
    }

    /// The items as a vector
    pub fn to_vec(self) -> Vec<T> {
        self.0
    }

    /// Reads `count` items with `read_item`, in order, stopping at the first error. A `u8`
    /// count is at most [`Self::MAX_LEN`], so the items always fit and there is no bounds error.
    pub(crate) fn read_with<E>(
        count: u8,
        mut read_item: impl FnMut() -> Result<T, E>,
    ) -> Result<Self, E> {
        (0..count)
            .map(|_| read_item())
            .collect::<Result<_, _>>()
            .map(TupleItems)
    }

    /// Each item mapped by `f`
    pub fn mapped<F: FnMut(T) -> N, N>(self, f: F) -> TupleItems<N> {
        TupleItems(self.0.into_iter().map(f).collect())
    }

    /// Each item, by reference, mapped by `f`
    pub fn mapped_ref<F: FnMut(&T) -> N, N>(&self, f: F) -> TupleItems<N> {
        TupleItems(self.0.iter().map(f).collect())
    }

    /// Each item mapped by `f`, or the first error
    pub fn try_mapped<F: FnMut(T) -> Result<N, E>, N, E>(self, f: F) -> Result<TupleItems<N>, E> {
        self.0
            .into_iter()
            .map(f)
            .collect::<Result<_, _>>()
            .map(TupleItems)
    }

    /// Each item, by reference, mapped by `f`, or the first error
    pub fn try_mapped_ref<F: FnMut(&T) -> Result<N, E>, N, E>(
        &self,
        f: F,
    ) -> Result<TupleItems<N>, E> {
        self.0
            .iter()
            .map(f)
            .collect::<Result<_, _>>()
            .map(TupleItems)
    }
}

// `TupleItems::read_with` takes a `u8` count because every such count fits
const _: () = assert!(TupleItems::<()>::MAX_LEN >= u8::MAX as usize);

impl<T> TryFrom<Vec<T>> for TupleItems<T> {
    type Error = BoundedVecOutOfBounds;

    fn try_from(items: Vec<T>) -> Result<Self, Self::Error> {
        if items.len() > Self::MAX_LEN {
            Err(BoundedVecOutOfBounds::UpperBoundError {
                upper_bound: Self::MAX_LEN,
                got: items.len(),
            })
        } else {
            Ok(TupleItems(items))
        }
    }
}

impl<T> From<[T; 2]> for TupleItems<T> {
    fn from(items: [T; 2]) -> Self {
        TupleItems(items.into())
    }
}

impl<T> From<TupleItems<T>> for Vec<T> {
    fn from(items: TupleItems<T>) -> Self {
        items.0
    }
}

impl<T> IntoIterator for TupleItems<T> {
    type Item = T;
    type IntoIter = alloc::vec::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a, T> IntoIterator for &'a TupleItems<T> {
    type Item = &'a T;
    type IntoIter = core::slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl TryFrom<Vec<SType>> for STuple {
    type Error = BoundedVecOutOfBounds;

    fn try_from(value: Vec<SType>) -> Result<Self, Self::Error> {
        Ok(STuple {
            items: value.try_into()?,
        })
    }
}

/// Tuple type
#[derive(PartialEq, Eq, Clone)]
pub struct STuple {
    /// Tuple element types
    pub items: TupleItems<SType>,
}

impl core::fmt::Debug for STuple {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.items.clone().to_vec().fmt(f)
    }
}

impl core::fmt::Display for STuple {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "(")?;
        for (i, item) in self.items.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            item.fmt(f)?;
        }
        write!(f, ")")
    }
}

impl STuple {
    /// Create a tuple type for a given type pair
    pub fn pair(t1: SType, t2: SType) -> Self {
        STuple {
            items: [t1, t2].into(),
        }
    }

    /// Create a triple type
    pub fn triple(t1: SType, t2: SType, t3: SType) -> Self {
        #[allow(clippy::unwrap_used)]
        STuple {
            items: vec![t1, t2, t3].try_into().unwrap(),
        }
    }

    /// Create a quadruple type
    pub fn quadruple(t1: SType, t2: SType, t3: SType, t4: SType) -> Self {
        #[allow(clippy::unwrap_used)]
        STuple {
            items: vec![t1, t2, t3, t4].try_into().unwrap(),
        }
    }

    pub(crate) fn with_subst(&self, subst: &HashMap<STypeVar, SType>) -> Self {
        STuple {
            items: self.items.clone().mapped(|a| a.with_subst(subst)),
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn tuple_items_hold_zero_to_255() {
        // sigmastate reads a tuple type of 0 to 255 items (v6.0.7 `TypeSerializer.scala:192-198`)
        assert_eq!(TupleItems::<i32>::MAX_LEN, 255);
        assert!(TupleItems::<i32>::try_from(vec![]).is_ok());
        assert!(TupleItems::try_from(vec![1]).is_ok());
        assert!(TupleItems::try_from(vec![1; 255]).is_ok());
        assert_eq!(
            TupleItems::try_from(vec![1; 256]),
            Err(BoundedVecOutOfBounds::UpperBoundError {
                upper_bound: 255,
                got: 256
            })
        );
        assert_eq!(
            TupleItems::from([1, 2]).mapped(|x| x * 10).as_slice(),
            [10, 20]
        );
    }

    #[test]
    fn read_with_reads_count_items_and_stops_at_the_first_error() {
        let mut calls = 0u8;
        let empty = TupleItems::<u8>::read_with(0, || -> Result<u8, ()> {
            calls += 1;
            Ok(calls)
        });
        assert_eq!(empty, Ok(TupleItems(vec![])));
        assert_eq!(calls, 0);

        let mut next = 0u8;
        let full = TupleItems::read_with(255, || -> Result<u8, ()> {
            let item = next;
            next = next.wrapping_add(1);
            Ok(item)
        })
        .unwrap();
        assert_eq!(full.as_slice(), (0..=254).collect::<Vec<u8>>().as_slice());

        let mut calls = 0u8;
        let failed = TupleItems::<u8>::read_with(5, || {
            calls += 1;
            if calls == 3 {
                Err("the third item")
            } else {
                Ok(calls)
            }
        });
        assert_eq!(failed, Err("the third item"));
        assert_eq!(calls, 3);
    }
}
