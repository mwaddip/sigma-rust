//! A collection value as sigmastate's evaluator holds it

use alloc::format;
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;
use thiserror::Error;

use super::constant::Literal;
use super::constant::TryExtractFromError;
use super::constant::TryExtractInto;
use super::value::debug_start;
use super::value::CollKind;
use super::value::NativeColl;
use super::value::Value;
use crate::types::stuple::STuple;
use crate::types::stype::SType;

/// The array of a `CollOverArray`: at most `MAX_ARRAY_LENGTH` items, as sigmastate's class
/// requires in its constructor (v6.0.6 `CollsOverArrays.scala:13-14`). The limit is this type's:
/// its field is private and every way to build one asks [`check_coll_len`], so no collection
/// value holds a longer array, whoever built it. Two constructors ask nothing because they cannot
/// lengthen anything: [`CollArray::from_fixed`] (a count the type bounds) and the crate's own
/// conversion of an array item by item
#[derive(Clone, PartialEq, Eq)]
pub struct CollArray<T>(Arc<[T]>);

/// The text of the items alone, as the array printed before it had a type of its own: a value's
/// `Debug` text is part of the text of a refusal
impl<T: core::fmt::Debug> core::fmt::Debug for CollArray<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        core::fmt::Debug::fmt(&self.0, f)
    }
}

/// A collection of more than `MAX_ARRAY_LENGTH` items: sigmastate's `CollOverArray` refuses it in
/// its constructor (v6.0.6 `CollsOverArrays.scala:13-14`). Holds the refused length
#[derive(Error, Debug, Clone, PartialEq, Eq)]
#[error("Cannot create collection with size {0} greater than 100000")]
pub struct CollLengthError(pub usize);

/// sigmastate's limit on a collection's length, `MaxArrayLength` (v6.0.6 `util/package.scala:7`):
/// `len` items are refused above it. The array's constructor asks it ([`CollArray::new`]), and so
/// does whatever sigmastate has check a length before it builds (`safeConcatArrays_v5`, `:25-28`)
pub fn check_coll_len(len: usize) -> Result<(), CollLengthError> {
    if len > crate::serialization::sigma_byte_reader::MAX_ARRAY_LENGTH {
        return Err(CollLengthError(len));
    }
    Ok(())
}

/// Why a collection could not be built from items
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum CollBuildError {
    /// More than `MAX_ARRAY_LENGTH` items
    #[error("{0}")]
    TooLong(#[from] CollLengthError),
    /// An item the array of the item type cannot hold
    #[error("{0}")]
    Item(#[from] TryExtractFromError),
}

impl From<CollBuildError> for TryExtractFromError {
    fn from(e: CollBuildError) -> Self {
        match e {
            CollBuildError::TooLong(e) => e.into(),
            CollBuildError::Item(e) => e,
        }
    }
}

impl<T> CollArray<T> {
    /// The array of `items`, refused above the limit: sigmastate's
    /// `require(length <= MaxArrayLength)`. The limit is asked before the items are moved
    pub fn new(items: impl AsRef<[T]> + Into<Arc<[T]>>) -> Result<Self, CollLengthError> {
        check_coll_len(items.as_ref().len())?;
        Ok(CollArray(items.into()))
    }
    /// An array whose count its TYPE bounds: `N` above the limit does not compile
    pub fn from_fixed<const N: usize>(items: [T; N]) -> Self {
        const {
            assert!(N <= crate::serialization::sigma_byte_reader::MAX_ARRAY_LENGTH);
        }
        CollArray(Arc::from(items))
    }
    /// The shared array
    pub fn arc(&self) -> &Arc<[T]> {
        &self.0
    }
    /// An array of as many items as this one: `items` is this array converted item by item, so
    /// its length passed the limit when this one was built
    pub(crate) fn same_len<U>(&self, items: Arc<[U]>) -> CollArray<U> {
        debug_assert_eq!(self.0.len(), items.len());
        CollArray(items)
    }
}

impl<T> TryFrom<Vec<T>> for CollArray<T> {
    type Error = CollLengthError;
    fn try_from(items: Vec<T>) -> Result<Self, Self::Error> {
        CollArray::new(items)
    }
}

impl<T: Clone> TryFrom<&[T]> for CollArray<T> {
    type Error = CollLengthError;
    fn try_from(items: &[T]) -> Result<Self, Self::Error> {
        CollArray::new(items)
    }
}

impl<T, const N: usize> TryFrom<[T; N]> for CollArray<T> {
    type Error = CollLengthError;
    fn try_from(items: [T; N]) -> Result<Self, Self::Error> {
        CollArray::new(items)
    }
}

impl<T> core::ops::Deref for CollArray<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        &self.0
    }
}

/// A collection as sigmastate's evaluator holds it: one form per JVM collection class (v6.0.7
/// `CollsOverArrays.scala`). The class decides how some operations answer (`getOrElse`,
/// `startsWith`, `append`), so a value keeps the class its producer built. Equality looks
/// through the forms, as the JVM's comparer does: it compares the item types, then the items.
/// No form holds more than `MAX_ARRAY_LENGTH` items: the four over an array by the array's type
/// ([`CollArray`]), a `PairOfCols` by its sides, as on the JVM, where it has no check of its own
/// (`CollsOverArrays.scala:274`)
#[derive(Debug, Clone)]
pub enum CollValue<'ctx> {
    /// `CollOverArray$mcB$sp`: a `Coll[Byte]` specialised over `byte[]`
    Bytes(CollArray<i8>),
    /// `CollOverArray$mcZ$sp`: a `Coll[Boolean]` specialised over `boolean[]`
    Booleans(CollArray<bool>),
    /// `CollOverArray$mcI$sp`: a `Coll[Int]` specialised over `int[]`
    Ints(CollArray<i32>),
    /// A generic `CollOverArray`
    Generic {
        /// The item type (the JVM's `tItem`)
        elem_tpe: SType,
        /// The items
        items: CollArray<Value<'ctx>>,
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
    pub fn generic(
        elem_tpe: SType,
        items: impl AsRef<[Value<'ctx>]> + Into<Arc<[Value<'ctx>]>>,
    ) -> Result<Self, CollLengthError> {
        Ok(CollValue::Generic {
            elem_tpe,
            items: CollArray::new(items)?,
        })
    }

    /// A specialised `Coll[Byte]` (`CollOverArray$mcB$sp`)
    pub fn bytes(items: impl AsRef<[i8]> + Into<Arc<[i8]>>) -> Result<Self, CollLengthError> {
        Ok(CollValue::Bytes(CollArray::new(items)?))
    }

    /// A specialised `Coll[Boolean]` (`CollOverArray$mcZ$sp`)
    pub fn booleans(
        items: impl AsRef<[bool]> + Into<Arc<[bool]>>,
    ) -> Result<Self, CollLengthError> {
        Ok(CollValue::Booleans(CollArray::new(items)?))
    }

    /// A specialised `Coll[Int]` (`CollOverArray$mcI$sp`)
    pub fn ints(items: impl AsRef<[i32]> + Into<Arc<[i32]>>) -> Result<Self, CollLengthError> {
        Ok(CollValue::Ints(CollArray::new(items)?))
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
    pub fn from_array(elem_tpe: SType, items: Vec<Value<'ctx>>) -> Result<Self, CollBuildError> {
        match split_pairs(&elem_tpe, items) {
            Ok((lt, rt, ls, rs)) => {
                // both components of every pair are stored, the left one first, before either
                // side is built (`fromBoxedPairs`, `:205-215`): a component its side's array
                // cannot hold is refused in item order, and before any side's length
                for (l, r) in ls.iter().zip(rs.iter()) {
                    for (side_tpe, component) in [(&lt, l), (&rt, r)] {
                        if !fits_array(side_tpe, component) {
                            return Err(cannot_hold(side_tpe, component).into());
                        }
                    }
                }
                Ok(CollValue::pairs(
                    CollValue::from_array(lt, ls)?,
                    CollValue::from_array(rt, rs)?,
                ))
            }
            Err(_) if pair_item_types(&elem_tpe).is_some() => Err(TryExtractFromError(format!(
                "a collection of {elem_tpe:?} holds an item that is not a pair"
            ))
            .into()),
            Err(items) => match items.iter().find(|v| !fits_array(&elem_tpe, v)) {
                Some(item) => Err(cannot_hold(&elem_tpe, item).into()),
                None => Ok(CollValue::generic(elem_tpe, items)?),
            },
        }
    }

    /// The collection sigmastate's `ConcreteCollection.value` builds from a collection
    /// EXPRESSION held in a register or a context variable (v6.0.6 `values.scala:879-884`):
    /// `Colls.fromArray` at the item type `stypeToRType(elementType)`. So one of tuples of other
    /// than two items keeps the tuple type as its item type, where decoded data of the same
    /// declared type has `Coll[Any]` ([`From<CollKind<Literal>>`]); equality compares item types
    /// first, so the two differ. `items` are the items' values. `Colls.fromArray` at an abstract
    /// type is never specialised ([`CollValue::from_array`]): a `Coll[Byte]` and a
    /// `Coll[Boolean]` expression are GENERIC collections over their primitive array, where the
    /// same items as data are specialised; pairs are their two sides. Neither parser reads a
    /// collection expression holding an item of another type (v6.0.6
    /// `ConcreteCollectionSerializer.scala:38`); handed such items directly, a `Coll[Byte]` is
    /// refused and a `Coll[Boolean]` takes the form data takes
    pub fn from_expression_items(
        elem_tpe: SType,
        items: Vec<Value<'static>>,
    ) -> Result<CollValue<'static>, CollBuildError> {
        let all_booleans = || items.iter().all(|v| matches!(v, Value::Boolean(_)));
        match elem_tpe {
            SType::SByte => CollValue::from_array(SType::SByte, items),
            SType::SBoolean if all_booleans() => CollValue::from_array(SType::SBoolean, items),
            elem_tpe => Ok(from_data(elem_tpe, items)?),
        }
    }

    /// The result of an operation that keeps its receiver's class: sigmastate builds it with
    /// `builder.fromArray` inside the receiver (v6.0.7 `CollsOverArrays.scala:31-111`), so a
    /// specialised receiver gives the same specialised form (an item of another kind is an error, as
    /// the store into its primitive array throws), a generic one goes through
    /// [`CollValue::from_array`], and a `PairOfCols` rebuilds each side the same way
    pub fn rebuild_like(&self, items: Vec<Value<'ctx>>) -> Result<Self, CollBuildError> {
        match self {
            CollValue::Bytes(_) => {
                let array = items
                    .into_iter()
                    .map(|v| v.try_extract_into::<i8>())
                    .collect::<Result<Arc<[_]>, _>>()?;
                Ok(CollValue::Bytes(CollArray::new(array)?))
            }
            CollValue::Booleans(_) => {
                let array = items
                    .into_iter()
                    .map(|v| v.try_extract_into::<bool>())
                    .collect::<Result<Arc<[_]>, _>>()?;
                Ok(CollValue::Booleans(CollArray::new(array)?))
            }
            CollValue::Ints(_) => {
                let array = items
                    .into_iter()
                    .map(|v| v.try_extract_into::<i32>())
                    .collect::<Result<Arc<[_]>, _>>()?;
                Ok(CollValue::Ints(CollArray::new(array)?))
            }
            CollValue::Generic { elem_tpe, .. } => CollValue::from_array(elem_tpe.clone(), items),
            CollValue::Pairs { ls, rs } => match split_pairs(&self.elem_tpe(), items) {
                Ok((_, _, l, r)) => Ok(CollValue::pairs(ls.rebuild_like(l)?, rs.rebuild_like(r)?)),
                Err(_) => Err(TryExtractFromError(format!(
                    "a collection of {:?} holds an item that is not a pair",
                    self.elem_tpe()
                ))
                .into()),
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
                SType::STuple(STuple::pair(ls.elem_tpe(), rs.elem_tpe()).into())
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

/// The refusal of an item the array of `elem_tpe`'s items cannot hold. It names the start of the
/// type and of the item: an item built at evaluation can be large
fn cannot_hold(elem_tpe: &SType, item: &Value<'_>) -> TryExtractFromError {
    TryExtractFromError(format!(
        "a collection of {} cannot hold {}",
        debug_start(elem_tpe),
        debug_start(item)
    ))
}

/// The first item of `coll`, with its index, that the array of `elem_tpe`'s items cannot hold. A
/// specialised collection and a `PairOfCols` say by their class what every item of theirs is, so
/// they are answered without reading an item; a generic one is read
fn first_misfit<'ctx>(elem_tpe: &SType, coll: &CollValue<'ctx>) -> Option<(usize, Value<'ctx>)> {
    let known_to_fit = match coll {
        CollValue::Bytes(_) => *elem_tpe == SType::SByte,
        CollValue::Booleans(_) => *elem_tpe == SType::SBoolean,
        CollValue::Ints(_) => *elem_tpe == SType::SInt,
        CollValue::Pairs { .. } => pair_item_types(elem_tpe).is_some(),
        CollValue::Generic { .. } => false,
    };
    if known_to_fit || *elem_tpe == SType::SAny {
        return None;
    }
    (0..coll.len()).find_map(|i| {
        coll.get_val(i)
            .filter(|item| !fits_array(elem_tpe, item))
            .map(|item| (i, item))
    })
}

/// The left components of a collection of pairs, as a collection: a `PairOfCols`'s own left
/// side, or each pair's first component
fn left_side<'ctx>(pairs: &CollValue<'ctx>) -> CollValue<'ctx> {
    match pairs {
        CollValue::Pairs { ls, .. } => (**ls).clone(),
        CollValue::Generic { items, .. } => CollValue::Generic {
            elem_tpe: SType::SAny,
            items: items.same_len(
                items
                    .iter()
                    .map(|item| match item {
                        Value::Pair(pair) => pair.0.clone(),
                        other => other.clone(),
                    })
                    .collect(),
            ),
        },
        specialised => specialised.clone(),
    }
}

/// The first component, in item order and the left one before the right one, that its side's
/// array cannot hold, among the pairs of `pairs`
fn component_misfit(lt: &SType, rt: &SType, pairs: &CollValue<'_>) -> Option<TryExtractFromError> {
    match pairs {
        CollValue::Pairs { ls, rs } => match (first_misfit(lt, ls), first_misfit(rt, rs)) {
            (Some((i, l)), Some((j, _))) if i <= j => Some(cannot_hold(lt, &l)),
            (_, Some((_, r))) => Some(cannot_hold(rt, &r)),
            (Some((_, l)), None) => Some(cannot_hold(lt, &l)),
            (None, None) => None,
        },
        other => (0..other.len()).find_map(|i| match other.get_val(i) {
            Some(Value::Pair(pair)) if !fits_array(lt, &pair.0) => Some(cannot_hold(lt, &pair.0)),
            Some(Value::Pair(pair)) if !fits_array(rt, &pair.1) => Some(cannot_hold(rt, &pair.1)),
            _ => None,
        }),
    }
}

/// A collection built part by part, as sigmastate's `flatMap` builds its result. Scala's
/// `flatMap` hands each part to an array builder of the result's item class with `++=` (v6.0.6
/// `CollsOverArrays.scala:70-73`, `:410-411`), and `builder.fromArray` then builds the collection
/// (`:226-232`). So a result is refused, in this order: for an item the array cannot hold, at its
/// part, before the next part exists; for a result of pairs, for a component its side's array
/// cannot hold, in item order and down the LEFT sides only, since both components of every pair
/// are stored before a side is built and the left side is built first (`fromBoxedPairs`,
/// `:205-215`); and last for its length (`:13-14`). `ArrayBuilder.ofUnit` only counts what it is
/// handed, so a `Coll[Unit]` takes items of any class.
///
/// sigmastate keeps every item until the length is asked. This keeps the items only while they
/// can still be a collection, and beyond that the parts themselves, which are shared: what a
/// script can make it hold does not grow with the result
pub struct CollBuilder<'ctx> {
    elem_tpe: SType,
    items: Vec<Value<'ctx>>,
    len: usize,
    parts: Vec<CollValue<'ctx>>,
}

impl<'ctx> CollBuilder<'ctx> {
    /// A builder of a collection of `elem_tpe`
    pub fn new(elem_tpe: SType) -> Self {
        CollBuilder {
            elem_tpe,
            items: Vec::new(),
            len: 0,
            parts: Vec::new(),
        }
    }

    /// Hands the builder a part: its items are stored, or the first one the array cannot hold
    /// is refused
    pub fn append(&mut self, part: &CollValue<'ctx>) -> Result<(), CollBuildError> {
        let of_units = self.elem_tpe == SType::SUnit;
        if !of_units {
            if let Some((_, item)) = first_misfit(&self.elem_tpe, part) {
                return Err(cannot_hold(&self.elem_tpe, &item).into());
            }
        }
        self.len += part.len();
        if check_coll_len(self.len).is_err() {
            self.items = Vec::new();
        } else if of_units {
            self.items
                .extend(core::iter::repeat_n(Value::Unit, part.len()));
        } else {
            self.items.extend(part.as_vec());
        }
        if pair_item_types(&self.elem_tpe).is_some() {
            self.parts.push(part.clone());
        }
        Ok(())
    }

    /// The collection of everything the builder was handed (`fromArray` at the item type)
    pub fn build(self) -> Result<CollValue<'ctx>, CollBuildError> {
        if check_coll_len(self.len).is_ok() {
            return CollValue::from_array(self.elem_tpe, self.items);
        }
        // over the limit: what `fromBoxedPairs` refuses before a side's constructor is reached
        let mut side_tpe = self.elem_tpe;
        let mut depth = 0;
        while let Some((lt, rt)) = pair_item_types(&side_tpe) {
            for part in &self.parts {
                let pairs = (0..depth).fold(part.clone(), |side, _| left_side(&side));
                if let Some(refusal) = component_misfit(&lt, &rt, &pairs) {
                    return Err(refusal.into());
                }
            }
            side_tpe = lt;
            depth += 1;
        }
        Err(CollLengthError(self.len).into())
    }

    /// How many items the builder holds (not how many it was handed)
    #[cfg(test)]
    fn kept(&self) -> usize {
        self.items.len()
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
fn from_data(
    elem_tpe: SType,
    items: Vec<Value<'static>>,
) -> Result<CollValue<'static>, CollLengthError> {
    match split_pairs(&elem_tpe, items) {
        Ok((lt, rt, ls, rs)) => Ok(CollValue::pairs(from_data(lt, ls)?, from_data(rt, rs)?)),
        Err(items) => CollValue::generic(elem_tpe, items),
    }
}

impl From<CollLengthError> for TryExtractFromError {
    fn from(e: CollLengthError) -> Self {
        TryExtractFromError(format!("{e}"))
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
impl TryFrom<CollKind<Literal>> for CollValue<'static> {
    type Error = CollLengthError;

    fn try_from(coll: CollKind<Literal>) -> Result<Self, Self::Error> {
        Ok(match coll {
            CollKind::NativeColl(NativeColl::CollByte(bytes)) => {
                CollValue::Bytes(CollArray::new(bytes)?)
            }
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
                let values = || {
                    items
                        .iter()
                        .cloned()
                        .map(Value::try_from)
                        .collect::<Result<Vec<_>, _>>()
                };
                match (bits, elem_tpe) {
                    (Some(bits), _) => CollValue::Booleans(CollArray::new(bits)?),
                    // the collection the decoder reads: one of tuples of other than two items
                    // is built at the item type `Coll[Any]`, the class of such a tuple's data,
                    // not at the tuple type (v6.0.6 `CoreDataSerializer.scala:160-163`; SANTA
                    // `Tuple.non_pair_value_producers` #5). Only that collection: a pair
                    // collection's sides keep their own types ([`from_data`])
                    (None, SType::STuple(t)) if t.items.len() != 2 => {
                        CollValue::generic(SType::SColl(SType::SAny.into()), values()?)?
                    }
                    (None, elem_tpe) => from_data(elem_tpe, values()?)?,
                }
            }
        })
    }
}

/// A runtime collection as data, in the data representation whatever its form: a `Coll[Byte]` as
/// native bytes, anything else as a list of items, so that it serializes the same
impl<'ctx> TryFrom<CollValue<'ctx>> for CollKind<Literal> {
    type Error = String;

    fn try_from(coll: CollValue<'ctx>) -> Result<Self, Self::Error> {
        match coll {
            CollValue::Bytes(bytes) => Ok(CollKind::NativeColl(NativeColl::CollByte(
                bytes.arc().clone(),
            ))),
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
            (
                SType::SAvlTree,
                Value::try_from(force_any_val::<AvlTreeData>()).unwrap(),
            ),
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
                Value::Coll(CollValue::Ints([1].try_into().unwrap())),
            ),
            (
                SType::SFunc(SFunc::new(vec![SType::SInt], SType::SInt).into()),
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
        let coll_of_longs =
            Value::Coll(CollValue::generic(SType::SLong, vec![Value::Long(1)]).unwrap());
        let option_of_long = Value::Opt(Some(Box::new(Value::Long(1))));
        let triple = SType::STuple(STuple::triple(SType::SInt, SType::SInt, SType::SInt).into());
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
        let tpe = SType::STuple(STuple::pair(SType::SInt, SType::SBox).into());
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
        CollValue::Bytes(b.try_into().unwrap())
    }

    fn generic(t: SType, items: Vec<Value<'static>>) -> CollValue<'static> {
        CollValue::generic(t, items).unwrap()
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
            CollValue::Booleans([true].as_slice().try_into().unwrap()),
            generic(SType::SBoolean, vec![Value::Boolean(true)])
        );
        assert_eq!(
            CollValue::Ints([0, 1].as_slice().try_into().unwrap()),
            generic(SType::SInt, ints(&[0, 1]))
        );
        assert_eq!(
            CollValue::pairs(
                generic(SType::SInt, ints(&[1, 3])),
                generic(SType::SInt, ints(&[2, 4]))
            ),
            generic(
                SType::STuple(STuple::pair(SType::SInt, SType::SInt).into()),
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
        let v = Value::Coll(CollValue::Booleans(
            [true, false].as_slice().try_into().unwrap(),
        ));
        assert_eq!(
            v.try_extract_into::<Vec<bool>>().unwrap(),
            vec![true, false]
        );
        let v = Value::Coll(CollValue::Ints([0, 1].as_slice().try_into().unwrap()));
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
            CollValue::Ints([0, 1, 2].as_slice().try_into().unwrap()),
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
        let triple = SType::STuple(STuple::triple(SType::SInt, SType::SInt, SType::SInt).into());
        let item = Literal::Tup(
            vec![Literal::Int(1), Literal::Int(2), Literal::Int(3)]
                .try_into()
                .unwrap(),
        );
        let coll_of_any = SType::SColl(SType::SAny.into());
        for items in [vec![item], vec![]] {
            let len = items.len();
            let coll = CollValue::try_from(CollKind::WrappedColl {
                elem_tpe: triple.clone(),
                items: items.into(),
            })
            .unwrap();
            assert!(matches!(&coll, CollValue::Generic { .. }));
            assert_eq!(coll.elem_tpe(), coll_of_any);
            assert_eq!(coll.len(), len);
            if len == 1 {
                let ints = vec![Value::Int(1), Value::Int(2), Value::Int(3)];
                assert_eq!(
                    coll.get_val(0),
                    Some(Value::Coll(CollValue::generic(SType::SAny, ints).unwrap()))
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
        let sides = CollValue::try_from(CollKind::WrappedColl {
            elem_tpe: SType::STuple(STuple::pair(triple.clone(), SType::SInt).into()),
            items: vec![Literal::Tup([triple_lit(), Literal::Int(4)].into())].into(),
        })
        .unwrap();
        match &sides {
            CollValue::Pairs { ls, .. } => assert_eq!(ls.elem_tpe(), triple),
            other => panic!("{other:?}"),
        }
        // and a collection EXPRESSION's value is built at the element type
        // (`ConcreteCollection.value`), so it differs from the same collection as data
        let ints = vec![Value::Int(1), Value::Int(2), Value::Int(3)];
        let item = Value::Coll(CollValue::generic(SType::SAny, ints).unwrap());
        let expr_coll = CollValue::from_expression_items(triple.clone(), vec![item]).unwrap();
        assert_eq!(expr_coll.elem_tpe(), triple);
        let data_coll = CollValue::try_from(CollKind::WrappedColl {
            elem_tpe: triple.clone(),
            items: vec![triple_lit()].into(),
        })
        .unwrap();
        assert_ne!(expr_coll, data_coll);
        // a collection of pairs is still a `PairOfCols`
        let pair = Literal::Tup([Literal::Int(1), Literal::Int(2)].into());
        let pairs = CollValue::try_from(CollKind::WrappedColl {
            elem_tpe: SType::STuple(STuple::pair(SType::SInt, SType::SInt).into()),
            items: vec![pair].into(),
        })
        .unwrap();
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
                coll(SType::STuple(int_pair.clone().into())),
                wrapped(
                    SType::STuple(int_pair.clone().into()),
                    vec![
                        tup(Literal::Int(1), Literal::Int(2)),
                        tup(Literal::Int(3), Literal::Int(4)),
                    ],
                ),
            ),
            (
                coll(SType::STuple(
                    STuple::pair(SType::SByte, SType::SByte).into(),
                )),
                wrapped(
                    SType::STuple(STuple::pair(SType::SByte, SType::SByte).into()),
                    vec![tup(Literal::Byte(1), Literal::Byte(2))],
                ),
            ),
            (
                coll(coll(SType::SByte)),
                wrapped(coll(SType::SByte), vec![bytes(&[1]), bytes(&[])]),
            ),
            (
                coll(SType::STuple(
                    STuple::pair(SType::STuple(int_pair.clone().into()), SType::SInt).into(),
                )),
                wrapped(
                    SType::STuple(STuple::pair(SType::STuple(int_pair.into()), SType::SInt).into()),
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
            .map(|(_, lit)| match Value::try_from(lit).unwrap() {
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
            let Value::Coll(coll) = Value::try_from(lit.clone()).unwrap() else {
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
                elem_tpe: SType::STuple(STuple::pair(SType::SInt, SType::SInt).into()),
                items: [pair_lit].as_slice().into(),
            }),
            Literal::Coll(CollKind::WrappedColl {
                elem_tpe: SType::SColl(SType::SByte.into()),
                items: [inner].as_slice().into(),
            }),
        ] {
            assert_eq!(
                Literal::try_from(Value::try_from(lit.clone()).unwrap()),
                Ok(lit)
            );
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod length_limit {
    //! sigmastate's collection class refuses an array of more than 100,000 items in its
    //! constructor (v6.0.6 `CollsOverArrays.scala:13-14`, `MaxArrayLength`, `util/package.scala:7`):
    //! here the array's type does, so every way to a collection value passes it

    use alloc::boxed::Box;
    use alloc::format;
    use alloc::vec;
    use alloc::vec::Vec;

    use super::{CollArray, CollBuildError, CollLengthError, CollValue};
    use crate::mir::avl_tree_data::{AvlTreeData, AvlTreeFlags, AvlTreeValue};
    use crate::mir::constant::Literal;
    use crate::mir::value::{CollKind, NativeColl, Value};
    use crate::serialization::sigma_byte_reader::MAX_ARRAY_LENGTH;
    use crate::types::stuple::STuple;
    use crate::types::stype::SType;

    const AT: usize = MAX_ARRAY_LENGTH;
    const OVER: usize = MAX_ARRAY_LENGTH + 1;
    const TOO_LONG: CollLengthError = CollLengthError(OVER);

    fn longs(n: usize) -> Vec<Value<'static>> {
        vec![Value::Long(1); n]
    }

    #[test]
    fn the_limit_is_sigmastates() {
        assert_eq!(MAX_ARRAY_LENGTH, 100_000);
        assert_eq!(
            format!("{TOO_LONG}"),
            "Cannot create collection with size 100001 greater than 100000"
        );
    }

    #[test]
    fn an_array_of_the_limit_builds_and_one_more_does_not() {
        assert_eq!(CollArray::new(vec![0i8; AT]).unwrap().len(), AT);
        assert_eq!(CollArray::new(vec![0i8; OVER]), Err(TOO_LONG));
        assert_eq!(CollArray::new(longs(AT)).unwrap().len(), AT);
        assert_eq!(CollArray::new(longs(OVER)), Err(TOO_LONG));
        assert_eq!(CollArray::<i32>::try_from(vec![0; OVER]), Err(TOO_LONG));
        assert_eq!(
            CollArray::<bool>::try_from(vec![true; OVER].as_slice()),
            Err(TOO_LONG)
        );
        // a count the TYPE bounds needs no check
        assert_eq!(&*CollArray::from_fixed([1i8, 2, 3]), &[1i8, 2, 3]);
    }

    #[test]
    fn every_constructor_of_a_collection_refuses() {
        assert!(CollValue::bytes(vec![0i8; AT]).is_ok());
        assert_eq!(CollValue::bytes(vec![0i8; OVER]), Err(TOO_LONG));
        assert!(CollValue::booleans(vec![true; AT]).is_ok());
        assert_eq!(CollValue::booleans(vec![true; OVER]), Err(TOO_LONG));
        assert!(CollValue::ints(vec![0i32; AT]).is_ok());
        assert_eq!(CollValue::ints(vec![0i32; OVER]), Err(TOO_LONG));
        assert!(CollValue::generic(SType::SLong, longs(AT)).is_ok());
        assert_eq!(CollValue::generic(SType::SLong, longs(OVER)), Err(TOO_LONG));
        let too_long = Err(CollBuildError::TooLong(TOO_LONG));
        assert!(CollValue::from_array(SType::SLong, longs(AT)).is_ok());
        assert_eq!(CollValue::from_array(SType::SLong, longs(OVER)), too_long);
        assert_eq!(
            CollValue::from_expression_items(SType::SLong, longs(OVER)),
            too_long
        );
        // a pair item type builds its two sides, each through the same constructor
        let pair_tpe = SType::STuple(STuple::pair(SType::SLong, SType::SLong).into());
        let pairs = |n: usize| vec![Value::pair(Value::Long(1), Value::Long(2)); n];
        assert!(CollValue::from_array(pair_tpe.clone(), pairs(AT)).is_ok());
        assert_eq!(CollValue::from_array(pair_tpe, pairs(OVER)), too_long);
    }

    #[test]
    fn an_operation_that_keeps_its_receivers_class_refuses() {
        let too_long = Err(CollBuildError::TooLong(TOO_LONG));
        let receivers = [
            (CollValue::bytes([1i8]).unwrap(), Value::Byte(1)),
            (CollValue::booleans([true]).unwrap(), Value::Boolean(true)),
            (CollValue::ints([1i32]).unwrap(), Value::Int(1)),
            (
                CollValue::generic(SType::SLong, longs(1)).unwrap(),
                Value::Long(1),
            ),
            (
                CollValue::pairs(
                    CollValue::ints([1i32]).unwrap(),
                    CollValue::ints([2i32]).unwrap(),
                ),
                Value::pair(Value::Int(1), Value::Int(2)),
            ),
        ];
        for (receiver, item) in receivers {
            assert_eq!(
                receiver.rebuild_like(vec![item.clone(); AT]).unwrap().len(),
                AT
            );
            assert_eq!(
                receiver.rebuild_like(vec![item; OVER]),
                too_long,
                "{receiver:?}"
            );
        }
    }

    #[test]
    fn data_becomes_a_value_through_the_same_constructor() {
        let bytes = |n: usize| {
            Literal::Coll(CollKind::NativeColl(NativeColl::CollByte(
                vec![0i8; n].into(),
            )))
        };
        let ints = |n: usize| {
            Literal::Coll(CollKind::WrappedColl {
                elem_tpe: SType::SInt,
                items: vec![Literal::Int(1); n].into(),
            })
        };
        let booleans = |n: usize| {
            Literal::Coll(CollKind::WrappedColl {
                elem_tpe: SType::SBoolean,
                items: vec![Literal::Boolean(true); n].into(),
            })
        };
        for data in [bytes, ints, booleans] {
            assert!(Value::try_from(data(AT)).is_ok());
            assert_eq!(Value::try_from(data(OVER)), Err(TOO_LONG));
            // wherever the collection stands in the data
            assert_eq!(
                Value::try_from(Literal::Opt(Some(Box::new(data(OVER))))),
                Err(TOO_LONG)
            );
            assert_eq!(
                Value::try_from(Literal::Tup([Literal::Int(1), data(OVER)].into())),
                Err(TOO_LONG)
            );
            assert_eq!(
                Value::try_from(Literal::Tup(
                    vec![Literal::Int(1), Literal::Int(2), data(OVER)]
                        .try_into()
                        .unwrap()
                )),
                Err(TOO_LONG)
            );
            assert_eq!(
                Value::try_from(Literal::Coll(CollKind::WrappedColl {
                    elem_tpe: SType::SColl(SType::SAny.into()),
                    items: vec![data(OVER)].into(),
                })),
                Err(TOO_LONG)
            );
        }
        assert!(Value::try_from(vec![0i8; AT]).is_ok());
        assert_eq!(Value::try_from(vec![0i8; OVER]), Err(TOO_LONG));
        assert_eq!(Value::try_from(vec![0u8; OVER]), Err(TOO_LONG));
        assert_eq!(Value::try_from(vec![0i64; OVER]), Err(TOO_LONG));
        // a tree's digest is a collection too
        let tree = |n: usize| AvlTreeData {
            digest: vec![0u8; n],
            tree_flags: AvlTreeFlags::new(true, true, true),
            key_length: 32,
            value_length_opt: None,
        };
        assert!(AvlTreeValue::try_from(tree(AT)).is_ok());
        assert_eq!(
            AvlTreeValue::try_from(tree(OVER)).map(|_| ()),
            Err(TOO_LONG)
        );
        assert_eq!(Value::try_from(tree(OVER)), Err(TOO_LONG));
    }

    // sigmastate's `PairOfCols` has no check of its own (`CollsOverArrays.scala:274`): its sides
    // are the collections that have one
    #[test]
    fn a_collection_of_pairs_is_as_long_as_its_sides() {
        let side = || CollValue::bytes(vec![0i8; AT]).unwrap();
        assert_eq!(CollValue::pairs(side(), side()).len(), AT);
    }

    // a refusal's text holds the `Debug` text of the value it names: the array prints as its items
    #[test]
    fn the_text_of_a_collection_is_its_items() {
        assert_eq!(
            format!("{:?}", CollValue::bytes([1i8, 2]).unwrap()),
            "Bytes([1, 2])"
        );
        assert_eq!(
            format!("{:?}", CollValue::generic(SType::SLong, longs(1)).unwrap()),
            "Generic { elem_tpe: SLong, items: [Long(1)] }"
        );
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod part_by_part {
    //! `flatMap`'s builder: sigmastate's order of refusals (measured on the sigma-state 6.0.7
    //! jars: queue item 33's `measure/rows/flat.txt`), and nothing kept past the limit

    use alloc::format;
    use alloc::vec;

    use super::{CollBuildError, CollBuilder, CollLengthError, CollValue};
    use crate::mir::constant::{Literal, TryExtractFromError};
    use crate::mir::value::{CollKind, Value};
    use crate::types::stuple::STuple;
    use crate::types::stype::SType;

    fn too_long<T>(len: usize) -> Result<T, CollBuildError> {
        Err(CollBuildError::TooLong(CollLengthError(len)))
    }

    fn ints(n: usize) -> CollValue<'static> {
        CollValue::generic(SType::SInt, vec![Value::Int(1); n]).unwrap()
    }

    fn longs(n: usize) -> CollValue<'static> {
        CollValue::generic(SType::SLong, vec![Value::Long(1); n]).unwrap()
    }

    fn pair(l: SType, r: SType) -> SType {
        SType::STuple(STuple::pair(l, r).into())
    }

    fn zip(l: &CollValue<'static>, r: &CollValue<'static>) -> CollValue<'static> {
        CollValue::pairs(l.clone(), r.clone())
    }

    fn build(
        elem_tpe: SType,
        parts: &[&CollValue<'static>],
    ) -> Result<CollValue<'static>, CollBuildError> {
        let mut builder = CollBuilder::new(elem_tpe);
        for part in parts {
            builder.append(part)?;
        }
        builder.build()
    }

    /// The refusal of `item` by the array of `tpe`
    fn refuses(result: Result<CollValue<'static>, CollBuildError>, tpe: &str, item: &str) -> bool {
        let text = format!("a collection of {tpe} cannot hold {item}");
        matches!(result, Err(CollBuildError::Item(e)) if e.0 == text)
    }

    // a script can hand the builder one shared part many times: what it keeps does not grow
    #[test]
    fn nothing_is_kept_past_the_limit() {
        let part = CollValue::bytes(vec![0i8; 60_000]).unwrap();
        let mut builder = CollBuilder::new(SType::SByte);
        assert!(builder.append(&part).is_ok());
        assert_eq!(builder.kept(), 60_000);
        for _ in 0..3 {
            assert!(builder.append(&part).is_ok());
            assert_eq!(builder.kept(), 0);
        }
        assert_eq!(builder.build(), too_long(240_000));
    }

    #[test]
    fn an_item_is_refused_at_its_part_and_the_length_last() {
        let (l, w) = (longs(60_000), ints(60_000));
        assert_eq!(build(SType::SLong, &[&l]).map(|c| c.len()), Ok(60_000));
        assert_eq!(build(SType::SLong, &[&l, &l]), too_long(120_000));
        // over the limit before the part of another class: the item is refused all the same
        assert!(refuses(
            build(SType::SLong, &[&l, &l, &w]),
            "SLong",
            "Int(1)"
        ));
        assert!(refuses(
            build(SType::SLong, &[&w, &l, &l]),
            "SLong",
            "Int(1)"
        ));
        // an empty part is never read, whatever its class
        assert_eq!(
            build(SType::SLong, &[&l, &ints(0)]).map(|c| c.len()),
            Ok(60_000)
        );
        // a specialised part is known by its class
        let bytes = CollValue::bytes(vec![0i8; 3]).unwrap();
        assert_eq!(
            build(SType::SByte, &[&bytes, &bytes]).map(|c| c.len()),
            Ok(6)
        );
        assert!(refuses(build(SType::SInt, &[&bytes]), "SInt", "Byte(0)"));
    }

    // `ArrayBuilder.ofUnit` only counts what it is handed
    #[test]
    fn a_builder_of_units_takes_any_item() {
        let w = ints(60_000);
        let units = CollValue::generic(SType::SUnit, vec![Value::Unit; 60_000]);
        assert_eq!(build(SType::SUnit, &[&w]).ok(), units.ok());
        assert_eq!(build(SType::SUnit, &[&w, &w]), too_long(120_000));
    }

    #[test]
    fn a_pairs_component_is_refused_before_a_length() {
        let (w, l) = (ints(60_000), longs(60_000));
        let int_long = pair(SType::SInt, SType::SLong);
        let ok = zip(&w, &l);
        assert_eq!(build(int_long.clone(), &[&ok]).map(|c| c.len()), Ok(60_000));
        assert_eq!(build(int_long.clone(), &[&ok, &ok]), too_long(120_000));
        // the right component of another class, under and over the limit
        assert!(refuses(
            build(int_long.clone(), &[&zip(&w, &w)]),
            "SLong",
            "Int(1)"
        ));
        assert!(refuses(
            build(int_long.clone(), &[&ok, &ok, &zip(&w, &w)]),
            "SLong",
            "Int(1)"
        ));
        // both components of another class: the left one is stored first
        assert!(refuses(
            build(int_long.clone(), &[&ok, &ok, &zip(&l, &w)]),
            "SInt",
            "Long(1)"
        ));
        // a part that holds no pairs is refused at its part
        assert!(matches!(
            build(int_long.clone(), &[&ok, &ok, &w]),
            Err(CollBuildError::Item(_))
        ));
        // a generic collection of pairs is read item by item
        let boxed = CollValue::generic(
            int_long.clone(),
            vec![Value::pair(Value::Int(1), Value::Int(1)); 60_000],
        )
        .unwrap();
        assert!(refuses(
            build(int_long, &[&ok, &ok, &boxed]),
            "SLong",
            "Int(1)"
        ));
    }

    // the left side is built before the right one: a nested component on the left is refused
    // before the length, one on the right after it
    #[test]
    fn nested_pairs_are_refused_down_the_left_sides_only() {
        let (w, l) = (ints(60_000), longs(60_000));
        let int_long = pair(SType::SInt, SType::SLong);
        let (ok, wrong) = (zip(&w, &l), zip(&w, &w));
        let on_the_left = pair(int_long.clone(), SType::SInt);
        assert!(refuses(
            build(on_the_left.clone(), &[&zip(&ok, &w), &zip(&wrong, &w)]),
            "SLong",
            "Int(1)"
        ));
        assert_eq!(
            build(on_the_left, &[&zip(&ok, &w), &zip(&ok, &w)]),
            too_long(120_000)
        );
        let on_the_right = pair(SType::SInt, int_long);
        assert_eq!(
            build(on_the_right.clone(), &[&zip(&w, &ok), &zip(&w, &wrong)]),
            too_long(120_000)
        );
        assert!(refuses(
            build(on_the_right, &[&zip(&w, &wrong)]),
            "SLong",
            "Int(1)"
        ));
    }

    // `fromBoxedPairs` stores both components of every pair before it builds a side
    #[test]
    fn from_array_stores_both_components_before_it_builds_a_side() {
        let int_long = pair(SType::SInt, SType::SLong);
        let over = vec![Value::pair(Value::Int(1), Value::Int(2)); 100_001];
        assert!(refuses(
            CollValue::from_array(int_long.clone(), over),
            "SLong",
            "Int(2)"
        ));
        // in item order: the first pair's right component before the second pair's left one
        let items = vec![
            Value::pair(Value::Int(1), Value::Int(2)),
            Value::pair(Value::Long(3), Value::Long(4)),
        ];
        assert!(refuses(
            CollValue::from_array(int_long.clone(), items),
            "SLong",
            "Int(2)"
        ));
        let items = vec![Value::pair(Value::Long(3), Value::Int(2))];
        assert!(refuses(
            CollValue::from_array(int_long, items),
            "SInt",
            "Long(3)"
        ));
    }

    // a specialised part and a `PairOfCols` say by their class what they hold: one of another
    // class is refused at its part like any other, over the limit too, and an empty one holds
    // nothing to refuse (measured: queue item 33's `measure/rows/flat2.txt`)
    #[test]
    fn a_specialised_part_of_another_class_is_refused_at_its_part() {
        let item_refused = |r: Result<CollValue<'static>, CollBuildError>| {
            matches!(r, Err(CollBuildError::Item(_)))
        };
        let l = longs(60_000);
        let z = zip(&ints(60_000), &l);
        let parts = [
            CollValue::bytes(vec![1i8; 3]).unwrap(),
            CollValue::booleans(vec![true; 3]).unwrap(),
            CollValue::ints(vec![1i32; 3]).unwrap(),
            zip(&ints(3), &longs(3)),
        ];
        for part in &parts {
            assert!(item_refused(build(SType::SLong, &[part, &l])), "{part:?}");
            assert!(
                item_refused(build(SType::SLong, &[&l, &l, part])),
                "{part:?}"
            );
        }
        for part in &parts[..3] {
            let pairs = pair(SType::SInt, SType::SLong);
            assert!(item_refused(build(pairs.clone(), &[part, &z])), "{part:?}");
            assert!(item_refused(build(pairs, &[&z, &z, part])), "{part:?}");
        }
        let empty = CollValue::bytes(vec![0i8; 0]).unwrap();
        assert_eq!(
            build(SType::SLong, &[&empty, &l]).map(|c| c.len()),
            Ok(60_000)
        );
        assert_eq!(build(SType::SLong, &[&l, &l, &empty]), too_long(120_000));
    }

    // an item built at evaluation can be large: its refusal names its start
    #[test]
    fn the_refusal_of_an_item_is_short() {
        let large = Value::Coll(CollValue::bytes(vec![0i8; 100_000]).unwrap());
        let refused = CollValue::from_array(SType::SInt, vec![large]);
        assert!(
            matches!(&refused, Err(CollBuildError::Item(e)) if e.0.len() < 300),
            "{refused:?}"
        );
        assert!(refuses(
            CollValue::from_array(SType::SInt, vec![Value::Long(1)]),
            "SInt",
            "Long(1)"
        ));
    }

    // data of tuples of three is built at the item type `Coll[Any]`, through the same constructor
    #[test]
    fn data_of_tuples_of_three_passes_the_constructor() {
        let triple = SType::STuple(STuple::triple(SType::SInt, SType::SInt, SType::SInt).into());
        let data = |n: usize| {
            let item = Literal::Tup(vec![Literal::Int(1); 3].try_into().unwrap());
            Literal::Coll(CollKind::WrappedColl {
                elem_tpe: triple.clone(),
                items: vec![item; n].into(),
            })
        };
        assert!(Value::try_from(data(100_000)).is_ok());
        assert_eq!(
            Value::try_from(data(100_001)),
            Err(CollLengthError(100_001))
        );
    }

    // the refusal carried as another error keeps sigmastate's sentence
    #[test]
    fn the_refusal_keeps_its_text_in_another_error() {
        let sentence = "Cannot create collection with size 100001 greater than 100000";
        let refusal = CollLengthError(100_001);
        assert_eq!(TryExtractFromError::from(refusal.clone()).0, sentence);
        assert_eq!(
            TryExtractFromError::from(CollBuildError::TooLong(refusal.clone())).0,
            sentence
        );
        assert_eq!(format!("{}", CollBuildError::TooLong(refusal)), sentence);
        let item = TryExtractFromError("an item of another class".into());
        assert_eq!(
            TryExtractFromError::from(CollBuildError::Item(item.clone())),
            item
        );
        assert_eq!(
            format!("{}", CollBuildError::Item(item.clone())),
            format!("{item}")
        );
    }
}
