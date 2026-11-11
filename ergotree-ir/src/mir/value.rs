//! Ergo data type

use alloc::boxed::Box;

use alloc::string::ToString;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::convert::TryInto;
use core::fmt::Formatter;
use hashbrown::HashMap;

use impl_trait_for_tuples::impl_for_tuples;
use sigma_util::AsVecI8;

use crate::bigint256::BigInt256;
use crate::chain::ergo_box::ErgoBox;
use crate::reference::Ref;
use crate::sigma_protocol::sigma_boolean::SigmaProp;
use crate::types::stuple::TupleItems;
use crate::types::stype::LiftIntoSType;
use crate::types::stype::SType;
use crate::unsignedbigint256::UnsignedBigInt;
use ergo_chain_types::{EcPoint, Header, PreHeader};

use super::avl_tree_data::AvlTreeData;
use super::coll_value::CollValue;
use super::constant::Literal;
use super::constant::TryExtractFrom;
use super::constant::TryExtractFromError;
use super::constant::TryExtractInto;
use super::expr::Expr;
use super::func_value::FuncArg;
use super::val_def::ValId;

extern crate derive_more;
use derive_more::From;

#[derive(PartialEq, Eq, Debug, Clone)]
/// Collection for primitive values (i.e byte array)
pub enum NativeColl {
    /// Collection of bytes
    CollByte(Arc<[i8]>),
}

impl NativeColl {
    /// Collection element type
    pub fn elem_tpe(&self) -> &SType {
        match self {
            NativeColl::CollByte(_) => &SType::SByte,
        }
    }
}

/// Collection elements
#[derive(PartialEq, Eq, Debug, Clone)]
pub enum CollKind<T> {
    /// Collection elements stored as a vector of Rust values
    NativeColl(NativeColl),
    /// Collection elements stored as a vector of Value's
    WrappedColl {
        /// Collection element type
        elem_tpe: SType,
        /// Collection elements
        items: Arc<[T]>,
    },
}

impl<T> CollKind<T>
where
    T: PartialEq + Eq + Clone,
    T: From<i8>,
    i8: TryExtractFrom<T>,
    Vec<i8>: TryExtractFrom<T>,
    Vec<T>: TryExtractFrom<T>,
{
    /// Build a collection from items, storing them as Rust types values when neccessary
    pub fn from_collection(
        elem_tpe: SType,
        items: impl Into<Arc<[T]>>,
    ) -> Result<CollKind<T>, TryExtractFromError> {
        match elem_tpe {
            SType::SByte => items
                .into()
                .iter()
                .cloned()
                .map(|v| v.try_extract_into::<i8>())
                .collect::<Result<Arc<[_]>, _>>()
                .map(|bytes| CollKind::NativeColl(NativeColl::CollByte(bytes))),
            _ => Ok(CollKind::WrappedColl {
                elem_tpe,
                items: items.into(),
            }),
        }
    }

    /// Build a collection from items where each is a collection as well, flattening the arrays
    /// This will convert a Coll[Coll\[T\]] to a Coll\[T\]
    pub fn from_vec_vec(
        elem_tpe: SType,
        items: Vec<T>,
    ) -> Result<CollKind<T>, TryExtractFromError> {
        match elem_tpe {
            SType::SColl(inner_type) if matches!(&*inner_type, SType::SByte) => items
                .into_iter()
                .map(|v| v.try_extract_into::<Vec<i8>>())
                .collect::<Result<Vec<_>, _>>()
                .map(|bytes| CollKind::NativeColl(NativeColl::CollByte(bytes.concat().into()))),
            SType::SColl(flat_type) => items
                .into_iter()
                .map(|v| v.try_extract_into::<Vec<T>>())
                .collect::<Result<Vec<_>, _>>()
                .map(|v| CollKind::WrappedColl {
                    elem_tpe: (*flat_type).clone(),
                    items: v.into_iter().flat_map(Vec::into_iter).collect(),
                }),
            _ => Err(TryExtractFromError(format!(
                "Expected Value::Coll, got: {:?}",
                elem_tpe
            ))),
        }
    }

    /// Collection element type
    pub fn elem_tpe(&self) -> &SType {
        match self {
            CollKind::NativeColl(ncoll) => match ncoll {
                NativeColl::CollByte(_) => &SType::SByte,
            },
            CollKind::WrappedColl { elem_tpe, .. } => elem_tpe,
        }
    }

    /// Return items, as vector of Values
    pub fn as_vec(&self) -> Vec<T> {
        match self {
            CollKind::NativeColl(NativeColl::CollByte(coll_byte)) => {
                coll_byte.clone().iter().map(|&byte| byte.into()).collect()
            }
            CollKind::WrappedColl {
                elem_tpe: _,
                items: v,
            } => v.clone().to_vec(),
        }
    }

    /// Return size of Coll
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        match self {
            CollKind::NativeColl(NativeColl::CollByte(coll_byte)) => coll_byte.len(),
            CollKind::WrappedColl { elem_tpe: _, items } => items.len(),
        }
    }

    /// Reverse Coll
    pub fn reverse(&self) -> Self {
        match self {
            CollKind::NativeColl(NativeColl::CollByte(arr)) => {
                Self::NativeColl(NativeColl::CollByte(arr.iter().copied().rev().collect()))
            }
            CollKind::WrappedColl { elem_tpe, items } => Self::WrappedColl {
                elem_tpe: elem_tpe.clone(),
                items: items.iter().cloned().rev().collect(),
            },
        }
    }

    /// Returns `true` if `self` begins with `prefix`
    pub fn starts_with(&self, prefix: &CollKind<T>) -> bool {
        match (self, prefix) {
            (
                CollKind::NativeColl(NativeColl::CollByte(s)),
                CollKind::NativeColl(NativeColl::CollByte(prefix)),
            ) => s.starts_with(prefix),
            (
                CollKind::WrappedColl { elem_tpe: _, items },
                CollKind::WrappedColl {
                    elem_tpe: _,
                    items: prefix,
                },
            ) => items.starts_with(prefix),
            _ => false,
        }
    }

    /// Returns `true` if `self` ends with suffix
    pub fn ends_with(&self, suffix: &CollKind<T>) -> bool {
        match (self, suffix) {
            (
                CollKind::NativeColl(NativeColl::CollByte(s)),
                CollKind::NativeColl(NativeColl::CollByte(suffix)),
            ) => s.ends_with(suffix),
            (
                CollKind::WrappedColl { elem_tpe: _, items },
                CollKind::WrappedColl {
                    elem_tpe: _,
                    items: suffix,
                },
            ) => items.ends_with(suffix),
            _ => false,
        }
    }

    /// Index the array. Returns None if out of bounds
    pub fn get_val(&self, index: usize) -> Option<T> {
        match self {
            CollKind::NativeColl(NativeColl::CollByte(coll_byte)) => {
                coll_byte.get(index).map(|&byte| byte.into())
            }
            CollKind::WrappedColl { elem_tpe: _, items } => items.get(index).cloned(),
        }
    }
}

/// Bindings of an environment, shared: a lambda captures its defining environment by holding
/// them, as the JVM's closure holds its immutable `DataEnv` (v6.0.7 `values.scala:1069-1086`).
/// A captured lambda holds its own, so the values form a graph whose shared parts must not be
/// copied or walked once per path: [`Value::to_static`] and a lambda's `Debug` do neither
pub type Bindings<'ctx> = Arc<HashMap<ValId, Value<'ctx>>>;

/// Converts `bindings` to `'static`, each shared map once (see [`Value::to_static`])
pub fn bindings_to_static<'ctx>(bindings: &'ctx Bindings<'ctx>) -> Bindings<'static> {
    bindings_to_static_shared(bindings, &mut StaticMemo::default())
}

/// What one conversion has converted so far, by the address it came from: captured maps,
/// collections' items and borrowed boxes, each shared by many values, so each is converted
/// once however many paths reach it
#[derive(Default)]
struct StaticMemo {
    maps: HashMap<usize, Bindings<'static>>,
    items: HashMap<usize, Arc<[Value<'static>]>>,
    boxes: HashMap<usize, Arc<ErgoBox>>,
}

/// [`Value::to_static_shared`] for a collection: its form kept, each generic item list converted
/// once per conversion
fn coll_to_static_shared<'ctx>(
    coll: &'ctx CollValue<'ctx>,
    memo: &mut StaticMemo,
) -> CollValue<'static> {
    match coll {
        CollValue::Bytes(items) => CollValue::Bytes(items.clone()),
        CollValue::Booleans(items) => CollValue::Booleans(items.clone()),
        CollValue::Ints(items) => CollValue::Ints(items.clone()),
        CollValue::Generic { elem_tpe, items } => {
            let key = Arc::as_ptr(items) as *const Value as usize;
            let items = match memo.items.get(&key) {
                Some(done) => done.clone(),
                None => {
                    let done: Arc<[Value<'static>]> =
                        items.iter().map(|v| v.to_static_shared(memo)).collect();
                    memo.items.insert(key, done.clone());
                    done
                }
            };
            CollValue::Generic {
                elem_tpe: elem_tpe.clone(),
                items,
            }
        }
        CollValue::Pairs { ls, rs } => CollValue::pairs(
            coll_to_static_shared(ls, memo),
            coll_to_static_shared(rs, memo),
        ),
    }
}

fn bindings_to_static_shared<'ctx>(
    bindings: &'ctx Bindings<'ctx>,
    memo: &mut StaticMemo,
) -> Bindings<'static> {
    let key = Arc::as_ptr(bindings) as usize;
    if let Some(done) = memo.maps.get(&key) {
        return done.clone();
    }
    let done: Bindings<'static> = Arc::new(
        bindings
            .iter()
            .map(|(k, v)| (*k, v.to_static_shared(memo)))
            .collect(),
    );
    memo.maps.insert(key, done.clone());
    done
}

/// Lambda
#[derive(PartialEq, Eq, Clone)]
pub struct Lambda<'ctx> {
    /// Argument placeholders
    pub args: Vec<FuncArg>,
    /// Body, shared with the `FuncValue` it was evaluated from
    pub body: Arc<Expr>,
    /// Bindings captured from the defining environment. The JVM's
    /// `FuncValue.eval` returns a closure over the env it was created in, so
    /// a lambda that escapes its creation site (returned from another lambda,
    /// bound to a `val`) still sees those bindings when applied — the
    /// caller's environment at application time plays no role.
    pub captured: Bindings<'ctx>,
}

impl core::fmt::Debug for Lambda<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        // the ids only: a captured lambda holds its own captured values, so printing them
        // printed every earlier definition again, once per path to it
        let mut captured: Vec<&ValId> = self.captured.keys().collect();
        captured.sort();
        f.debug_struct("Lambda")
            .field("args", &self.args)
            .field("body", &self.body)
            .field("captured", &captured)
            .finish()
    }
}

/// Runtime value
#[derive(PartialEq, Eq, Debug, Clone, From)]
pub enum Value<'ctx> {
    /// Boolean
    Boolean(bool),
    /// Byte
    Byte(i8),
    /// Short
    Short(i16),
    /// Int
    Int(i32),
    /// Long
    Long(i64),
    /// Unit struct
    Unit,
    /// Big integer
    BigInt(BigInt256),
    /// GroupElement
    GroupElement(Ref<'ctx, EcPoint>),
    /// Sigma property
    SigmaProp(Box<SigmaProp>),
    /// UnsignedBigInt
    UnsignedBigInt(UnsignedBigInt),
    /// Ergo box
    CBox(Ref<'ctx, ErgoBox>),
    /// AVL tree
    AvlTree(Box<AvlTreeData>),
    /// Collection of values of the same type, in the form of the JVM object that holds it
    Coll(CollValue<'ctx>),
    /// Tuple (arbitrary type values)
    Tup(TupleItems<Value<'ctx>>),
    /// Transaction(and blockchain) context info
    Context,
    /// String type
    String(Arc<str>),
    /// Block header
    Header(Box<Header>),
    /// Header with predictable data
    PreHeader(Box<PreHeader>),
    /// Global which is used to define global methods
    Global,
    /// Optional value
    Opt(Option<Box<Value<'ctx>>>),
    /// lambda
    Lambda(Lambda<'ctx>),
}

impl<'ctx> Value<'ctx> {
    /// Convert a Value<'ctx> to a Value<'static>.
    /// Useful for returning errors especially in bindings where we can't return  borrowed values
    pub fn to_static(&'ctx self) -> Value<'static> {
        self.to_static_shared(&mut StaticMemo::default())
    }

    /// [`Value::to_static`], converting each shared captured map once per conversion
    fn to_static_shared(&'ctx self, memo: &mut StaticMemo) -> Value<'static> {
        match self {
            Value::Boolean(b) => Value::Boolean(*b),
            Value::Byte(b) => Value::Byte(*b),
            Value::Short(b) => Value::Short(*b),
            Value::Int(b) => Value::Int(*b),
            Value::Long(b) => Value::Long(*b),
            Value::Unit => Value::Unit,
            Value::BigInt(b) => Value::BigInt(*b),
            Value::GroupElement(b) => Value::GroupElement(b.to_static()),
            Value::SigmaProp(p) => Value::SigmaProp(p.clone()),
            Value::UnsignedBigInt(b) => Value::UnsignedBigInt(*b),
            Value::AvlTree(t) => Value::AvlTree(t.clone()),
            Value::Coll(coll) => Value::Coll(coll_to_static_shared(coll, memo)),
            Value::Tup(tup) => Value::Tup(tup.mapped_ref(|v| v.to_static_shared(memo))),
            Value::Context => Value::Context,
            Value::String(s) => Value::String(s.clone()),
            Value::Header(h) => Value::Header(h.clone()),
            Value::PreHeader(h) => Value::PreHeader(h.clone()),
            Value::Global => Value::Global,
            Value::CBox(Ref::Arc(b)) => Value::CBox(Ref::Arc(b.clone())),
            Value::CBox(Ref::Borrowed(b)) => {
                let key = *b as *const ErgoBox as usize;
                let owned = memo
                    .boxes
                    .entry(key)
                    .or_insert_with(|| Arc::new((*b).clone()))
                    .clone();
                Value::CBox(Ref::Arc(owned))
            }
            Value::Opt(opt) => Value::Opt(
                Option::as_ref(opt)
                    .map(|o| o.to_static_shared(memo))
                    .map(Box::new),
            ),
            Value::Lambda(l) => Value::Lambda(Lambda {
                args: l.args.clone(),
                body: l.body.clone(),
                captured: bindings_to_static_shared(&l.captured, memo),
            }),
        }
    }
}

impl<'ctx> Value<'ctx> {
    /// Create Sigma property constant
    pub fn sigma_prop(prop: SigmaProp) -> Value<'ctx> {
        Value::SigmaProp(Box::new(prop))
    }
}

impl<'ctx, T: Into<SigmaProp>> From<T> for Value<'ctx> {
    fn from(t: T) -> Self {
        Value::SigmaProp(Box::new(t.into()))
    }
}

impl From<EcPoint> for Value<'static> {
    fn from(v: EcPoint) -> Self {
        Value::GroupElement(Ref::from(v))
    }
}

impl From<Arc<EcPoint>> for Value<'static> {
    fn from(v: Arc<EcPoint>) -> Self {
        Value::GroupElement(Ref::from(v))
    }
}

impl<'ctx> From<&'ctx EcPoint> for Value<'ctx> {
    fn from(v: &'ctx EcPoint) -> Self {
        Value::GroupElement(Ref::from(v))
    }
}

impl<'ctx> From<Vec<i8>> for Value<'ctx> {
    fn from(v: Vec<i8>) -> Self {
        Value::Coll(CollValue::Bytes(v.into()))
    }
}

impl<'ctx> From<Vec<u8>> for Value<'ctx> {
    fn from(v: Vec<u8>) -> Self {
        Value::Coll(CollValue::Bytes(v.as_vec_i8().into()))
    }
}

impl<'ctx, T: Into<Value<'ctx>>> From<Option<T>> for Value<'ctx> {
    fn from(opt: Option<T>) -> Self {
        Value::Opt(opt.map(|v| v.into()).map(Box::new))
    }
}

impl From<Literal> for Value<'static> {
    fn from(lit: Literal) -> Self {
        match lit {
            Literal::Boolean(b) => Value::Boolean(b),
            Literal::Byte(b) => Value::Byte(b),
            Literal::Short(s) => Value::Short(s),
            Literal::Int(i) => Value::Int(i),
            Literal::Long(l) => Value::Long(l),
            Literal::BigInt(b) => Value::BigInt(b),
            Literal::String(s) => Value::String(s),
            Literal::Unit => Value::Unit,
            Literal::SigmaProp(s) => Value::SigmaProp(s),
            Literal::UnsignedBigInt(b) => Value::UnsignedBigInt(b),
            Literal::GroupElement(e) => Value::GroupElement(e.into()),
            Literal::CBox(b) => Value::CBox(b),
            Literal::Coll(coll) => Value::Coll(CollValue::from(coll)),
            Literal::AvlTree(a) => Value::AvlTree(a),
            Literal::Header(h) => Value::Header(h),
            Literal::Opt(lit) => Value::Opt(lit.map(|boxed| *boxed).map(Value::from).map(Box::new)),
            // sigmastate's data of a tuple: a `Tuple2` for a pair, a generic `Coll[Any]` for any
            // other count (v6.0.6 `CoreDataSerializer.scala:134-139`, `Evaluation.scala:99-102`)
            Literal::Tup(t) if t.len() == 2 => Value::Tup(t.mapped(Value::from)),
            Literal::Tup(t) => Value::Coll(CollValue::generic(
                SType::SAny,
                t.to_vec().into_iter().map(Value::from).collect::<Vec<_>>(),
            )),
        }
    }
}

impl core::fmt::Display for Value<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        match self {
            Value::Coll(CollValue::Bytes(i8_bytes)) => {
                write!(f, "Coll[Byte](")?;
                for (i, b) in i8_bytes.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", b)?;
                }
                write!(f, ")")
            }
            Value::Coll(coll) => {
                write!(f, "Coll[{}](", coll.elem_tpe())?;
                for (i, item) in coll.as_vec().iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    item.fmt(f)?;
                }
                write!(f, ")")
            }
            Value::Opt(boxed_opt) => {
                if let Some(v) = boxed_opt {
                    write!(f, "Some(")?;
                    v.fmt(f)?;
                    write!(f, ")")
                } else {
                    write!(f, "None")
                }
            }
            Value::Tup(items) => {
                write!(f, "(")?;
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    item.fmt(f)?;
                }
                write!(f, ")")
            }
            Value::Unit => write!(f, "()"),
            Value::Boolean(v) => v.fmt(f),
            Value::Byte(v) => v.fmt(f),
            Value::Short(v) => v.fmt(f),
            Value::Int(v) => v.fmt(f),
            Value::Long(v) => write!(f, "{}L", v),
            Value::BigInt(v) => v.fmt(f),
            Value::String(v) => v.fmt(f),
            Value::SigmaProp(v) => v.fmt(f),
            Value::UnsignedBigInt(v) => v.fmt(f),
            Value::GroupElement(v) => v.fmt(f),
            Value::AvlTree(v) => write!(f, "AvlTree({:?})", v),
            Value::CBox(v) => write!(f, "ErgoBox({:?})", v),
            Value::Context => write!(f, "CONTEXT"),
            Value::Header(_) => write!(f, "HEADER"),
            Value::PreHeader(_) => write!(f, "PREHEADER"),
            Value::Global => write!(f, "GLOBAL"),
            Value::Lambda(v) => write!(f, "{v:?}"),
        }
    }
}

/// Marker trait to select types which stored as Vec of wrapped Value's
pub trait StoreWrapped {}

impl StoreWrapped for bool {}
impl StoreWrapped for i16 {}
impl StoreWrapped for i32 {}
impl StoreWrapped for i64 {}
impl StoreWrapped for BigInt256 {}
impl StoreWrapped for Header {}
impl StoreWrapped for ErgoBox {}
impl StoreWrapped for Arc<str> {}
impl StoreWrapped for Ref<'_, ErgoBox> {}
impl StoreWrapped for EcPoint {}
impl StoreWrapped for SigmaProp {}
impl StoreWrapped for UnsignedBigInt {}
impl<T: StoreWrapped> StoreWrapped for Option<T> {}
impl<T> StoreWrapped for Vec<T> {}
impl StoreWrapped for Value<'_> {}
impl StoreWrapped for Literal {}

#[impl_for_tuples(2, 4)]
impl StoreWrapped for Tuple {}

impl<'ctx, T: LiftIntoSType + StoreWrapped + Into<Value<'ctx>>> From<Vec<T>> for Value<'ctx> {
    fn from(v: Vec<T>) -> Self {
        Value::Coll(CollValue::generic(
            T::stype(),
            v.into_iter().map(|i| i.into()).collect::<Arc<[_]>>(),
        ))
    }
}

#[allow(clippy::from_over_into)]
#[allow(clippy::unwrap_used)]
#[impl_for_tuples(2, 4)]
impl<'ctx> Into<Value<'ctx>> for Tuple {
    fn into(self) -> Value<'ctx> {
        let v: Vec<Value> = [for_tuples!(  #( Tuple.into() ),* )].to_vec();
        Value::Tup(v.try_into().unwrap())
    }
}

impl<'ctx> From<Vec<Ref<'ctx, ErgoBox>>> for Value<'ctx> {
    fn from(v: Vec<Ref<'ctx, ErgoBox>>) -> Self {
        Value::Coll(CollValue::generic(
            SType::SBox,
            v.into_iter().map(|i| i.into()).collect::<Arc<[_]>>(),
        ))
    }
}

impl TryExtractFrom<Value<'_>> for bool {
    fn try_extract_from(cv: Value) -> Result<bool, TryExtractFromError> {
        match cv {
            Value::Boolean(v) => Ok(v),
            _ => Err(TryExtractFromError(format!(
                "expected bool, found {:?}",
                cv
            ))),
        }
    }
}

impl TryExtractFrom<Value<'_>> for i8 {
    fn try_extract_from(cv: Value) -> Result<i8, TryExtractFromError> {
        match cv {
            Value::Byte(v) => Ok(v),
            _ => Err(TryExtractFromError(format!("expected i8, found {:?}", cv))),
        }
    }
}

impl TryExtractFrom<Value<'_>> for i16 {
    fn try_extract_from(cv: Value) -> Result<i16, TryExtractFromError> {
        match cv {
            Value::Short(v) => Ok(v),
            _ => Err(TryExtractFromError(format!("expected i16, found {:?}", cv))),
        }
    }
}

impl TryExtractFrom<Value<'_>> for i32 {
    fn try_extract_from(cv: Value) -> Result<i32, TryExtractFromError> {
        match cv {
            Value::Int(v) => Ok(v),
            _ => Err(TryExtractFromError(format!("expected i32, found {:?}", cv))),
        }
    }
}

impl TryExtractFrom<Value<'_>> for i64 {
    fn try_extract_from(cv: Value) -> Result<i64, TryExtractFromError> {
        match cv {
            Value::Long(v) => Ok(v),
            _ => Err(TryExtractFromError(format!("expected i64, found {:?}", cv))),
        }
    }
}

impl TryExtractFrom<Value<'_>> for EcPoint {
    fn try_extract_from(cv: Value) -> Result<EcPoint, TryExtractFromError> {
        match cv {
            Value::GroupElement(v) => Ok(*v),
            _ => Err(TryExtractFromError(format!(
                "expected EcPoint, found {:?}",
                cv
            ))),
        }
    }
}

impl TryExtractFrom<Value<'_>> for SigmaProp {
    fn try_extract_from(cv: Value) -> Result<SigmaProp, TryExtractFromError> {
        match cv {
            Value::SigmaProp(v) => Ok(*v),
            _ => Err(TryExtractFromError(format!(
                "expected SigmaProp, found {:?}",
                cv
            ))),
        }
    }
}

impl TryExtractFrom<Value<'_>> for UnsignedBigInt {
    fn try_extract_from(cv: Value) -> Result<UnsignedBigInt, TryExtractFromError> {
        match cv {
            Value::UnsignedBigInt(v) => Ok(v),
            _ => Err(TryExtractFromError(format!(
                "expected UnsignedBigInt, found {:?}",
                cv
            ))),
        }
    }
}

impl<'ctx> TryExtractFrom<Value<'ctx>> for Ref<'ctx, ErgoBox> {
    fn try_extract_from(c: Value<'ctx>) -> Result<Self, TryExtractFromError> {
        match c {
            Value::CBox(b) => Ok(b),
            _ => Err(TryExtractFromError(format!(
                "expected ErgoBox, found {:?}",
                c
            ))),
        }
    }
}

impl TryExtractFrom<Value<'_>> for Header {
    fn try_extract_from(c: Value) -> Result<Self, TryExtractFromError> {
        match c {
            Value::Header(h) => Ok(*h),
            _ => Err(TryExtractFromError(format!(
                "expected Header, found {:?}",
                c
            ))),
        }
    }
}

impl TryExtractFrom<Value<'_>> for PreHeader {
    fn try_extract_from(c: Value) -> Result<Self, TryExtractFromError> {
        match c {
            Value::PreHeader(ph) => Ok(*ph),
            _ => Err(TryExtractFromError(format!(
                "expected PreHeader, found {:?}",
                c
            ))),
        }
    }
}

impl<'ctx, T: TryExtractFrom<Value<'ctx>> + StoreWrapped> TryExtractFrom<Value<'ctx>> for Vec<T> {
    fn try_extract_from(c: Value<'ctx>) -> Result<Self, TryExtractFromError> {
        match c {
            Value::Coll(coll) => coll.as_vec().into_iter().map(T::try_extract_from).collect(),
            _ => Err(TryExtractFromError(format!(
                "expected {:?}, found {:?}",
                core::any::type_name::<Self>(),
                c
            ))),
        }
    }
}

impl<'ctx, T: TryExtractFrom<Value<'ctx>> + StoreWrapped, const N: usize>
    TryExtractFrom<Value<'ctx>> for [T; N]
{
    fn try_extract_from(c: Value<'ctx>) -> Result<Self, TryExtractFromError> {
        match c {
            Value::Coll(coll) => {
                let v = coll
                    .as_vec()
                    .into_iter()
                    .map(T::try_extract_from)
                    .collect::<Result<Vec<_>, _>>()?;
                let len = v.len();
                v.try_into().map_err(|_| TryExtractFromError(format!("can't convert vec of {:?} with length of {:?} to array with length of {:?}", core::any::type_name::<T>(), len, N)))
            }
            _ => Err(TryExtractFromError(format!(
                "expected {:?}, found {:?}",
                core::any::type_name::<Self>(),
                c
            ))),
        }
    }
}

impl TryExtractFrom<Value<'_>> for Vec<i8> {
    fn try_extract_from(v: Value) -> Result<Self, TryExtractFromError> {
        match v {
            Value::Coll(CollValue::Bytes(bs)) => Ok(bs.iter().copied().collect()),
            Value::Coll(coll) if coll.elem_tpe() == SType::SByte => coll
                .as_vec()
                .into_iter()
                .map(|v| v.try_extract_into::<i8>())
                .collect(),
            _ => Err(TryExtractFromError(format!(
                "expected {:?}, found {:?}",
                core::any::type_name::<Self>(),
                v
            ))),
        }
    }
}

impl TryExtractFrom<Value<'_>> for Vec<u8> {
    fn try_extract_from(v: Value) -> Result<Self, TryExtractFromError> {
        use sigma_util::FromVecI8;
        Vec::<i8>::try_extract_from(v).map(Vec::<u8>::from_vec_i8)
    }
}

impl<'ctx> TryExtractFrom<Value<'ctx>> for Value<'ctx> {
    fn try_extract_from(v: Value<'ctx>) -> Result<Self, TryExtractFromError> {
        Ok(v)
    }
}

impl TryExtractFrom<Value<'_>> for BigInt256 {
    fn try_extract_from(v: Value) -> Result<Self, TryExtractFromError> {
        match v {
            Value::BigInt(bi) => Ok(bi),
            _ => Err(TryExtractFromError(format!(
                "expected {:?}, found {:?}",
                core::any::type_name::<Self>(),
                v
            ))),
        }
    }
}

impl TryExtractFrom<Value<'_>> for AvlTreeData {
    fn try_extract_from(v: Value) -> Result<Self, TryExtractFromError> {
        match v {
            Value::AvlTree(a) => Ok(*a),
            _ => Err(TryExtractFromError(format!(
                "expected {:?}, found {:?}",
                core::any::type_name::<Self>(),
                v
            ))),
        }
    }
}

impl<'ctx, T: TryExtractFrom<Value<'ctx>> + StoreWrapped> TryExtractFrom<Vec<Value<'ctx>>>
    for Vec<T>
{
    fn try_extract_from(v: Vec<Value<'ctx>>) -> Result<Self, TryExtractFromError> {
        v.into_iter().map(|it| it.try_extract_into::<T>()).collect()
    }
}

impl<'ctx, T: TryExtractFrom<Value<'ctx>>> TryExtractFrom<Value<'ctx>> for Option<T> {
    fn try_extract_from(v: Value<'ctx>) -> Result<Self, TryExtractFromError> {
        match v {
            Value::Opt(opt) => opt.map(|boxed| *boxed).map(T::try_extract_from).transpose(),
            _ => Err(TryExtractFromError(format!(
                "expected Option, found {:?}",
                v
            ))),
        }
    }
}

#[impl_for_tuples(2, 4)]
impl<'ctx> TryExtractFrom<Value<'ctx>> for Tuple {
    fn try_extract_from(v: Value<'ctx>) -> Result<Self, TryExtractFromError> {
        match v {
            Value::Tup(items) => {
                let mut iter = items.iter();
                Ok(for_tuples!( ( #(
                                Tuple::try_extract_from(
                                    iter
                                        .next()
                                        .cloned()
                                        .ok_or_else(|| TryExtractFromError("not enough items in STuple".to_string()))?
                                )?
                                ),* ) ))
            }
            _ => Err(TryExtractFromError(format!(
                "expected Context, found {:?}",
                v
            ))),
        }
    }
}

#[cfg(test)]
#[cfg(feature = "arbitrary")]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn tuple_data_is_a_pair_or_a_collection_of_any() {
        // sigmastate's data of a tuple is a `Tuple2` for a pair and a `Coll[Any]` for any other
        // count (v6.0.6 `CoreDataSerializer.scala:134-139`, `Evaluation.scala:99-102`)
        let tuple = |xs: &[i32]| -> Literal {
            let items: Vec<Literal> = xs.iter().map(|x| Literal::Int(*x)).collect();
            Literal::Tup(items.try_into().unwrap())
        };
        for xs in [&[][..], &[1], &[1, 2, 3]] {
            let items: Vec<Value> = xs.iter().map(|x| Value::Int(*x)).collect();
            assert_eq!(
                Value::from(tuple(xs)),
                Value::Coll(CollValue::generic(SType::SAny, items)),
                "{xs:?}"
            );
        }
        assert!(matches!(Value::from(tuple(&[1, 2])), Value::Tup(items) if items.len() == 2));
        // a tuple inside a pair converts the same way
        let nested = Literal::Tup([tuple(&[1, 2, 3]), Literal::Int(4)].into());
        let triple = CollValue::generic(
            SType::SAny,
            vec![Value::Int(1), Value::Int(2), Value::Int(3)],
        );
        assert_eq!(
            Value::from(nested),
            Value::Tup([Value::Coll(triple), Value::Int(4)].into())
        );
    }

    #[test]
    fn byte_u8_array_into() {
        let bytes = vec![1u8, 2u8, 3u8];
        let value: Value = bytes.into();
        assert!(matches!(value, Value::Coll(CollValue::Bytes(_))))
    }

    #[test]
    fn byte_i8_array_into() {
        let bytes = vec![1i8, 2i8, 3i8];
        let value: Value = bytes.into();
        assert!(matches!(value, Value::Coll(CollValue::Bytes(_))))
    }

    #[test]
    fn byte_from_array_is_generic() {
        // `fromArray` at an abstract type never specialises, a `Coll[Byte]` included
        let bytes = vec![1i8, 2i8, 3i8];
        let wrapped: Vec<Value> = bytes.into_iter().map(|b| b.into()).collect();
        let coll = CollValue::from_array(SType::SByte, wrapped.clone()).unwrap();
        assert!(matches!(
            coll,
            CollValue::Generic {
                elem_tpe: SType::SByte,
                ..
            }
        ));
        let as_vec = coll.as_vec();
        assert_eq!(as_vec, wrapped);
    }

    #[test]
    fn wrapped_from_array_roundtrip() {
        let longs = vec![1i64, 2i64, 3i64];
        let wrapped: Vec<Value> = longs.into_iter().map(|b| b.into()).collect();
        let coll = CollValue::from_array(SType::SLong, wrapped.clone()).unwrap();
        assert!(matches!(
            coll,
            CollValue::Generic {
                elem_tpe: SType::SLong,
                items: _,
            }
        ));
        let as_vec = coll.as_vec();
        assert_eq!(as_vec, wrapped);
    }

    #[test]
    fn to_static_converts_a_shared_collection_and_box_once() {
        // a collection bound in two distinct captured maps, holding one borrowed box twice:
        // converting each copy of it rebuilt the collection and cloned the box again, so the
        // diagnostics of many lambdas over `OUTPUTS` cloned every output once per lambda
        let ergo_box = sigma_test_util::force_any_val::<ErgoBox>();
        let coll = Value::Coll(CollValue::generic(
            SType::SBox,
            Arc::from(vec![
                Value::CBox(Ref::Borrowed(&ergo_box)),
                Value::CBox(Ref::Borrowed(&ergo_box)),
            ]),
        ));
        fn lambda(captured: HashMap<ValId, Value<'_>>) -> Value<'_> {
            Value::Lambda(Lambda {
                args: vec![],
                body: Arc::new(Expr::Const(true.into())),
                captured: Arc::new(captured),
            })
        }
        let inner_1 = lambda([(ValId::from(1), coll.clone())].into_iter().collect());
        let inner_2 = lambda([(ValId::from(1), coll)].into_iter().collect());
        let outer = lambda(
            [(ValId::from(2), inner_1), (ValId::from(3), inner_2)]
                .into_iter()
                .collect(),
        );
        let converted = outer.to_static();
        let items_of = |id: u32| -> Arc<[Value<'static>]> {
            let Value::Lambda(outer) = &converted else {
                return Arc::from(vec![]);
            };
            match outer.captured.get(&ValId::from(id)) {
                Some(Value::Lambda(inner)) => match inner.captured.get(&ValId::from(1)) {
                    Some(Value::Coll(CollValue::Generic { items, .. })) => items.clone(),
                    _ => Arc::from(vec![]),
                },
                _ => Arc::from(vec![]),
            }
        };
        let (items_2, items_3) = (items_of(2), items_of(3));
        assert_eq!(items_2.len(), 2);
        assert!(Arc::ptr_eq(&items_2, &items_3));
        match (&items_2[0], &items_2[1]) {
            (Value::CBox(Ref::Arc(a)), Value::CBox(Ref::Arc(b))) => assert!(Arc::ptr_eq(a, b)),
            other => assert!(
                matches!(other, (Value::CBox(_), Value::CBox(_))),
                "{other:?}"
            ),
        }
    }
}
