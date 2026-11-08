//! Serialization of Ergo types
use crate::chain::ergo_box::RegisterValueError;
use crate::ergo_tree::{ErgoTreeError, ErgoTreeHeaderError, ErgoTreeVersion};
use crate::mir::val_def::ValId;
use crate::mir::{constant::TryExtractFromError, expr::InvalidArgumentError};
use crate::types::type_unify::TypeUnificationError;

use super::{
    constant_store::ConstantStore,
    sigma_byte_reader::{SigmaByteRead, SigmaByteReader, MAX_ARRAY_LENGTH},
    sigma_byte_writer::{SigmaByteWrite, SigmaByteWriter},
};
use crate::types::smethod::MethodId;
use alloc::boxed::Box;

use alloc::string::{String, ToString};
use alloc::vec::Vec;
use bounded_vec::BoundedVec;
use bounded_vec::BoundedVecOutOfBounds;
use core::convert::TryInto;
use core3::io;
use io::Cursor;
use sigma_ser::{vlq_encode, ScorexParsingError, ScorexSerializationError};
use thiserror::Error;

/// Ways serialization might fail
#[derive(Error, Eq, PartialEq, Debug, Clone)]
pub enum SigmaSerializationError {
    /// IO fail (EOF, etc.)
    #[error("IO error: {0}")]
    Io(String),
    /// Serialization not yet implemented
    #[error("serialization not yet implemented: {0}")]
    NotImplementedYet(&'static str),
    /// Unexpected value type
    #[error("Unexpected value: {0:?}")]
    UnexpectedValue(#[from] TryExtractFromError),
    /// Serialization not supported
    #[error("serialization not supported: {0}")]
    NotSupported(String),
    /// Scorex serialization error
    #[error("Scorex serialization error: {0}")]
    ScorexSerializationError(#[from] ScorexSerializationError),
}

impl From<io::Error> for SigmaSerializationError {
    fn from(error: io::Error) -> Self {
        SigmaSerializationError::Io(error.to_string())
    }
}

/// Ways parsing might fail
#[derive(Error, Eq, PartialEq, Debug, Clone)]
pub enum SigmaParsingError {
    /// Invalid op code
    #[error("invalid op code: {0}")]
    InvalidOpCode(u8),
    /// Lacking support for the op
    #[error("not implemented op error: {0}")]
    NotImplementedOpCode(String),
    /// Failed to parse type
    #[error("type parsing error, invalid type code: {0}({0:#04X})")]
    InvalidTypeCode(u8),
    /// V6 type error
    #[error("Can't use v6 types (UnsignedBigInt, Header, Option) in ContextExtension/Registers ")]
    V6TypeError,
    /// Failed to decode VLQ
    #[error("vlq encode error: {0}")]
    VlqEncode(#[from] vlq_encode::VlqEncodingError),
    /// IO fail (EOF, etc.)
    #[error("IO error: {0}")]
    Io(String),
    /// Misc fail
    #[error("misc error: {0}")]
    Misc(String),
    /// Feature not yet implemented
    #[error("parsing not yet implemented: {0}")]
    NotImplementedYet(String),
    /// Constant with given index not found in constant store
    #[error("Constant with index {0} not found in constant store")]
    ConstantForPlaceholderNotFound(u32),
    /// Value out of bounds
    #[error("Value out of bounds: {0}")]
    ValueOutOfBounds(String),
    /// Tuple items out of bounds
    #[error("Tuple items out of bounds: {0}")]
    TupleItemsOutOfBounds(usize),
    /// Nested value deserialization went deeper than
    /// [`MAX_TREE_DEPTH`](crate::serialization::sigma_byte_reader::MAX_TREE_DEPTH)
    /// (sigmastate `DeserializeCallDepthExceeded`)
    #[error(
        "nested value deserialization call depth({0}) exceeds allowed maximum {max}",
        max = crate::serialization::sigma_byte_reader::MAX_TREE_DEPTH
    )]
    DeserializeCallDepthExceeded(usize),
    /// A type nested deeper than
    /// `MAX_TYPE_DEPTH`: sigmastate's
    /// `TypeSerializer` throws `DeserializeCallDepthExceeded` (v6.0.7
    /// `TypeSerializer.scala:133-136`), a `SerializerException`
    #[error(
        "type nesting depth({0}) exceeds allowed maximum {max}",
        max = crate::serialization::types::MAX_TYPE_DEPTH
    )]
    TypeDepthExceeded(usize),
    /// A declared item count above [`MAX_ARRAY_LENGTH`]: sigmastate's `safeNewArray` refuses it
    /// before any item is read
    #[error(
        "cannot allocate an array of {0} items: the limit is {max}",
        max = crate::serialization::sigma_byte_reader::MAX_ARRAY_LENGTH
    )]
    ArrayLengthExceeded(usize),
    /// A CTHRESHOLD whose `k` and number of children break `0 <= k <= n <= 255`:
    /// sigmastate's `require` throws an `IllegalArgumentException`
    #[error("CTHRESHOLD needs 0 <= k <= n <= 255, got k = {0}, n = {1}")]
    CthresholdOutOfBounds(u16, usize),
    /// A tree whose version is above the activated script version, from activated 2:
    /// sigmastate's `VersionContext` refuses the pair (`VersionContext.scala:17-21`), and
    /// `deserializeErgoTree` rethrows it as a `SerializerException`
    /// (`ErgoTreeSerializer.scala:191-193`)
    #[error("Tree version ({0}) is above activated script version ({1})")]
    TreeVersionAboveActivated(u8, u8),
    /// A bitwise operation with a non-numeric operand: sigmastate's `BitOp` requires
    /// numeric operands and throws an `IllegalArgumentException`
    #[error("bitwise operation on non-numeric operands: {0}")]
    BitOpOperandsNotNumeric(String),
    /// A tree without the size flag failed with an error that would only degrade a
    /// size-flagged tree. sigmastate cannot keep it as `UnparsedErgoTree` and throws a
    /// `SerializerException` ("ErgoTree serialized without size bit") instead, which rejects a
    /// size-flagged tree around it as well.
    #[error("ErgoTree serialized without size bit: {0}")]
    UnsizedTreeValidationError(Box<ErgoTreeError>),
    /// Type code 0, which sigmastate's `TypeSerializer` refuses with `InvalidTypePrefix`
    #[error("type code 0 is not a type")]
    InvalidTypePrefix,
    /// A type parameter sigmastate cannot build: a negative `FunDef` count, or an `SFunc` type
    /// parameter that is not a type variable (a `NegativeArraySizeException`, a failed
    /// `require`)
    #[error("invalid type parameter: {0}")]
    InvalidTypeParameter(String),
    /// A `FunDef` type parameter that is not a type variable: `ValDefSerializer` casts it
    /// with `.asInstanceOf[STypeVar]`, a `ClassCastException`
    #[error("a FunDef type parameter that is not a type variable: {0}")]
    TypeParameterNotTypeVar(String),
    /// A SigmaBoolean opcode sigmastate's parser does not match (a `MatchError`)
    #[error("unexpected SigmaBoolean opcode {0:#04x}")]
    InvalidSigmaBooleanOpCode(u8),
    /// A BigInt or UnsignedBigInt data size above 32 bytes, which sigmastate refuses with a
    /// `SerializerException`, or a BigInt of 0 bytes, which `new BigInteger` refuses with a
    /// `NumberFormatException`
    #[error("invalid BigInt size {0}")]
    InvalidBigIntSize(usize),
    /// A value sigmastate reads with `getUIntExact` that does not fit an `Int` (an
    /// `ArithmeticException`)
    #[error("{0} {1} exceeds Int.MaxValue (getUIntExact)")]
    ExceedsIntMax(&'static str, u32),
    /// A register or context-extension value that is not an `EvaluatedValue`: sigmastate's
    /// cast after `getValue` throws a `ClassCastException` (v6.0.6
    /// `ErgoBoxCandidate.scala:231`, `ContextExtension.scala:61`)
    #[error("not a value: {0}")]
    UnevaluatedValue(String),
    /// A tuple whose size, a signed byte, is negative: sigmastate's `safeNewArray` throws a
    /// `NegativeArraySizeException` (v6.0.6 `TupleSerializer.scala:28-31`)
    #[error("negative tuple size {0}")]
    NegativeTupleSize(i8),
    /// A Box constant with more than the six non-mandatory registers: sigmastate's
    /// register-id lookup throws an `ArrayIndexOutOfBoundsException` at the seventh
    #[error("{0} registers, where only R4 to R9 exist")]
    TooManyRegisters(u8),
    /// An `Append` or `Slice` whose input is not a collection: sigmastate types the node
    /// eagerly as its input's type, cast to a collection (a `ClassCastException`)
    #[error("a collection input expected: {0}")]
    CollectionInputExpected(String),
    /// A concrete collection's item whose type is not the declared element type: sigmastate
    /// asserts each item's type, an `AssertionError`
    #[error("collection item type mismatch: {0}")]
    CollectionItemTypeMismatch(String),
    /// A numeric cast to a type that is not numeric: sigmastate's `NumericCastSerializer` casts
    /// the target with `asNumType`, a `ClassCastException` (v6.0.7
    /// `NumericCastSerializer.scala:20-24`)
    #[error("a numeric cast to a non-numeric type: {0}")]
    NumericCastTargetNotNumeric(String),
    /// A node whose constructor requires a numeric operand got another type: the `require` of
    /// `Upcast` or `Downcast` (v6.0.7 `trees.scala:398`, `:431`), or of `Negation` or
    /// `BitInversion`, which admit NoType too (`:882`, `:900`), an `IllegalArgumentException`
    #[error("a numeric operand expected: {0}")]
    NumericOperandExpected(String),
    /// A relation whose operands fail the deserializing builder's `check2`: a comparison of a
    /// non-numeric operand, or two operands of different types after the builder's upcast
    /// (v6.0.7 `SigmaBuilder.scala:286-313`, `:686-704`), a `ConstraintFailed`
    #[error("constraint failed: {0}")]
    ConstraintFailed(String),
    /// Below tree version 3, a `ByIndex` index `upcastTo(SInt)` refuses: a non-numeric index,
    /// or one wider than `Int` (v6.0.7 `ByIndexSerializer.scala:27-36`,
    /// `syntax.scala:168-177`), an `AssertionError`
    #[error("ByIndex index cannot be upcast to Int: {0}")]
    IndexUpcastFailed(String),
    /// A block item that is not a `ValDef`: sigmastate casts each item to `BlockItem` as it
    /// reads it (v6.0.7 `BlockValueSerializer.scala:39`), a `ClassCastException`
    #[error("a block item that is not a ValDef: {0}")]
    BlockItemNotValDef(String),
    /// A register id outside R0 to R9 in `ExtractRegisterAs` or `DeserializeRegister`:
    /// sigmastate's `findRegisterByIndex(id).get` right after the id byte (v6.0.7
    /// `ExtractRegisterAsSerializer.scala:28`, `DeserializeRegisterSerializer.scala:28`), a
    /// `NoSuchElementException`
    #[error("no register with id {0}")]
    RegisterIdOutOfRange(i8),
    /// From tree version 3, a `MethodCall` with no arguments: sigmastate asserts there are
    /// some after reading them (v6.0.7 `MethodCallSerializer.scala:52-55`), an `AssertionError`
    #[error("a MethodCall with no arguments in a tree of version 3 or later")]
    MethodCallWithoutArguments,
    /// A node whose constructor casts its input's type to `SOption` got another type:
    /// `OptionGet` and `OptionGetOrElse` (v6.0.7 `transformers.scala:600-601`, `:625-626`), a
    /// `ClassCastException`
    #[error("an Option input expected: {0}")]
    OptionInputExpected(String),
    /// A `MapCollection` whose mapper is not a function: its constructor casts the mapper's
    /// type to `SFunc` (v6.0.7 `transformers.scala:38`), a `ClassCastException`
    #[error("a function expected: {0}")]
    FunctionExpected(String),
    /// A node's type read that sigmastate's throws on: `Filter`'s `def tpe = input.tpe` casts
    /// the input's type to `SCollection` on every read (v6.0.7 `transformers.scala:121`), a
    /// `ClassCastException`
    #[error("a type read that throws: {0}")]
    TypeReadFailed(String),
    /// A `SelectField` whose input is not a tuple: its `val tpe` casts the input's type to
    /// `STuple` (v6.0.7 `transformers.scala:294`), a `ClassCastException`
    #[error("a tuple input expected: {0}")]
    TupleInputExpected(String),
    /// A `SelectField` whose field index, a signed byte, is not one of the tuple's: its
    /// `val tpe` takes `items(fieldIndex - 1)` (`transformers.scala:294`), an
    /// `IndexOutOfBoundsException`
    #[error("a tuple field index out of bounds: {0}")]
    TupleFieldOutOfBounds(String),
    /// Collection data whose element type is zero-width: `SUnit`, or a collection or tuple
    /// built only from it. Its elements take no bytes, so its declared length escapes the
    /// position limit. Rule 1020 `CheckZeroWidthCollection` (sigmastate 6.0.7
    /// `CoreDataSerializer.scala:159-169`, `ValidationRules.scala:199-210`), a
    /// `ValidationException`: a size-flagged tree degrades on it, and a box register or a
    /// context-extension value, which no tree encloses, rejects
    #[error("collection with zero-width element type {0} cannot be deserialized (rule 1020)")]
    ZeroWidthCollection(String),
    /// ValDef type for a given index not found in ValDefTypeStore store
    #[error("ValDef type for an index {0:?} not found in ValDefTypeStore store")]
    ValDefIdNotFound(ValId),
    /// Invalid argument on node creation
    #[error("Invalid argument: {0:?}")]
    InvalidArgument(#[from] InvalidArgumentError),
    /// Unknown method ID for given type code
    #[error("No method id {0:?} found in type companion with type id {1:?} ")]
    UnknownMethodId(MethodId, u8),
    /// Feature not supported
    #[error("parsing not supported: {0}")]
    NotSupported(&'static str),
    /// Serialization error
    #[error("serialization error: {0}")]
    SerializationError(#[from] SigmaSerializationError),
    /// Invalid item quantity for BoundedVec
    #[error("Invalid item quantity for BoundedVec: {0}")]
    BoundedVecOutOfBounds(#[from] BoundedVecOutOfBounds),
    /// Scorex parsing error
    #[error("Scorex parsing error: {0}")]
    ScorexParsingError(#[from] ScorexParsingError),
    /// ErgoTreeHeaderError
    #[error("ErgoTreeHeaderError: {0}")]
    ErgoTreeHeaderError(#[from] ErgoTreeHeaderError),
    /// Invalid register value
    #[error("Invalid register value: {0}")]
    InvalidRegisterValue(#[from] RegisterValueError),
    /// Data value of a type whose type code has no `DataSerializer` and is NOT
    /// soft-forkable per sigma-state rule 1009 (`CheckSerializableTypeCode`): the
    /// code is neither `OptionTypeCode` (36) nor `> LastDataType` (111). Mirrors the
    /// JVM's hard `SerializerException` ("Not defined DataSerializer for type ..."),
    /// which escapes `ErgoTreeSerializer.deserializeErgoTree`'s `UnparsedErgoTree`
    /// soft-fork fallback — so a size-flagged tree carrying such a constant is
    /// rejected, not degraded to `Unparsed`.
    #[error("data value of type code {0} cannot be deserialized (rule 1009: not soft-forkable)")]
    NonSerializableTypeCode(u8),
    /// A read started past the position limit (the reference impl's rule 1014
    /// `CheckPositionLimit`, set as the `MaxBoxSize` window during box parse).
    /// SOFT-FORKABLE: a size-flagged tree whose body overruns its position
    /// window degrades to `Unparsed` (matching the JVM's `ValidationException`),
    /// unlike a hard EOF/structural failure which rejects.
    #[error("read position exceeds the position limit")]
    PositionLimitExceeded,
}

impl From<io::Error> for SigmaParsingError {
    fn from(error: io::Error) -> Self {
        // `InvalidData` is the position-limit signal (rule 1014); see
        // `VlqEncodingError`'s `From<io::Error>`. Keep it apart from `Io` (EOF).
        if error.kind() == io::ErrorKind::InvalidData {
            SigmaParsingError::PositionLimitExceeded
        } else {
            SigmaParsingError::Io(error.to_string())
        }
    }
}

impl From<&io::Error> for SigmaParsingError {
    fn from(error: &io::Error) -> Self {
        if error.kind() == io::ErrorKind::InvalidData {
            SigmaParsingError::PositionLimitExceeded
        } else {
            SigmaParsingError::Io(error.to_string())
        }
    }
}

impl SigmaParsingError {
    /// True if this is the soft-forkable position-limit error (rule 1014),
    /// whether at the top level or nested in `VlqEncode` / `ScorexParsingError`
    /// (a windowed read can trip the limit through either channel).
    pub fn is_position_limit_exceeded(&self) -> bool {
        match self {
            SigmaParsingError::PositionLimitExceeded => true,
            SigmaParsingError::VlqEncode(vlq_encode::VlqEncodingError::PositionLimitExceeded) => {
                true
            }
            SigmaParsingError::ScorexParsingError(e) => e.is_position_limit_exceeded(),
            _ => false,
        }
    }

    /// True if a size-flagged `ErgoTree` whose body fails with this error must reject (escape
    /// the soft-fork degrade) instead of degrading to `Unparsed`. sigmastate's
    /// `deserializeErgoTree` keeps a size-flagged tree as `UnparsedErgoTree` on one exception
    /// class, the `ValidationException` a failed validation rule throws, and turns it into a
    /// `SerializerException` for an unsized tree; every other exception propagates and rejects
    /// the tree (v6.0.7 `ErgoTreeSerializer.scala:196-209`). So this error escapes unless
    /// sigmastate fails at its site through a validation rule; `jvm_failure()` holds that class
    /// for every variant. Whether the rule is soft-forkable does not enter into it.
    pub fn escapes_sized_tree_degrade(&self) -> bool {
        self.jvm_failure() != JvmFailure::Validation
    }
}

impl From<TypeUnificationError> for SigmaParsingError {
    fn from(e: TypeUnificationError) -> Self {
        SigmaParsingError::Misc(format!("{:?}", e))
    }
}

/// Result type for [`SigmaSerializable::sigma_serialize`]
pub type SigmaSerializeResult = Result<(), SigmaSerializationError>;

/// Consensus-critical serialization for Ergo
pub trait SigmaSerializable: Sized {
    /// Write `self` to the given `writer`.
    /// This function has a `sigma_` prefix to alert the reader that the
    /// serialization in use is consensus-critical serialization
    // fn sigma_serialize<W: SigmaByteWrite>(&self, w: &mut W) -> SigmaSerializeResult;
    fn sigma_serialize<W: SigmaByteWrite>(&self, w: &mut W) -> SigmaSerializeResult;

    /// Try to read `self` from the given `reader`.
    /// `sigma-` prefix to alert the reader that the serialization in use
    /// is consensus-critical
    fn sigma_parse<R: SigmaByteRead>(r: &mut R) -> Result<Self, SigmaParsingError>;

    /// Serialize any SigmaSerializable value into bytes
    fn sigma_serialize_bytes(&self) -> Result<Vec<u8>, SigmaSerializationError> {
        let mut data = Vec::new();
        let mut w = SigmaByteWriter::new(&mut data, None);
        w.with_tree_version(ErgoTreeVersion::MAX_SCRIPT_VERSION, |w| {
            self.sigma_serialize(w)
        })?;
        Ok(data)
    }

    /// Parse `self` from the bytes
    fn sigma_parse_bytes(bytes: &[u8]) -> Result<Self, SigmaParsingError> {
        let cursor = Cursor::new(bytes);
        let mut sr = SigmaByteReader::new(cursor, ConstantStore::empty());
        // Set version to max for convenience when parsing new types like UnsignedBigInt from bytes/base16
        sr.with_tree_version(ErgoTreeVersion::MAX_SCRIPT_VERSION, |sr| {
            Self::sigma_parse(sr)
        })
    }
}

impl<T: SigmaSerializable> SigmaSerializable for Vec<T> {
    fn sigma_serialize<W: SigmaByteWrite>(&self, w: &mut W) -> SigmaSerializeResult {
        w.put_u32(self.len() as u32)?;
        self.iter().try_for_each(|i| i.sigma_serialize(w))
    }

    /// Every counted list sigmastate reads this way (the values of `getValues`, SigmaAnd's
    /// and SigmaOr's items, a block's items, a function's arguments) reads the count with
    /// `getUIntExact` and allocates with `safeNewArray`, which refuses more than
    /// `MaxArrayLength` before reading an item (v6.0.6 `SigmaByteReader.scala:53-59`,
    /// `SigmaTransformerSerializer.scala:21-25`, `BlockValueSerializer.scala:28-37`,
    /// `FuncValueSerializer.scala:30-34`)
    fn sigma_parse<R: SigmaByteRead>(r: &mut R) -> Result<Self, SigmaParsingError> {
        let items_count = r.get_u32()?;
        if items_count as usize > MAX_ARRAY_LENGTH {
            return Err(SigmaParsingError::ArrayLengthExceeded(items_count as usize));
        }
        let mut items = Vec::new();
        for _ in 0..items_count {
            items.push(T::sigma_parse(r)?);
        }
        Ok(items)
    }
}

impl<T: SigmaSerializable, const L: usize, const U: usize> SigmaSerializable
    for BoundedVec<T, L, U>
{
    fn sigma_serialize<W: SigmaByteWrite>(&self, w: &mut W) -> SigmaSerializeResult {
        self.as_vec().sigma_serialize(w)
    }

    fn sigma_parse<R: SigmaByteRead>(r: &mut R) -> Result<Self, SigmaParsingError> {
        Ok(Vec::<T>::sigma_parse(r)?.try_into()?)
    }
}

/// Corresponds to `VLQ(UInt)` format from `ErgoTree` spec.
impl SigmaSerializable for u32 {
    fn sigma_serialize<W: SigmaByteWrite>(&self, w: &mut W) -> SigmaSerializeResult {
        w.put_u32(*self)?;
        Ok(())
    }
    fn sigma_parse<R: SigmaByteRead>(r: &mut R) -> Result<Self, SigmaParsingError> {
        let v = r.get_u32()?;
        Ok(v)
    }
}

impl<T: SigmaSerializable> SigmaSerializable for Option<Box<T>> {
    fn sigma_serialize<W: SigmaByteWrite>(&self, w: &mut W) -> SigmaSerializeResult {
        match self {
            Some(v) => {
                w.put_u8(1)?;
                v.sigma_serialize(w)
            }
            None => Ok(w.put_u8(0)?),
        }
    }

    fn sigma_parse<R: SigmaByteRead>(r: &mut R) -> Result<Self, SigmaParsingError> {
        let tag = r.get_u8()?;
        Ok(if tag != 0 {
            Some(T::sigma_parse(r)?.into())
        } else {
            None
        })
    }
}

/// serialization roundtrip
#[allow(clippy::expect_used)]
pub fn sigma_serialize_roundtrip<T: SigmaSerializable>(v: &T) -> T {
    let mut data = Vec::new();
    let mut w = SigmaByteWriter::new(&mut data, None);
    v.sigma_serialize(&mut w).expect("serialization failed");
    let cursor = Cursor::new(&mut data[..]);
    let mut sr = SigmaByteReader::new(cursor, ConstantStore::empty());
    T::sigma_parse(&mut sr).expect("parse failed")
}

/// Perform versioned serialization
pub fn sigma_serialize_roundtrip_versioned<T: SigmaSerializable>(
    v: &T,
    tree_version: ErgoTreeVersion,
) -> Result<T, Box<dyn core::error::Error>> {
    let mut data = Vec::new();
    let mut w = SigmaByteWriter::new(&mut data, None);
    w.with_tree_version(tree_version, |w| v.sigma_serialize(w))?;
    let cursor = Cursor::new(&mut data[..]);
    let mut sr = SigmaByteReader::new(cursor, ConstantStore::empty());
    sr.with_tree_version(tree_version, T::sigma_parse)
        .map_err(From::from)
}

/// Perform serialization roundtrip for a feature that's only supported after `since`
#[allow(clippy::expect_used)]
pub fn roundtrip_new_feature<T: SigmaSerializable + core::fmt::Debug + PartialEq>(
    v: &T,
    since: ErgoTreeVersion,
) {
    for version in 0u8..since.into() {
        assert!(sigma_serialize_roundtrip_versioned(v, version.into()).is_err());
    }
    for version in u8::from(since)..=ErgoTreeVersion::MAX_SCRIPT_VERSION.into() {
        assert_eq!(
            *v,
            sigma_serialize_roundtrip_versioned(v, version.into()).expect("roundtrip failed")
        );
    }
}

/// The class of exception sigmastate 6.0.7 throws where sigma-rust raises a parse error. Two
/// JVM handlers key on it: `deserializeErgoTree` degrades a size-flagged tree on a
/// `ValidationException` only (`ErgoTreeSerializer.scala:141-205`), and the substitution at
/// spend leaves a node where it is on a `ClassCastException` only, raised inside the rule
/// (Kiama's `strategy`, `Rewriter.scala:180-191`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum JvmFailure {
    /// A validation rule's `ValidationException`
    Validation,
    /// A `ClassCastException`
    ClassCast,
    /// Any other exception, or a refusal sigma-rust makes where sigmastate builds the node
    Other,
}

impl SigmaParsingError {
    /// What sigmastate throws where this error is raised (v6.0.7). Every variant has its row,
    /// so a new variant cannot be added without deciding it.
    pub(crate) fn jvm_failure(&self) -> JvmFailure {
        use JvmFailure::{ClassCast, Other, Validation};
        use SigmaParsingError::*;
        // rule 1014 `CheckPositionLimit`, through whichever reader channel tripped it
        if self.is_position_limit_exceeded() {
            return Validation;
        }
        match self {
            // rule 1002 `CheckValidOpCode` (`ValueSerializer.scala:160-170`)
            InvalidOpCode(_) | NotImplementedOpCode(_) => Validation,
            // rules 1008 / 1018 `CheckPrimitiveTypeCode` / `CheckTypeCodeV6`
            // (`TypeSerializer.scala:226-233`)
            InvalidTypeCode(_) => Validation,
            // rule 1019, a v6 type in a register or context extension before v6
            // (`ErgoBoxCandidate.scala:234`, `ContextExtension.scala:62`)
            V6TypeError => Validation,
            // rule 1010 `CheckTypeWithMethods`
            NotImplementedYet(_) => Validation,
            // rules 1011 / 1016 `CheckAndGetMethod` (`methods.scala`)
            UnknownMethodId(_, _) => Validation,
            // rule 1009 `CheckSerializableTypeCode` (`CoreDataSerializer.scala:149-170`). Two
            // sites are a `SerializerException` on the JVM, and a tree reaches neither:
            // UnsignedBigInt data below v3, whose type code the type parse refuses first
            // (`types.rs`), and `NoType` data, which has no type code (`data.rs`)
            NotSupported(_) => Validation,
            // rule 1020 `CheckZeroWidthCollection` (`CoreDataSerializer.scala:168`)
            ZeroWidthCollection(_) => Validation,
            // rule 1012 `CheckHeaderSizeBit` for a missing size bit. `IoError` is the end of
            // input at the header byte, a `BufferUnderflowException`; it is never the position
            // limit, as a tree's window starts at its header byte
            // (`ErgoTreeSerializer.scala:144-145`). The header parse does not raise
            // `VersionError`.
            ErgoTreeHeaderError(e) => match e {
                crate::ergo_tree::ErgoTreeHeaderError::InvalidSizeBit(_) => Validation,
                crate::ergo_tree::ErgoTreeHeaderError::VersionError(_)
                | crate::ergo_tree::ErgoTreeHeaderError::IoError(_) => Other,
            },
            // casts to a concrete class: `asInstanceOf[EvaluatedValue[_]]`
            // (`ErgoBoxCandidate.scala:234`, `ContextExtension.scala:61`), the input's type to
            // `SCollection` (`transformers.scala:62`, `:89`, `:254`) or `SOption` (`:600-601`,
            // `:625-626`), a mapper's to `SFunc` (`:38`), a block item to `BlockItem`
            // (`BlockValueSerializer.scala:39`), a cast target to `SNumericType`
            // (`NumericCastSerializer.scala:20-24`), a `FunDef` type parameter to `STypeVar`
            // (`ValDefSerializer.scala:41`), a `SelectField`'s input type to `STuple`
            // (`transformers.scala:294`), and `Filter`'s type read, its input's cast to
            // `SCollection` (`:121`)
            UnevaluatedValue(_)
            | CollectionInputExpected(_)
            | OptionInputExpected(_)
            | FunctionExpected(_)
            | BlockItemNotValDef(_)
            | NumericCastTargetNotNumeric(_)
            | TypeParameterNotTypeVar(_)
            | TupleInputExpected(_)
            | TypeReadFailed(_) => ClassCast,
            // the reader's other failures: an `IllegalArgumentException` for a malformed VLQ,
            // an `IOException` at the end of the input
            VlqEncode(_) | ScorexParsingError(_) | Io(_) | PositionLimitExceeded => Other,
            // `SerializerException`s: the depth guards (`DeserializeCallDepthExceeded`), an
            // unsized tree re-throwing a `ValidationException`
            // (`ErgoTreeSerializer.scala:197-207`), type code 0 (`TypeSerializer.scala:138`),
            // a data type with no serializer (`CoreDataSerializer.scala:149-151`), a version
            // above the activated one (`VersionContext.scala:17-21`)
            DeserializeCallDepthExceeded(_)
            | TypeDepthExceeded(_)
            | UnsizedTreeValidationError(_)
            | InvalidTypePrefix
            | NonSerializableTypeCode(_)
            | TreeVersionAboveActivated(_, _) => Other,
            // `IllegalArgumentException`s from `require`: `Cthreshold`
            // (`SigmaBoolean.scala:223`), `BitOp` (`trees.scala:913`), the numeric nodes
            // (`:398`, `:431`, `:882`, `:900`), an `SFunc` type parameter; and a negative
            // `FunDef` count's `NegativeArraySizeException`
            CthresholdOutOfBounds(_, _)
            | BitOpOperandsNotNumeric(_)
            | NumericOperandExpected(_)
            | InvalidTypeParameter(_) => Other,
            // the builder's `check2` throws `ConstraintFailed`, a `BuilderException`, not a
            // `ValidationException` (`SigmaBuilder.scala:293-295`)
            ConstraintFailed(_) => Other,
            // `AssertionError`s: a collection item's type (`ConcreteCollectionSerializer.scala:37`),
            // a `ByIndex` index's upcast (`syntax.scala:168-177`), a v3 call without arguments
            // (`MethodCallSerializer.scala:58`)
            CollectionItemTypeMismatch(_) | IndexUpcastFailed(_) | MethodCallWithoutArguments => {
                Other
            }
            // `NoSuchElementException`, `IndexOutOfBoundsException`, `MatchError`,
            // `ArithmeticException`, `NegativeArraySizeException`, `RuntimeException`; the
            // placeholder and `ValUse` lookups are `ConstantPlaceholderSerializer.scala:19` and
            // `ValDefTypeStore.scala:11`
            RegisterIdOutOfRange(_)
            | ValDefIdNotFound(_)
            | ConstantForPlaceholderNotFound(_)
            | TooManyRegisters(_)
            | InvalidSigmaBooleanOpCode(_)
            | ExceedsIntMax(_, _)
            | NegativeTupleSize(_)
            | ArrayLengthExceeded(_)
            | InvalidBigIntSize(_)
            | TupleFieldOutOfBounds(_) => Other,
            // no validation rule, and none is raised on a path from a tree's body: an input's
            // context extension (`SigmaSerializer.error`, a `SerializerException`,
            // `ContextExtension.scala:55`, `:60`), a transaction output's token digest index
            // (`sys.error`, `ErgoBoxCandidate.scala:210-211`), a P2SH address's tree, and
            // conversions that cannot fail
            Misc(_) | ValueOutOfBounds(_) | InvalidArgument(_) | BoundedVecOutOfBounds(_) => Other,
            // not raised in this crate
            TupleItemsOutOfBounds(_) | SerializationError(_) | InvalidRegisterValue(_) => Other,
        }
    }
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
#[allow(clippy::panic)]
mod tests {
    use super::*;
    use sigma_ser::vlq_encode::WriteSigmaVlqExt;

    /// A huge declared count followed by no elements must return Err without
    /// allocating a multi-gigabyte Vec.
    #[test]
    fn vec_sigma_parse_huge_count_no_data_returns_err() {
        // VLQ-encode u32::MAX as the count, provide zero element bytes
        let mut data = Vec::new();
        let mut w = SigmaByteWriter::new(&mut data, None);
        w.put_u32(u32::MAX).unwrap();
        let cursor = Cursor::new(&data[..]);
        let mut r = SigmaByteReader::new(cursor, ConstantStore::empty());
        let result = Vec::<u32>::sigma_parse(&mut r);
        assert!(result.is_err());
    }

    #[test]
    fn each_parse_error_maps_to_the_exception_sigmastate_throws() {
        let s = String::new;
        for e in [
            SigmaParsingError::UnevaluatedValue(s()),
            SigmaParsingError::CollectionInputExpected(s()),
            SigmaParsingError::OptionInputExpected(s()),
            SigmaParsingError::FunctionExpected(s()),
            SigmaParsingError::BlockItemNotValDef(s()),
            SigmaParsingError::NumericCastTargetNotNumeric(s()),
            SigmaParsingError::TypeParameterNotTypeVar(s()),
            SigmaParsingError::TupleInputExpected(s()),
            SigmaParsingError::TypeReadFailed(s()),
        ] {
            assert_eq!(e.jvm_failure(), JvmFailure::ClassCast, "{e:?}");
        }
        for e in [
            SigmaParsingError::ConstraintFailed(s()),
            SigmaParsingError::NumericOperandExpected(s()),
            SigmaParsingError::InvalidTypeParameter(s()),
            SigmaParsingError::TupleFieldOutOfBounds(s()),
            SigmaParsingError::ErgoTreeHeaderError(ErgoTreeHeaderError::IoError(s())),
        ] {
            assert_eq!(e.jvm_failure(), JvmFailure::Other, "{e:?}");
        }
        for e in [
            SigmaParsingError::InvalidOpCode(0),
            SigmaParsingError::ErgoTreeHeaderError(ErgoTreeHeaderError::InvalidSizeBit(0x0b)),
            SigmaParsingError::PositionLimitExceeded,
            SigmaParsingError::VlqEncode(vlq_encode::VlqEncodingError::PositionLimitExceeded),
        ] {
            assert_eq!(e.jvm_failure(), JvmFailure::Validation, "{e:?}");
        }
    }
}
