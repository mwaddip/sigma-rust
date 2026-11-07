//! Traits for IR nodes with one input value(expr)

use crate::serialization::sigma_byte_reader::SigmaByteRead;
use crate::serialization::sigma_byte_writer::SigmaByteWrite;
use crate::serialization::SigmaParsingError;
use crate::serialization::SigmaSerializable;
use crate::serialization::SigmaSerializeResult;

use super::bit_inversion::BitInversion;
use super::bool_to_sigma::BoolToSigmaProp;
use super::byte_array_to_bigint::ByteArrayToBigInt;
use super::byte_array_to_long::ByteArrayToLong;
use super::calc_blake2b256::CalcBlake2b256;
use super::calc_sha256::CalcSha256;
use super::coll_size::SizeOf;
use super::create_provedlog::CreateProveDlog;
use super::decode_point::DecodePoint;
use super::expr::Expr;
use super::expr::InvalidArgumentError;
use super::extract_amount::ExtractAmount;
use super::extract_bytes::ExtractBytes;
use super::extract_bytes_with_no_ref::ExtractBytesWithNoRef;
use super::extract_creation_info::ExtractCreationInfo;
use super::extract_id::ExtractId;
use super::extract_script_bytes::ExtractScriptBytes;
use super::logical_not::LogicalNot;
use super::long_to_byte_array::LongToByteArray;
use super::negation::Negation;
use super::option_get::OptionGet;
use super::option_is_defined::OptionIsDefined;
use super::sigma_prop_bytes::SigmaPropBytes;
use alloc::boxed::Box;

/// IR nodes with one input value(expr)
pub trait OneArgOp {
    /// Input value(expr) of the IR node
    fn input(&self) -> &Expr;
    /// Mutable reference to input value(expr) of the IR node
    fn input_mut(&mut self) -> &mut Expr;
}

/// Constructor for unary IR nodes that check the validity of the argument
pub trait OneArgOpTryBuild: Sized {
    /// Create new IR node, returns an error if any of the requirements failed
    fn try_build(input: Expr) -> Result<Self, InvalidArgumentError>;
}

/// The parser's constructor for unary IR nodes: the node over its parsed input, with no check.
/// sigmastate builds most of these nodes without reading their input's type, and the checks it
/// does make (`Negation`, `BitInversion`, `OptionGet`) run in the parser's `check_build`.
pub trait OneArgOpParseBuild: Sized {
    /// The node over `input`, unchecked
    fn build_parsed(input: Expr) -> Self;
}

impl<T: OneArgOp + OneArgOpTryBuild + OneArgOpParseBuild> SigmaSerializable for T {
    fn sigma_serialize<W: SigmaByteWrite>(&self, w: &mut W) -> SigmaSerializeResult {
        self.input().sigma_serialize(w)
    }

    fn sigma_parse<R: SigmaByteRead>(r: &mut R) -> Result<Self, SigmaParsingError> {
        let input = Expr::sigma_parse(r)?;
        Ok(T::build_parsed(input))
    }
}

macro_rules! impl_one_arg_op_parse_build {
    ($($node:ident),* $(,)?) => {
        $(impl OneArgOpParseBuild for $node {
            fn build_parsed(input: Expr) -> Self {
                Self {
                    input: Box::new(input),
                }
            }
        })*
    };
}

impl_one_arg_op_parse_build!(
    BitInversion,
    BoolToSigmaProp,
    ByteArrayToBigInt,
    ByteArrayToLong,
    CalcBlake2b256,
    CalcSha256,
    SizeOf,
    CreateProveDlog,
    DecodePoint,
    ExtractAmount,
    ExtractBytes,
    ExtractBytesWithNoRef,
    ExtractCreationInfo,
    ExtractId,
    ExtractScriptBytes,
    LogicalNot,
    LongToByteArray,
    Negation,
    OptionGet,
    OptionIsDefined,
    SigmaPropBytes,
);
