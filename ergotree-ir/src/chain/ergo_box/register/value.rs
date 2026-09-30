use alloc::string::String;
use alloc::string::ToString;
use alloc::vec::Vec;
use derive_more::From;
use thiserror::Error;

use crate::chain::evaluated_value::EvaluatedValue;
use crate::mir::constant::Constant;
use crate::mir::constant::TryExtractFromError;
use crate::serialization::SigmaSerializable;

/// Register value (either a value or bytes if it's unparseable)
#[derive(PartialEq, Eq, Debug, Clone, From)]
pub enum RegisterValue {
    /// Constant value
    Parsed(Constant),
    /// A value that is not a constant: a tuple, a concrete collection or the group generator,
    /// which sigmastate's cast to `EvaluatedValue` accepts too (v6.0.6
    /// `ErgoBoxCandidate.scala:231`)
    ParsedExpr(RegisterExpr),
    /// Unparseable bytes
    Invalid {
        /// Bytes that were not parsed (whole register bytes)
        bytes: Vec<u8>,
        /// Error message on parsing
        error_msg: String,
    },
}

/// A register value that is not a constant, with its data as a constant where it has one (see
/// [`EvaluatedValue::to_constant`])
#[derive(PartialEq, Eq, Debug, Clone)]
pub struct RegisterExpr {
    /// Never an `EvaluatedValue::Constant`: `RegisterValue::from` keeps a constant in `Parsed`
    value: EvaluatedValue,
    constant: Result<Constant, TryExtractFromError>,
}

impl RegisterExpr {
    /// The value
    pub fn value(&self) -> &EvaluatedValue {
        &self.value
    }
}

impl From<EvaluatedValue> for RegisterValue {
    fn from(value: EvaluatedValue) -> Self {
        match value {
            EvaluatedValue::Constant(c) => RegisterValue::Parsed(c),
            value => {
                let constant = value.to_constant();
                RegisterValue::ParsedExpr(RegisterExpr { value, constant })
            }
        }
    }
}

/// Errors on parsing register values
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum RegisterValueError {
    /// Invalid register value
    #[error("Invalid register value: {0}")]
    Invalid(String),
    /// A value with no constant form: an item that is not a value, or a tuple of fewer than two
    /// items
    #[error("Unexpected register value: {0}")]
    UnexpectedRegisterValue(String),
}

impl RegisterValue {
    /// The value as a constant: a constant as is, and a value that is not a constant as its data,
    /// where it has a constant form (see [`EvaluatedValue::to_constant`])
    pub fn as_constant(&self) -> Result<&Constant, RegisterValueError> {
        match self {
            RegisterValue::Parsed(c) => Ok(c),
            RegisterValue::ParsedExpr(e) => e
                .constant
                .as_ref()
                .map_err(|err| RegisterValueError::UnexpectedRegisterValue(err.0.clone())),
            RegisterValue::Invalid {
                bytes: _,
                error_msg,
            } => Err(RegisterValueError::Invalid(error_msg.to_string())),
        }
    }

    /// Return a seraialized bytes of the register value
    #[allow(clippy::unwrap_used)] // it could only fail on OOM, etc.
    pub fn sigma_serialize_bytes(&self) -> Vec<u8> {
        match self {
            RegisterValue::Parsed(c) => c.sigma_serialize_bytes().unwrap(),
            RegisterValue::ParsedExpr(e) => e.value.sigma_serialize_bytes().unwrap(),
            RegisterValue::Invalid {
                bytes,
                error_msg: _,
            } => bytes.clone(),
        }
    }

    /// The register value as ergo's default version context (1, 1) writes it, below ErgoTree
    /// version 3; an unparsed value keeps its bytes
    #[cfg(feature = "json")]
    pub(crate) fn default_context_bytes(
        &self,
    ) -> Result<Vec<u8>, crate::serialization::SigmaSerializationError> {
        use crate::serialization::sigma_byte_writer::default_context_bytes;
        match self {
            RegisterValue::Parsed(c) => default_context_bytes(c),
            RegisterValue::ParsedExpr(e) => default_context_bytes(&e.value),
            RegisterValue::Invalid {
                bytes,
                error_msg: _,
            } => Ok(bytes.clone()),
        }
    }

    /// Parse bytes to RegisterValue: any value sigmastate's cast to `EvaluatedValue` accepts, or
    /// the bytes as `Invalid`
    pub fn sigma_parse_bytes(bytes: &[u8]) -> Self {
        match EvaluatedValue::sigma_parse_bytes(bytes) {
            Ok(value) => value.into(),
            Err(e) => RegisterValue::Invalid {
                bytes: bytes.to_vec(),
                error_msg: format!("failed to parse register value {bytes:?}: {e}"),
            },
        }
    }
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    use crate::mir::constant::Literal;
    use crate::types::stuple::STuple;
    use crate::types::stype::SType;

    use super::*;

    #[test]
    fn test_tuple_expr_i700() {
        let tuple_expr_bytes_str = "860202660263";
        let tuple_expr_bytes = base16::decode(tuple_expr_bytes_str).unwrap();
        assert!(
            Constant::sigma_parse_bytes(&tuple_expr_bytes).is_err(),
            "constant cannot be parsed from tuple expr"
        );
        let reg_value = RegisterValue::sigma_parse_bytes(&tuple_expr_bytes);
        // now let's construct a Constant for (102, 99) byte tuple
        let expected_constant: Constant = Constant {
            tpe: SType::STuple(STuple::pair(SType::SByte, SType::SByte)),
            v: Literal::Tup([Literal::Byte(102), Literal::Byte(99)].into()),
        };
        assert_eq!(
            reg_value.as_constant().unwrap(),
            &expected_constant,
            "should be accessible as Constant"
        );
        assert_eq!(
            reg_value.sigma_serialize_bytes(),
            tuple_expr_bytes,
            "preserve tuple expr on serialization"
        );
    }
}
