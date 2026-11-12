use alloc::boxed::Box;

use alloc::sync::Arc;

use crate::chain::ergo_box::RegisterId;
use crate::serialization::op_code::OpCode;
use crate::serialization::sigma_byte_reader::SigmaByteRead;
use crate::serialization::sigma_byte_writer::SigmaByteWrite;
use crate::serialization::SigmaParsingError;
use crate::serialization::SigmaSerializable;
use crate::serialization::SigmaSerializeResult;
use crate::traversable::impl_traversable_expr;
use crate::types::stype::SType;

use super::expr::Expr;
use super::expr::InvalidArgumentError;
use crate::has_opcode::HasStaticOpCode;

/// Get box register value (Box.R0 - R9)
#[derive(PartialEq, Eq, Debug, Clone)]
pub struct ExtractRegisterAs {
    /// Box
    pub input: Box<Expr>,
    /// Register id to extract value from (0 is R0 .. 9 for R9)
    pub register_id: i8,
    /// Result type, to be wrapped in SOption
    pub elem_tpe: Arc<SType>,
}

impl ExtractRegisterAs {
    /// Create new object, returns an error if any of the requirements failed
    pub fn new(input: Expr, register_id: i8, tpe: SType) -> Result<Self, InvalidArgumentError> {
        if input.post_eval_tpe() != SType::SBox {
            return Err(InvalidArgumentError(format!(
                "expected input to be SBox, got {0:?}",
                input
            )));
        }
        let elem_tpe = match tpe {
            SType::SOption(t) => Ok(t),
            _ => Err(InvalidArgumentError(format!(
                "expected tpe to be SOption, got {0:?}",
                tpe
            ))),
        }?;

        Ok(ExtractRegisterAs {
            input: input.into(),
            register_id,
            elem_tpe,
        })
    }

    /// Type
    pub fn tpe(&self) -> SType {
        SType::SOption(self.elem_tpe.clone())
    }
}

impl HasStaticOpCode for ExtractRegisterAs {
    const OP_CODE: OpCode = OpCode::EXTRACT_REGISTER_AS;
}

impl SigmaSerializable for ExtractRegisterAs {
    fn sigma_serialize<W: SigmaByteWrite>(&self, w: &mut W) -> SigmaSerializeResult {
        self.input.sigma_serialize(w)?;
        w.put_i8(self.register_id)?;
        self.elem_tpe.sigma_serialize(w)
    }

    fn sigma_parse<R: SigmaByteRead>(r: &mut R) -> Result<Self, SigmaParsingError> {
        let input = Expr::sigma_parse(r)?;
        let register_id = r.get_i8()?;
        // `findRegisterByIndex(id).get` right after the id byte, before the type (v6.0.7
        // `ExtractRegisterAsSerializer.scala:25-31`)
        if RegisterId::try_from(register_id).is_err() {
            return Err(SigmaParsingError::RegisterIdOutOfRange(register_id));
        }
        let elem_tpe = SType::sigma_parse(r)?;
        // sigmastate casts the input to a box value with no check (the cast is erased) and
        // builds the node over it, so evaluating it fails if the value is not a box (v6.0.7
        // `ExtractRegisterAsSerializer.scala:25-31`, `transformers.scala:484-496`); `new`
        // stays strict for library users
        Ok(ExtractRegisterAs {
            input: input.into(),
            register_id,
            elem_tpe: elem_tpe.into(),
        })
    }
}

impl_traversable_expr!(ExtractRegisterAs, boxed input);

#[cfg(test)]
#[cfg(feature = "arbitrary")]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::mir::global_vars::GlobalVars;
    use crate::serialization::sigma_serialize_roundtrip;

    #[test]
    fn ser_roundtrip() {
        let e: Expr = ExtractRegisterAs::new(
            GlobalVars::SelfBox.into(),
            0,
            SType::SOption(SType::SLong.into()),
        )
        .unwrap()
        .into();
        assert_eq![sigma_serialize_roundtrip(&e), e];
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod register_id_tests {
    use crate::ergo_tree::ErgoTree;
    use crate::serialization::SigmaParsingError;

    fn tree(hex: &str) -> Result<ErgoTree, SigmaParsingError> {
        ErgoTree::sigma_parse_bytes_no_fallback(&base16::decode(hex.as_bytes()).unwrap())
    }

    #[test]
    fn a_register_id_outside_r0_to_r9_rejects() {
        // SANTA `tree_parse_acceptance` #38, #39: SELF.R10[Int].isDefined and the id byte
        // 0x80 (-128); #40: SELF.R9[Int].isDefined parses. The size-flagged form rejects too.
        assert!(matches!(
            tree("00d1e6c6a70a04"),
            Err(SigmaParsingError::RegisterIdOutOfRange(10))
        ));
        assert!(matches!(
            tree("00d1e6c6a78004"),
            Err(SigmaParsingError::RegisterIdOutOfRange(-128))
        ));
        assert!(matches!(
            tree("0805d1e6c6a70a04"),
            Err(SigmaParsingError::RegisterIdOutOfRange(10))
        ));
        assert!(matches!(tree("00d1e6c6a70904"), Ok(ErgoTree::Parsed(_))));
    }
}
