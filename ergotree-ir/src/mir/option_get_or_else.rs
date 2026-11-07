use super::expr::Expr;
use super::expr::InvalidArgumentError;
use crate::has_opcode::HasStaticOpCode;
use crate::serialization::op_code::OpCode;
use crate::serialization::sigma_byte_reader::SigmaByteRead;
use crate::serialization::sigma_byte_writer::SigmaByteWrite;
use crate::serialization::SigmaParsingError;
use crate::serialization::SigmaSerializable;
use crate::serialization::SigmaSerializeResult;
use crate::traversable::impl_traversable_expr;
use crate::types::stype::SType;
use alloc::boxed::Box;

/// Returns the Option's value or error if no value
#[derive(PartialEq, Eq, Debug, Clone)]
pub struct OptionGetOrElse {
    /// Object of SOption type
    pub input: Box<Expr>,
    /// Default value if option is empty
    pub default: Box<Expr>,
}

impl OptionGetOrElse {
    /// Create new object, returns an error if any of the requirements failed
    pub fn new(input: Expr, default: Expr) -> Result<Self, InvalidArgumentError> {
        match input.post_eval_tpe() {
            SType::SOption(elem_tpe) => {
                default.check_post_eval_tpe(elem_tpe.as_ref())?;
                Ok(OptionGetOrElse {
                    input: Box::new(input),
                    default: Box::new(default),
                })
            }
            _ => Err(InvalidArgumentError(format!(
                "expected OptionGetOrElse::input type to be SOption, got: {0:?}",
                input.tpe(),
            ))),
        }
    }

    /// Rebuild this node with new children. See `ByIndex::with_children` for why the printer
    /// cannot go back through `new()`.
    pub(crate) fn with_children(&self, input: Expr, default: Expr) -> Self {
        Self {
            input: Box::new(input),
            default: Box::new(default),
        }
    }

    /// Type: the input's Option element type, computed on access as sigmastate's `def tpe` is
    /// (v6.0.7 `transformers.scala:626`). NoType for an input that is not an Option, which the
    /// parser refuses (`check_build`) and `new()` refuses.
    pub fn tpe(&self) -> SType {
        match self.input.tpe() {
            SType::SOption(elem_tpe) => (*elem_tpe).clone(),
            _ => SType::SNoType,
        }
    }
}

impl HasStaticOpCode for OptionGetOrElse {
    const OP_CODE: OpCode = OpCode::OPTION_GET_OR_ELSE;
}

impl SigmaSerializable for OptionGetOrElse {
    fn sigma_serialize<W: SigmaByteWrite>(&self, w: &mut W) -> SigmaSerializeResult {
        self.input.sigma_serialize(w)?;
        self.default.sigma_serialize(w)
    }

    fn sigma_parse<R: SigmaByteRead>(r: &mut R) -> Result<Self, SigmaParsingError> {
        let input = Expr::sigma_parse(r)?;
        let default = Expr::sigma_parse(r)?;
        // built as sigmastate builds it, its input's cast in `check_build`; the default is not
        // checked (v6.0.7 `transformers.scala:622-626`)
        Ok(OptionGetOrElse {
            input: Box::new(input),
            default: Box::new(default),
        })
    }
}

impl_traversable_expr!(OptionGetOrElse, boxed input, boxed default);

#[cfg(test)]
#[cfg(feature = "arbitrary")]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::mir::constant::Constant;
    use crate::mir::expr::Expr;
    use crate::mir::extract_reg_as::ExtractRegisterAs;
    use crate::mir::global_vars::GlobalVars;
    use crate::serialization::sigma_serialize_roundtrip;
    use crate::types::stype::SType;

    #[test]
    fn ser_roundtrip() {
        let get_reg_expr: Expr = ExtractRegisterAs::new(
            GlobalVars::SelfBox.into(),
            0,
            SType::SOption(SType::SLong.into()),
        )
        .unwrap()
        .into();
        let default_expr: Constant = 1i64.into();
        let e: Expr = OptionGetOrElse::new(get_reg_expr, default_expr.into())
            .unwrap()
            .into();
        assert_eq![sigma_serialize_roundtrip(&e), e];
    }
}
