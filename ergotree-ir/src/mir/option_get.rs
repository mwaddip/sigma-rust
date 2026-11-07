use alloc::boxed::Box;

use super::expr::Expr;
use super::expr::InvalidArgumentError;
use super::unary_op::OneArgOp;
use super::unary_op::OneArgOpTryBuild;
use crate::has_opcode::HasStaticOpCode;
use crate::serialization::op_code::OpCode;
use crate::types::stype::SType;

/// Returns the Option's value or error if no value
#[derive(PartialEq, Eq, Debug, Clone)]
pub struct OptionGet {
    /// Object of SOption type
    pub input: Box<Expr>,
}

impl OptionGet {
    /// Rebuild this node with a new input. See `ByIndex::with_children` for why the printer
    /// cannot go back through its strict constructor.
    pub(crate) fn with_children(&self, input: Expr) -> Self {
        Self {
            input: Box::new(input),
        }
    }

    /// Type: the input's Option element type, computed on access as sigmastate's `def tpe`
    /// is (v6.0.7 `transformers.scala:601`), so it cannot go stale when the input is
    /// replaced. NoType for an input that is not an Option, which the parser refuses
    /// (`check_build`) and `try_build` refuses.
    pub fn tpe(&self) -> SType {
        match self.input.tpe() {
            SType::SOption(elem_tpe) => (*elem_tpe).clone(),
            _ => SType::SNoType,
        }
    }
}

impl HasStaticOpCode for OptionGet {
    const OP_CODE: OpCode = OpCode::OPTION_GET;
}

impl OneArgOp for OptionGet {
    fn input(&self) -> &Expr {
        &self.input
    }
    fn input_mut(&mut self) -> &mut Expr {
        &mut self.input
    }
}

impl OneArgOpTryBuild for OptionGet {
    fn try_build(input: Expr) -> Result<Self, InvalidArgumentError> {
        match input.post_eval_tpe() {
            SType::SOption(_) => Ok(OptionGet {
                input: Box::new(input),
            }),
            _ => Err(InvalidArgumentError(format!(
                "expected OptionGet::input type to be SOption, got: {0:?}",
                input.tpe(),
            ))),
        }
    }
}

#[cfg(test)]
#[cfg(feature = "arbitrary")]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
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
        let e: Expr = OptionGet::try_build(get_reg_expr).unwrap().into();
        assert_eq![sigma_serialize_roundtrip(&e), e];
    }
}
