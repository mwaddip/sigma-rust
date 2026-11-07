//! Extract register of SELF box as `Coll[Byte]`, deserialize it into Value and inline into executing script.

use alloc::boxed::Box;

use super::expr::Expr;
use super::expr::InvalidArgumentError;
use crate::chain::ergo_box::RegisterId;
use crate::has_opcode::HasStaticOpCode;
use crate::serialization::op_code::OpCode;
use crate::serialization::sigma_byte_reader::SigmaByteRead;
use crate::serialization::sigma_byte_writer::SigmaByteWrite;
use crate::serialization::SigmaParsingError;
use crate::serialization::SigmaSerializable;
use crate::serialization::SigmaSerializeResult;
use crate::traversable::impl_traversable_expr;
use crate::types::stype::SType;

/// Extract register of SELF box as `Coll[Byte]`, deserialize and execute Expr out of it
#[derive(PartialEq, Eq, Debug, Clone)]
pub struct DeserializeRegister {
    /// Register number (0 .. =9 for R0-R9 registers)
    pub reg: RegisterId,
    /// Type of expression serialized in register
    pub tpe: SType,
    /// Default value (expression that would be executed if register is empty)
    pub default: Option<Box<Expr>>,
}

impl DeserializeRegister {
    /// Create new object, returns an error if any of the requirements failed
    pub fn new(
        reg: RegisterId,
        tpe: SType,
        default: Option<Box<Expr>>,
    ) -> Result<Self, InvalidArgumentError> {
        Ok(Self { reg, tpe, default })
    }
    /// Type
    pub fn tpe(&self) -> SType {
        self.tpe.clone()
    }
}

impl HasStaticOpCode for DeserializeRegister {
    const OP_CODE: OpCode = OpCode::DESERIALIZE_REGISTER;
}

impl SigmaSerializable for DeserializeRegister {
    fn sigma_serialize<W: SigmaByteWrite>(&self, w: &mut W) -> SigmaSerializeResult {
        w.put_u8(self.reg.into())?;
        self.tpe.sigma_serialize(w)?;
        self.default.sigma_serialize(w)
    }

    fn sigma_parse<R: SigmaByteRead>(r: &mut R) -> Result<Self, SigmaParsingError> {
        r.set_deserialize(true);
        // `findRegisterByIndex(id).get` right after the id byte (v6.0.7
        // `DeserializeRegisterSerializer.scala:27-33`), a `NoSuchElementException`
        let id = r.get_i8()?;
        let reg =
            RegisterId::try_from(id).map_err(|_| SigmaParsingError::RegisterIdOutOfRange(id))?;
        let tpe = SType::sigma_parse(r)?;
        let default = Option::<Box<Expr>>::sigma_parse(r)?;
        Ok(Self::new(reg, tpe, default)?)
    }
}

impl_traversable_expr!(DeserializeRegister, opt default);

#[cfg(feature = "arbitrary")]
#[allow(clippy::unwrap_used)]
/// Arbitrary impl
mod arbitrary {
    use crate::mir::expr::arbitrary::ArbExprParams;

    use super::*;

    use alloc::boxed::Box;
    use proptest::option;
    use proptest::prelude::*;

    impl Arbitrary for DeserializeRegister {
        type Strategy = BoxedStrategy<Self>;
        type Parameters = usize;

        fn arbitrary_with(args: Self::Parameters) -> Self::Strategy {
            (
                0_u8..9,
                any::<SType>(),
                option::of(
                    any_with::<Expr>(ArbExprParams {
                        tpe: SType::SColl(SType::SBoolean.into()),
                        depth: args,
                    })
                    .prop_map(Box::new),
                ),
            )
                .prop_map(|(reg, tpe, default)| Self {
                    reg: reg.try_into().unwrap(),
                    tpe,
                    default,
                })
                .boxed()
        }
    }
}

#[cfg(test)]
#[cfg(feature = "arbitrary")]
#[allow(clippy::panic)]
mod tests {
    use crate::mir::expr::Expr;
    use crate::serialization::sigma_serialize_roundtrip;

    use super::*;

    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(16))]

        #[test]
        fn ser_roundtrip(v in any::<DeserializeRegister>()) {
            let expr: Expr = v.into();
            prop_assert_eq![sigma_serialize_roundtrip(&expr), expr];
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod register_id_tests {
    use crate::ergo_tree::ErgoTree;
    use crate::serialization::{SigmaParsingError, SigmaSerializable};

    #[test]
    fn a_register_id_outside_r0_to_r9_rejects_a_size_flagged_tree() {
        // SANTA `tree_parse_acceptance` #41: DeserializeRegister(R10, SigmaProp) as the root,
        // here size-flagged: the NoSuchElementException does not degrade
        for hex in ["00d50a0800", "0804d50a0800"] {
            assert!(matches!(
                ErgoTree::sigma_parse_bytes(&base16::decode(hex.as_bytes()).unwrap()),
                Err(SigmaParsingError::RegisterIdOutOfRange(10))
            ));
        }
    }
}
