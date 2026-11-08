//! `TaggedVariable`: a node sigmastate parses and never evaluates. sigmastate marks its reader
//! "scheduled for version-gated rejection in v7.0" (v6.0.7 `TaggedVariableSerializer`); 6.0.7
//! reads it, so sigma-rust does.

use crate::has_opcode::HasStaticOpCode;
use crate::serialization::op_code::OpCode;
use crate::serialization::sigma_byte_reader::SigmaByteRead;
use crate::serialization::sigma_byte_writer::SigmaByteWrite;
use crate::serialization::SigmaParsingError;
use crate::serialization::SigmaSerializable;
use crate::serialization::SigmaSerializeResult;
use crate::traversable::impl_traversable_expr;
use crate::types::stype::SType;

use super::expr::Expr;

/// A context variable by id and type, the predecessor of `GetVar` (v6.0.7 `values.scala:455-461`).
/// sigmastate builds it with no check and defines no `eval` for it
#[derive(PartialEq, Eq, Debug, Clone)]
pub struct TaggedVariable {
    /// ID of the variable, a signed byte to sigmastate, kept as read
    pub var_id: u8,
    /// Type of the variable
    pub var_tpe: SType,
}

impl TaggedVariable {
    /// Creates the node
    pub fn new(var_id: u8, var_tpe: SType) -> Self {
        Self { var_id, var_tpe }
    }

    /// Type
    pub fn tpe(&self) -> SType {
        self.var_tpe.clone()
    }
}

impl HasStaticOpCode for TaggedVariable {
    const OP_CODE: OpCode = OpCode::TAGGED_VARIABLE;
}

impl SigmaSerializable for TaggedVariable {
    fn sigma_serialize<W: SigmaByteWrite>(&self, w: &mut W) -> SigmaSerializeResult {
        w.put_u8(self.var_id)?;
        self.var_tpe.sigma_serialize(w)
    }

    fn sigma_parse<R: SigmaByteRead>(r: &mut R) -> Result<Self, SigmaParsingError> {
        let var_id = r.get_u8()?;
        let var_tpe = SType::sigma_parse(r)?;
        Ok(Self { var_id, var_tpe })
    }
}

impl_traversable_expr!(TaggedVariable);

#[cfg(feature = "arbitrary")]
mod arbitrary {
    use super::*;
    use proptest::prelude::*;

    impl Arbitrary for TaggedVariable {
        type Strategy = BoxedStrategy<Self>;
        type Parameters = ();

        fn arbitrary_with(_: Self::Parameters) -> Self::Strategy {
            (any::<u8>(), any::<SType>())
                .prop_map(|(var_id, var_tpe)| Self::new(var_id, var_tpe))
                .boxed()
        }
    }
}

#[cfg(test)]
#[cfg(feature = "arbitrary")]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::ergo_tree::ErgoTree;
    use crate::serialization::sigma_serialize_roundtrip;
    use crate::serialization::SigmaSerializable;
    use proptest::prelude::*;

    /// The `TaggedVariable` under the root `sigmaProp(...)` of a parsed tree
    fn parse_tagged_variable(hex: &str) -> TaggedVariable {
        let bytes = base16::decode(hex).unwrap();
        let tree = ErgoTree::sigma_parse_bytes(&bytes).unwrap();
        assert!(matches!(tree, ErgoTree::Parsed(_)), "{hex}: {tree:?}");
        assert_eq!(tree.sigma_serialize_bytes().unwrap(), bytes, "{hex}");
        let Expr::BoolToSigmaProp(b) = tree.root_expr().unwrap() else {
            panic!("root: {tree:?}")
        };
        let Expr::TaggedVariable(v) = &*b.input else {
            panic!("input: {b:?}")
        };
        v.clone()
    }

    /// `sigmaProp(TaggedVariable(1, Boolean))`, an unsized v0 tree: sigmastate reads the id, a
    /// signed byte, then a type (v6.0.7 `TaggedVariableSerializer`)
    #[test]
    fn tagged_variable_parses_and_round_trips() {
        let v = parse_tagged_variable("00d1710101");
        assert_eq!(v.var_id, 1);
        assert_eq!(v.var_tpe, SType::SBoolean);
    }

    /// An id above 127, a negative byte to sigmastate, is kept as read
    #[test]
    fn tagged_variable_high_id_round_trips() {
        let v = parse_tagged_variable("00d171ff01");
        assert_eq!(v.var_id, 255);
    }

    proptest! {

        #[test]
        fn ser_roundtrip(v in any::<TaggedVariable>()) {
            let expr: Expr = v.into();
            prop_assert_eq![sigma_serialize_roundtrip(&expr), expr];
        }
    }
}
