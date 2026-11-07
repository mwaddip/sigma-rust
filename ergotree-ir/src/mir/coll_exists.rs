use alloc::boxed::Box;

use alloc::sync::Arc;

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

/// Tests whether a predicate holds for at least one element of this collection
#[derive(PartialEq, Eq, Debug, Clone)]
pub struct Exists {
    /// Collection
    pub input: Box<Expr>,
    /// Function (lambda) to test each element
    pub condition: Box<Expr>,
    /// Collection element type
    pub elem_tpe: Arc<SType>,
}

impl Exists {
    /// Create new object, returns an error if any of the requirements failed
    pub fn new(input: Expr, condition: Expr) -> Result<Self, InvalidArgumentError> {
        let input_elem_type = match input.post_eval_tpe() {
            SType::SColl(elem_type) => Ok(elem_type),
            _ => Err(InvalidArgumentError(format!(
                "Expected Exists input to be SColl, got {0:?}",
                input.tpe()
            ))),
        }?;
        match condition.tpe() {
            SType::SFunc(sfunc)
                if sfunc.t_dom.len() == 1
                    && sfunc.t_dom[0] == *input_elem_type
                    && *sfunc.t_range == SType::SBoolean =>
            {
                Ok(Exists {
                    input: input.into(),
                    condition: condition.into(),
                    elem_tpe: input_elem_type,
                })
            }
            _ => Err(InvalidArgumentError(format!(
                "Invalid condition tpe: {0:?}",
                condition.tpe()
            ))),
        }
    }

    /// Type
    pub fn tpe(&self) -> SType {
        SType::SBoolean
    }
}

impl HasStaticOpCode for Exists {
    const OP_CODE: OpCode = OpCode::EXISTS;
}

impl SigmaSerializable for Exists {
    fn sigma_serialize<W: SigmaByteWrite>(&self, w: &mut W) -> SigmaSerializeResult {
        self.input.sigma_serialize(w)?;
        self.condition.sigma_serialize(w)
    }

    /// sigmastate builds `Exists` with no check of its input or condition (v6.0.7
    /// `transformers.scala:155`)
    fn sigma_parse<R: SigmaByteRead>(r: &mut R) -> Result<Self, SigmaParsingError> {
        let input = Expr::sigma_parse(r)?;
        let condition = Expr::sigma_parse(r)?;
        Ok(Exists {
            elem_tpe: parsed_elem_tpe(&input),
            input: input.into(),
            condition: condition.into(),
        })
    }
}

/// The element type of a collection node the parser builds over `input` without checking it:
/// the collection's element type, `SAny` for a tuple (`STuple` is an `SCollection[SAny]`,
/// `SType.scala:838-841`), NoType for anything else
pub(crate) fn parsed_elem_tpe(input: &Expr) -> Arc<SType> {
    match input.tpe() {
        SType::SColl(elem_tpe) => elem_tpe,
        SType::STuple(_) => SType::SAny.into(),
        _ => SType::SNoType.into(),
    }
}

impl_traversable_expr!(Exists, boxed input, boxed condition);

#[allow(clippy::unwrap_used)]
#[cfg(feature = "arbitrary")]
/// Arbitrary impl
mod arbitrary {
    use super::*;
    use crate::mir::expr::arbitrary::ArbExprParams;
    use crate::types::sfunc::SFunc;
    use alloc::vec;
    use proptest::prelude::*;

    impl Arbitrary for Exists {
        type Strategy = BoxedStrategy<Self>;
        type Parameters = ();

        fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
            (
                any_with::<Expr>(ArbExprParams {
                    tpe: SType::SColl(SType::SBoolean.into()),
                    depth: 1,
                }),
                any_with::<Expr>(ArbExprParams {
                    tpe: SType::SFunc(SFunc {
                        t_dom: vec![SType::SBoolean],
                        t_range: SType::SBoolean.into(),
                        tpe_params: vec![],
                    }),
                    depth: 0,
                }),
            )
                .prop_map(|(input, mapper)| Exists::new(input, mapper).unwrap())
                .boxed()
        }
    }
}

#[cfg(test)]
#[cfg(feature = "arbitrary")]
#[allow(clippy::panic)]
mod tests {
    use super::*;
    use crate::mir::expr::Expr;
    use crate::serialization::sigma_serialize_roundtrip;
    use proptest::prelude::*;

    proptest! {

        #![proptest_config(ProptestConfig::with_cases(16))]

        #[test]
        fn ser_roundtrip(v in any::<Exists>()) {
            let expr: Expr = v.into();
            prop_assert_eq![sigma_serialize_roundtrip(&expr), expr];
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod parse_tests {
    use crate::ergo_tree::ErgoTree;
    use crate::serialization::SigmaSerializable;

    #[test]
    fn exists_over_a_non_collection_parses() {
        // SANTA `tree_parse_acceptance` #0: sigmaProp(Exists(Int 1, (x: Int) => true))
        let bytes = base16::decode(b"00d1ae0402d90101040101").unwrap();
        let tree = ErgoTree::sigma_parse_bytes(&bytes).unwrap();
        assert!(matches!(tree, ErgoTree::Parsed(_)));
        assert_eq!(tree.sigma_serialize_bytes().unwrap(), bytes);
    }
}
