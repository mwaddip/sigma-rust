use alloc::boxed::Box;
use alloc::format;

use crate::ergo_tree::ErgoTreeVersion;
use crate::mir::upcast::Upcast;
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

/// Get collection element by index
#[derive(PartialEq, Eq, Debug, Clone)]
pub struct ByIndex {
    /// Collection
    pub input: Box<Expr>,
    /// Element index
    pub index: Box<Expr>,
    /// Default value, returned if index is out of bounds in "Coll.getOrElse()" op
    pub default: Option<Box<Expr>>,
}

impl ByIndex {
    /// Create new object, returns an error if any of the requirements failed
    pub fn new(
        input: Expr,
        index: Expr,
        default: Option<Box<Expr>>,
    ) -> Result<Self, InvalidArgumentError> {
        let input_elem_type = match input.post_eval_tpe() {
            SType::SColl(elem_type) => Ok(elem_type),
            _ => Err(InvalidArgumentError(format!(
                "Expected ByIndex input to be SColl, got {0:?}",
                input.tpe()
            ))),
        }?;
        if index.post_eval_tpe() != SType::SInt {
            return Err(InvalidArgumentError(format!(
                "ByIndex: expected index type to be SInt, got {0:?}",
                index
            )));
        }
        if !default
            .as_ref()
            .map(|expr| expr.post_eval_tpe() == *input_elem_type)
            .unwrap_or(true)
        {
            return Err(InvalidArgumentError(format!(
                "ByIndex: expected default type to be {0:?}, got {1:?}",
                input_elem_type, default
            )));
        }
        Ok(Self {
            input: input.into(),
            index: index.into(),
            default,
        })
    }

    /// Rebuild this node with new children.
    ///
    /// The printer walks a tree and rebuilds every node it passes from the children it has
    /// printed. It cannot go back through `new()`: the parser accepts nodes `new()` refuses,
    /// and printing one must not fail.
    pub(crate) fn with_children(
        &self,
        input: Expr,
        index: Expr,
        default: Option<Box<Expr>>,
    ) -> Self {
        Self {
            input: Box::new(input),
            index: Box::new(index),
            default,
        }
    }

    /// Type: the input's element type, computed on access so that it cannot go stale when the
    /// input is replaced (sigmastate computes its `val tpe = input.tpe.elemType` whenever it
    /// builds the node, v6.0.7 `transformers.scala:254`). `SAny` for a tuple, an
    /// `SCollection[SAny]` (`SType.scala:838-841`); NoType for anything else, which the
    /// parser refuses (`check_build`) and `new()` refuses.
    pub fn tpe(&self) -> SType {
        match self.input.tpe() {
            SType::SColl(elem_tpe) => (*elem_tpe).clone(),
            SType::STuple(_) => SType::SAny,
            _ => SType::SNoType,
        }
    }
}

impl HasStaticOpCode for ByIndex {
    const OP_CODE: OpCode = OpCode::BY_INDEX;
}

impl SigmaSerializable for ByIndex {
    fn sigma_serialize<W: SigmaByteWrite>(&self, w: &mut W) -> SigmaSerializeResult {
        self.input.sigma_serialize(w)?;
        self.index.sigma_serialize(w)?;
        self.default.sigma_serialize(w)
    }

    fn sigma_parse<R: SigmaByteRead>(r: &mut R) -> Result<Self, SigmaParsingError> {
        let input = Expr::sigma_parse(r)?;
        let index = Expr::sigma_parse(r)?;
        // Below tree version 3 sigmastate upcasts the index to Int right after reading it,
        // before the default (v6.0.7 `ByIndexSerializer.scala:27-36`): `upcastTo(SInt)`
        // asserts a numeric index no wider than Int and wraps a Byte or Short one in an
        // `Upcast` (`syntax.scala:168-177`). From version 3 the index is taken as it is.
        let index = if r.tree_version() < ErgoTreeVersion::V3 {
            upcast_index_to_int(index)?
        } else {
            index
        };
        let default = Option::<Box<Expr>>::sigma_parse(r)?;
        // built as sigmastate builds it, its input's cast in `check_build`; its index and its
        // default are not checked
        Ok(Self {
            input: input.into(),
            index: index.into(),
            default,
        })
    }
}

/// `upcastTo(SInt)` of a `ByIndex` index read below tree version 3
fn upcast_index_to_int(index: Expr) -> Result<Expr, SigmaParsingError> {
    match index.tpe() {
        SType::SInt => Ok(index),
        SType::SByte | SType::SShort => Ok(Upcast {
            input: Box::new(index),
            tpe: SType::SInt,
        }
        .into()),
        tpe => Err(SigmaParsingError::IndexUpcastFailed(format!(
            "an index of type {tpe:?}"
        ))),
    }
}

impl_traversable_expr!(ByIndex, boxed input, boxed index, opt default);

#[cfg(feature = "arbitrary")]
#[allow(clippy::unwrap_used)]
/// Arbitrary impl
mod arbitrary {
    use super::*;
    use crate::mir::expr::arbitrary::ArbExprParams;
    use proptest::prelude::*;
    use proptest::result::Probability;

    impl Arbitrary for ByIndex {
        type Strategy = BoxedStrategy<Self>;
        type Parameters = ArbExprParams;

        fn arbitrary_with(args: Self::Parameters) -> Self::Strategy {
            (
                any_with::<Expr>(ArbExprParams {
                    tpe: SType::SColl(args.tpe.clone().into()),
                    depth: args.depth,
                }),
                any_with::<Expr>(ArbExprParams {
                    tpe: SType::SInt,
                    depth: 0,
                }),
                any_with::<Option<Box<Expr>>>((
                    Probability::default(),
                    ArbExprParams {
                        tpe: args.tpe,
                        depth: 0,
                    },
                )),
            )
                .prop_map(|(input, index, default)| Self::new(input, index, default).unwrap())
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
        fn ser_roundtrip(v in any::<ByIndex>()) {
            let expr: Expr = v.into();
            prop_assert_eq![sigma_serialize_roundtrip(&expr), expr];
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod index_upcast_tests {
    use crate::ergo_tree::ErgoTree;
    use crate::serialization::{SigmaParsingError, SigmaSerializable};

    fn tree(hex: &str) -> Result<ErgoTree, SigmaParsingError> {
        ErgoTree::sigma_parse_bytes_no_fallback(&base16::decode(hex.as_bytes()).unwrap())
    }

    #[test]
    fn below_tree_version_3_a_byte_index_is_upcast_to_int() {
        // v0 sigmaProp(Coll[Int](1)(Byte 0) == 1): the index is wrapped in an Upcast to Int,
        // which the serializer writes back as the constant
        let hex = "00d193b21001020200000402";
        assert_eq!(
            tree(hex).unwrap().sigma_serialize_bytes().unwrap(),
            base16::decode(hex.as_bytes()).unwrap()
        );
    }

    #[test]
    fn below_tree_version_3_an_index_wider_than_int_rejects() {
        // SANTA `tree_parse_acceptance` #33: v0 ByIndex(Coll[Boolean](true), Long 0), an
        // AssertionError in `upcastTo`
        assert!(matches!(
            tree("00d1b20d0101050000"),
            Err(SigmaParsingError::IndexUpcastFailed(_))
        ));
    }
}
