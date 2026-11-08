use core::convert::TryInto;

use alloc::vec::Vec;

use crate::serialization::op_code::OpCode;
use crate::serialization::sigma_byte_reader::SigmaByteRead;
use crate::serialization::sigma_byte_writer::SigmaByteWrite;
use crate::serialization::SigmaParsingError;
use crate::serialization::SigmaSerializable;
use crate::serialization::SigmaSerializeResult;
use crate::traversable::impl_traversable_expr;
use crate::types::stuple::STuple;
use crate::types::stuple::TupleItems;
use crate::types::stype::SType;

use super::expr::Expr;
use super::expr::InvalidArgumentError;
use crate::has_opcode::HasStaticOpCode;

/// Tuple of elements
#[derive(PartialEq, Eq, Debug, Clone)]
pub struct Tuple {
    /// Tuple items
    pub items: TupleItems<Expr>,
}

impl Tuple {
    /// Create new object
    pub fn new(items: Vec<Expr>) -> Result<Self, InvalidArgumentError> {
        Ok(Tuple {
            items: items.try_into()?,
        })
    }

    /// Type
    pub fn tpe(&self) -> SType {
        SType::STuple(STuple {
            items: self.items.mapped_ref(|it| it.tpe()),
        })
    }
}

impl HasStaticOpCode for Tuple {
    const OP_CODE: OpCode = OpCode::TUPLE;
}

impl SigmaSerializable for Tuple {
    fn sigma_serialize<W: SigmaByteWrite>(&self, w: &mut W) -> SigmaSerializeResult {
        w.put_u8(self.items.len() as u8)?;
        // item count is one byte (PutByteCost) -- Scala `TupleSerializer`'s `putUByte`;
        // charged when a box-register tuple expression is serialized under `Global.serialize`
        w.add_put_byte_cost();
        self.items.iter().try_for_each(|i| i.sigma_serialize(w))
    }

    fn sigma_parse<R: SigmaByteRead>(r: &mut R) -> Result<Self, SigmaParsingError> {
        Ok(Tuple {
            items: tuple_items_sigma_parse(r)?.try_into()?,
        })
    }
}

/// `TupleSerializer.parse` (v6.0.7 `TupleSerializer.scala:27-36`): a signed-byte size, as
/// many values, and no arity check. Read the same way in a tree and in a register.
pub(crate) fn tuple_items_sigma_parse<R: SigmaByteRead>(
    r: &mut R,
) -> Result<Vec<Expr>, SigmaParsingError> {
    let size = r.get_i8()?;
    if size < 0 {
        // `safeNewArray` throws a `NegativeArraySizeException`
        return Err(SigmaParsingError::NegativeTupleSize(size));
    }
    (0..size).map(|_| Expr::sigma_parse(r)).collect()
}

impl_traversable_expr!(Tuple, arr items);

#[cfg(feature = "arbitrary")]
#[allow(clippy::unwrap_used)]
/// Arbitrary impl
mod arbitrary {

    use super::*;
    use crate::mir::constant::Constant;
    use proptest::collection::*;
    use proptest::prelude::*;

    impl Arbitrary for Tuple {
        type Strategy = BoxedStrategy<Self>;
        type Parameters = ();

        fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
            prop_oneof![
                vec(any::<Expr>(), 2..4),
                vec(any::<Constant>().prop_map_into(), 2..4)
            ]
            .prop_map(move |items| Self::new(items).unwrap())
            .boxed()
        }
    }
}

#[cfg(test)]
#[cfg(feature = "arbitrary")]
#[allow(clippy::panic)]
mod tests {
    use super::*;
    use crate::serialization::sigma_serialize_roundtrip;
    use proptest::prelude::*;

    proptest! {

        #[test]
        fn ser_roundtrip(v in any::<Tuple>()) {
            let expr: Expr = v.into();
            prop_assert_eq![sigma_serialize_roundtrip(&expr), expr];
        }

    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod size_tests {
    use crate::ergo_tree::ErgoTree;
    use crate::serialization::SigmaParsingError;
    use crate::serialization::SigmaSerializable;
    use alloc::format;

    #[test]
    fn a_script_tuple_of_128_items_is_refused() {
        // `sigmaProp(Tuple(1, ..., 1)._1 == 1)` with 128 items: sigmastate reads the size as a
        // signed byte, and `safeNewArray(-128)` throws `NegativeArraySizeException`, at both
        // versions, without a degrade (SANTA case 5b)
        let body = format!("d1938c8680{}010402", "0402".repeat(128));
        for hex in [format!("00{body}"), format!("0b8802{body}")] {
            let res = ErgoTree::sigma_parse_bytes(&base16::decode(&hex).unwrap());
            assert!(
                matches!(res, Err(SigmaParsingError::NegativeTupleSize(-128))),
                "{res:?}"
            );
        }
    }

    #[test]
    fn a_script_tuple_of_127_items_parses() {
        // the largest size a signed byte holds (SANTA case 5a): parsed and written back as is
        let body = format!("d1938c867f{}010402", "0402".repeat(127));
        for hex in [format!("00{body}"), format!("0b8602{body}")] {
            let bytes = base16::decode(&hex).unwrap();
            let tree = ErgoTree::sigma_parse_bytes(&bytes).unwrap();
            assert!(matches!(tree, ErgoTree::Parsed(_)), "{hex}");
            assert_eq!(tree.sigma_serialize_bytes().unwrap(), bytes);
        }
    }
}
