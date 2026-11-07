use core::convert::TryFrom;

use alloc::boxed::Box;

use crate::serialization::op_code::OpCode;
use crate::serialization::sigma_byte_reader::SigmaByteRead;
use crate::serialization::sigma_byte_writer::SigmaByteWrite;
use crate::serialization::SigmaParsingError;
use crate::serialization::SigmaSerializable;
use crate::serialization::SigmaSerializeResult;
use crate::traversable::impl_traversable_expr;
use crate::types::stuple::STuple;
use crate::types::stype::SType;

use super::expr::Expr;
use super::expr::InvalidArgumentError;
use crate::has_opcode::HasStaticOpCode;

/// Tuple field access index (1..=255)
#[derive(PartialEq, Eq, Debug, Copy, Clone, derive_more::Display)]
pub struct TupleFieldIndex(u8);

/// Error for tuple index being out of bounds (1..=255)
#[derive(Debug)]
pub struct TupleFieldIndexOutBounds;

impl TryFrom<u8> for TupleFieldIndex {
    type Error = TupleFieldIndexOutBounds;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        if value >= 1 {
            Ok(TupleFieldIndex(value))
        } else {
            Err(TupleFieldIndexOutBounds)
        }
    }
}

impl TupleFieldIndex {
    /// Returns a zero-based index
    pub fn zero_based_index(&self) -> usize {
        (self.0 - 1) as usize
    }
}

impl SigmaSerializable for TupleFieldIndex {
    fn sigma_serialize<W: SigmaByteWrite>(&self, w: &mut W) -> SigmaSerializeResult {
        w.put_u8(self.0)?;
        Ok(())
    }

    fn sigma_parse<R: SigmaByteRead>(r: &mut R) -> Result<Self, SigmaParsingError> {
        let field_index = r.get_u8()?;
        TupleFieldIndex::try_from(field_index).map_err(|_| {
            SigmaParsingError::ValueOutOfBounds(format!(
                "invalid tuple field index: {0}",
                field_index
            ))
        })
    }
}

/// Select a field of the tuple value
#[derive(PartialEq, Eq, Debug, Clone)]
pub struct SelectField {
    /// Tuple value
    pub input: Box<Expr>,
    /// 1-based tuple field index (input._1 has field_index of 1)
    pub field_index: TupleFieldIndex,
    /// Field type
    field_tpe: SType,
}

impl SelectField {
    /// Create new object, returns an error if any of the requirements failed
    pub fn new(input: Expr, field_index: TupleFieldIndex) -> Result<Self, InvalidArgumentError> {
        match input.tpe() {
            SType::STuple(STuple { items }) => {
                if let Some(field_tpe) = items.get(field_index.zero_based_index()) {
                    Ok(SelectField {
                        input: Box::new(input),
                        field_index,
                        field_tpe: field_tpe.clone(),
                    })
                } else {
                    Err(InvalidArgumentError(format!(
                        "SelectField field index is out of bounds {0:?}, tuple type: {1:?}",
                        field_index,
                        input.tpe()
                    )))
                }
            }
            tpe => Err(InvalidArgumentError(format!(
                "expected SelectField input type to be STuple, got {0:?}",
                tpe
            ))),
        }
    }
}

impl HasStaticOpCode for SelectField {
    const OP_CODE: OpCode = OpCode::SELECT_FIELD;
}

impl SelectField {
    /// Rebuild this node with a new input, keeping the field type this one was built with.
    /// See `ByIndex::with_children` for why the printer cannot go back through `new()`.
    pub(crate) fn with_children(&self, input: Expr) -> Self {
        Self {
            input: Box::new(input),
            field_index: self.field_index,
            field_tpe: self.field_tpe.clone(),
        }
    }

    /// Type
    pub fn tpe(&self) -> SType {
        self.field_tpe.clone()
    }
}

impl SigmaSerializable for SelectField {
    fn sigma_serialize<W: SigmaByteWrite>(&self, w: &mut W) -> SigmaSerializeResult {
        self.input.sigma_serialize(w)?;
        self.field_index.sigma_serialize(w)
    }

    /// sigmastate reads the input and the field index, a signed byte, and builds the node
    /// (v6.0.7 `SelectFieldSerializer.scala:26-30`), whose `val tpe = input.tpe.items(fieldIndex
    /// - 1)` casts the input's type to `STuple`, then takes the field (`transformers.scala:294`)
    fn sigma_parse<R: SigmaByteRead>(r: &mut R) -> Result<Self, SigmaParsingError> {
        let input = Expr::sigma_parse(r)?;
        let field_index = r.get_i8()?;
        let field_tpe = field_tpe(&input, field_index)?;
        Ok(SelectField {
            input: Box::new(input),
            // 1..=127 here, as the index is a positive signed byte that names an item
            field_index: TupleFieldIndex(field_index as u8),
            field_tpe,
        })
    }
}

/// sigmastate's `val tpe = input.tpe.items(fieldIndex - 1)` (v6.0.7 `transformers.scala:294`):
/// the input's type cast to `STuple`, then the field, by a signed index
fn field_tpe(input: &Expr, field_index: i8) -> Result<SType, SigmaParsingError> {
    let items = match input.try_tpe()? {
        SType::STuple(STuple { items }) => items,
        tpe => {
            return Err(SigmaParsingError::TupleInputExpected(format!(
                "SelectField of an input of type {tpe:?}"
            )))
        }
    };
    Ok(usize::try_from(field_index)
        .ok()
        .and_then(|i| i.checked_sub(1))
        .and_then(|i| items.get(i))
        .ok_or_else(|| {
            SigmaParsingError::TupleFieldOutOfBounds(format!(
                "field {field_index} of a tuple of {} items",
                items.len()
            ))
        })?
        .clone())
}

impl SelectField {
    /// Recomputes the field's type from the current input, as sigmastate's constructor does
    /// on a node it rebuilds (the substitution at spend)
    pub(crate) fn rebuild(&mut self) -> Result<(), SigmaParsingError> {
        self.field_tpe = field_tpe(&self.input, self.field_index.0 as i8)?;
        Ok(())
    }
}

impl_traversable_expr!(SelectField, boxed input);

#[cfg(test)]
#[cfg(feature = "arbitrary")]
#[allow(clippy::unwrap_used)]
mod tests {
    use core::convert::TryInto;

    use crate::serialization::sigma_serialize_roundtrip;

    use super::*;

    #[test]
    fn ser_roundtrip() {
        let e: Expr = SelectField::new(Expr::Const((1i64, true).into()), 1u8.try_into().unwrap())
            .unwrap()
            .into();
        assert_eq![sigma_serialize_roundtrip(&e), e];
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod build_tests {
    use crate::ergo_tree::ErgoTree;
    use crate::serialization::{SigmaParsingError, SigmaSerializable};

    fn tree(hex: &str) -> Result<ErgoTree, SigmaParsingError> {
        ErgoTree::sigma_parse_bytes(&base16::decode(hex.as_bytes()).unwrap())
    }

    #[test]
    fn select_field_casts_its_input_to_a_tuple_and_then_indexes_it() {
        // `SelectField`'s `val tpe = input.tpe.items(fieldIndex - 1)` (v6.0.7
        // `transformers.scala:294`): an input that is not a tuple is a `ClassCastException`,
        // a field index (a signed byte, `SelectFieldSerializer.scala:28`) outside 1..=n an
        // `IndexOutOfBoundsException`; neither is a `ValidationException`, so a size-flagged
        // tree rejects. sigmaProp(EQ(SelectField(x, i), Int 1)), v0, size-flagged:
        // over Int 1, then over the pair (1, 2) with i = 0, 3 and 200 (-56).
        assert!(matches!(
            tree("0808d1938c0402010402"),
            Err(SigmaParsingError::TupleInputExpected(_))
        ));
        for i in ["00", "03", "c8"] {
            let hex = format!("080cd1938c8602040204 04{i}0402").replace(' ', "");
            assert!(
                matches!(tree(&hex), Err(SigmaParsingError::TupleFieldOutOfBounds(_))),
                "{hex}: {:?}",
                tree(&hex)
            );
        }
        // the second field of (1, 2) parses and round-trips
        let bytes = base16::decode(b"080cd1938c860204020404020402").unwrap();
        let parsed = ErgoTree::sigma_parse_bytes(&bytes).unwrap();
        assert!(matches!(parsed, ErgoTree::Parsed(_)));
        assert_eq!(parsed.sigma_serialize_bytes().unwrap(), bytes);
    }
}
