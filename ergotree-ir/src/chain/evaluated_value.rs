//! Register and context-extension values

use alloc::format;
use alloc::vec::Vec;

use ergo_chain_types::ec_point::generator;

use crate::has_opcode::HasOpCode;
use crate::mir::collection::coll_sigma_serialize;
use crate::mir::collection::Collection;
use crate::mir::constant::Constant;
use crate::mir::constant::Literal;
use crate::mir::constant::TryExtractFromError;
use crate::mir::expr::Expr;
use crate::mir::global_vars::GlobalVars;
use crate::mir::value::CollKind;
use crate::serialization::op_code::OpCode;
use crate::serialization::sigma_byte_reader::SigmaByteRead;
use crate::serialization::sigma_byte_writer::SigmaByteWrite;
use crate::serialization::SigmaParsingError;
use crate::serialization::SigmaSerializable;
use crate::serialization::SigmaSerializationError;
use crate::serialization::SigmaSerializeResult;
use crate::types::stuple::STuple;
use crate::types::stype::SType;

/// A register or context-extension value: what sigmastate's `getValue` reads there and casts
/// to its sealed `EvaluatedValue` (v6.0.6 `ErgoBoxCandidate.scala:231`,
/// `ContextExtension.scala:61`, `values.scala:310`)
#[derive(PartialEq, Eq, Debug, Clone)]
pub enum EvaluatedValue {
    /// A constant, `TrueLeaf` and `FalseLeaf` included
    Constant(Constant),
    /// A value that is not a constant
    Expr(EvaluatedExpr),
}

/// The `EvaluatedValue` nodes that are not constants. Their items may be any expression:
/// sigmastate's cast checks only the outer node.
#[derive(PartialEq, Eq, Debug, Clone)]
pub enum EvaluatedExpr {
    /// A `Tuple` of 0 to 127 items: its size is a signed byte, and nothing checks its arity
    /// (`TupleSerializer.scala:27-36`)
    Tuple(Vec<Expr>),
    /// A `ConcreteCollection`
    Collection(Collection),
    /// `GroupGenerator`
    GroupGenerator,
}

impl<T: Into<Constant>> From<T> for EvaluatedValue {
    fn from(c: T) -> Self {
        EvaluatedValue::Constant(c.into())
    }
}

/// What a script reads from a register or a context variable: sigmastate's
/// `toAnyValue(v.value)(stypeToRType(v.tpe))`, which `toSigmaContext` builds for every context
/// variable and `CBox` for every register (v6.0.6 `ErgoLikeContext.scala:158-161`,
/// `CBox.scala:83-92`)
#[derive(PartialEq, Eq, Debug, Clone)]
pub struct ScriptValue {
    /// The type a script reads the value at: `None` for a tuple of fewer than two items, which no
    /// script type here can name
    pub tpe: Option<SType>,
    /// The data
    pub v: Literal,
}

impl EvaluatedValue {
    /// sigmastate's `v.tpe`: `None` for a tuple of fewer than two items
    pub fn tpe(&self) -> Option<SType> {
        match self {
            EvaluatedValue::Constant(c) => Some(c.tpe.clone()),
            EvaluatedValue::Expr(EvaluatedExpr::Tuple(items)) => {
                STuple::try_from(items.iter().map(Expr::tpe).collect::<Vec<_>>())
                    .ok()
                    .map(SType::STuple)
            }
            EvaluatedValue::Expr(EvaluatedExpr::Collection(c)) => Some(c.tpe()),
            EvaluatedValue::Expr(EvaluatedExpr::GroupGenerator) => Some(SType::SGroupElement),
        }
    }

    /// The value as a script reads it (see [`ScriptValue`]). Fails where sigmastate's conversion
    /// throws: an item that is not a value itself, a tuple expression inside a collection of
    /// pairs, or a type with no runtime form.
    pub fn to_script_value(&self) -> Result<ScriptValue, TryExtractFromError> {
        // `stypeToRType(v.tpe)`, then `v.value` (`ErgoLikeContext.scala:159-160`)
        let tpe = self.script_type()?;
        let v = match self {
            EvaluatedValue::Constant(c) => c.v.clone(),
            EvaluatedValue::Expr(EvaluatedExpr::Tuple(items)) => tuple_script_data(items)?,
            EvaluatedValue::Expr(EvaluatedExpr::Collection(c)) => collection_script_data(c)?,
            EvaluatedValue::Expr(EvaluatedExpr::GroupGenerator) => {
                Literal::GroupElement(generator().into())
            }
        };
        Ok(ScriptValue { tpe, v })
    }

    /// The type a script reads the value at (see [`ScriptValue::tpe`]), before its data is read:
    /// sigmastate's `stypeToRType(v.tpe)`, which fails for a type with no runtime form
    pub fn script_type(&self) -> Result<Option<SType>, TryExtractFromError> {
        let has_runtime_type = match self {
            EvaluatedValue::Constant(c) => has_runtime_type(&c.tpe),
            EvaluatedValue::Expr(EvaluatedExpr::Tuple(items)) => {
                items.iter().all(|item| has_runtime_type(&item.tpe()))
            }
            EvaluatedValue::Expr(EvaluatedExpr::Collection(c)) => has_runtime_type(&c.tpe()),
            EvaluatedValue::Expr(EvaluatedExpr::GroupGenerator) => true,
        };
        if !has_runtime_type {
            return Err(TryExtractFromError(format!(
                "{self:?} has a type with no runtime form"
            )));
        }
        Ok(self.tpe())
    }
}

/// Whether sigmastate's `stypeToRType` converts `tpe` (v6.0.6 `Evaluation.scala:18-56`): not a
/// type variable, nor a function of other than one argument or with type parameters
fn has_runtime_type(tpe: &SType) -> bool {
    match tpe {
        SType::STypeVar(_) => false,
        SType::SFunc(f) => {
            f.t_dom.len() == 1
                && f.tpe_params.is_empty()
                && f.t_dom.iter().all(has_runtime_type)
                && has_runtime_type(&f.t_range)
        }
        SType::SColl(elem) | SType::SOption(elem) => has_runtime_type(elem),
        SType::STuple(t) => t.items.iter().all(has_runtime_type),
        _ => true,
    }
}

/// An item's data, where the item is a value itself: sigmastate casts each item to
/// `EvaluatedValue`, an `AssertionError` otherwise (`CollectionUtil.scala:188-193`)
fn item_script_data(item: &Expr) -> Result<Literal, TryExtractFromError> {
    match item {
        Expr::Const(c) => Ok(c.v.clone()),
        Expr::Tuple(t) => tuple_script_data(t.items.as_slice()),
        Expr::Collection(c) => collection_script_data(c),
        Expr::GlobalVars(GlobalVars::GroupGenerator) => {
            Ok(Literal::GroupElement(generator().into()))
        }
        other => Err(TryExtractFromError(format!("{other:?} is not a value"))),
    }
}

/// `Tuple.value` (`values.scala:818-822`): a `Coll[Any]` of the items' data. sigmastate
/// represents every tuple but a pair that way (`TupleData`, `package.scala:67`), which sigma-rust's
/// tuple data stands for, so only a pair differs: a pair is a `Tuple2` in sigmastate, and a pair
/// expression's data stays the collection, which the pair's type checks reject
/// (`Value.checkType`). A tuple of fewer than two items has no tuple data here, so it keeps the
/// collection too.
fn tuple_script_data(items: &[Expr]) -> Result<Literal, TryExtractFromError> {
    let data = items
        .iter()
        .map(item_script_data)
        .collect::<Result<Vec<_>, _>>()?;
    if data.len() <= 2 {
        return Ok(Literal::Coll(CollKind::from_collection(SType::SAny, data)?));
    }
    Ok(Literal::Tup(data.try_into().map_err(|_| {
        TryExtractFromError(format!("a tuple of {} items", items.len()))
    })?))
}

/// `ConcreteCollection.value` (`values.scala:882-885`): the items' data in an array of the
/// element type's class. For a pair that class is `Tuple2`, which a tuple expression's data, a
/// collection (`Tuple.value`), is not: an `ArrayStoreException`. Other tuples are collections
/// themselves (`TupleData`, `package.scala:67`).
fn collection_script_data(coll: &Collection) -> Result<Literal, TryExtractFromError> {
    let (elem_tpe, data) = match coll {
        Collection::BoolConstants(bools) => (
            SType::SBoolean,
            bools.iter().map(|b| Literal::Boolean(*b)).collect(),
        ),
        Collection::Exprs { elem_tpe, items } => {
            let of_pairs = matches!(elem_tpe, SType::STuple(t) if t.items.len() == 2);
            let data = items
                .iter()
                .map(|item| match item {
                    Expr::Tuple(_) if of_pairs => Err(TryExtractFromError(format!(
                        "{item:?} is not a pair's data"
                    ))),
                    _ => item_script_data(item),
                })
                .collect::<Result<Vec<_>, _>>()?;
            (elem_tpe.clone(), data)
        }
    };
    Ok(Literal::Coll(CollKind::from_collection(elem_tpe, data)?))
}

impl EvaluatedValue {
    /// sigmastate's `CheckV6Type` (rule 1019, v6.0.6 `ValidationRules.scala:165-194`), which
    /// the register and extension parsers run on each value: no `Option`, `Header` or
    /// `UnsignedBigInt` in a constant's type, a tuple item's type or a collection's element type
    pub fn check_v6_type(&self) -> Result<(), SigmaParsingError> {
        match self {
            EvaluatedValue::Constant(c) => c.tpe.check_v6_type(),
            EvaluatedValue::Expr(EvaluatedExpr::Tuple(items)) => {
                items.iter().try_for_each(|item| item.tpe().check_v6_type())
            }
            EvaluatedValue::Expr(EvaluatedExpr::Collection(c)) => c.tpe().check_v6_type(),
            EvaluatedValue::Expr(EvaluatedExpr::GroupGenerator) => Ok(()),
        }
    }

    /// The value's data as a constant, as a wallet reads it: a tuple's items as a tuple
    /// constant, a collection's items as a collection constant, the generator as a group
    /// element. Fails when an item is not a value itself, and for a tuple of fewer than two
    /// items, which has no type here. A script does not see this form: sigmastate hands it a
    /// tuple expression's data as a collection (`values.scala:818-822`).
    pub fn to_constant(&self) -> Result<Constant, TryExtractFromError> {
        match self {
            EvaluatedValue::Constant(c) => Ok(c.clone()),
            EvaluatedValue::Expr(EvaluatedExpr::Tuple(items)) => tuple_constant(items),
            EvaluatedValue::Expr(EvaluatedExpr::Collection(c)) => collection_constant(c),
            EvaluatedValue::Expr(EvaluatedExpr::GroupGenerator) => Ok(generator().into()),
        }
    }

    /// sigmastate's cast to `EvaluatedValue`, which checks only the outer node
    fn from_expr(expr: Expr) -> Result<Self, SigmaParsingError> {
        match expr {
            Expr::Const(c) => Ok(EvaluatedValue::Constant(c)),
            Expr::Tuple(t) => Ok(EvaluatedValue::Expr(EvaluatedExpr::Tuple(t.items.to_vec()))),
            Expr::Collection(c) => Ok(EvaluatedValue::Expr(EvaluatedExpr::Collection(c))),
            Expr::GlobalVars(GlobalVars::GroupGenerator) => {
                Ok(EvaluatedValue::Expr(EvaluatedExpr::GroupGenerator))
            }
            other => Err(SigmaParsingError::UnevaluatedValue(format!("{other:?}"))),
        }
    }
}

/// An item's data, where the item is a value itself
fn item_constant(item: &Expr) -> Result<Constant, TryExtractFromError> {
    match item {
        Expr::Const(c) => Ok(c.clone()),
        Expr::Tuple(t) => tuple_constant(t.items.as_slice()),
        Expr::Collection(c) => collection_constant(c),
        Expr::GlobalVars(GlobalVars::GroupGenerator) => Ok(generator().into()),
        other => Err(TryExtractFromError(format!("{other:?} is not a value"))),
    }
}

fn tuple_constant(items: &[Expr]) -> Result<Constant, TryExtractFromError> {
    let no_type = || TryExtractFromError(format!("a tuple of {} items has no type", items.len()));
    let constants = items
        .iter()
        .map(item_constant)
        .collect::<Result<Vec<_>, _>>()?;
    let tpe = STuple::try_from(constants.iter().map(|c| c.tpe.clone()).collect::<Vec<_>>())
        .map_err(|_| no_type())?;
    let v = constants
        .into_iter()
        .map(|c| c.v)
        .collect::<Vec<_>>()
        .try_into()
        .map_err(|_| no_type())?;
    Ok(Constant {
        tpe: SType::STuple(tpe),
        v: Literal::Tup(v),
    })
}

fn collection_constant(coll: &Collection) -> Result<Constant, TryExtractFromError> {
    match coll {
        Collection::BoolConstants(bools) => Ok(bools.clone().into()),
        Collection::Exprs { elem_tpe, items } => {
            let literals = items
                .iter()
                .map(|item| item_constant(item).map(|c| c.v))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Constant {
                tpe: SType::SColl(elem_tpe.clone().into()),
                v: Literal::Coll(CollKind::from_collection(elem_tpe.clone(), literals)?),
            })
        }
    }
}

/// `TupleSerializer.parse` (v6.0.6 `TupleSerializer.scala:27-36`): a signed-byte size, as
/// many values, and no arity check
fn tuple_items_sigma_parse<R: SigmaByteRead>(r: &mut R) -> Result<Vec<Expr>, SigmaParsingError> {
    let size = r.get_i8()?;
    if size < 0 {
        // `safeNewArray` throws a `NegativeArraySizeException`
        return Err(SigmaParsingError::NegativeTupleSize(size));
    }
    (0..size).map(|_| Expr::sigma_parse(r)).collect()
}

impl SigmaSerializable for EvaluatedValue {
    fn sigma_serialize<W: SigmaByteWrite>(&self, w: &mut W) -> SigmaSerializeResult {
        match self {
            EvaluatedValue::Constant(c) => c.sigma_serialize(w),
            EvaluatedValue::Expr(EvaluatedExpr::Tuple(items)) => {
                // as `Tuple`'s own serializer writes it, costs included (`mir/tuple.rs`)
                let size = u8::try_from(items.len())
                    .ok()
                    .filter(|size| *size <= i8::MAX as u8)
                    .ok_or_else(|| {
                        SigmaSerializationError::NotSupported(format!(
                            "a tuple of {} items, where its size is a signed byte",
                            items.len()
                        ))
                    })?;
                OpCode::TUPLE.sigma_serialize(w)?;
                w.put_u8(size)?;
                w.add_put_byte_cost();
                items.iter().try_for_each(|item| item.sigma_serialize(w))
            }
            EvaluatedValue::Expr(EvaluatedExpr::Collection(c)) => {
                c.op_code().sigma_serialize(w)?;
                coll_sigma_serialize(c, w)
            }
            EvaluatedValue::Expr(EvaluatedExpr::GroupGenerator) => {
                OpCode::GROUP_GENERATOR.sigma_serialize(w)
            }
        }
    }

    /// `getValue`, then the cast to `EvaluatedValue`. `getValue` is
    /// `ValueSerializer.deserialize` (v6.0.6 `ValueSerializer.scala:396-409`): one nesting
    /// level, a peek, then the value. A tuple is read here rather than as an `Expr`, since it
    /// may have 0 or 1 items.
    fn sigma_parse<R: SigmaByteRead>(r: &mut R) -> Result<Self, SigmaParsingError> {
        let depth = r.level();
        r.set_level(depth + 1)?;
        r.peek_u8()?;
        let tag = r.get_u8()?;
        let value = if tag == OpCode::TUPLE.value() {
            Ok(EvaluatedValue::Expr(EvaluatedExpr::Tuple(
                tuple_items_sigma_parse(r)?,
            )))
        } else {
            Self::from_expr(Expr::parse_tagged(r, tag)?)
        };
        r.set_level(r.level().saturating_sub(1))?;
        value
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::ergo_tree::ErgoTreeVersion;
    use crate::serialization::sigma_byte_writer::SigmaByteWriter;
    use alloc::format;
    use alloc::string::String;

    fn parse(hex: &str) -> Result<EvaluatedValue, SigmaParsingError> {
        EvaluatedValue::sigma_parse_bytes(&base16::decode(hex).unwrap())
    }

    fn constant(hex: &str) -> Constant {
        Constant::sigma_parse_bytes(&base16::decode(hex).unwrap()).unwrap()
    }

    #[test]
    fn values_the_jvm_accepts_parse_and_are_written_back_as_it_writes_them() {
        // SANTA `extension_evaluated_values` X1-X14: `TrueLeaf`, `FalseLeaf`, `GroupGenerator`;
        // `Coll[Int](1, 2)`, Boolean constants read as `83` (constant and leaf items, and the
        // empty collection), the bit-packed form; tuples of 2, 3, 1 and 0 items; then a tuple
        // and a collection that hold `HEIGHT`
        for (hex, written_back) in [
            ("7f", "0101"),
            ("80", "0100"),
            ("82", "82"),
            ("83020404020404", "83020404020404"),
            ("83020101010100", "850201"),
            ("8302017f80", "850201"),
            ("830001", "8500"),
            ("850201", "850201"),
            ("860204020404", "860204020404"),
            ("8603040204040406", "8603040204040406"),
            ("86010402", "86010402"),
            ("8600", "8600"),
            ("86020402a3", "86020402a3"),
            ("830104a3", "830104a3"),
        ] {
            let value = parse(hex).unwrap();
            assert_eq!(
                base16::encode_lower(&value.sigma_serialize_bytes().unwrap()),
                written_back,
                "{hex}"
            );
        }
    }

    #[test]
    fn values_the_jvm_rejects_do_not_parse() {
        // SANTA `extension_evaluated_values` N1-N4, N6: a placeholder with no constants, a bare
        // `HEIGHT` and `Plus(1, 2)` (the cast), a tuple holding a placeholder, and a tuple size
        // of 0x80 (-128 as a signed byte) followed by 128 items
        let n6 = format!("8680{}", "0402".repeat(128));
        assert!(matches!(
            parse("7300"),
            Err(SigmaParsingError::ConstantForPlaceholderNotFound(_))
        ));
        assert!(matches!(
            parse("a3"),
            Err(SigmaParsingError::UnevaluatedValue(_))
        ));
        assert!(matches!(
            parse("9a04020404"),
            Err(SigmaParsingError::UnevaluatedValue(_))
        ));
        assert!(matches!(
            parse("860204027300"),
            Err(SigmaParsingError::ConstantForPlaceholderNotFound(_))
        ));
        assert!(matches!(
            parse(&n6),
            Err(SigmaParsingError::NegativeTupleSize(-128))
        ));
    }

    /// `value` as a writer at `version` writes it
    fn written_at(value: &EvaluatedValue, version: ErgoTreeVersion) -> String {
        let mut data = Vec::new();
        let mut w = SigmaByteWriter::new(&mut data, None);
        w.with_tree_version(version, |w| value.sigma_serialize(w))
            .unwrap();
        base16::encode_lower(&data)
    }

    #[test]
    fn below_tree_version_3_an_upcast_of_a_constant_is_written_as_the_constant() {
        // SANTA X15, `Tuple(1, Upcast(1, Long))`; `Coll[Long](Upcast(1, Long))`;
        // `Tuple(Plus(Upcast(1, Long), 2L))`; and `Tuple(Upcast(Upcast(1.toByte, Int), Long))`,
        // where only the inner `Upcast` holds a constant (`ValueSerializer.scala:157-169`)
        for (hex, below_v3) in [
            ("860204027e040205", "860204020402"),
            ("8301057e040205", "8301050402"),
            ("86019a7e0402050504", "86019a04020504"),
            ("86017e7e02010405", "86017e020105"),
        ] {
            let value = parse(hex).unwrap();
            for version in [
                ErgoTreeVersion::V0,
                ErgoTreeVersion::V1,
                ErgoTreeVersion::V2,
            ] {
                assert_eq!(written_at(&value, version), below_v3, "{hex} {version:?}");
            }
            assert_eq!(written_at(&value, ErgoTreeVersion::V3), hex);
        }
    }

    #[test]
    fn a_tuple_of_more_than_127_items_is_not_written() {
        // its size byte would read back as negative
        let items: Vec<Expr> = core::iter::repeat_n(Expr::Const(1i32.into()), 128).collect();
        let tuple = EvaluatedValue::Expr(EvaluatedExpr::Tuple(items));
        assert!(tuple.sigma_serialize_bytes().is_err());
    }

    #[test]
    fn check_v6_type_walks_tuple_items_and_the_element_type() {
        // SANTA N5: a tuple holding `GetVar[Int](0)`, of type `Option[Int]`; an empty
        // `Coll[Option[Int]]`; then the generator, which has no type to check, and `Tuple(1, 2)`
        for hex in ["86020402e30004", "830028"] {
            assert!(
                matches!(
                    parse(hex).unwrap().check_v6_type(),
                    Err(SigmaParsingError::V6TypeError)
                ),
                "{hex}"
            );
        }
        for hex in ["82", "860204020404"] {
            assert!(parse(hex).unwrap().check_v6_type().is_ok(), "{hex}");
        }
    }

    #[test]
    fn to_constant_reads_a_value_whose_items_are_values() {
        // each value against the constant encoding of the same data: true, the pair (1, 2),
        // `Coll[Int](1, 2)`, `Coll(true, false)`, the generator, and a `Coll[(Int, Int)]` whose
        // one item is the tuple expression (1, 2)
        for (hex, constant_hex) in [
            ("7f", "0101"),
            ("860204020404", "580204"),
            ("83020404020404", "10020204"),
            ("850201", "0d0201"),
            (
                "82",
                "070279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798",
            ),
            ("830158860204020404", "0c58010204"),
        ] {
            assert_eq!(
                parse(hex).unwrap().to_constant().unwrap(),
                constant(constant_hex),
                "{hex}"
            );
        }
        // an item that is no value, and tuples of 1 and 0 items, which have no type here
        for hex in ["86020402a3", "86010402", "8600"] {
            assert!(parse(hex).unwrap().to_constant().is_err(), "{hex}");
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod script_value_tests {
    use super::*;

    fn parse(hex: &str) -> EvaluatedValue {
        EvaluatedValue::sigma_parse_bytes(&base16::decode(hex).unwrap()).unwrap()
    }

    #[test]
    fn a_value_reads_at_its_type_with_its_data() {
        // each value against the constant encoding of the same data: `TrueLeaf`, the generator,
        // `Coll[Int](1, 2)`, and a `Coll[(Int, Int)]` holding the pair constant
        for (hex, constant_hex) in [
            ("7f", "0101"),
            (
                "82",
                "070279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798",
            ),
            ("83020404020404", "10020204"),
            ("830158580204", "0c58010204"),
        ] {
            let c = Constant::sigma_parse_bytes(&base16::decode(constant_hex).unwrap()).unwrap();
            assert_eq!(
                parse(hex).to_script_value().unwrap(),
                ScriptValue {
                    tpe: Some(c.tpe),
                    v: c.v
                },
                "{hex}"
            );
        }
    }

    #[test]
    fn a_pair_tuple_expression_reads_as_a_collection() {
        // `Tuple.value` is a `Coll[Any]` of the items' data (`values.scala:818-822`) while the
        // type is a pair (SANTA V3, V9): a script reads `Tuple(1, 2)` as that collection
        let sv = parse("860204020404").to_script_value().unwrap();
        assert_eq!(
            sv.tpe,
            Some(SType::STuple(STuple::pair(SType::SInt, SType::SInt)))
        );
        assert_eq!(
            sv.v,
            Literal::Coll(
                CollKind::from_collection(SType::SAny, vec![Literal::Int(1), Literal::Int(2)])
                    .unwrap()
            )
        );
    }

    #[test]
    fn a_tuple_of_any_size_converts() {
        // `stypeToRType` builds a tuple type of any arity (`Evaluation.scala:37-48`): 1 and 0
        // items convert with no type here, and 3 items (SANTA V7) at their tuple type
        for hex in ["86010402", "8600"] {
            assert_eq!(parse(hex).to_script_value().unwrap().tpe, None, "{hex}");
        }
        assert!(matches!(
            parse("8603040204040406").to_script_value().unwrap().tpe,
            Some(SType::STuple(t)) if t.items.len() == 3
        ));
    }

    #[test]
    fn to_script_value_fails_where_the_jvm_conversion_throws() {
        // SANTA V1, `Tuple(1, HEIGHT)` (the items' cast, an `AssertionError`); C1, a
        // `Coll[(Int, Int)]` holding a tuple expression (an `ArrayStoreException`); C2, an empty
        // `Coll` of a 2-argument function (`stypeToRType`); and `Coll[Int](HEIGHT)`
        for hex in [
            "86020402a3",
            "830158860204020404",
            "8300700204040400",
            "830104a3",
        ] {
            assert!(parse(hex).to_script_value().is_err(), "{hex}");
        }
        // C2's twin, a function of one argument, converts
        assert!(parse("83007001040400").to_script_value().is_ok());
    }
}
