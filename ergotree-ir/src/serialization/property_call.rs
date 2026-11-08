use alloc::format;
use alloc::vec::Vec;
use hashbrown::HashMap;

use crate::mir::expr::Expr;
use crate::mir::property_call::PropertyCall;
use crate::types::smethod::MethodId;
use crate::types::smethod::SMethod;
use crate::types::stype::SType;
use crate::types::stype_param::STypeVar;

use super::sigma_byte_reader::SigmaByteRead;
use super::sigma_byte_writer::SigmaByteWrite;
use super::types::TypeCode;
use super::SigmaParsingError;
use super::SigmaSerializable;
use super::SigmaSerializationError;
use super::SigmaSerializeResult;

/// Reads one type per explicit type argument `method` declares, in the declaration order, as
/// sigmastate's call serializers do (v6.0.7 `PropertyCallSerializer.scala:36-43`,
/// `MethodCallSerializer.scala:58-67`)
pub(crate) fn parse_explicit_type_args<R: SigmaByteRead>(
    method: &SMethod,
    r: &mut R,
) -> Result<HashMap<STypeVar, SType>, SigmaParsingError> {
    method
        .method_raw
        .explicit_type_args
        .iter()
        .map(|var| Ok((var.clone(), SType::sigma_parse(r)?)))
        .collect()
}

/// Writes one type per explicit type argument `method` declares, in the declaration order
/// (`PropertyCallSerializer.scala:24-27`, `MethodCallSerializer.scala:29-32`). sigmastate's
/// node constructor guarantees each has a type (`values.scala:1348`); here a missing one is an
/// error
pub(crate) fn serialize_explicit_type_args<W: SigmaByteWrite>(
    method: &SMethod,
    type_args: &HashMap<STypeVar, SType>,
    w: &mut W,
) -> SigmaSerializeResult {
    for var in &method.method_raw.explicit_type_args {
        let tpe = type_args.get(var).ok_or_else(|| {
            SigmaSerializationError::NotSupported(format!(
                "{}: no type for explicit type argument {var:?}",
                method.name()
            ))
        })?;
        tpe.sigma_serialize(w)?;
    }
    Ok(())
}

/// A PropertyCall's bytes after its opcode: type code, method id, object, explicit types
/// (`PropertyCallSerializer.scala:20-28`). sigmastate writes any MethodCall without arguments
/// this way (`values.scala:1351`)
pub(crate) fn serialize_property_call_body<W: SigmaByteWrite>(
    method: &SMethod,
    obj: &Expr,
    type_args: &HashMap<STypeVar, SType>,
    w: &mut W,
) -> SigmaSerializeResult {
    method.obj_type.type_code().sigma_serialize(w)?;
    method.method_id().sigma_serialize(w)?;
    obj.sigma_serialize(w)?;
    serialize_explicit_type_args(method, type_args, w)
}

impl SigmaSerializable for PropertyCall {
    fn sigma_serialize<W: SigmaByteWrite>(&self, w: &mut W) -> SigmaSerializeResult {
        serialize_property_call_body(&self.method, &self.obj, &self.explicit_type_args, w)
    }

    fn sigma_parse<R: SigmaByteRead>(r: &mut R) -> Result<Self, SigmaParsingError> {
        let type_id = TypeCode::sigma_parse(r)?;
        let method_id = MethodId::sigma_parse(r)?;
        let obj = Expr::sigma_parse(r)?;
        let method = SMethod::from_ids(type_id, method_id)?;
        // Reject a method whose required ErgoTree version exceeds the tree's, at
        // DESERIALIZE — the JVM gates this in `SMethod.fromIds`, so it fires
        // structurally over the whole tree (a dead `if` branch is still parsed,
        // not lazily skipped). `MethodCall` already does this; `PropertyCall`
        // (e.g. `Global.none[T]`) was missing it, so a v6 property in a pre-v3
        // tree was accepted.
        if r.tree_version() < method.method_raw.min_version {
            return Err(SigmaParsingError::UnknownMethodId(
                method_id,
                type_id.value(),
            ));
        }
        // v6.0.7 `PropertyCallSerializer.scala:30-52`. A method with explicit type arguments
        // reads their types and substitutes them, and nothing else: the object's type is not
        // read. Any other is specialized over the object's type, `specializeFor`, which returns
        // the method as it is when the types do not unify (`SMethod.scala:193-199`).
        // sigmastate builds the call with no check of its own
        let (method, explicit_type_args) = if method.method_raw.explicit_type_args.is_empty() {
            let method = method
                .clone()
                .specialize_for(obj.try_tpe()?, Vec::new())
                .unwrap_or(method);
            (method, HashMap::new())
        } else {
            let explicit_type_args = parse_explicit_type_args(&method, r)?;
            (
                method.with_concrete_types(&explicit_type_args),
                explicit_type_args,
            )
        };
        Ok(PropertyCall {
            obj: obj.into(),
            method,
            explicit_type_args,
        })
    }
}

#[cfg(test)]
#[cfg(feature = "arbitrary")]
#[allow(clippy::unwrap_used)]
mod tests {
    use crate::chain::ergo_box::ErgoBox;
    use crate::ergo_tree::ErgoTree;
    use crate::ergo_tree::ErgoTreeVersion;
    use crate::mir::expr::Expr;
    use crate::mir::property_call::PropertyCall;
    use crate::serialization::roundtrip_new_feature;
    use crate::serialization::sigma_byte_reader::from_bytes;
    use crate::serialization::sigma_byte_reader::SigmaByteRead;
    use crate::serialization::sigma_serialize_roundtrip;
    use crate::serialization::SigmaParsingError;
    use crate::serialization::SigmaSerializable;
    use crate::types::scontext;
    use crate::types::sglobal;
    use crate::types::stype::SType;
    use crate::types::stype_param::STypeVar;
    use alloc::vec::Vec;
    use hashbrown::HashMap;

    fn t_is(tpe: SType) -> HashMap<STypeVar, SType> {
        core::iter::once((STypeVar::t(), tpe)).collect()
    }

    fn hex(s: &str) -> Vec<u8> {
        base16::decode(s.as_bytes()).unwrap()
    }

    /// `body` read as an expression of a tree of version 3
    fn parse_v3(body: &str) -> Result<Expr, SigmaParsingError> {
        from_bytes(hex(body)).with_tree_version(ErgoTreeVersion::V3, Expr::sigma_parse)
    }

    fn property_call(e: &Expr) -> Option<&PropertyCall> {
        match e {
            Expr::PropertyCall(pc) => Some(pc.expr()),
            _ => None,
        }
    }

    /// `tree_hex` parses as a tree (not a degraded one) and is written back unchanged: an
    /// unparsed tree round-trips its bytes too, so the bytes alone prove nothing
    fn assert_parsed_tree_round_trips(tree_hex: &str) {
        let tree = ErgoTree::sigma_parse_bytes(&hex(tree_hex)).unwrap();
        assert!(matches!(tree, ErgoTree::Parsed(_)), "{tree_hex}: {tree:?}");
        assert_eq!(
            base16::encode_lower(&tree.sigma_serialize_bytes().unwrap()),
            tree_hex
        );
    }

    #[test]
    fn ser_roundtrip_property() {
        let mc = PropertyCall::new(Expr::Context, scontext::DATA_INPUTS_PROPERTY.clone()).unwrap();
        let expr = Expr::PropertyCall(mc.into());
        assert_eq![sigma_serialize_roundtrip(&expr), expr];
    }

    #[test]
    fn versioned_roundtrip() {
        // `Global.none` is a v6 (V3) property; a pre-V3 ErgoTree must reject it
        // at parse, mirroring MethodCall's version gate. (SANTA v6 audit: the
        // dead-branch `Global.none[UnsignedBigInt]` over-acceptance,
        // V6-PROPERTY-TYPEARG-GATE-01.)
        let pc: Expr = PropertyCall::with_type_args(
            Expr::Global,
            sglobal::NONE_METHOD.clone(),
            t_is(SType::SInt),
        )
        .unwrap()
        .into();
        roundtrip_new_feature(&pc, ErgoTreeVersion::V3);
    }

    #[test]
    fn none_reads_and_writes_its_explicit_type() {
        // `Global.none[Int]`: the type follows the object (v6.0.7 `PropertyCallSerializer.scala:24`)
        let expr = parse_v3("db6a0add04").unwrap();
        let pc = property_call(&expr).unwrap();
        assert_eq!(pc.explicit_type_args, t_is(SType::SInt));
        assert_eq!(pc.tpe(), SType::SOption(SType::SInt.into()));
        assert_eq!(
            base16::encode_lower(&expr.sigma_serialize_bytes().unwrap()),
            "db6a0add04"
        );
    }

    #[test]
    fn santa_none_tree_and_box_round_trip() {
        // SANTA `vectors/wire/v6/authored/{ErgoTree,Box}.explicit_type_args.json`, `none`
        assert_parsed_tree_round_trips("1b0800d1e6db6a0add04");
        let box_hex = "c0843d1b0800d1e6db6a0add04010000cf1c7558ff524edd17b4d5ce9e80042a49ad566bf522d8089e147c29522307cc00";
        let b = ErgoBox::sigma_parse_bytes(&hex(box_hex)).unwrap();
        assert!(
            matches!(b.ergo_tree, ErgoTree::Parsed(_)),
            "{:?}",
            b.ergo_tree
        );
        assert_eq!(
            base16::encode_lower(&b.sigma_serialize_bytes().unwrap()),
            box_hex
        );
    }

    #[test]
    fn explicit_types_do_not_read_the_receivers_type() {
        // sigmaProp(Filter(1, (x: Int) => true).none[Int].isDefined): with explicit types
        // sigmastate never reads the object's type (`PropertyCallSerializer.scala:36-45`), so a
        // receiver whose type read throws (`Filter` over an `Int`) still parses
        assert_parsed_tree_round_trips("0b0fd1e6db6a0ab50402d9010104010104");
    }

    #[test]
    fn a_typed_method_written_as_a_property_reads_its_type() {
        // SELF.getReg[Int] as a PropertyCall, no argument: the type byte is read all the same
        assert_parsed_tree_round_trips("0b07d1e6db6313a704");
    }

    #[test]
    fn new_refuses_a_method_with_explicit_types() {
        assert!(PropertyCall::new(Expr::Global, sglobal::NONE_METHOD.clone()).is_err());
        let pc = PropertyCall::with_type_args(
            Expr::Global,
            sglobal::NONE_METHOD.clone(),
            t_is(SType::SInt),
        )
        .unwrap();
        assert_eq!(pc.tpe(), SType::SOption(SType::SInt.into()));
    }

    #[test]
    fn a_missing_type_byte_is_a_parse_error() {
        // the end of input, which refuses a size-flagged tree rather than degrading it, as the
        // JVM's end of input is no ValidationException either
        let err = parse_v3("db6a0add").unwrap_err();
        assert!(matches!(err, SigmaParsingError::Io(_)), "{err:?}");
        assert!(err.escapes_sized_tree_degrade());
    }

    #[test]
    fn printing_keeps_explicit_type_args() {
        let pc: Expr = PropertyCall::with_type_args(
            Expr::Global,
            sglobal::NONE_METHOD.clone(),
            t_is(SType::SInt),
        )
        .unwrap()
        .into();
        let (printed, _) = pc.pretty_print().unwrap();
        assert_eq!(
            property_call(&printed).unwrap().explicit_type_args,
            t_is(SType::SInt)
        );
    }
}
