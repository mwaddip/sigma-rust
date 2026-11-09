use crate::ergo_tree::ErgoTreeVersion;

use alloc::vec::Vec;

use crate::mir::expr::Expr;
use crate::mir::method_call::MethodCall;
use crate::types::smethod::MethodId;
use crate::types::smethod::SMethod;

use super::property_call::{parse_explicit_type_args, serialize_explicit_type_args};
use super::sigma_byte_reader::SigmaByteRead;
use super::sigma_byte_writer::SigmaByteWrite;
use super::SigmaParsingError;
use super::SigmaSerializable;
use super::SigmaSerializeResult;

impl SigmaSerializable for MethodCall {
    fn sigma_serialize<W: SigmaByteWrite>(&self, w: &mut W) -> SigmaSerializeResult {
        self.method.obj_type.type_code().sigma_serialize(w)?;
        self.method.method_id().sigma_serialize(w)?;
        self.obj.sigma_serialize(w)?;
        self.args.sigma_serialize(w)?;
        serialize_explicit_type_args(&self.method, &self.explicit_type_args, w)
    }

    fn sigma_parse<R: SigmaByteRead>(r: &mut R) -> Result<Self, SigmaParsingError> {
        // the type id is a raw byte, looked up with the method once the call is read (v6.0.7
        // `MethodCallSerializer.scala:47-56`)
        let type_id = r.get_u8()?;
        let method_id = MethodId::sigma_parse(r)?;
        let obj = Expr::sigma_parse(r)?;
        let args = Vec::<Expr>::sigma_parse(r)?;
        // From tree version 3 sigmastate asserts there are arguments, before the method lookup
        // (v6.0.7 `MethodCallSerializer.scala:52-55`)
        if r.tree_version() >= ErgoTreeVersion::V3 && args.is_empty() {
            return Err(SigmaParsingError::MethodCallWithoutArguments);
        }
        let method = SMethod::from_type_byte(type_id, method_id)?;
        if r.tree_version() < method.method_raw.min_version {
            return Err(SigmaParsingError::UnknownMethodId(method_id, type_id));
        }
        let explicit_type_args = parse_explicit_type_args(&method, r)?;
        // `getSpecializedMethodFor`: the explicit types, then `specializeFor` over the
        // arguments' types and the object's, which returns the method as it is when they do
        // not unify (v6.0.7 `MethodCallSerializer.scala:77-97`, `SMethod.scala:185-191`).
        // sigmastate builds the call with no arity or type check of its own.
        // The arguments' types are read first, then the object's (`:84-96`), where a read can
        // throw ([`Expr::try_tpe`])
        let method = method.with_concrete_types(&explicit_type_args);
        let arg_types = args
            .iter()
            .map(|arg| arg.try_tpe())
            .collect::<Result<_, _>>()?;
        let obj_tpe = obj.try_tpe()?;
        let method = method
            .clone()
            .specialize_for(obj_tpe, arg_types)
            .unwrap_or(method);
        Ok(MethodCall {
            obj: obj.into(),
            method,
            args,
            explicit_type_args,
        })
    }
}

#[cfg(test)]
#[cfg(feature = "arbitrary")]
#[allow(clippy::unwrap_used)]
mod tests {
    use alloc::vec;

    use crate::chain::ergo_box::ErgoBox;
    use crate::ergo_tree::ErgoTreeVersion;
    use crate::mir::constant::Constant;
    use crate::mir::expr::Expr;
    use crate::mir::method_call::MethodCall;
    use crate::serialization::sigma_byte_reader::{from_bytes, SigmaByteRead};
    use crate::serialization::{roundtrip_new_feature, sigma_serialize_roundtrip};
    use crate::types::stype::SType;
    use crate::types::stype_param::STypeVar;
    use crate::types::{scoll, sglobal};
    use hashbrown::HashMap;

    fn t_is(tpe: SType) -> HashMap<STypeVar, SType> {
        core::iter::once((STypeVar::t(), tpe)).collect()
    }

    fn hex(s: &str) -> alloc::vec::Vec<u8> {
        base16::decode(s.as_bytes()).unwrap()
    }

    fn method_call(e: &Expr) -> Option<&MethodCall> {
        match e {
            Expr::MethodCall(mc) => Some(mc.expr()),
            _ => None,
        }
    }

    /// `tree_hex` parses as a tree (not a degraded one) and is written back unchanged: an
    /// unparsed tree round-trips its bytes too, so the bytes alone prove nothing
    fn assert_parsed_tree_round_trips(tree_hex: &str) {
        use crate::ergo_tree::ErgoTree;
        use crate::serialization::SigmaSerializable;
        let tree = ErgoTree::sigma_parse_bytes(&hex(tree_hex)).unwrap();
        assert!(matches!(tree, ErgoTree::Parsed(_)), "{tree_hex}: {tree:?}");
        assert_eq!(
            base16::encode_lower(&tree.sigma_serialize_bytes().unwrap()),
            tree_hex
        );
    }

    #[test]
    fn with_type_args_builds_some_from_its_generic_method() {
        use crate::serialization::SigmaSerializable;
        // `some`'s type variable is in its domain, so the arguments are checked against the
        // domain with the explicit types substituted
        let mc = MethodCall::with_type_args(
            Expr::Global,
            sglobal::SOME_METHOD.clone(),
            vec![5i32.into()],
            t_is(SType::SInt),
        )
        .unwrap();
        assert_eq!(mc.tpe(), SType::SOption(SType::SInt.into()));
        let expr: Expr = mc.into();
        assert_eq!(
            base16::encode_lower(&expr.sigma_serialize_bytes().unwrap()),
            "dc6a09dd01040a04"
        );
    }

    #[test]
    fn a_method_call_without_arguments_is_written_as_a_property_call() {
        use crate::mir::property_call::PropertyCall;
        use crate::serialization::SigmaSerializable;
        use crate::types::snumeric;
        // sigmastate's MethodCall is written with PropertyCall's opcode and layout when it has
        // no arguments (v6.0.7 `values.scala:1351`); from tree version 3 a MethodCall without
        // arguments is refused (`MethodCallSerializer.scala:52-55`)
        let none: Expr = MethodCall::with_type_args(
            Expr::Global,
            sglobal::NONE_METHOD.clone(),
            vec![],
            t_is(SType::SInt),
        )
        .unwrap()
        .into();
        assert_eq!(
            base16::encode_lower(&none.sigma_serialize_bytes().unwrap()),
            "db6a0add04"
        );
        let to_bytes = snumeric::sint::METHODS
            .iter()
            .find(|m| m.method_id() == snumeric::TO_BYTES_METHOD_ID)
            .unwrap()
            .clone();
        let obj = Expr::Const(1i32.into());
        let mc: Expr = MethodCall::new(obj.clone(), to_bytes.clone(), vec![])
            .unwrap()
            .into();
        let pc: Expr = PropertyCall::new(obj, to_bytes).unwrap().into();
        assert_eq!(
            mc.sigma_serialize_bytes().unwrap(),
            pc.sigma_serialize_bytes().unwrap()
        );
    }

    #[test]
    fn a_missing_explicit_type_is_an_error() {
        use crate::mir::property_call::PropertyCall;
        use crate::serialization::SigmaSerializable;
        use alloc::boxed::Box;
        // the fields are public, so a literal can skip the constructors' checks
        let mc: Expr = MethodCall {
            obj: Box::new(Expr::Global),
            method: sglobal::SOME_METHOD.clone(),
            args: vec![5i32.into()],
            explicit_type_args: HashMap::new(),
        }
        .into();
        assert!(mc.sigma_serialize_bytes().is_err());
        let pc: Expr = PropertyCall {
            obj: Box::new(Expr::Global),
            method: sglobal::NONE_METHOD.clone(),
            explicit_type_args: HashMap::new(),
        }
        .into();
        assert!(pc.sigma_serialize_bytes().is_err());
    }

    #[test]
    fn the_other_four_explicit_type_methods_round_trip_unchanged() {
        // SANTA `vectors/wire/v6/authored/ErgoTree.explicit_type_args.json`: deserializeTo (a
        // type byte mid-body, before the EQ's second operand), fromBigEndianBytes, getReg,
        // getVarFromInput (one type after two arguments)
        for tree_hex in [
            "1b15020e0401020304040ed193dc6a04dd017300047301",
            "1b15020e0401020304040ed193dc6a05dd017300047301",
            "1b0d010408d1e6dc6313a701730004",
            "1b110203000201d1e6dc650cfe027300730104",
        ] {
            assert_parsed_tree_round_trips(tree_hex);
        }
    }

    #[test]
    fn some_reads_and_writes_its_explicit_type() {
        use crate::ergo_tree::ErgoTree;
        use crate::serialization::SigmaSerializable;
        // SANTA `vectors/wire/v6/authored/{ErgoTree,Box}.explicit_type_args.json`, `some`: the
        // type follows the arguments (v6.0.7 `MethodCallSerializer.scala:29-32`)
        assert_parsed_tree_round_trips("1b0d01040ad1e6dc6a09dd01730004");
        let box_hex = "c0843d1b0d01040ad1e6dc6a09dd01730004010000a66b4e70ff8bf3d9dc7c600e4980ededaaeeaadfba7e47e84ec865211f4ae2a000";
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
    fn an_explicit_type_wins_over_an_argument_of_another_type() {
        use crate::serialization::SigmaSerializable;
        // Global.some[Int](5L): the explicit type is substituted first, then `specializeFor`
        // keeps the method when Long does not unify with Int (`MethodCallSerializer.scala:77-97`)
        let expr = from_bytes(hex("dc6a09dd01050a04"))
            .with_tree_version(ErgoTreeVersion::V3, Expr::sigma_parse)
            .unwrap();
        let mc = method_call(&expr).unwrap();
        assert_eq!(mc.tpe(), SType::SOption(SType::SInt.into()));
        assert_eq!(mc.explicit_type_args, t_is(SType::SInt));
        assert_eq!(
            base16::encode_lower(&expr.sigma_serialize_bytes().unwrap()),
            "dc6a09dd01050a04"
        );
    }

    #[test]
    fn from_tree_version_3_a_method_call_needs_arguments() {
        use crate::ergo_tree::ErgoTree;
        use crate::serialization::{SigmaParsingError, SigmaSerializable};
        // SANTA `tree_parse_acceptance` #43: v3 sigmaProp(CONTEXT method 11 ().isDefined), an
        // AssertionError; #45: the same call with its argument Byte 0 parses
        let tree =
            |hex: &str| ErgoTree::sigma_parse_bytes(&base16::decode(hex.as_bytes()).unwrap());
        assert!(matches!(
            tree("0b07d1e6dc650bfe00"),
            Err(SigmaParsingError::MethodCallWithoutArguments)
        ));
        assert!(matches!(
            tree("0b09d1e6dc650bfe010200"),
            Ok(ErgoTree::Parsed(_))
        ));
    }

    #[test]
    fn ser_roundtrip() {
        let mc: Expr = MethodCall::new(
            vec![1i64, 2i64].into(),
            scoll::INDEX_OF_METHOD
                .clone()
                .with_concrete_types(&[(STypeVar::t(), SType::SLong)].iter().cloned().collect()),
            vec![2i64.into(), 0i32.into()],
        )
        .unwrap()
        .into();
        assert_eq![sigma_serialize_roundtrip(&mc), mc];
    }
    // test that methodcalls that are added in later versions via soft-fork can't be parsed with older version
    #[test]
    fn versioned_roundtrip() {
        let mc: Expr = MethodCall::new(
            Expr::Global,
            sglobal::SERIALIZE_METHOD
                .clone()
                .specialize_for(SType::SGlobal, vec![SType::SInt])
                .unwrap(),
            vec![Constant::from(1i32).into()],
        )
        .unwrap()
        .into();
        roundtrip_new_feature(&mc, ErgoTreeVersion::V3);
    }

    #[test]
    fn the_type_id_is_resolved_after_the_call_is_read() {
        // sigmastate reads the type id as a raw byte and looks it up in `SMethod.fromIds`
        // (rule 1010) only after the method id, the object, the arguments and the v3 args
        // assert (v6.0.7 `MethodCallSerializer.scala:47-56`). A failure on the way is no
        // `ValidationException` and rejects the tree; only a complete call degrades it.
        use crate::ergo_tree::{ErgoTree, ErgoTreeError};
        use crate::serialization::{SigmaParsingError, SigmaSerializable};
        let parse = |tree_hex: &str| ErgoTree::sigma_parse_bytes(&hex(tree_hex));
        // the input ends at the method id
        let parsed = parse("0902dcff");
        assert!(
            matches!(parsed, Err(SigmaParsingError::Io(_))),
            "{parsed:?}"
        );
        // v3, no arguments
        assert_eq!(
            parse("0b05dcff017f00"),
            Err(SigmaParsingError::MethodCallWithoutArguments)
        );
        // the object is a constant of type `Any`, which has no data serializer
        assert_eq!(
            parse("0905dcff016100"),
            Err(SigmaParsingError::NonSerializableTypeCode(0x61))
        );
        // v1, the call complete: type id 0xff has no methods, rule 1010
        let parsed = parse("0905dcff017f00");
        assert!(
            matches!(
                parsed,
                Ok(ErgoTree::Unparsed {
                    error: ErgoTreeError::SigmaParsingError(SigmaParsingError::NotImplementedYet(
                        _
                    )),
                    ..
                })
            ),
            "{parsed:?}"
        );
    }

    /// `expr_hex` parses as an expression from tree version `since`, is written back as read,
    /// and is refused below `since`
    fn assert_expr_parses_from(expr_hex: &str, since: ErgoTreeVersion) {
        use crate::serialization::SigmaSerializable;
        let bytes = hex(expr_hex);
        let expr = from_bytes(&bytes)
            .with_tree_version(since, Expr::sigma_parse)
            .map_err(|e| alloc::format!("{expr_hex}: {e:?}"))
            .unwrap();
        assert_eq!(expr.sigma_serialize_bytes().unwrap(), bytes, "{expr_hex}");
        roundtrip_new_feature(&expr, since);
    }

    #[test]
    fn scollection_v5_ids_parse_at_every_version() {
        // SANTA `ErgoTree.scollection_methodcall_v5_ids.json`, the v0 unsized bodies: the JVM
        // resolves ids 1-10 at every tree version (`methods.scala:821-945`)
        for expr_hex in [
            "db0c0183020404020404",                             // size
            "dc0c028302040402040402040a0412",                   // getOrElse
            "dc0c038302040402040401d90101049a72010402",         // map
            "dc0c048302040402040401d90101049172010402",         // exists
            "dc0c0583020404020404020400d902010402049a72017202", // fold
            "dc0c068302040402040401d90101049172010400",         // forall
            "dc0c078303040402040404060204020406",               // slice
            "dc0c088302040402040401d90101049172010402",         // filter
            "dc0c098301040402018301040404",                     // append
            "dc0c0a83020404020404010402",                       // apply
        ] {
            assert_expr_parses_from(expr_hex, ErgoTreeVersion::V0);
        }
    }

    #[test]
    fn scollection_size_tree_parses() {
        // sigmaProp(Coll(1, 2).size == 2), v0 unsized and v3 sized
        assert_parsed_tree_round_trips("00d193db0c01830204040204040404");
        assert_parsed_tree_round_trips("0b0ed193db0c01830204040204040404");
    }

    #[test]
    fn to_big_int_parses_from_v3() {
        // numeric id 5 is parser-visible from tree version 3 only, as ids 1-4 and 6-13
        use crate::ergo_tree::{ErgoTree, ErgoTreeError};
        use crate::serialization::{SigmaParsingError, SigmaSerializable};
        use crate::types::smethod::MethodId;
        assert_expr_parses_from("db04050402", ErgoTreeVersion::V3);
        // sigmaProp(1.toBigInt == 1.toBigInt)
        assert_parsed_tree_round_trips("0b0cd193db04050402db04050402");
        let v2 = ErgoTree::sigma_parse_bytes(&hex("0a0cd193db04050402db04050402"));
        assert!(
            matches!(
                v2,
                Ok(ErgoTree::Unparsed {
                    error: ErgoTreeError::SigmaParsingError(SigmaParsingError::UnknownMethodId(
                        MethodId(5),
                        4
                    )),
                    ..
                })
            ),
            "{v2:?}"
        );
    }

    #[test]
    fn v5_ids_missing_in_rust_parse() {
        // SANTA `ErgoTree.v5_methods_missing_in_rust.json`, v0 unsized and v3 sized: the JVM
        // resolves GroupElement 3-4, Box 9-18 and Option 2-4 at every tree version
        for tree_hex in [
            "00cddc0703db6a01dd01060101", // proveDlog(g.exp(BigInt 1))
            "0b0ccddc0703db6a01dd01060101",
            "00cddc0704db6a01dd01db6a01dd", // proveDlog(g.multiply(g))
            "0b0dcddc0704db6a01dd01db6a01dd",
            "00d1e6db6309a7", // sigmaProp(SELF.R0.isDefined)
            "0b06d1e6db6309a7",
            "00d1e6db630da7", // sigmaProp(SELF.R4.isDefined)
            "0b06d1e6db630da7",
            "00d1db2402e30104", // sigmaProp(getVar[Int](1).isDefined)
            "0b07d1db2402e30104",
            "00d1db2403e30101", // sigmaProp(getVar[Boolean](1).get)
            "0b07d1db2403e30101",
            "00d1dc2404e30101010101", // sigmaProp(getVar[Boolean](1).getOrElse(true))
            "0b0ad1dc2404e30101010101",
        ] {
            assert_parsed_tree_round_trips(tree_hex);
        }
        // SELF.R0 ... SELF.R9
        for id in 0x09u8..=0x12 {
            assert_expr_parses_from(&alloc::format!("db63{id:02x}a7"), ErgoTreeVersion::V0);
        }
    }
}
