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
        let method = method.with_concrete_types(&explicit_type_args)?;
        let arg_types = args
            .iter()
            .map(|arg| arg.try_tpe())
            .collect::<Result<_, _>>()?;
        let obj_tpe = obj.try_tpe()?;
        let method = method.specialize_for(obj_tpe, arg_types)?;
        // the call's type is a `val` of the node sigmastate builds (`values.scala:1355-1358`):
        // a second substitution of the explicit types that can throw
        method.try_call_tpe(&explicit_type_args)?;
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
                .with_concrete_types(&[(STypeVar::t(), SType::SLong)].iter().cloned().collect())
                .unwrap(),
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

    #[test]
    fn sigmaprop_methods_parse() {
        // SANTA `ErgoTree.sigmaprop_methodcall_ids.json`: the JVM resolves SSigmaProp 1-2
        // (`propBytes`, `isProven`) at every tree version (`methods.scala:703-707`)
        for tree_hex in [
            "00d191b1db0801d101010400", // sigmaProp(sigmaProp(true).propBytes.size > 0)
            "0b0bd191b1db0801d101010400",
            "00d1db0802d10101", // sigmaProp(sigmaProp(true).isProven)
            "0b07d1db0802d10101",
        ] {
            assert_parsed_tree_round_trips(tree_hex);
        }
    }

    /// A call of a method that takes an explicit type argument, with `type_arg` as that
    /// type's bytes. `getReg` reads register 4 of a box constant with no registers, the
    /// byte arguments are one byte, `{(x: Int) => 1}` is the lambda.
    fn call(method: &str, type_arg: &str) -> alloc::string::String {
        let start = match method {
            "getReg" => "dc631363c0843d0008cd0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798000000000000000000000000000000000000000000000000000000000000000000000000010408",
            "getVarFromInput" => "dc650cfe0203000201",
            "none" => "db6a0add",
            "some(1)" => "dc6a09dd010402",
            "some(lambda)" => "dc6a09dd01d90101040402",
            "some(Coll(lambda))" => "dc6a09dd0183017001040400d90101040402",
            "some(Coll[Int]())" => "dc6a09dd011000",
            "fromBigEndianBytes" => "dc6a05dd010e0101",
            "deserializeTo" => "dc6a04dd010e0100",
            // `some` of `{(x: T) => 1}`, of `{(f: [T](Int) => Int) => 1}` and of a collection of it
            "some(lambda of T)" => "dc6a09dd01d901016701540402",
            "some(lambda of [T](Int)=>Int)" => "dc6a09dd01d9010170010404016701540402",
            "some(Coll(lambda of [T](Int)=>Int))" => {
                "dc6a09dd018301700170010404016701540400d9010170010404016701540402"
            }
            _ => "",
        };
        assert!(!start.is_empty(), "{method}");
        alloc::format!("{start}{type_arg}")
    }

    fn parse_v3(body_hex: &str) -> Result<Expr, crate::serialization::SigmaParsingError> {
        use crate::serialization::SigmaSerializable;
        from_bytes(&hex(body_hex)).with_tree_version(ErgoTreeVersion::V3, Expr::sigma_parse)
    }

    fn tpe(type_hex: &str) -> SType {
        use crate::serialization::SigmaSerializable;
        from_bytes(&hex(type_hex))
            .with_tree_version(ErgoTreeVersion::V3, SType::sigma_parse)
            .unwrap()
    }

    #[test]
    fn a_calls_type_is_the_range_with_the_type_argument_substituted_again() {
        // sigma-state 6.0.7, read from the jars: the method, its type argument, the type of
        // the call. A type argument that names `T` is substituted into itself; a function
        // type at the top of the range loses its parameter `T`; `some`'s argument can bind a
        // variable of the type argument (`specializeFor`).
        for (method, type_arg, call_tpe) in [
            ("getReg", "04", "28"),                                   // Int
            ("getReg", "670154", "24670154"),                         // T
            ("getReg", "0c670154", "240c0c670154"),                   // Coll[T]
            ("getReg", "40670154", "244040670154"),                   // (Int,T)
            ("getReg", "70010467015400", "247001047001046701540000"), // (Int)=>T
            ("getReg", "7001040401670155", "247001040401670155"),     // [U](Int)=>Int
            (
                "getReg",
                "0c70016701540401670155",
                "240c70010c700167015404016701550401670155",
            ), // Coll[[U](T)=>Int]
            ("getVarFromInput", "24670154", "242424670154"),          // Option[T]
            ("none", "0c0c670154", "240c0c0c0c670154"),               // Coll[Coll[T]]
            ("none", "70016701540400", "247001700167015404000400"),   // (T)=>Int
            ("some(1)", "0c670154", "240c0c670154"),                  // Coll[T]
            ("some(1)", "670154", "28"),                              // T
            ("some(Coll[Int]())", "0c670154", "34"),                  // Coll[T]
            ("fromBigEndianBytes", "0c670154", "0c0c670154"),         // Coll[T]
            ("fromBigEndianBytes", "7001040401670154", "7001040400"), // [T](Int)=>Int
            (
                "fromBigEndianBytes",
                "7001040402670154670155",
                "7001040401670155",
            ), // [T,U](Int)=>Int
            ("deserializeTo", "0c670154", "0c0c670154"),              // Coll[T]
            ("deserializeTo", "7001040401670154", "7001040400"),      // [T](Int)=>Int
            ("deserializeTo", "7001040401670155", "7001040401670155"), // [U](Int)=>Int
            (
                "deserializeTo",
                "70016701540401670154",
                "7001700167015404016701540400",
            ), // [T](T)=>Int
            (
                "deserializeTo",
                "70010467015401670154",
                "7001047001046701540167015400",
            ), // [T](Int)=>T
            (
                "deserializeTo",
                "700104700104040167015400",
                "700104700104040000",
            ), // (Int)=>[T](Int)=>Int
            (
                "deserializeTo",
                "700170010404016701540400",
                "700170010404000400",
            ), // ([T](Int)=>Int)=>Int
            (
                "deserializeTo",
                "700104700104040167015401670154",
                "700104700104040000",
            ), // [T](Int)=>[T](Int)=>Int
            ("some(lambda)", "70016701550400", "247001040400"),       // (U)=>Int
            ("some(lambda)", "70016701550401670156", "247001040401670156"), // [V](U)=>Int
            ("some(Coll(lambda))", "0c70016701540400", "240c7001040400"), // Coll[(T)=>Int]
            (
                "some(Coll(lambda))",
                "0c70016701540401670155",
                "240c7001040401670155",
            ), // Coll[[U](T)=>Int]
            // the `T` that `some`'s ARGUMENT brought into the range is substituted too: `U`
            // is the type argument here
            ("some(lambda of T)", "670155", "2470016701550400"), // Option[(U) => Int]
            // ... and a function type's parameter `T` there is renamed
            (
                "some(lambda of [T](Int)=>Int)",
                "670155",
                "24700170010404016701550400",
            ), // Option[([U](Int) => Int) => Int]
        ] {
            let expr = parse_v3(&call(method, type_arg)).unwrap();
            assert_eq!(expr.tpe(), tpe(call_tpe), "{method}[{type_arg}]");
        }
        // a call with no type argument has its method's range
        let index_of = MethodCall::new(
            vec![1i64, 2i64].into(),
            scoll::INDEX_OF_METHOD
                .clone()
                .with_concrete_types(&t_is(SType::SLong))
                .unwrap(),
            vec![2i64.into(), 0i32.into()],
        )
        .unwrap();
        assert_eq!(index_of.tpe(), *index_of.method.tpe().t_range.clone());
        assert_eq!(index_of.tpe(), SType::SInt);
    }

    #[test]
    fn a_rebuilt_call_keeps_its_type() {
        // sigmastate's `dup` builds the call again from the same method and the same type
        // arguments (`Rewriter.scala:241-260`): its type is computed from what it was
        let mut expr = parse_v3(&call("deserializeTo", "0c670154")).unwrap();
        crate::serialization::check_build::rebuild(&mut expr).unwrap();
        assert_eq!(expr.tpe(), tpe("0c0c670154"));
    }

    #[test]
    fn a_call_whose_type_sigmastate_cannot_build_does_not_parse() {
        // sigma-state 6.0.7, read from the jars: each throws kiama's `RuntimeException` when
        // the call is built. A function type with a type parameter named by the substitution,
        // anywhere but at the top of the range of `deserializeTo` or `fromBigEndianBytes`
        for (method, type_arg) in [
            ("getReg", "7001040401670154"),                       // [T](Int)=>Int
            ("getReg", "700104700104040167015400"),               // (Int)=>[T](Int)=>Int
            ("getVarFromInput", "70016701540401670154"),          // [T](T)=>Int
            ("none", "7001040401670154"),                         // [T](Int)=>Int
            ("none", "0c7001040401670154"),                       // Coll[[T](Int)=>Int]
            ("some(1)", "7001040402670154670155"),                // [T,U](Int)=>Int
            ("some(lambda)", "70016701540401670154"),             // [T](T)=>Int
            ("some(lambda)", "70010467015401670154"),             // [T](Int)=>T
            ("fromBigEndianBytes", "247001040401670154"),         // Option[[T](Int)=>Int]
            ("fromBigEndianBytes", "7001040c700104040167015400"), // (Int)=>Coll[[T](Int)=>Int]
            ("deserializeTo", "0c7001040401670154"),              // Coll[[T](Int)=>Int]
            ("deserializeTo", "407001040401670154"),              // (Int,[T](Int)=>Int)
            ("deserializeTo", "0c700104700104040167015400"),      // Coll[(Int)=>[T](Int)=>Int]
            // the type argument holds no type parameter; the argument's type does
            ("some(lambda of [T](Int)=>Int)", "70016701550400"), // (U)=>Int
            ("some(Coll(lambda of [T](Int)=>Int))", "0c670155"), // Coll[U]
        ] {
            assert!(
                matches!(
                    parse_v3(&call(method, type_arg)),
                    Err(crate::serialization::SigmaParsingError::TypeSubstFailed(_))
                ),
                "{method}[{type_arg}]"
            );
        }
    }

    #[test]
    fn a_call_that_specialising_cannot_build_does_not_parse() {
        // sigma-state 6.0.7, read from the jars: `specializeFor` binds a variable of the type
        // argument from `some`'s argument (`U` is `Int` here), and its substitution throws
        // kiama's `RuntimeException` on a function type inside the range whose type parameter
        // is that variable
        for (method, type_arg) in [
            ("some(lambda)", "70016701550401670155"), // [U](U)=>Int
            ("some(lambda)", "700167015567015602670155670156"), // [U,V](U)=>V
            ("some(Coll(lambda))", "0c70016701550401670155"), // Coll[[U](U)=>Int]
            ("some(Coll(lambda))", "0c70016701550402670155670156"), // Coll[[U,V](U)=>Int]
        ] {
            assert!(
                matches!(
                    parse_v3(&call(method, type_arg)),
                    Err(crate::serialization::SigmaParsingError::TypeSubstFailed(_))
                ),
                "{method}[{type_arg}]"
            );
        }
    }

    #[test]
    fn the_calls_type_decides_the_checks_of_the_node_around_it() {
        use crate::serialization::SigmaParsingError;
        let by_index = |coll: &str| alloc::format!("b2{coll}040000");
        // box.getReg[Coll[T]](4) is an Option[Coll[Coll[T]]]: `.get(0)(0)` is a `T`
        let get = alloc::format!("e4{}", call("getReg", "0c670154"));
        assert_eq!(
            parse_v3(&by_index(&by_index(&get))).unwrap().tpe(),
            tpe("670154")
        );
        // deserializeTo[Coll[T]](bytes) is a Coll[Coll[T]]: its item is no `T`, its item's is
        let coll = call("deserializeTo", "0c670154");
        let t = call("deserializeTo", "670154");
        assert!(matches!(
            parse_v3(&alloc::format!("93{}{t}", by_index(&coll))),
            Err(SigmaParsingError::ConstraintFailed(_))
        ));
        assert_eq!(
            parse_v3(&alloc::format!("93{}{t}", by_index(&by_index(&coll))))
                .unwrap()
                .tpe(),
            SType::SBoolean
        );
        // deserializeTo[[T](Int) => Int](bytes) is an Int => Int, a lambda's type
        let func = call("deserializeTo", "7001040401670154");
        assert_eq!(
            parse_v3(&alloc::format!("93{func}d90101040402"))
                .unwrap()
                .tpe(),
            SType::SBoolean
        );
        // some[U]({(x: T) => 1}) is an Option[(U) => Int]: its content is no `(T) => Int`
        let get = alloc::format!("e4{}", call("some(lambda of T)", "670155"));
        assert!(matches!(
            parse_v3(&alloc::format!("93{get}d901016701540402")),
            Err(SigmaParsingError::ConstraintFailed(_))
        ));
        assert_eq!(
            parse_v3(&alloc::format!("93{get}d901016701550402"))
                .unwrap()
                .tpe(),
            SType::SBoolean
        );
    }

    #[allow(clippy::panic)]
    fn tuple_of(tpe: &SType) -> &alloc::sync::Arc<crate::types::stuple::STuple> {
        match tpe {
            SType::STuple(tuple) => tuple,
            _ => panic!("not a tuple type: {tpe:?}"),
        }
    }

    #[test]
    fn a_calls_grown_type_is_shared() {
        // sigmastate puts ONE object at every place the second substitution replaces `T`, so
        // a call's type costs what its type argument costs however often it names `T`.
        // deserializeTo[(T, T)](bytes) is a ((T, T), (T, T)) whose two items are one
        // allocation
        let expr = parse_v3(&call("deserializeTo", "3c670154670154")).unwrap();
        let call_tpe = expr.tpe();
        assert_eq!(call_tpe, tpe("3c3c6701546701543c670154670154"));
        let items = tuple_of(&call_tpe).items.as_slice();
        assert!(alloc::sync::Arc::ptr_eq(
            tuple_of(&items[0]),
            tuple_of(&items[1])
        ));
    }

    #[test]
    fn a_call_assembled_by_hand_falls_back_to_the_range() {
        use crate::mir::property_call::PropertyCall;
        // none[[T](Int) => Int] cannot be built by the parser or by the constructor
        // (sigmastate cannot build its type); put together field by field, its type is the
        // method's range. That value is this library's choice: the JVM has no such call
        let fp = tpe("7001040401670154");
        let method = sglobal::NONE_METHOD
            .clone()
            .with_concrete_types(&t_is(fp.clone()))
            .unwrap();
        let call = PropertyCall {
            obj: Expr::Global.into(),
            method,
            explicit_type_args: t_is(fp.clone()),
        };
        assert_eq!(call.tpe(), SType::SOption(fp.into()));
    }

    #[test]
    fn a_call_built_in_memory_is_refused_where_the_parser_refuses_it() {
        use crate::mir::property_call::PropertyCall;
        // none[[T](Int) => Int] and deserializeTo[Coll[[T](Int) => Int]](bytes)
        assert!(PropertyCall::with_type_args(
            Expr::Global,
            sglobal::NONE_METHOD.clone(),
            t_is(tpe("7001040401670154")),
        )
        .is_err());
        assert!(MethodCall::with_type_args(
            Expr::Global,
            sglobal::DESERIALIZE_METHOD.clone(),
            vec![Constant::from(vec![0i8]).into()],
            t_is(tpe("0c7001040401670154")),
        )
        .is_err());
        // the control: deserializeTo[[T](Int) => Int](bytes), whose type is Int => Int
        let call = MethodCall::with_type_args(
            Expr::Global,
            sglobal::DESERIALIZE_METHOD.clone(),
            vec![Constant::from(vec![0i8]).into()],
            t_is(tpe("7001040401670154")),
        )
        .unwrap();
        assert_eq!(call.tpe(), tpe("7001040400"));
    }
}
