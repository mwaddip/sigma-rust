//! The JVM's node construction at parse.
//!
//! sigmastate builds every node it parses through `DeserializationSigmaBuilder` and then the
//! node's constructor (`SigmaBuilder.scala:750-764`), right after the node's own bytes are
//! read. Building reads the types of some children, casts them and checks them, and a failed
//! check throws then, before any later byte is read: the first failure in byte order decides
//! (SANTA `tree_parse_acceptance` #46). None of these throws is a `ValidationException`, so
//! none degrades a size-flagged tree (`ErgoTreeSerializer.scala:141-205`).
//!
//! [`check_build`] runs when an opcode's parse arm returns. Checks the JVM makes in the middle
//! of a node's bytes stay in that node's parse arm, at the JVM's read position.

use crate::ergo_tree::ErgoTreeVersion;
use crate::mir::bin_op::{BinOp, BinOpKind, RelationOp};
use crate::mir::expr::Expr;
use crate::mir::upcast::Upcast;
use crate::serialization::SigmaParsingError;
use crate::types::stype::SType;
use alloc::boxed::Box;
use alloc::format;

/// The checks sigmastate makes when it builds `expr`, which the parser has just read from a
/// tree of version `tree_version`, in the JVM's order: the builder's, then the constructor's.
/// Returns the node as the JVM builds it, which below tree version 3 may wrap an operand in
/// an `Upcast`.
pub(crate) fn check_build(
    expr: Expr,
    tree_version: ErgoTreeVersion,
) -> Result<Expr, SigmaParsingError> {
    let expr = apply_builder(expr, tree_version)?;
    check_constructor(&expr)?;
    Ok(expr)
}

/// `DeserializationSigmaBuilder`'s checks (v6.0.7 `SigmaBuilder.scala:286-340`, `:674-704`,
/// `:750-764`). Only the relations and the arithmetic operations go through one; every other
/// `mk*` is a plain constructor call.
fn apply_builder(expr: Expr, tree_version: ErgoTreeVersion) -> Result<Expr, SigmaParsingError> {
    let Expr::BinOp(spanned) = expr else {
        return Ok(expr);
    };
    let op = spanned.expr;
    let op = match op.kind {
        // `equalityOp`: `applyUpcast`, then `check2(SameType)` (`:298-301`, `:686-693`)
        BinOpKind::Relation(RelationOp::Eq | RelationOp::NEq) => {
            let op = apply_upcast(op, tree_version);
            check2_same_type(&op)?;
            op
        }
        // `comparisonOp`: `check2(OnlyNumeric)` on the operands as read, `applyUpcast`, then
        // `check2(SameType)` (`:304-313`, `:696-704`)
        BinOpKind::Relation(_) => {
            check2_only_numeric(&op)?;
            let op = apply_upcast(op, tree_version);
            check2_same_type(&op)?;
            op
        }
        // `arithOp`: `applyUpcast` only (`:316-340`, `:707-712`)
        BinOpKind::Arith(_) => apply_upcast(op, tree_version),
        BinOpKind::Logical(_) | BinOpKind::Bit(_) => op,
    };
    Ok(op.into())
}

/// `applyUpcast` (`:674-683`): two numeric operands of different types are each upcast to the
/// wider (`upcastTo`, `syntax.scala:168-177`), which wraps the narrower in an `Upcast` node and
/// so rewrites the tree. A no-op from tree version 3 (`:757-763`).
fn apply_upcast(op: BinOp, tree_version: ErgoTreeVersion) -> BinOp {
    if tree_version >= ErgoTreeVersion::V3 {
        return op;
    }
    let (lt, rt) = (op.left.tpe(), op.right.tpe());
    match (numeric_type_index(&lt), numeric_type_index(&rt)) {
        (Some(li), Some(ri)) if li != ri => {
            let tmax = if li > ri { lt } else { rt };
            let upcast = |e: Box<Expr>| -> Box<Expr> {
                if e.tpe() == tmax {
                    e
                } else {
                    Box::new(
                        Upcast {
                            input: e,
                            tpe: tmax.clone(),
                        }
                        .into(),
                    )
                }
            };
            BinOp {
                kind: op.kind,
                left: upcast(op.left),
                right: upcast(op.right),
            }
        }
        _ => op,
    }
}

/// `check2(SameType)`: `left.tpe == right.tpe`, else `ConstraintFailed` (`:286-295`,
/// `:786-788`)
fn check2_same_type(op: &BinOp) -> Result<(), SigmaParsingError> {
    let (lt, rt) = (op.left.tpe(), op.right.tpe());
    if lt != rt {
        return Err(SigmaParsingError::ConstraintFailed(format!(
            "{:?} needs operands of one type, got {lt:?} and {rt:?}",
            op.kind
        )));
    }
    Ok(())
}

/// `check2(OnlyNumeric)`: both operands numeric (NoType is not), else `ConstraintFailed`
/// (`:775-784`)
fn check2_only_numeric(op: &BinOp) -> Result<(), SigmaParsingError> {
    let (lt, rt) = (op.left.tpe(), op.right.tpe());
    if !(lt.is_numeric() && rt.is_numeric()) {
        return Err(SigmaParsingError::ConstraintFailed(format!(
            "{:?} needs numeric operands, got {lt:?} and {rt:?}",
            op.kind
        )));
    }
    Ok(())
}

/// `SNumericType.numericTypeIndex`, the order `max` upcasts by (`SType.scala:412-556`)
fn numeric_type_index(tpe: &SType) -> Option<u8> {
    Some(match tpe {
        SType::SByte => 0,
        SType::SShort => 1,
        SType::SInt => 2,
        SType::SLong => 3,
        SType::SBigInt => 4,
        SType::SUnsignedBigInt => 5,
        _ => return None,
    })
}

/// `isNumTypeOrNoType` (`ast/package.scala:135`): a numeric type, or NoType
fn is_num_type_or_no_type(tpe: &SType) -> bool {
    tpe.is_numeric() || *tpe == SType::SNoType
}

/// The constructor's `require`s and eager casts (trees.scala, transformers.scala)
fn check_constructor(expr: &Expr) -> Result<(), SigmaParsingError> {
    match expr {
        // `NumericCastSerializer.parse` casts the target type with `asNumType`, a
        // `ClassCastException` for a non-numeric type (v6.0.7
        // `NumericCastSerializer.scala:20-24`), then the constructor requires a numeric input,
        // an `IllegalArgumentException` (`trees.scala:398`, `:431`). NoType is not numeric.
        Expr::Upcast(cast) => require_numeric_cast("Upcast", &cast.input, &cast.tpe),
        Expr::Downcast(cast) => require_numeric_cast("Downcast", &cast.input, &cast.tpe),
        // `Negation` and `BitInversion` require a numeric input or NoType (`trees.scala:882`,
        // `:900`), an `IllegalArgumentException`
        Expr::Negation(op) => require_num_type_or_no_type("Negation", &op.expr.input),
        Expr::BitInversion(op) => require_num_type_or_no_type("BitInversion", &op.input),
        // `OptionGet`'s `val opType` reads its type, `input.tpe.elemType`, a cast to `SOption`
        // (`transformers.scala:600-601`), a `ClassCastException`
        Expr::OptionGet(op) => require_option("OptionGet", &op.expr.input),
        // `OptionGetOrElse`'s `val opType` reads `tpe`, `input.tpe.elemType`, the same cast
        // (`transformers.scala:625-626`)
        Expr::OptionGetOrElse(op) => require_option("OptionGetOrElse", &op.expr.input),
        // `Append`'s and `Slice`'s `val tpe = input.tpe` and `ByIndex`'s
        // `val tpe = input.tpe.elemType` cast the input's type to `SCollection`, which a tuple
        // passes (`transformers.scala:62`, `:89`, `:254`; `SType.scala:838-841`), a
        // `ClassCastException`
        Expr::Append(op) => require_collection("Append", &op.expr.input),
        Expr::Slice(op) => require_collection("Slice", &op.expr.input),
        Expr::ByIndex(op) => require_collection("ByIndex", &op.expr.input),
        // `BitOp` requires both operands numeric or NoType (`trees.scala:913`), an
        // `IllegalArgumentException`
        Expr::BinOp(op) if matches!(op.expr.kind, BinOpKind::Bit(_)) => {
            let (lt, rt) = (op.expr.left.tpe(), op.expr.right.tpe());
            if !(is_num_type_or_no_type(&lt) && is_num_type_or_no_type(&rt)) {
                return Err(SigmaParsingError::BitOpOperandsNotNumeric(format!(
                    "left: {lt:?}, right: {rt:?}"
                )));
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

fn require_num_type_or_no_type(node: &str, input: &Expr) -> Result<(), SigmaParsingError> {
    let input_tpe = input.tpe();
    if !is_num_type_or_no_type(&input_tpe) {
        return Err(SigmaParsingError::NumericOperandExpected(format!(
            "{node} of an input of type {input_tpe:?}"
        )));
    }
    Ok(())
}

fn require_collection(node: &str, input: &Expr) -> Result<(), SigmaParsingError> {
    let input_tpe = input.tpe();
    if !matches!(input_tpe, SType::SColl(_) | SType::STuple(_)) {
        return Err(SigmaParsingError::CollectionInputExpected(format!(
            "{node} of an input of type {input_tpe:?}"
        )));
    }
    Ok(())
}

fn require_option(node: &str, input: &Expr) -> Result<(), SigmaParsingError> {
    let input_tpe = input.tpe();
    if !matches!(input_tpe, SType::SOption(_)) {
        return Err(SigmaParsingError::OptionInputExpected(format!(
            "{node} of an input of type {input_tpe:?}"
        )));
    }
    Ok(())
}

fn require_numeric_cast(
    node: &str,
    input: &Expr,
    target: &crate::types::stype::SType,
) -> Result<(), SigmaParsingError> {
    if !target.is_numeric() {
        return Err(SigmaParsingError::NumericCastTargetNotNumeric(format!(
            "{node} to {target:?}"
        )));
    }
    let input_tpe = input.tpe();
    if !input_tpe.is_numeric() {
        return Err(SigmaParsingError::NumericOperandExpected(format!(
            "{node} of an input of type {input_tpe:?}"
        )));
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use crate::ergo_tree::{ErgoTree, ErgoTreeError};
    use crate::serialization::{SigmaParsingError, SigmaSerializable};

    fn tree(hex: &str) -> Result<ErgoTree, SigmaParsingError> {
        ErgoTree::sigma_parse_bytes(&base16::decode(hex.as_bytes()).unwrap())
    }

    #[test]
    fn a_numeric_cast_of_a_non_numeric_input_rejects_a_size_flagged_tree() {
        // SANTA `Box.tree_parse_acceptance` #25 to #28: size-flagged v0 trees whose root is
        // Upcast(true, Long), Upcast(Coll[Int](), Long), Downcast(true, Byte),
        // Downcast(Coll[Int](), Byte). The constructor's require throws an
        // IllegalArgumentException, which does not degrade.
        for hex in [
            "08047e010105",
            "08047e100005",
            "08047d010102",
            "08047d100002",
        ] {
            assert!(
                matches!(tree(hex), Err(SigmaParsingError::NumericOperandExpected(_))),
                "{hex}: {:?}",
                tree(hex)
            );
        }
    }

    #[test]
    fn a_numeric_cast_to_a_non_numeric_type_rejects_a_size_flagged_tree() {
        // SANTA #30: Upcast(Int 1, Boolean), `asNumType` throws a ClassCastException
        assert!(matches!(
            tree("08047e040201"),
            Err(SigmaParsingError::NumericCastTargetNotNumeric(_))
        ));
    }

    #[test]
    fn a_numeric_cast_that_builds_leaves_the_root_rule_to_degrade_the_tree() {
        // SANTA #29: Upcast(Int 1, Long) builds, and its Long root fails rule 1001
        assert!(matches!(
            tree("08047e040205"),
            Ok(ErgoTree::Unparsed {
                error: ErgoTreeError::RootTpeError(_),
                ..
            })
        ));
    }

    fn roundtrips(hex: &str) -> bool {
        let bytes = base16::decode(hex.as_bytes()).unwrap();
        match ErgoTree::sigma_parse_bytes(&bytes) {
            Ok(t @ ErgoTree::Parsed(_)) => t.sigma_serialize_bytes().unwrap() == bytes,
            _ => false,
        }
    }

    #[test]
    fn a_relation_needs_operands_of_one_type_from_tree_version_3() {
        // SANTA #11: v3 EQ(Int 1, Long 1), size-flagged: ConstraintFailed, no degrade.
        // #12, #13: the same EQ at v0 parses (the Int is upcast), and v3 EQ(Int 1, Int 1).
        assert!(matches!(
            tree("0b06d19304020502"),
            Err(SigmaParsingError::ConstraintFailed(_))
        ));
        assert!(roundtrips("00d19304020502"));
        assert!(roundtrips("0b06d19304020402"));
    }

    #[test]
    fn a_comparison_needs_numeric_operands() {
        // SANTA #14, #15: GT(true, true), unsized and size-flagged; #49: v3 GT(1.toBytes,
        // Int 0); #51: v3 GT(UnsignedBigInt 5, true). #16: GT(Int 1, Int 1) parses.
        for hex in [
            "00d19101010101",
            "0806d19101010101",
            "0b09d191db040604020400",
            "0b07d1910901050101",
        ] {
            assert!(
                matches!(tree(hex), Err(SigmaParsingError::ConstraintFailed(_))),
                "{hex}: {:?}",
                tree(hex)
            );
        }
        assert!(roundtrips("00d19104020402"));
    }

    #[test]
    fn below_tree_version_3_arithmetic_takes_the_wider_operand_type() {
        // SANTA #31: v0 EQ(SizeOf(Coll[Long](Plus(Int 1, Long 2))), Int 1): the Int is
        // upcast, so the Plus is a Long item, and the Upcast of a constant is written back
        // as the constant. #32: the same at v3 is an Int item, an AssertionError.
        assert!(roundtrips("00d193b18301059a040205040402"));
        assert!(matches!(
            tree("0b0dd193b18301059a040205040402"),
            Err(SigmaParsingError::CollectionItemTypeMismatch(_))
        ));
    }

    #[test]
    fn below_tree_version_3_the_upcast_of_a_non_constant_is_written_back() {
        // v0 sigmaProp(HEIGHT > 1L): the builder wraps HEIGHT in an Upcast to Long, and the
        // serializer writes that Upcast (`ValueSerializer.scala:157-169`), so the tree comes
        // back one node longer, as the JVM writes it.
        let bytes = base16::decode(b"00d191a30502").unwrap();
        let tree = ErgoTree::sigma_parse_bytes(&bytes).unwrap();
        assert_eq!(
            base16::encode_lower(&tree.sigma_serialize_bytes().unwrap()),
            "00d1917ea3050502"
        );
        // v0 sigmaProp(1L < HEIGHT): the narrower operand on the right is upcast the same way
        let bytes = base16::decode(b"00d18f0502a3").unwrap();
        let tree = ErgoTree::sigma_parse_bytes(&bytes).unwrap();
        assert_eq!(
            base16::encode_lower(&tree.sigma_serialize_bytes().unwrap()),
            "00d18f05027ea305"
        );
    }

    #[test]
    fn a_unary_node_is_built_without_reading_its_input_type() {
        // SANTA `tree_parse_acceptance` #1: LogicalNot(Int 1); #10: OptionIsDefined(Int 1);
        // EQ(SizeOf(Int 1), Int 0). sigmastate builds each with no check of its input.
        for hex in ["00d1ef0402", "00d1e60402", "00d193b104020400"] {
            assert!(roundtrips(hex), "{hex}: {:?}", tree(hex));
        }
        // OptionIsDefined(ExtractRegisterAs(Int 1, R4, Int)), unsized and size-flagged:
        // the serializer casts the input to a box value with no check
        for hex in ["00d1e6c604020404", "0b07d1e6c604020404"] {
            assert!(roundtrips(hex), "{hex}: {:?}", tree(hex));
        }
    }

    #[test]
    fn negation_needs_a_numeric_input_or_no_type() {
        // size-flagged root Negation(true): the require throws; Negation(Int 1) builds, and
        // its Int root degrades the tree on rule 1001
        assert!(matches!(
            tree("0803f00101"),
            Err(SigmaParsingError::NumericOperandExpected(_))
        ));
        assert!(matches!(
            tree("0803f00402"),
            Ok(ErgoTree::Unparsed {
                error: ErgoTreeError::RootTpeError(_),
                ..
            })
        ));
    }

    #[test]
    fn option_get_needs_an_option_input() {
        // size-flagged root OptionGet(Int 1): the cast to SOption throws
        assert!(matches!(
            tree("0803e40402"),
            Err(SigmaParsingError::OptionInputExpected(_))
        ));
    }

    #[test]
    fn a_call_whose_types_do_not_unify_keeps_its_method() {
        // SANTA `tree_parse_acceptance` #4: sigmaProp(SBox.value on Int 1 == Long 1).
        // `specializeFor` keeps the method, typed Long, so the EQ is well-typed and parses.
        assert!(roundtrips("00d193db630104020502"));
        // #53: v3 sigmaProp(a type-9 property on Int 1 > true): the call builds, and the
        // comparison's operands are not both numeric
        assert!(matches!(
            tree("0b09d191db090104020101"),
            Err(SigmaParsingError::ConstraintFailed(_))
        ));
        // #48: v0 sigmaProp(1.toBytes > 0), size-flagged: below v3 no numeric method is
        // found by id (rule 1016), which degrades the tree before GT is built
        assert!(matches!(
            tree("0809d191db040604020400"),
            Ok(ErgoTree::Unparsed { .. })
        ));
    }

    #[test]
    fn a_call_of_the_wrong_arity_specializes_from_the_arguments_it_has() {
        // v3 sigmaProp(Coll(1).updated(0) == Coll(1)), the same call as updated(0, 5, 6), and
        // as a PropertyCall. `unifyTypeLists` zips the method's domain with the object's and
        // the arguments' types, so `updated`'s T binds to Int from the object and the EQ is
        // well-typed (v6.0.7 `sigma/ast/package.scala:17-33`)
        for hex in [
            "0b0ed193dc0c14100102010400100102",
            "0b12d193dc0c14100102030400040a040c100102",
            "0b0bd193db0c14100102100102",
        ] {
            assert!(roundtrips(hex), "{hex}: {:?}", tree(hex));
        }
    }

    #[test]
    fn below_tree_version_3_a_method_call_with_no_arguments_is_written_as_a_property_call() {
        // SANTA `tree_parse_acceptance` #44: v0 sigmaProp(CONTEXT method 11 ().isDefined)
        // parses and comes back as the PropertyCall `db 65 0b fe`, one byte shorter
        let bytes = base16::decode(b"00d1e6dc650bfe00").unwrap();
        let tree = ErgoTree::sigma_parse_bytes(&bytes).unwrap();
        assert_eq!(
            base16::encode_lower(&tree.sigma_serialize_bytes().unwrap()),
            "00d1e6db650bfe"
        );
    }

    #[test]
    fn a_collection_node_takes_a_tuple_and_checks_nothing_else() {
        // sigmaProp(EQ(SizeOf(Append((1, 2), (3, 4))), 0)), the same with Slice((1, 2), 0, 1),
        // sigmaProp(ByIndex((true, false), 0)), sigmaProp(EQ(SizeOf(Map(Int 1, (x: Int) =>
        // true)), 0)), and SANTA #34: v3 ByIndex(Coll[Boolean](true), Long 0)
        for hex in [
            "00d193b1b3860204020404860204060408 0400",
            "00d193b1b48602040204040400040204 00",
            "00d1b2860201010100040000",
            "00d193b1ad0402d901010401010400",
            "0b08d1b20d0101050000",
        ] {
            let hex = hex.replace(' ', "");
            assert!(roundtrips(&hex), "{hex}: {:?}", tree(&hex));
        }
    }

    #[test]
    fn a_collection_node_over_a_non_collection_rejects() {
        // size-flagged: ByIndex(Int 1, 0); Map(Coll[Int](), Int 1), whose mapper is no function
        assert!(matches!(
            tree("0807d1b20402040000"),
            Err(SigmaParsingError::CollectionInputExpected(_))
        ));
        assert!(matches!(
            tree("080ad193b1ad100004020400"),
            Err(SigmaParsingError::FunctionExpected(_))
        ));
    }

    #[test]
    fn option_get_or_else_casts_its_input_and_not_its_default() {
        // sigmaProp(getVar[Int](1).getOrElse(true) == 1) parses; the size-flagged
        // sigmaProp(OptionGetOrElse(Int 1, Int 0)) rejects
        assert!(roundtrips("00d193e5e3010401010402"));
        assert!(matches!(
            tree("0806d1e504020400"),
            Err(SigmaParsingError::OptionInputExpected(_))
        ));
    }

    #[test]
    fn the_sigma_combinators_check_nothing() {
        // SigmaAnd(true, SigmaProp(true)); AtLeast(true, Coll[Int]())
        for hex in ["00ea02010108d3", "009801011000"] {
            assert!(roundtrips(hex), "{hex}: {:?}", tree(hex));
        }
    }

    #[test]
    fn a_failed_build_decides_before_a_later_soft_failure() {
        // SANTA #46's order: sigmaProp(if (Upcast(x, Long) == 0L) <0xfd> else ...). The
        // Upcast is built right after its target type, before opcode 0xfd (rule 1002) is
        // read: Upcast(true, Long) rejects, Upcast(Int 1, Long) reaches 0xfd and degrades.
        assert!(matches!(
            tree("080ad195937e0101050500fd"),
            Err(SigmaParsingError::NumericOperandExpected(_))
        ));
        assert!(matches!(
            tree("080ad195937e0402050500fd"),
            Ok(ErgoTree::Unparsed { .. })
        ));
    }
}
