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
