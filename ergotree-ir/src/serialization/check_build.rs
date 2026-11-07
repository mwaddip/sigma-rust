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
use crate::mir::expr::Expr;
use crate::serialization::SigmaParsingError;
use alloc::format;

/// The checks sigmastate makes when it builds `expr`, which the parser has just read from a
/// tree of version `tree_version`, in the JVM's order. Returns the node as the JVM builds it.
pub(crate) fn check_build(
    expr: Expr,
    _tree_version: ErgoTreeVersion,
) -> Result<Expr, SigmaParsingError> {
    check_constructor(&expr)?;
    Ok(expr)
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
