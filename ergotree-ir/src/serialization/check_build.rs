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

/// sigmastate rebuilding a node one of whose children the substitution at spend replaced:
/// `dup` calls the node's constructor, not the builder (v6.0.7 `Rewriter.scala:241-260`), so
/// its `val`s are computed afresh and its checks run again. Recomputes the types sigma-rust
/// caches from the children, then makes the constructor's checks.
pub(crate) fn rebuild(expr: &mut Expr) -> Result<(), SigmaParsingError> {
    use crate::mir::coll_exists::parsed_elem_tpe;
    match expr {
        Expr::Exists(e) => e.expr.elem_tpe = parsed_elem_tpe(&e.expr.input),
        Expr::ForAll(e) => e.expr.elem_tpe = parsed_elem_tpe(&e.expr.input),
        Expr::Filter(e) => e.expr.elem_tpe = parsed_elem_tpe(&e.expr.input),
        // `MapCollection`'s `val tpe` casts the mapper's type to `SFunc` (`transformers.scala:38`)
        Expr::Map(m) => {
            m.expr.mapper_sfunc = match m.expr.mapper.try_tpe()? {
                SType::SFunc(sfunc) => (*sfunc).clone(),
                tpe => {
                    return Err(SigmaParsingError::FunctionExpected(format!(
                        "a Map mapper of type {tpe:?}"
                    )))
                }
            }
        }
        Expr::SelectField(s) => s.expr.rebuild()?,
        Expr::FuncValue(f) => f.rebuild(),
        _ => (),
    }
    check_constructor(expr)
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
            let op = apply_upcast(op, tree_version)?;
            check2_same_type(&op)?;
            op
        }
        // `comparisonOp`: `check2(OnlyNumeric)` on the operands as read, `applyUpcast`, then
        // `check2(SameType)` (`:304-313`, `:696-704`)
        BinOpKind::Relation(_) => {
            check2_only_numeric(&op)?;
            let op = apply_upcast(op, tree_version)?;
            check2_same_type(&op)?;
            op
        }
        // `arithOp`: `applyUpcast` only (`:316-340`, `:707-712`)
        BinOpKind::Arith(_) => apply_upcast(op, tree_version)?,
        BinOpKind::Logical(_) | BinOpKind::Bit(_) => op,
    };
    Ok(op.into())
}

/// `applyUpcast` (`:674-683`): two numeric operands of different types are each upcast to the
/// wider (`upcastTo`, `syntax.scala:168-177`), which wraps the narrower in an `Upcast` node and
/// so rewrites the tree. A no-op from tree version 3 (`:757-763`).
fn apply_upcast(op: BinOp, tree_version: ErgoTreeVersion) -> Result<BinOp, SigmaParsingError> {
    if tree_version >= ErgoTreeVersion::V3 {
        return Ok(op);
    }
    let (lt, rt) = (op.left.try_tpe()?, op.right.try_tpe()?);
    Ok(match (numeric_type_index(&lt), numeric_type_index(&rt)) {
        (Some(li), Some(ri)) if li != ri => {
            let tmax = if li > ri { lt.clone() } else { rt.clone() };
            let upcast = |e: Box<Expr>, tpe: SType| -> Box<Expr> {
                if tpe == tmax {
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
                left: upcast(op.left, lt),
                right: upcast(op.right, rt),
            }
        }
        _ => op,
    })
}

/// `check2(SameType)`: `left.tpe == right.tpe`, else `ConstraintFailed` (`:286-295`,
/// `:786-788`)
fn check2_same_type(op: &BinOp) -> Result<(), SigmaParsingError> {
    let (lt, rt) = (op.left.try_tpe()?, op.right.try_tpe()?);
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
    let (lt, rt) = (op.left.try_tpe()?, op.right.try_tpe()?);
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
            let (lt, rt) = (op.expr.left.try_tpe()?, op.expr.right.try_tpe()?);
            if !(is_num_type_or_no_type(&lt) && is_num_type_or_no_type(&rt)) {
                return Err(SigmaParsingError::BitOpOperandsNotNumeric(format!(
                    "left: {lt:?}, right: {rt:?}"
                )));
            }
            Ok(())
        }
        // Constructors that only read: a read that throws fails the build, nothing else does.
        // `ArithOp`'s `val opType` reads both operands' types (`trees.scala:707-708`)
        Expr::BinOp(op) if matches!(op.expr.kind, BinOpKind::Arith(_)) => {
            op.expr.left.try_tpe()?;
            op.expr.right.try_tpe()?;
            Ok(())
        }
        // `If`'s and `TreeLookup`'s `Quadruple.opType` reads all three children's types
        // (`trees.scala:1313`)
        Expr::If(op) => {
            op.condition.try_tpe()?;
            op.true_branch.try_tpe()?;
            op.false_branch.try_tpe()?;
            Ok(())
        }
        Expr::TreeLookup(op) => {
            op.expr.tree.try_tpe()?;
            op.expr.key.try_tpe()?;
            op.expr.proof.try_tpe()?;
            Ok(())
        }
        // `OptionIsDefined`'s and `SigmaPropBytes`' `val opType` reads the input's type
        // (`transformers.scala:656`, `:336`)
        Expr::OptionIsDefined(op) => op.expr.input.try_tpe().map(|_| ()),
        Expr::SigmaPropBytes(op) => op.input.try_tpe().map(|_| ()),
        // `SigmaPropIsProven`'s `val opType` reads its input's type (`transformers.scala:321-325`)
        Expr::SigmaPropIsProven(op) => op.input.try_tpe().map(|_| ()),
        _ => Ok(()),
    }
}

fn require_num_type_or_no_type(node: &str, input: &Expr) -> Result<(), SigmaParsingError> {
    let input_tpe = input.try_tpe()?;
    if !is_num_type_or_no_type(&input_tpe) {
        return Err(SigmaParsingError::NumericOperandExpected(format!(
            "{node} of an input of type {input_tpe:?}"
        )));
    }
    Ok(())
}

fn require_collection(node: &str, input: &Expr) -> Result<(), SigmaParsingError> {
    let input_tpe = input.try_tpe()?;
    if !matches!(input_tpe, SType::SColl(_) | SType::STuple(_)) {
        return Err(SigmaParsingError::CollectionInputExpected(format!(
            "{node} of an input of type {input_tpe:?}"
        )));
    }
    Ok(())
}

fn require_option(node: &str, input: &Expr) -> Result<(), SigmaParsingError> {
    let input_tpe = input.try_tpe()?;
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
    let input_tpe = input.try_tpe()?;
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
        ErgoTree::sigma_parse_bytes_no_fallback(&base16::decode(hex.as_bytes()).unwrap())
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
    fn a_type_read_that_throws_fails_the_build() {
        // Where sigmastate reads a child's type while it builds the node, a child whose read
        // throws (a `Filter` over a non-collection, v6.0.7 `transformers.scala:121`) fails the
        // build with a `ClassCastException`: the builder's `check2` and `applyUpcast`
        // (`SigmaBuilder.scala:286-340`), every constructor of §1a, the read-only ones too
        // (`ArithOp`'s and `If`'s `opType`, `trees.scala:707-708`, `:1313`)
        use crate::ergo_tree::ErgoTreeVersion;
        const V0: ErgoTreeVersion = ErgoTreeVersion::V0;
        const V3: ErgoTreeVersion = ErgoTreeVersion::V3;
        use crate::mir::bin_op::{ArithOp, BinOp, BitOp, RelationOp};
        use crate::mir::bit_inversion::BitInversion;
        use crate::mir::coll_by_index::ByIndex;
        use crate::mir::expr::type_read_tests::{bi, filter};
        use crate::mir::expr::Expr;
        use crate::mir::if_op::If;
        use crate::mir::negation::Negation;
        use crate::mir::option_get::OptionGet;
        use crate::mir::option_is_defined::OptionIsDefined;
        use crate::mir::sigma_prop_bytes::SigmaPropBytes;
        use crate::mir::tree_lookup::TreeLookup;
        use crate::mir::upcast::Upcast;
        use crate::types::stype::SType;
        use alloc::boxed::Box;
        let f = || Box::new(filter(bi()));
        let bin = |kind: crate::mir::bin_op::BinOpKind| -> Expr {
            BinOp {
                kind,
                left: f(),
                right: f(),
            }
            .into()
        };
        let cases: [(&str, Expr, ErgoTreeVersion); 15] = [
            (
                "If",
                If {
                    condition: Box::new(Expr::Const(true.into())),
                    true_branch: f(),
                    false_branch: f(),
                }
                .into(),
                V3,
            ),
            (
                "TreeLookup",
                TreeLookup {
                    tree: f(),
                    key: f(),
                    proof: f(),
                }
                .into(),
                V3,
            ),
            ("EQ v0", bin(RelationOp::Eq.into()), V0),
            ("EQ v3", bin(RelationOp::Eq.into()), V3),
            ("GT", bin(RelationOp::Gt.into()), V3),
            ("Plus v0", bin(ArithOp::Plus.into()), V0),
            ("Plus v3", bin(ArithOp::Plus.into()), V3),
            ("BitOr", bin(BitOp::BitOr.into()), V3),
            ("Negation", Negation { input: f() }.into(), V3),
            ("BitInversion", BitInversion { input: f() }.into(), V3),
            (
                "Upcast",
                Upcast {
                    input: f(),
                    tpe: SType::SLong,
                }
                .into(),
                V3,
            ),
            ("OptionGet", OptionGet { input: f() }.into(), V3),
            (
                "ByIndex",
                ByIndex {
                    input: f(),
                    index: Box::new(0i32.into()),
                    default: None,
                }
                .into(),
                V3,
            ),
            ("OptionIsDefined", OptionIsDefined { input: f() }.into(), V3),
            ("SigmaPropBytes", SigmaPropBytes { input: f() }.into(), V3),
        ];
        for (name, expr, version) in cases {
            assert!(
                matches!(
                    super::check_build(expr, version),
                    Err(SigmaParsingError::TypeReadFailed(_))
                ),
                "{name}"
            );
        }
    }

    /// A size-flagged v3 tree of `root`
    fn sized_v3(body: &[u8]) -> alloc::vec::Vec<u8> {
        assert!(body.len() < 128);
        let mut bytes = alloc::vec![0x0b, body.len() as u8];
        bytes.extend_from_slice(body);
        bytes
    }

    #[test]
    fn a_filter_is_built_without_reading_its_input_type() {
        // sigmastate's `Filter` checks nothing when built (v6.0.7 `transformers.scala:117-121`):
        // sigmaProp(SizeOf(F(BI)) > 0) parses and round-trips, F(BI) = Filter(ByIndex(Tuple(0,
        // 0), 0), (i: Int) => true) over an input typed SAny. Reading F(BI)'s type throws, so
        // sigmaProp(EQ(If(true, F(BI), F(BI)), F(BI))) rejects, size-flagged too: `If`'s
        // constructor reads it (`trees.scala:1313`), a `ClassCastException`
        use crate::mir::bin_op::{BinOp, RelationOp};
        use crate::mir::bool_to_sigma::BoolToSigmaProp;
        use crate::mir::coll_size::SizeOf;
        use crate::mir::expr::type_read_tests::{bi, filter};
        use crate::mir::expr::Expr;
        use crate::mir::if_op::If;
        use crate::serialization::SigmaSerializable;
        use alloc::boxed::Box;
        let root = |e: Expr| -> alloc::vec::Vec<u8> {
            sized_v3(
                &Expr::from(BoolToSigmaProp { input: Box::new(e) })
                    .sigma_serialize_bytes()
                    .unwrap(),
            )
        };
        let gt: Expr = BinOp {
            kind: RelationOp::Gt.into(),
            left: Box::new(
                SizeOf {
                    input: Box::new(filter(bi())),
                }
                .into(),
            ),
            right: Box::new(0i32.into()),
        }
        .into();
        let bytes = root(gt);
        let parsed = ErgoTree::sigma_parse_bytes(&bytes).unwrap();
        assert!(matches!(parsed, ErgoTree::Parsed(_)));
        assert_eq!(parsed.sigma_serialize_bytes().unwrap(), bytes);
        let eq: Expr = BinOp {
            kind: RelationOp::Eq.into(),
            left: Box::new(
                If {
                    condition: Box::new(Expr::Const(true.into())),
                    true_branch: Box::new(filter(bi())),
                    false_branch: Box::new(filter(bi())),
                }
                .into(),
            ),
            right: Box::new(filter(bi())),
        }
        .into();
        assert!(matches!(
            ErgoTree::sigma_parse_bytes(&root(eq)),
            Err(SigmaParsingError::TypeReadFailed(_))
        ));
    }

    #[test]
    fn a_type_read_in_a_parse_arm_that_throws_rejects_the_tree() {
        // Every read sigmastate makes while it parses, of a child whose type read throws
        // (F(BI), see above), is a `ClassCastException`: a method call's argument and object
        // types (`MethodCallSerializer.scala:91`, `:96`), a property call's object type
        // (`PropertyCallSerializer.scala:47`), a ValDef's right-hand side
        // (`ValDefSerializer.scala:48`), a Map's mapper (`transformers.scala:38`), a
        // SelectField's input (`:294`), a collection item (`ConcreteCollectionSerializer.scala
        // :38`), and the root's, for rule 1001 (`ErgoTreeSerializer.scala`)
        use crate::mir::bin_op::{BinOp, RelationOp};
        use crate::mir::block::BlockValue;
        use crate::mir::bool_to_sigma::BoolToSigmaProp;
        use crate::mir::coll_map::Map;
        use crate::mir::coll_size::SizeOf;
        use crate::mir::collection::Collection;
        use crate::mir::expr::type_read_tests::{bi, filter};
        use crate::mir::expr::Expr;
        use crate::mir::method_call::MethodCall;
        use crate::mir::property_call::PropertyCall;
        use crate::mir::val_def::ValDef;
        use crate::serialization::SigmaSerializable;
        use crate::types::scoll;
        use crate::types::sfunc::SFunc;
        use crate::types::stype::SType;
        use alloc::boxed::Box;
        use alloc::vec;
        let f = || filter(bi());
        let ser = |e: Expr| e.sigma_serialize_bytes().unwrap();
        // sigmaProp(EQ(SizeOf(e), Int 0))
        let size_is_0 = |e: Expr| -> alloc::vec::Vec<u8> {
            let eq: Expr = BinOp {
                kind: RelationOp::Eq.into(),
                left: Box::new(SizeOf { input: Box::new(e) }.into()),
                right: Box::new(0i32.into()),
            }
            .into();
            sized_v3(&ser(BoolToSigmaProp {
                input: Box::new(eq),
            }
            .into()))
        };
        let index_of = |obj: Expr, elem: Expr| -> Expr {
            MethodCall {
                obj: Box::new(obj),
                method: scoll::INDEX_OF_METHOD.clone(),
                args: vec![elem, 0i32.into()],
                explicit_type_args: Default::default(),
            }
            .into()
        };
        let index_of_is_0 = |e: Expr| -> alloc::vec::Vec<u8> {
            let eq: Expr = BinOp {
                kind: RelationOp::Eq.into(),
                left: Box::new(e),
                right: Box::new(0i32.into()),
            }
            .into();
            sized_v3(&ser(BoolToSigmaProp {
                input: Box::new(eq),
            }
            .into()))
        };
        let block: Expr = BlockValue {
            items: vec![ValDef {
                id: 1.into(),
                rhs: Box::new(f()),
                tpe_args: vec![],
            }
            .into()],
            result: Box::new(Expr::Const(true.into())),
        }
        .into();
        // sigmaProp(EQ(SelectField(F(BI), 1), Int 1)), written by hand: the node's type is
        // private
        let mut select = vec![0xd1, 0x93, 0x8c];
        select.extend(ser(f()));
        select.extend([0x01, 0x04, 0x02]);
        let cases: [(&str, alloc::vec::Vec<u8>); 8] = [
            (
                "MethodCall object",
                index_of_is_0(index_of(f(), 0i32.into())),
            ),
            (
                "MethodCall argument",
                index_of_is_0(index_of(vec![1i32].into(), f())),
            ),
            (
                "PropertyCall object",
                size_is_0(
                    PropertyCall {
                        obj: Box::new(f()),
                        method: scoll::INDICES_METHOD.clone(),
                        explicit_type_args: Default::default(),
                    }
                    .into(),
                ),
            ),
            (
                "ValDef",
                sized_v3(&ser(BoolToSigmaProp {
                    input: Box::new(block),
                }
                .into())),
            ),
            (
                "Map mapper",
                size_is_0(
                    Map {
                        input: Box::new(vec![1i32].into()),
                        mapper: Box::new(f()),
                        mapper_sfunc: SFunc::new(vec![SType::SInt], SType::SInt),
                    }
                    .into(),
                ),
            ),
            ("SelectField", sized_v3(&select)),
            (
                "collection item",
                size_is_0(
                    Collection::Exprs {
                        elem_tpe: SType::SInt,
                        items: vec![f()],
                    }
                    .into(),
                ),
            ),
            ("root", sized_v3(&ser(f()))),
        ];
        let wrong: alloc::vec::Vec<&str> = cases
            .iter()
            .filter(|(_, bytes)| {
                !matches!(
                    ErgoTree::sigma_parse_bytes(bytes),
                    Err(SigmaParsingError::TypeReadFailed(_))
                )
            })
            .map(|(name, _)| *name)
            .collect();
        assert!(wrong.is_empty(), "not a TypeReadFailed: {wrong:?}");
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
