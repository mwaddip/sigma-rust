use alloc::boxed::Box;

use crate::mir::bin_op::BinOp;
use crate::mir::bin_op::BinOpKind;
use crate::mir::constant::Constant;
use crate::mir::constant::TryExtractInto;
use crate::mir::expr::Expr;
use crate::types::stype::SType;

use super::op_code::OpCode;
use super::sigma_byte_reader::SigmaByteRead;
use super::sigma_byte_writer::SigmaByteWrite;
use super::SigmaParsingError;
use super::SigmaSerializable;
use super::SigmaSerializeResult;

/// sigmastate writes and reads the packed boolean pair after `85` only in `Relation2Serializer`,
/// which serves the relations and the logical operations (v6.0.6 `ValueSerializer.scala:48-58`).
/// The arithmetic and bitwise operations go through `TwoArgumentsSerializer`, which always
/// writes and reads two values (`TwoArgumentsSerializer.scala:15-25`), so there `85` starts a
/// `Coll[Boolean]` constant.
fn has_pair_form(kind: BinOpKind) -> bool {
    matches!(kind, BinOpKind::Relation(_) | BinOpKind::Logical(_))
}

pub fn bin_op_sigma_serialize<W: SigmaByteWrite>(
    bin_op: &BinOp,
    w: &mut W,
) -> SigmaSerializeResult {
    match (*bin_op.clone().left, *bin_op.clone().right) {
        (
            Expr::Const(Constant {
                tpe: SType::SBoolean,
                v: l,
            }),
            Expr::Const(Constant {
                tpe: SType::SBoolean,
                v: r,
            }),
        ) if has_pair_form(bin_op.kind) => {
            OpCode::COLL_OF_BOOL_CONST.sigma_serialize(w)?;
            let arr = [l.try_extract_into::<bool>()?, r.try_extract_into::<bool>()?];
            w.put_bits(&arr)?;
            Ok(())
        }
        _ => {
            bin_op.left.sigma_serialize(w)?;
            bin_op.right.sigma_serialize(w)
        }
    }
}

pub fn bin_op_sigma_parse<R: SigmaByteRead>(
    op_kind: BinOpKind,
    r: &mut R,
) -> Result<Expr, SigmaParsingError> {
    // `Relation2Serializer.parse` (sigmastate v6.0.6 `Relation2Serializer.scala:40-52`)
    // peeks for the packed boolean pair without the position check, then reads either
    // the pair or two values.
    if has_pair_form(op_kind) && r.peek_u8()? == OpCode::COLL_OF_BOOL_CONST.value() {
        r.get_u8()?;
        let bools = r.get_bits(2)?;
        #[allow(clippy::unwrap_used)]
        return Ok(BinOp {
            kind: op_kind,
            left: Box::new(Expr::Const((*bools.first().unwrap()).into())),
            right: Box::new(Expr::Const((*bools.get(1).unwrap()).into())),
        }
        .into());
    }
    let left = Expr::sigma_parse(r)?;
    let right = Expr::sigma_parse(r)?;
    // `BitOp` requires numeric operands (`trees.scala:913`)
    if matches!(op_kind, BinOpKind::Bit(_))
        && !(left.tpe().is_numeric() && right.tpe().is_numeric())
    {
        return Err(SigmaParsingError::BitOpOperandsNotNumeric(format!(
            "left: {:?}, right: {:?}",
            left.tpe(),
            right.tpe()
        )));
    }
    Ok(BinOp {
        kind: op_kind,
        left: Box::new(left),
        right: Box::new(right),
    }
    .into())
}

#[cfg(test)]
#[cfg(feature = "arbitrary")]
#[allow(clippy::panic)]
mod proptests {
    use super::*;
    use crate::mir::expr::arbitrary::ArbExprParams;
    use crate::serialization::sigma_serialize_roundtrip;

    use proptest::prelude::*;

    proptest! {

        #[test]
        fn ser_roundtrip(v in any_with::<BinOp>(ArbExprParams {tpe: SType::SAny, depth: 0})) {
            let expr: Expr = v.into();
            prop_assert_eq![sigma_serialize_roundtrip(&expr), expr];
        }
    }
}

#[cfg(test)]
#[cfg(feature = "arbitrary")]
mod tests {
    use sigma_test_util::force_any_val_with;

    use super::*;
    use crate::mir::bin_op::RelationOp;
    use crate::mir::expr::arbitrary::ArbExprParams;
    use crate::serialization::sigma_serialize_roundtrip;
    use crate::types::stype::SType;

    fn test_ser_roundtrip(kind: BinOpKind, left: Expr, right: Expr) {
        let eq_op: Expr = BinOp {
            kind,
            left: Box::new(left),
            right: Box::new(right),
        }
        .into();
        assert_eq![sigma_serialize_roundtrip(&eq_op), eq_op];
    }

    #[test]
    fn ser_roundtrip_eq() {
        test_ser_roundtrip(
            RelationOp::Eq.into(),
            force_any_val_with::<Expr>(ArbExprParams {
                tpe: SType::SAny,
                depth: 1,
            }),
            force_any_val_with::<Expr>(ArbExprParams {
                tpe: SType::SAny,
                depth: 1,
            }),
        )
    }

    #[test]
    fn ser_roundtrip_neq() {
        test_ser_roundtrip(
            RelationOp::NEq.into(),
            force_any_val_with::<Expr>(ArbExprParams {
                tpe: SType::SAny,
                depth: 1,
            }),
            force_any_val_with::<Expr>(ArbExprParams {
                tpe: SType::SAny,
                depth: 1,
            }),
        )
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod pair_form_tests {
    //! JVM parity: only the relations and logical operations read a packed boolean pair after
    //! `85` (`Relation2Serializer`); the arithmetic and bitwise operations read two values
    //! (`TwoArgumentsSerializer`), and `BitOp` requires numeric operands (`trees.scala:913`).
    use super::*;
    use crate::ergo_tree::ErgoTree;
    use crate::mir::bin_op::ArithOp;
    use crate::serialization::sigma_serialize_roundtrip;
    use alloc::vec;
    use alloc::vec::Vec;

    /// `sigmaProp(EQ(op(C, C), C))`, `C = Coll[Boolean](true)` (SANTA `tree_bool_pair_form`)
    fn tree(op: u8) -> Vec<u8> {
        vec![
            0x00, 0xd1, 0x93, op, 0x85, 0x01, 0x01, 0x85, 0x01, 0x01, 0x85, 0x01, 0x01,
        ]
    }

    #[test]
    fn arithmetic_reads_85_as_a_collection() {
        // Plus and Minus (SANTA #0, #1), then Multiply, Division, Modulo, Min and Max
        for op in [0x9a, 0x99, 0x9c, 0x9d, 0x9e, 0xa1, 0xa2] {
            let bytes = tree(op);
            let parsed = ErgoTree::sigma_parse_bytes(&bytes).unwrap();
            assert!(matches!(parsed, ErgoTree::Parsed(_)), "{op:02x}");
            assert_eq!(parsed.sigma_serialize_bytes().unwrap(), bytes, "{op:02x}");
        }
    }

    #[test]
    fn a_bitwise_operation_on_a_collection_rejects() {
        // BitOr, BitAnd and BitXor fail `BitOp`'s `require`, unsized and size-flagged alike
        for op in [0xf2, 0xf3, 0xf5] {
            let unsized_tree = tree(op);
            let sized_tree = [&[0x08, 0x0c][..], &unsized_tree[1..]].concat();
            for bytes in [unsized_tree, sized_tree] {
                assert!(
                    matches!(
                        ErgoTree::sigma_parse_bytes(&bytes),
                        Err(SigmaParsingError::BitOpOperandsNotNumeric(..))
                    ),
                    "{bytes:02x?}"
                );
            }
        }
    }

    #[test]
    fn a_relation_reads_the_pair() {
        // SANTA #2: `sigmaProp(EQ(true, true))` in the packed form
        let bytes = [0x00, 0xd1, 0x93, 0x85, 0x03];
        let parsed = ErgoTree::sigma_parse_bytes(&bytes).unwrap();
        assert!(matches!(parsed, ErgoTree::Parsed(_)));
        assert_eq!(parsed.sigma_serialize_bytes().unwrap(), bytes);
    }

    #[test]
    fn arithmetic_on_boolean_constants_writes_two_values() {
        let expr: Expr = BinOp {
            kind: ArithOp::Plus.into(),
            left: Box::new(Expr::Const(true.into())),
            right: Box::new(Expr::Const(false.into())),
        }
        .into();
        assert_eq!(
            expr.sigma_serialize_bytes().unwrap(),
            [0x9a, 0x01, 0x01, 0x01, 0x00]
        );
        assert_eq!(sigma_serialize_roundtrip(&expr), expr);
    }
}
