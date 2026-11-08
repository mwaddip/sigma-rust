//! Create an AVL tree

use alloc::boxed::Box;
use alloc::sync::Arc;

use super::expr::Expr;
use crate::has_opcode::HasStaticOpCode;
use crate::mir::expr::InvalidArgumentError;
use crate::serialization::op_code::OpCode;
use crate::serialization::sigma_byte_reader::SigmaByteRead;
use crate::serialization::sigma_byte_writer::SigmaByteWrite;
use crate::serialization::{SigmaParsingError, SigmaSerializable, SigmaSerializeResult};
use crate::traversable::impl_traversable_expr;
use crate::types::stype::SType;

/// Creates an AVL tree
#[derive(PartialEq, Eq, Debug, Clone)]
pub struct CreateAvlTree {
    /// Value of type Byte
    pub flags: Box<Expr>,
    /// Byte array with SColl(SByte) expr type
    pub digest: Box<Expr>,
    /// Value of type Int
    pub key_length: Box<Expr>,
    /// Value of type `Option[Int]`, the optional length of the tree's values
    pub value_length: Box<Expr>,
}

impl CreateAvlTree {
    /// Creates new AVL Tree, checking the four arguments' types
    pub fn new(
        flags: Expr,
        digest: Expr,
        key_length: Expr,
        value_length: Expr,
    ) -> Result<Self, InvalidArgumentError> {
        flags.check_post_eval_tpe(&SType::SByte)?;
        digest.check_post_eval_tpe(&SType::SColl(Arc::new(SType::SByte)))?;
        key_length.check_post_eval_tpe(&SType::SInt)?;
        value_length.check_post_eval_tpe(&SType::SOption(Arc::new(SType::SInt)))?;
        Ok(Self {
            flags: flags.into(),
            digest: digest.into(),
            key_length: key_length.into(),
            value_length: value_length.into(),
        })
    }

    /// Type
    pub fn tpe(&self) -> SType {
        SType::SAvlTree
    }
}

impl HasStaticOpCode for CreateAvlTree {
    const OP_CODE: OpCode = OpCode::AVL_TREE;
}

impl SigmaSerializable for CreateAvlTree {
    /// sigmastate reads four values and builds the node with no check of them (v6.0.7
    /// `CreateAvlTreeSerializer.scala:30-36`, `SigmaBuilder.scala:499-503`; its casts are to an
    /// erased `Value[T]`)
    fn sigma_parse<R: SigmaByteRead>(r: &mut R) -> Result<Self, SigmaParsingError> {
        let flags = Expr::sigma_parse(r)?;
        let digest = Expr::sigma_parse(r)?;
        let key_length = Expr::sigma_parse(r)?;
        let value_length = Expr::sigma_parse(r)?;
        Ok(Self {
            flags: flags.into(),
            digest: digest.into(),
            key_length: key_length.into(),
            value_length: value_length.into(),
        })
    }

    fn sigma_serialize<W: SigmaByteWrite>(&self, w: &mut W) -> SigmaSerializeResult {
        self.flags.sigma_serialize(w)?;
        self.digest.sigma_serialize(w)?;
        self.key_length.sigma_serialize(w)?;
        self.value_length.sigma_serialize(w)
    }
}

impl_traversable_expr!(
    CreateAvlTree,
    boxed flags,
    boxed digest,
    boxed key_length,
    boxed value_length
);

#[allow(clippy::unwrap_used)]
#[cfg(feature = "arbitrary")]
/// Arbitrary impl
mod arbitrary {
    use crate::mir::expr::arbitrary::ArbExprParams;
    use crate::mir::get_var::GetVar;

    use super::*;
    use proptest::prelude::*;

    impl Arbitrary for CreateAvlTree {
        type Strategy = BoxedStrategy<Self>;
        type Parameters = ();

        fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
            (
                any_with::<Expr>(ArbExprParams {
                    tpe: SType::SByte,
                    depth: 0,
                }),
                any_with::<Expr>(ArbExprParams {
                    tpe: SType::SColl(Arc::new(SType::SByte)),
                    depth: 0,
                }),
                any_with::<Expr>(ArbExprParams {
                    tpe: SType::SInt,
                    depth: 0,
                }),
                // `Option[Int]` data is serializable from tree version 3 only (rule 1009 below
                // it), so the generated argument is a context variable of type `Option[Int]`
                any::<u8>().prop_map(|var_id| {
                    Expr::GetVar(
                        GetVar {
                            var_id,
                            var_tpe: SType::SInt,
                        }
                        .into(),
                    )
                }),
            )
                .prop_map(|(flags, digest, key_length, value_length)| {
                    Self::new(flags, digest, key_length, value_length).unwrap()
                })
                .boxed()
        }
    }
}

#[cfg(test)]
#[cfg(feature = "arbitrary")]
#[allow(clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::ergo_tree::ErgoTree;
    use crate::mir::constant::Constant;
    use crate::serialization::sigma_serialize_roundtrip;
    use alloc::vec;
    use alloc::vec::Vec;
    use proptest::prelude::*;

    proptest! {

        #[test]
        fn ser_roundtrip(v in any::<CreateAvlTree>()) {
            let expr: Expr = v.into();
            prop_assert_eq![sigma_serialize_roundtrip(&expr), expr];
        }
    }

    /// A `Coll[Byte]` constant of 33 zero bytes, the digest argument
    fn digest_constant() -> Vec<u8> {
        [vec![0x0e, 0x21], vec![0u8; 33]].concat()
    }

    /// sigmastate reads four values (v6.0.7 `CreateAvlTreeSerializer.scala:30-36`): an unsized v0
    /// tree with segregated constants `7: Byte`, the digest and `32: Int`, whose body is
    /// `sigmaProp(avlTree(c0, c1, c2, getVar[Int](1)) == avlTree(c0, c1, c2, getVar[Int](1)))`.
    /// The fourth argument is not a constant: `Option[Int]` data is refused below tree version 3
    /// (rule 1009, `CoreDataSerializer.scala:149-170`).
    #[test]
    fn create_avl_tree_reads_four_values() {
        let call = [0xb6, 0x73, 0x00, 0x73, 0x01, 0x73, 0x02, 0xe3, 0x01, 0x04];
        let bytes = [
            &[0x10, 0x03, 0x02, 0x07][..],
            &digest_constant(),
            &[0x04, 0x40, 0xd1, 0x93],
            &call,
            &call,
        ]
        .concat();
        let tree = ErgoTree::sigma_parse_bytes(&bytes).unwrap();
        assert!(matches!(tree, ErgoTree::Parsed(_)), "{tree:?}");
        assert_eq!(tree.sigma_serialize_bytes().unwrap(), bytes);
        let Expr::BoolToSigmaProp(b) = tree.root_expr().unwrap() else {
            panic!("root: {tree:?}")
        };
        let Expr::BinOp(eq) = &*b.input else {
            panic!("input: {b:?}")
        };
        let Expr::CreateAvlTree(avl) = &*eq.expr().left else {
            panic!("left: {eq:?}")
        };
        assert!(matches!(&*avl.value_length, Expr::GetVar(v) if v.expr().var_id == 1));
        assert_eq!(
            avl.value_length.tpe(),
            SType::SOption(alloc::sync::Arc::new(SType::SInt))
        );
    }

    /// SANTA's `create_avl_tree-errored#1` (`AvlTree.unsupported_eval_nodes_v6.json`), written by
    /// sigmastate: a v3 tree whose root is the `CreateAvlTree` over four placeholders
    #[test]
    fn create_avl_tree_placeholders_round_trip() {
        let bytes = base16::decode(
            "1b330402070e21fb2b77372d81da43ce2d72714aec79ae5fcac20a9aff426fe6afb476a6fbc02c04\
             04402800b67300730173027303",
        )
        .unwrap();
        let tree = ErgoTree::sigma_parse_bytes_lenient(&bytes).unwrap();
        assert!(matches!(tree, ErgoTree::Parsed(_)), "{tree:?}");
        assert_eq!(tree.sigma_serialize_bytes().unwrap(), bytes);
    }

    /// The layout sigma-rust used to write, three values and a `00` tag for `None`: the fourth
    /// read meets `00`, a constant of type code 0, which sigmastate refuses
    /// (`TypeSerializer.scala:138`)
    #[test]
    fn the_old_three_value_layout_is_refused() {
        let bytes = [
            &[0x00, 0xd1, 0x93, 0xb6, 0x02, 0x07][..],
            &digest_constant(),
            &[0x04, 0x40, 0x00],
        ]
        .concat();
        assert!(matches!(
            ErgoTree::sigma_parse_bytes(&bytes),
            Err(SigmaParsingError::InvalidTypePrefix)
        ));
    }

    #[test]
    fn printing_keeps_all_four_arguments() {
        let args: [Expr; 4] = [
            Constant::from(7i8).into(),
            Constant::from(vec![0u8; 33]).into(),
            Constant::from(32i32).into(),
            Constant::from(Some(5i32)).into(),
        ];
        let expr: Expr = CreateAvlTree::new(
            args[0].clone(),
            args[1].clone(),
            args[2].clone(),
            args[3].clone(),
        )
        .unwrap()
        .into();
        let (printed, text) = expr.pretty_print().unwrap();
        assert_eq!(printed, expr);
        for arg in &args {
            let (_, arg_text) = arg.pretty_print().unwrap();
            assert!(text.contains(&arg_text), "{arg_text} not in {text}");
        }
    }
}
