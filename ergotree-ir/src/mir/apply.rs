//! Application of function

use alloc::boxed::Box;

use alloc::vec::Vec;

use crate::serialization::op_code::OpCode;
use crate::serialization::sigma_byte_reader::SigmaByteRead;
use crate::serialization::sigma_byte_writer::SigmaByteWrite;
use crate::serialization::SigmaParsingError;
use crate::serialization::SigmaSerializable;
use crate::serialization::SigmaSerializeResult;
use crate::traversable::impl_traversable_expr;
use crate::types::stype::SType;

use super::expr::Expr;
use super::expr::InvalidArgumentError;
use crate::has_opcode::HasStaticOpCode;

/// Application of function `func` to given arguments `args`
#[derive(PartialEq, Eq, Debug, Clone)]
pub struct Apply {
    /// Function
    pub func: Box<Expr>,
    /// Arguments
    pub args: Vec<Expr>,
}

impl Apply {
    /// Create new object, returns an error if any of the requirements failed
    pub fn new(func: Expr, args: Vec<Expr>) -> Result<Self, InvalidArgumentError> {
        match func.tpe() {
            SType::SFunc(sfunc) => {
                let arg_types: Vec<SType> = args.iter().map(|a| a.tpe()).collect();
                if sfunc.t_dom != arg_types {
                    Err(InvalidArgumentError(format!(
                        "Expected args: {0:?}, got: {1:?}",
                        sfunc.t_dom, args
                    )))
                } else {
                    Ok(Apply {
                        func: Box::new(func),
                        args,
                    })
                }
            }
            _ => Err(InvalidArgumentError(format!(
                "unexpected Apply::func: {0:?}",
                func.tpe(),
            ))),
        }
    }

    /// Type
    ///
    /// sigmastate's `Apply.tpe` is a `lazy val` over the function's type
    /// (`data/shared/src/main/scala/sigma/ast/values.scala:1247-1251`): an `SFunc` gives its
    /// range, a collection gives its element type, and anything else gives `NoType`. It is
    /// computed on access rather than cached at construction, so a node whose child has been
    /// replaced in place still reports the right type — which is why the cached field is gone.
    pub fn tpe(&self) -> SType {
        match self.func.tpe() {
            SType::SFunc(sfunc) => (*sfunc.t_range).clone(),
            SType::SColl(elem_tpe) => (*elem_tpe).clone(),
            _ => SType::SNoType,
        }
    }
}

impl HasStaticOpCode for Apply {
    const OP_CODE: OpCode = OpCode::APPLY;
}

impl SigmaSerializable for Apply {
    fn sigma_serialize<W: SigmaByteWrite>(&self, w: &mut W) -> SigmaSerializeResult {
        self.func.sigma_serialize(w)?;
        self.args.sigma_serialize(w)
    }

    /// sigmastate builds `Apply` with no check: its type is a lazy val, NoType for a func that
    /// is neither a function nor a collection, and the parse reads it only where a later node
    /// does (v6.0.7 `values.scala:1242-1251`)
    fn sigma_parse<R: SigmaByteRead>(r: &mut R) -> Result<Self, SigmaParsingError> {
        let func = Expr::sigma_parse(r)?;
        let args = Vec::<Expr>::sigma_parse(r)?;
        Ok(Apply {
            func: Box::new(func),
            args,
        })
    }
}

impl_traversable_expr!(Apply, boxed func, arr args);

/// Arbitrary impl
#[cfg(feature = "arbitrary")]
#[allow(clippy::unwrap_used)]
pub mod arbitrary {
    use crate::mir::func_value::*;

    use super::*;

    use proptest::collection::vec;
    use proptest::prelude::*;

    impl Arbitrary for Apply {
        type Strategy = BoxedStrategy<Self>;
        type Parameters = ();

        fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
            (any::<Expr>(), vec(any::<Expr>(), 1..10))
                .prop_map(|(body, args)| {
                    let func = FuncValue::new(
                        args.iter()
                            .enumerate()
                            .map(|(idx, arg)| FuncArg {
                                idx: (idx as u32).into(),
                                tpe: arg.tpe(),
                            })
                            .collect(),
                        body,
                    )
                    .into();
                    Self::new(func, args).unwrap()
                })
                .boxed()
        }
    }
}

#[cfg(test)]
#[cfg(feature = "arbitrary")]
#[allow(clippy::panic)]
#[allow(clippy::unwrap_used)]
mod tests {

    use crate::serialization::sigma_serialize_roundtrip;

    use super::*;

    use crate::mir::constant::Constant;
    use crate::mir::func_value::{FuncArg, FuncValue};
    use proptest::prelude::*;

    /// `(Int) => Int` applied to one Int: the shape `Apply::new` accepts.
    fn int_to_int_func() -> Expr {
        FuncValue::new(
            vec![FuncArg {
                idx: 1u32.into(),
                tpe: SType::SInt,
            }],
            Expr::Const(Constant::from(1i32)),
        )
        .into()
    }

    #[test]
    fn apply_over_a_function_is_its_range() {
        // unchanged behaviour, pinned so the refactor cannot move it
        let a = Apply::new(int_to_int_func(), vec![Expr::Const(Constant::from(1i32))]).unwrap();
        assert_eq!(a.tpe(), SType::SInt);
    }

    #[test]
    fn apply_over_a_collection_is_its_element_type() {
        // JVM `values.scala:1247-1251`: a collection-typed func gives the element type.
        // Not reachable through new(); build the struct literal directly.
        let a = Apply {
            func: Box::new(Expr::Const(Constant::from(vec![1i8, 2i8]))),
            args: vec![],
        };
        assert_eq!(a.tpe(), SType::SByte);
    }

    #[test]
    fn apply_over_anything_else_is_no_type() {
        let a = Apply {
            func: Box::new(Expr::Const(Constant::from(1i32))),
            args: vec![],
        };
        assert_eq!(a.tpe(), SType::SNoType);
    }

    #[test]
    fn apply_new_still_refuses_a_non_function() {
        // new() stays strict for library users (spec decision 1)
        assert!(Apply::new(Expr::Const(Constant::from(1i32)), vec![]).is_err());
    }

    proptest! {

        #[test]
        fn ser_roundtrip(v in any::<Apply>()) {
            let expr: Expr = v.into();
            prop_assert_eq![sigma_serialize_roundtrip(&expr), expr];
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod parse_tests {
    use crate::ergo_tree::{ErgoTree, ErgoTreeError};
    use crate::serialization::{SigmaParsingError, SigmaSerializable};
    use crate::types::stype::SType;

    fn tree(hex: &str) -> Result<ErgoTree, SigmaParsingError> {
        ErgoTree::sigma_parse_bytes(&base16::decode(hex.as_bytes()).unwrap())
    }

    #[test]
    fn apply_of_a_non_function_parses() {
        // SANTA `tree_parse_acceptance` #5: sigmaProp(Apply(Int 1, [])); `tree_root_forms`
        // #0: the root Apply(Coll(sigmaProp(true)), [0]), whose type is the element type
        for hex in ["00d1da040200", "00da1401d3010400"] {
            let bytes = base16::decode(hex.as_bytes()).unwrap();
            let parsed = ErgoTree::sigma_parse_bytes(&bytes).unwrap();
            assert!(matches!(parsed, ErgoTree::Parsed(_)), "{hex}");
            assert_eq!(parsed.sigma_serialize_bytes().unwrap(), bytes);
        }
    }

    #[test]
    fn apply_of_an_int_is_no_type() {
        // `tree_root_forms` #1: the size-flagged root Apply(Int 0, [0]) is NoType, which fails
        // rule 1001; and Negation admits a NoType input, so the root Negation of it builds
        for hex in ["0806da04000104 00", "0807f0da0400010400"] {
            let hex = hex.replace(' ', "");
            assert!(
                matches!(
                    tree(&hex),
                    Ok(ErgoTree::Unparsed {
                        error: ErgoTreeError::RootTpeError(SType::SNoType),
                        ..
                    })
                ),
                "{hex}: {:?}",
                tree(&hex)
            );
        }
    }
}
