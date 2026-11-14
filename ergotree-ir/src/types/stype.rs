//! SType hierarchy

use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::convert::TryInto;
use core::fmt::Debug;
use hashbrown::HashMap;

use impl_trait_for_tuples::impl_for_tuples;

use crate::bigint256::BigInt256;
use crate::chain::ergo_box::ErgoBox;
use crate::serialization::SigmaParsingError;
use crate::sigma_protocol::sigma_boolean::SigmaBoolean;
use crate::sigma_protocol::sigma_boolean::SigmaProofOfKnowledgeTree;
use crate::sigma_protocol::sigma_boolean::SigmaProp;
use crate::sigma_protocol::sigma_boolean::{ProveDhTuple, ProveDlog};
use crate::unsignedbigint256::UnsignedBigInt;
use ergo_chain_types::EcPoint;

use super::sfunc::SFunc;
use super::stuple::STuple;
use super::stype_param::STypeParam;
use super::stype_param::STypeVar;
use crate::mir::avl_tree_data::AvlTreeData;

/// A substitution of type variables that sigmastate cannot build: its rewrite replaced the
/// variable of a function type's type parameter with a type that is no type variable
/// (`SType::with_subst`)
#[derive(PartialEq, Eq, Debug, Clone)]
pub struct TypeSubstError(pub String);

/// Every type descriptor is a tree represented by nodes in SType hierarchy.
#[derive(PartialEq, Eq, Debug, Clone)]
pub enum SType {
    /// Type variable (generic)
    STypeVar(STypeVar),
    /// TBD
    SAny,
    /// Unit struct
    SUnit,
    /// Boolean
    SBoolean,
    /// Signed byte
    SByte,
    /// Signed short (16-bit)
    SShort,
    /// Signed int (32-bit)
    SInt,
    /// Signed long (64-bit)
    SLong,
    /// 256-bit integer
    SBigInt,
    /// Discrete logarithm prime-order group element [`EcPoint`]
    SGroupElement,
    /// Proposition which can be proven and verified by sigma protocol.
    SSigmaProp,
    /// Unsigned 256-bit integer type
    SUnsignedBigInt,
    /// ErgoBox value
    SBox,
    /// AVL tree value
    SAvlTree,
    /// Optional value
    SOption(Arc<SType>),
    /// Collection of elements of the same type
    SColl(Arc<SType>),
    /// Tuple (elements can have different types)
    STuple(Arc<STuple>),
    /// Function (signature)
    SFunc(Arc<SFunc>),
    /// Context object ("CONTEXT" in ErgoScript)
    SContext,
    /// UTF-8 String type
    SString,
    /// Header of a block
    SHeader,
    /// Header of a block without solved mining puzzle
    SPreHeader,
    /// Data type introduced to unify handling of global and non-global (i.e. methods) operations.
    SGlobal,
    /// The JVM's `NoType` (`sigma/ast/SType.scala:278`): the type of a node whose type cannot be
    /// assigned statically. sigmastate accepts it wherever a numeric type is required
    /// (`isNumTypeOrNoType`, `sigma/ast/package.scala:135`, gating `Negation`, `BitInversion` and
    /// `BitOp`), which is wired up in a later round. It carries type code 0 and has **no wire
    /// form**: type code 0 is a hard parse reject here, so it must never be serialized.
    SNoType,
}

impl SType {
    /// Check if type is numeric
    pub fn is_numeric(&self) -> bool {
        matches!(
            self,
            SType::SByte
                | SType::SShort
                | SType::SInt
                | SType::SLong
                | SType::SBigInt
                | SType::SUnsignedBigInt
        )
    }

    /// Check if type is primitive
    pub fn is_prim(&self) -> bool {
        matches!(
            self,
            SType::SByte
                | SType::SShort
                | SType::SInt
                | SType::SLong
                | SType::SBigInt
                | SType::SAny
                | SType::SUnit
                | SType::SGroupElement
                | SType::SSigmaProp
                | SType::SUnsignedBigInt
                | SType::SBox
                | SType::SAvlTree
                | SType::SContext
                | SType::SString
                | SType::SBoolean
                | SType::SHeader
                | SType::SPreHeader
                | SType::SGlobal
        )
    }

    /// Whether sigmastate's `stypeToRType` converts this type (v6.0.7 `Evaluation.scala`): every
    /// type has an arm except a type variable, `NoType`, and a function type with other than one
    /// argument or with type parameters, at any depth. Where sigmastate resolves the runtime
    /// type of a type that has none it throws a `RuntimeException`, whatever it was about to do
    /// with it.
    pub fn has_runtime_type(&self) -> bool {
        match self {
            SType::STypeVar(_) | SType::SNoType => false,
            SType::SColl(t) | SType::SOption(t) => t.has_runtime_type(),
            SType::STuple(t) => t.items.iter().all(SType::has_runtime_type),
            SType::SFunc(f) => {
                f.t_dom.len() == 1
                    && f.tpe_params.is_empty()
                    && f.t_dom.iter().all(SType::has_runtime_type)
                    && f.t_range.has_runtime_type()
            }
            SType::SAny
            | SType::SUnit
            | SType::SBoolean
            | SType::SByte
            | SType::SShort
            | SType::SInt
            | SType::SLong
            | SType::SBigInt
            | SType::SUnsignedBigInt
            | SType::SGroupElement
            | SType::SSigmaProp
            | SType::SBox
            | SType::SAvlTree
            | SType::SContext
            | SType::SString
            | SType::SHeader
            | SType::SPreHeader
            | SType::SGlobal => true,
        }
    }

    pub(crate) fn check_v6_type(&self) -> Result<(), SigmaParsingError> {
        match self {
            SType::SUnsignedBigInt | SType::SOption(_) | SType::SHeader => {
                Err(SigmaParsingError::V6TypeError)
            }
            SType::SColl(elem_tpe) => elem_tpe.check_v6_type(),
            SType::STuple(tuple) => tuple.items.iter().try_for_each(SType::check_v6_type),
            _ => Ok(()),
        }
    }

    /// sigmastate's `applySubst` (v6.0.7 `sigma/ast/package.scala:72-81`), which treats a
    /// function type in two ways. One it is handed directly (the type itself, or the argument
    /// or the range of such a function type) loses the type parameters the substitution
    /// names, and its arguments and range are substituted the same way ([`SFunc::with_subst`]).
    /// Any other type is rewritten bottom-up ([`Self::rewrite_bottom_up`]), and that walk
    /// fails on a function type inside it whose type parameter the substitution names.
    pub(crate) fn with_subst(
        &self,
        subst: &HashMap<STypeVar, SType>,
    ) -> Result<Self, TypeSubstError> {
        match self {
            SType::SFunc(sfunc) => Ok(SType::SFunc(sfunc.with_subst(subst)?.into())),
            _ => self.rewrite_bottom_up(subst),
        }
    }

    /// The rewrite `applySubst` runs on a type that is not a function type: kiama's
    /// `everywherebu` over the rule "a type variable the substitution names becomes its
    /// type" (`package.scala:77-80`). The walk visits every child of every node, a function
    /// type's parameters included, and rebuilds a node whose child changed through its
    /// constructor. A type parameter holds a type variable (`STypeParam(ident: STypeVar)`),
    /// so a parameter whose variable was replaced by anything but a type variable cannot be
    /// rebuilt: kiama's `dup` throws a `RuntimeException` ("dup illegal arguments"). A
    /// replacement is not walked again.
    fn rewrite_bottom_up(&self, subst: &HashMap<STypeVar, SType>) -> Result<Self, TypeSubstError> {
        Ok(match self {
            SType::STypeVar(tpe_var) => subst.get(tpe_var).cloned().unwrap_or(self.clone()),
            SType::SOption(tpe) => SType::SOption(tpe.rewrite_bottom_up(subst)?.into()),
            SType::SColl(tpe) => SType::SColl(tpe.rewrite_bottom_up(subst)?.into()),
            SType::STuple(stup) => SType::STuple(
                STuple {
                    items: stup
                        .items
                        .try_mapped_ref(|item| item.rewrite_bottom_up(subst))?,
                }
                .into(),
            ),
            SType::SFunc(sfunc) => SType::SFunc(
                SFunc {
                    t_dom: sfunc
                        .t_dom
                        .iter()
                        .map(|arg| arg.rewrite_bottom_up(subst))
                        .collect::<Result<_, _>>()?,
                    t_range: sfunc.t_range.rewrite_bottom_up(subst)?.into(),
                    tpe_params: sfunc
                        .tpe_params
                        .iter()
                        .map(|param| match subst.get(&param.ident) {
                            None => Ok(param.clone()),
                            Some(SType::STypeVar(ident)) => Ok(STypeParam {
                                ident: ident.clone(),
                            }),
                            Some(tpe) => Err(TypeSubstError(format!(
                                "the type parameter {:?} of {sfunc:?} cannot become {tpe:?}",
                                param.ident
                            ))),
                        })
                        .collect::<Result<_, _>>()?,
                }
                .into(),
            ),
            SType::SAny
            | SType::SUnit
            | SType::SBoolean
            | SType::SByte
            | SType::SShort
            | SType::SInt
            | SType::SLong
            | SType::SBigInt
            | SType::SGroupElement
            | SType::SSigmaProp
            | SType::SUnsignedBigInt
            | SType::SBox
            | SType::SAvlTree
            | SType::SContext
            | SType::SString
            | SType::SHeader
            | SType::SPreHeader
            | SType::SGlobal
            | SType::SNoType => self.clone(),
        })
    }
}

impl From<STuple> for SType {
    fn from(v: STuple) -> Self {
        SType::STuple(v.into())
    }
}

impl From<STypeVar> for SType {
    fn from(v: STypeVar) -> Self {
        SType::STypeVar(v)
    }
}

impl From<SFunc> for SType {
    fn from(v: SFunc) -> Self {
        SType::SFunc(v.into())
    }
}

impl core::fmt::Display for SType {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            SType::STypeVar(t) => write!(f, "{}", t.as_string()),
            SType::SAny => write!(f, "Any"),
            SType::SNoType => write!(f, "NoType"),
            SType::SUnit => write!(f, "Unit"),
            SType::SBoolean => write!(f, "Boolean"),
            SType::SByte => write!(f, "Byte"),
            SType::SShort => write!(f, "Short"),
            SType::SInt => write!(f, "Int"),
            SType::SLong => write!(f, "Long"),
            SType::SBigInt => write!(f, "BigInt"),
            SType::SGroupElement => write!(f, "GroupElement"),
            SType::SSigmaProp => write!(f, "SigmaProp"),
            SType::SUnsignedBigInt => write!(f, "SUnsignedBigInt"),
            SType::SBox => write!(f, "Box"),
            SType::SAvlTree => write!(f, "AvlTree"),
            SType::SOption(t) => write!(f, "Option[{}]", t),
            SType::SColl(t) => write!(f, "Coll[{}]", t),
            SType::STuple(t) => write!(f, "{}", t),
            SType::SFunc(t) => write!(f, "{}", t),
            SType::SContext => write!(f, "Context"),
            SType::SString => write!(f, "String"),
            SType::SHeader => write!(f, "Header"),
            SType::SPreHeader => write!(f, "PreHeader"),
            SType::SGlobal => write!(f, "Global"),
        }
    }
}

/// Conversion to SType
pub trait LiftIntoSType {
    /// get SType
    fn stype() -> SType;
}

impl<T: LiftIntoSType> LiftIntoSType for Vec<T> {
    fn stype() -> SType {
        SType::SColl(Arc::new(T::stype()))
    }
}

impl LiftIntoSType for bool {
    fn stype() -> SType {
        SType::SBoolean
    }
}

impl LiftIntoSType for u8 {
    fn stype() -> SType {
        SType::SByte
    }
}

impl LiftIntoSType for i8 {
    fn stype() -> SType {
        SType::SByte
    }
}

impl LiftIntoSType for i16 {
    fn stype() -> SType {
        SType::SShort
    }
}

impl LiftIntoSType for i32 {
    fn stype() -> SType {
        SType::SInt
    }
}

impl LiftIntoSType for i64 {
    fn stype() -> SType {
        SType::SLong
    }
}

impl LiftIntoSType for ErgoBox {
    fn stype() -> SType {
        SType::SBox
    }
}

impl LiftIntoSType for SigmaBoolean {
    fn stype() -> SType {
        SType::SSigmaProp
    }
}

impl LiftIntoSType for SigmaProofOfKnowledgeTree {
    fn stype() -> SType {
        SType::SSigmaProp
    }
}

impl LiftIntoSType for SigmaProp {
    fn stype() -> SType {
        SType::SSigmaProp
    }
}

impl LiftIntoSType for ProveDlog {
    fn stype() -> SType {
        SType::SSigmaProp
    }
}

impl LiftIntoSType for EcPoint {
    fn stype() -> SType {
        SType::SGroupElement
    }
}

impl LiftIntoSType for BigInt256 {
    fn stype() -> SType {
        SType::SBigInt
    }
}

impl LiftIntoSType for UnsignedBigInt {
    fn stype() -> SType {
        SType::SUnsignedBigInt
    }
}

impl LiftIntoSType for ProveDhTuple {
    fn stype() -> SType {
        SType::SSigmaProp
    }
}

impl LiftIntoSType for AvlTreeData {
    fn stype() -> SType {
        SType::SAvlTree
    }
}

impl LiftIntoSType for String {
    fn stype() -> SType {
        SType::SString
    }
}

impl<T: LiftIntoSType> LiftIntoSType for Option<T> {
    fn stype() -> SType {
        SType::SOption(Arc::new(T::stype()))
    }
}

#[impl_for_tuples(2, 4)]
#[allow(clippy::unwrap_used)]
impl LiftIntoSType for Tuple {
    fn stype() -> SType {
        let v: Vec<SType> = [for_tuples!(  #( Tuple::stype() ),* )].to_vec();
        SType::STuple(Arc::new(v.try_into().unwrap()))
    }
}

#[cfg(feature = "arbitrary")]
#[allow(clippy::unwrap_used)]
pub(crate) mod tests {
    use super::*;
    use alloc::vec;
    use proptest::prelude::*;

    pub(crate) fn primitive_type(include_v6_types: bool) -> BoxedStrategy<SType> {
        let strategy = prop_oneof![
            Just(SType::SAny),
            Just(SType::SUnit),
            Just(SType::SBoolean),
            Just(SType::SByte),
            Just(SType::SShort),
            Just(SType::SInt),
            Just(SType::SLong),
            Just(SType::SBigInt),
            Just(SType::SGroupElement),
            Just(SType::SSigmaProp),
            Just(SType::SBox),
            Just(SType::SAvlTree),
            Just(SType::SContext),
            Just(SType::SString),
            Just(SType::SHeader),
            Just(SType::SPreHeader),
            Just(SType::SGlobal),
        ]
        .boxed();
        if include_v6_types {
            strategy
                .prop_union(prop_oneof![Just(SType::SUnsignedBigInt)].boxed())
                .boxed()
        } else {
            strategy
        }
    }
    impl Arbitrary for SType {
        type Parameters = bool;
        type Strategy = BoxedStrategy<Self>;

        fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
            prop_oneof![primitive_type(_args), Just(SType::STypeVar(STypeVar::t())),]
                .prop_recursive(
                    4,  // no more than this branches deep
                    64, // total elements target
                    16, // each collection max size
                    |elem| {
                        prop_oneof![
                            prop::collection::vec(elem.clone(), 2..=5).prop_map(|elems| {
                                SType::STuple(Arc::new(elems.try_into().unwrap()))
                            }),
                            elem.clone().prop_map(|tpe| SType::SColl(Arc::new(tpe))),
                            elem.prop_map(|tpe| SType::SOption(Arc::new(tpe))),
                        ]
                    },
                )
                .boxed()
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod no_type_tests {
    use super::*;
    use alloc::format;

    #[test]
    fn no_type_is_neither_primitive_nor_numeric() {
        assert!(!SType::SNoType.is_prim());
        assert!(!SType::SNoType.is_numeric());
    }

    #[test]
    fn no_type_displays_as_no_type() {
        assert_eq!(format!("{}", SType::SNoType), "NoType");
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod runtime_type_tests {
    use super::*;
    use crate::ergo_tree::ErgoTreeVersion;
    use crate::serialization::sigma_byte_reader::{from_bytes, SigmaByteRead};
    use crate::serialization::SigmaSerializable;

    fn tpe(hex: &str) -> SType {
        let bytes = base16::decode(hex.as_bytes()).unwrap();
        from_bytes(&bytes)
            .with_tree_version(ErgoTreeVersion::V3, SType::sigma_parse)
            .unwrap()
    }

    #[test]
    fn the_types_stype_to_rtype_converts() {
        // sigmastate 6.0.7 `Evaluation.stypeToRType`: each row measured on the jars as the
        // item type of an empty `ConcreteCollection`; `NoType`, which has no bytes, by reading
        // every type that is no type variable and has no parts: the 18 of them
        for hex in [
            "01", // Boolean
            "02", // Byte
            "03", // Short
            "04", // Int
            "05", // Long
            "06", // BigInt
            "07", // GroupElement
            "08", // SigmaProp
            "09", // UnsignedBigInt
            "61", // Any
            "62", // Unit
            "63", // Box
            "64", // AvlTree
            "65", // Context
            "66", // String
            "68", // Header
            "69", // PreHeader
            "6a", // Global
        ] {
            assert!(tpe(hex).has_runtime_type(), "{hex}");
        }
        for hex in [
            "10",               // Coll[Int]
            "28",               // Option[Int]
            "48040404",         // (Int, Int, Int)
            "7001040400",       // Int => Int
            "0c24407001040400", // Coll[Option[(Int, Int => Int)]]
        ] {
            assert!(tpe(hex).has_runtime_type(), "{hex}");
        }
        assert!(!SType::SNoType.has_runtime_type());
        for hex in [
            "670154",               // T
            "0c670154",             // Coll[T]
            "0c0c670154",           // Coll[Coll[T]]
            "24670154",             // Option[T]
            "3c67015404",           // (T, Int)
            "40670154",             // (Int, T)
            "480404670154",         // (Int, Int, T)
            "0c480404670154",       // Coll[(Int, Int, T)]
            "700204040400",         // (Int, Int) => Int
            "70000400",             // () => Int
            "7001040401670154",     // [T](Int) => Int
            "7001040401670155",     // [U](Int) => Int
            "70016701540400",       // T => Int
            "70010467015400",       // Int => T
            "0c700204040400",       // Coll[(Int, Int) => Int]
            "0c244070016701540400", // Coll[Option[(Int, T => Int)]]
        ] {
            assert!(!tpe(hex).has_runtime_type(), "{hex}");
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod subst_tests {
    //! sigmastate's `applySubst` as a function, each row from the sigma-state 6.0.7 jars
    //! (`sigma.ast.package.applySubst`, called directly). 6000 random rows agree as well.
    use super::*;
    use crate::ergo_tree::ErgoTreeVersion;
    use crate::serialization::sigma_byte_reader::{from_bytes, SigmaByteRead};
    use crate::serialization::SigmaSerializable;

    fn tpe(hex: &str) -> SType {
        let bytes = base16::decode(hex.as_bytes()).unwrap();
        from_bytes(&bytes)
            .with_tree_version(ErgoTreeVersion::V3, SType::sigma_parse)
            .unwrap()
    }

    /// `"T=04,U=05"`: each variable's name and the bytes of its type
    fn subst(pairs: &str) -> HashMap<STypeVar, SType> {
        pairs
            .split(',')
            .map(|pair| {
                let (name, hex) = pair.split_once('=').unwrap();
                (
                    STypeVar::new_from_bytes(name.as_bytes().to_vec()).unwrap(),
                    tpe(hex),
                )
            })
            .collect()
    }

    #[test]
    fn a_substitution_gives_the_jvms_type_and_fails_where_it_throws() {
        // the type, the substitution, the JVM's result (`None`: `applySubst` throws)
        for (hex, pairs, expected) in [
            // T
            ("670154", "T=04", Some("04")),
            // Coll[T]
            ("0c670154", "T=04", Some("10")),
            // Option[(T, U)]
            ("243c670154670155", "T=04", Some("2440670155")),
            // (Int, T, Coll[T])
            ("48046701540c670154", "T=05", Some("48040511")),
            // Coll[T], T is Coll[T]
            ("0c670154", "T=0c670154", Some("0c0c670154")),
            // (T, U), two variables
            ("3c670154670155", "T=04,U=05", Some("4005")),
            // T, another variable named
            ("670154", "U=04", Some("670154")),
            // T => T
            ("700167015467015400", "T=04", Some("7001040400")),
            // [T](T) => T
            ("700167015467015401670154", "T=04", Some("7001040400")),
            // [T, U](Int) => Int
            ("7001040402670154670155", "T=04", Some("7001040401670155")),
            // [U](T) => Int
            ("70016701540401670155", "T=04", Some("7001040401670155")),
            // Int => [T](T) => Int
            (
                "7001047001670154040167015400",
                "T=04",
                Some("700104700104040000"),
            ),
            // ([T](T) => Int) => Int
            (
                "7001700167015404016701540400",
                "T=04",
                Some("700170010404000400"),
            ),
            // [T](Int) => Int, T is U
            ("7001040401670154", "T=670155", Some("7001040400")),
            // [T](Int) => Int, another variable named
            ("7001040401670154", "U=04", Some("7001040401670154")),
            // Coll[T => Int]
            ("0c70016701540400", "T=04", Some("0c7001040400")),
            // Coll[[U](T) => Int]
            ("0c70016701540401670155", "T=04", Some("0c7001040401670155")),
            // Coll[[T](Int) => Int]
            ("0c7001040401670154", "T=04", None),
            // Option[[T](T) => Int]
            ("2470016701540401670154", "T=04", None),
            // ([T](Int) => Int, Int)
            ("3c700104040167015404", "T=04", None),
            // (Int, Int, [T](Int) => Int)
            ("4804047001040401670154", "T=04", None),
            // Int => Coll[[T](Int) => Int]
            ("7001040c700104040167015400", "T=04", None),
            // Coll[Int => [T](Int) => Int]
            ("0c700104700104040167015400", "T=04", None),
            // Coll[([T](Int) => Int) => Int]
            ("0c700170010404016701540400", "T=04", None),
            // Coll[[T](Int) => Int], T is U
            ("0c7001040401670154", "T=670155", Some("0c7001040401670155")),
            // Coll[[T](T) => Int], T is U
            (
                "0c70016701540401670154",
                "T=670155",
                Some("0c70016701550401670155"),
            ),
            // Coll[[T](Int) => Int], T is T
            ("0c7001040401670154", "T=670154", Some("0c7001040401670154")),
            // Coll[[T](Int) => Int], another variable named
            ("0c7001040401670154", "U=04", Some("0c7001040401670154")),
            // Coll[[T, U](Int) => Int], T is V and U is Int
            ("0c7001040402670154670155", "T=670156,U=04", None),
            // (Int, Int, Int, T), T is U and U is Int: a replacement is not walked again
            ("54040404670154", "T=670155,U=04", Some("54040404670155")),
        ] {
            assert_eq!(
                tpe(hex).with_subst(&subst(pairs)).ok(),
                expected.map(tpe),
                "{hex} with {pairs}"
            );
        }
    }

    #[allow(clippy::panic)]
    fn tuple_of(tpe: &SType) -> &Arc<STuple> {
        match tpe {
            SType::STuple(tuple) => tuple,
            _ => panic!("not a tuple type: {tpe:?}"),
        }
    }

    #[test]
    fn a_replacement_is_shared_not_copied() {
        // sigmastate's rule gives the substitution's own object at every place it replaces a
        // variable (`package.scala:77-79`), so a type grown by a substitution costs what its
        // parts cost. `(T, T)` with `T` as `(Int, Long)`; the jars: `((Int, Long), (Int, Long))`
        let subst = subst("T=4005");
        let own = tuple_of(&subst[&STypeVar::t()]);
        let replaced = tpe("3c670154670154").with_subst(&subst).unwrap();
        assert_eq!(replaced, tpe("3c40054005"));
        let items = tuple_of(&replaced).items.as_slice();
        assert!(Arc::ptr_eq(tuple_of(&items[0]), tuple_of(&items[1])));
        assert!(Arc::ptr_eq(tuple_of(&items[0]), own));
    }
}
