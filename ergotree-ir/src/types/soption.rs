use crate::ergo_tree::ErgoTreeVersion;
use crate::serialization::types::TypeCode;

use super::sfunc::SFunc;
use super::smethod::SMethod;
use super::smethod::SMethodDesc;
use super::stype::SType;
use super::stype_companion::STypeCompanion;
use super::stype_param::STypeVar;
use crate::types::smethod::MethodId;
use alloc::vec;
use alloc::vec::Vec;
use lazy_static::lazy_static;

/// SOption type code
pub const TYPE_CODE: TypeCode = TypeCode::OPTION;
/// SOption type name
pub static TYPE_NAME: &str = "Option";
/// Option.map
pub const MAP_METHOD_ID: MethodId = MethodId(7);
/// Option.filter
pub const FILTER_METHOD_ID: MethodId = MethodId(8);
/// Option.isDefined
pub const IS_DEFINED_METHOD_ID: MethodId = MethodId(2);
/// Option.get
pub const GET_METHOD_ID: MethodId = MethodId(3);
/// Option.getOrElse
pub const GET_OR_ELSE_METHOD_ID: MethodId = MethodId(4);

lazy_static! {
    /// Option method descriptors
    pub(crate) static ref METHOD_DESC: Vec<SMethodDesc> =
        vec![
            IS_DEFINED_METHOD_DESC.clone(),
            GET_METHOD_DESC.clone(),
            GET_OR_ELSE_METHOD_DESC.clone(),
            MAP_METHOD_DESC.clone(),
            FILTER_METHOD_DESC.clone(),
        ]
    ;
}

/// `Option[T]` (sigmastate's `SOptionMethods.ThisType`)
fn option_t() -> SType {
    SType::SOption(SType::STypeVar(STypeVar::t()).into())
}

lazy_static! {
    // sigmastate's `SOptionMethods` ids 2-4, at every tree version (v6.0.7 `methods.scala:750-774`)
    static ref IS_DEFINED_METHOD_DESC: SMethodDesc = SMethodDesc {
        method_id: IS_DEFINED_METHOD_ID,
        name: "isDefined",
        tpe: SFunc::new(vec![option_t()], SType::SBoolean),
        explicit_type_args: vec![],
        min_version: ErgoTreeVersion::V0
    };
    static ref GET_METHOD_DESC: SMethodDesc = SMethodDesc {
        method_id: GET_METHOD_ID,
        name: "get",
        tpe: SFunc::new(vec![option_t()], STypeVar::t().into()),
        explicit_type_args: vec![],
        min_version: ErgoTreeVersion::V0
    };
    static ref GET_OR_ELSE_METHOD_DESC: SMethodDesc = SMethodDesc {
        method_id: GET_OR_ELSE_METHOD_ID,
        name: "getOrElse",
        tpe: SFunc::new(vec![option_t(), STypeVar::t().into()], STypeVar::t().into()),
        explicit_type_args: vec![],
        min_version: ErgoTreeVersion::V0
    };
    /// Option.isDefined
    pub static ref IS_DEFINED_METHOD: SMethod =
        SMethod::new(STypeCompanion::Option, IS_DEFINED_METHOD_DESC.clone());
    /// Option.get
    pub static ref GET_METHOD: SMethod = SMethod::new(STypeCompanion::Option, GET_METHOD_DESC.clone());
    /// Option.getOrElse
    pub static ref GET_OR_ELSE_METHOD: SMethod =
        SMethod::new(STypeCompanion::Option, GET_OR_ELSE_METHOD_DESC.clone());
}

lazy_static! {
    static ref MAP_METHOD_DESC: SMethodDesc = SMethodDesc {
        method_id: MAP_METHOD_ID,
        name: "map",
        tpe: SFunc::new(
            vec![
                SType::SOption(SType::STypeVar(STypeVar::iv()).into()),
                SFunc::new(
                    vec![STypeVar::iv().into()],
                    STypeVar::ov().into(),
                ).into()
                ],
            SType::SOption(SType::STypeVar(STypeVar::ov()).into()),
        ),
        explicit_type_args: vec![],
        min_version: ErgoTreeVersion::V0
    };
    /// Option.map
    pub static ref MAP_METHOD: SMethod = SMethod::new(
         STypeCompanion::Option,
         MAP_METHOD_DESC.clone());
}

lazy_static! {
    static ref FILTER_METHOD_DESC: SMethodDesc = SMethodDesc {
        method_id: FILTER_METHOD_ID,
        name: "filter",
        tpe: SFunc::new(
            vec![
                SType::SOption(SType::STypeVar(STypeVar::iv()).into()),
                SFunc::new(
                    vec![STypeVar::iv().into()],
                    SType::SBoolean,
                ).into()
                ],
            SType::SOption(SType::STypeVar(STypeVar::iv()).into()),
        ),
        explicit_type_args: vec![],
        min_version: ErgoTreeVersion::V0
    };
    /// Option.map
    pub static ref FILTER_METHOD: SMethod = SMethod::new(
         STypeCompanion::Option,
         FILTER_METHOD_DESC.clone());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_ids() {
        assert!(SMethod::from_ids(TYPE_CODE, MAP_METHOD_ID).map(|e| e.name()) == Ok("map"));
        assert!(SMethod::from_ids(TYPE_CODE, FILTER_METHOD_ID).map(|e| e.name()) == Ok("filter"));
    }
}
