use crate::ergo_tree::ErgoTreeVersion;
use crate::serialization::types::TypeCode;
use crate::types::stype_companion::STypeCompanion;
use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;

use super::sfunc::SFunc;
use super::smethod::MethodId;
use super::smethod::SMethod;
use super::smethod::SMethodDesc;
use super::stype::SType;
use lazy_static::lazy_static;

/// SSigmaProp type code
pub const TYPE_CODE: TypeCode = TypeCode::SSIGMAPROP;
/// SSigmaProp type name
pub static TYPE_NAME: &str = "SigmaProp";
/// SigmaProp.propBytes
pub const PROP_BYTES_METHOD_ID: MethodId = MethodId(1);
/// SigmaProp.isProven
pub const IS_PROVEN_METHOD_ID: MethodId = MethodId(2);

lazy_static! {
    /// SigmaProp method descriptors: sigmastate's `SSigmaPropMethods`, at every tree version
    /// (v6.0.7 `methods.scala:703-713`). sigmastate parses both and cannot evaluate either
    /// (`propBytes` has a `PerItemCost` and no `propBytes_eval`, `isProven` no cost kind), so
    /// neither has an evaluator.
    pub(crate) static ref METHOD_DESC: Vec<SMethodDesc> =
        vec![
            PROP_BYTES_METHOD_DESC.clone(),
            IS_PROVEN_METHOD_DESC.clone(),
        ]
    ;
}

lazy_static! {
    static ref PROP_BYTES_METHOD_DESC: SMethodDesc = SMethodDesc {
        method_id: PROP_BYTES_METHOD_ID,
        name: "propBytes",
        tpe: SFunc::new(
            vec![SType::SSigmaProp],
            SType::SColl(Arc::new(SType::SByte)),
        ),
        explicit_type_args: vec![],
        min_version: ErgoTreeVersion::V0
    };
    /// SigmaProp.propBytes
    pub static ref PROP_BYTES_METHOD: SMethod = SMethod::new(STypeCompanion::SigmaProp, PROP_BYTES_METHOD_DESC.clone(),);
}

lazy_static! {
    static ref IS_PROVEN_METHOD_DESC: SMethodDesc = SMethodDesc {
        method_id: IS_PROVEN_METHOD_ID,
        name: "isProven",
        tpe: SFunc::new(vec![SType::SSigmaProp], SType::SBoolean),
        explicit_type_args: vec![],
        min_version: ErgoTreeVersion::V0
    };
    /// SigmaProp.isProven
    pub static ref IS_PROVEN_METHOD: SMethod = SMethod::new(STypeCompanion::SigmaProp, IS_PROVEN_METHOD_DESC.clone(),);
}
