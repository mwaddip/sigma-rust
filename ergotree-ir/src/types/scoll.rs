use crate::ergo_tree::ErgoTreeVersion;
use crate::serialization::types::TypeCode;
use crate::types::stuple::STuple;
use crate::types::stype_companion::STypeCompanion;
use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;

use super::sfunc::SFunc;
use super::smethod::MethodId;
use super::smethod::SMethod;
use super::smethod::SMethodDesc;
use super::stype::SType;
use super::stype_param::STypeVar;
use lazy_static::lazy_static;

/// SColl type code
pub const TYPE_CODE: TypeCode = TypeCode::COLL;
/// SColl type name
pub static TYPE_NAME: &str = "Coll";
/// Coll.indexOf
pub const INDEX_OF_METHOD_ID: MethodId = MethodId(26);
/// Coll.flatmap
pub const FLATMAP_METHOD_ID: MethodId = MethodId(15);
/// Coll.zip
pub const ZIP_METHOD_ID: MethodId = MethodId(29);
/// Coll.indices
pub const INDICES_METHOD_ID: MethodId = MethodId(14);
/// Coll.patch
pub const PATCH_METHOD_ID: MethodId = MethodId(19);
/// Coll.updated
pub const UPDATED_METHOD_ID: MethodId = MethodId(20);
/// Coll.updateMany
pub const UPDATE_MANY_METHOD_ID: MethodId = MethodId(21);
/// Coll.reverse
pub const REVERSE_METHOD_ID: MethodId = MethodId(30);
/// Coll.startsWith
pub const STARTS_WITH_METHOD_ID: MethodId = MethodId(31);
/// Coll.endsWith
pub const ENDS_WITH_METHOD_ID: MethodId = MethodId(32);
/// Coll.get
pub const GET_METHOD_ID: MethodId = MethodId(33);
/// Coll.size
pub const SIZE_METHOD_ID: MethodId = MethodId(1);
/// Coll.getOrElse
pub const GET_OR_ELSE_METHOD_ID: MethodId = MethodId(2);
/// Coll.map
pub const MAP_METHOD_ID: MethodId = MethodId(3);
/// Coll.exists
pub const EXISTS_METHOD_ID: MethodId = MethodId(4);
/// Coll.fold
pub const FOLD_METHOD_ID: MethodId = MethodId(5);
/// Coll.forall
pub const FORALL_METHOD_ID: MethodId = MethodId(6);
/// Coll.slice
pub const SLICE_METHOD_ID: MethodId = MethodId(7);
/// Coll.filter
pub const FILTER_METHOD_ID: MethodId = MethodId(8);
/// Coll.append
pub const APPEND_METHOD_ID: MethodId = MethodId(9);
/// Coll.apply
pub const APPLY_METHOD_ID: MethodId = MethodId(10);

lazy_static! {
    /// Coll method descriptors
    pub(crate) static ref METHOD_DESC: Vec<SMethodDesc> =
        vec![
            INDEX_OF_METHOD_DESC.clone(),
            FLATMAP_METHOD_DESC.clone(),
            ZIP_METHOD_DESC.clone(),
            INDICES_METHOD_DESC.clone(),
            UPDATED_METHOD_DESC.clone(),
            UPDATE_MANY_METHOD_DESC.clone(),
            PATCH_METHOD_DESC.clone(),
            REVERSE_METHOD_DESC.clone(),
            STARTS_WITH_METHOD_DESC.clone(),
            ENDS_WITH_METHOD_DESC.clone(),
            GET_METHOD_DESC.clone(),
            SIZE_METHOD_DESC.clone(),
            GET_OR_ELSE_METHOD_DESC.clone(),
            MAP_METHOD_DESC.clone(),
            EXISTS_METHOD_DESC.clone(),
            FOLD_METHOD_DESC.clone(),
            FORALL_METHOD_DESC.clone(),
            SLICE_METHOD_DESC.clone(),
            FILTER_METHOD_DESC.clone(),
            APPEND_METHOD_DESC.clone(),
            APPLY_METHOD_DESC.clone(),
        ];
}

lazy_static! {
    static ref INDEX_OF_METHOD_DESC: SMethodDesc = SMethodDesc {
        method_id: INDEX_OF_METHOD_ID,
        name: "indexOf",
        tpe: SFunc {
            t_dom: vec![
                SType::SColl(SType::STypeVar(STypeVar::t()).into()),
                STypeVar::t().into(),
                SType::SInt],
            t_range: SType::SInt.into(),
            tpe_params: vec![],
        },
        explicit_type_args: vec![],
        min_version: ErgoTreeVersion::V0
    };
    /// Coll.indexOf
    pub static ref INDEX_OF_METHOD: SMethod = SMethod::new(STypeCompanion::Coll, INDEX_OF_METHOD_DESC.clone());
}

lazy_static! {
    static ref FLATMAP_METHOD_DESC: SMethodDesc = SMethodDesc {
        method_id: FLATMAP_METHOD_ID,
        name: "flatMap",
        tpe: SFunc::new(
            vec![
                SType::SColl(SType::STypeVar(STypeVar::iv()).into()),
                SFunc::new(
                    vec![STypeVar::iv().into()],
                    SType::SColl(Arc::new(STypeVar::ov().into())),
                ).into()
                ],
            SType::SColl(SType::STypeVar(STypeVar::ov()).into()),
        ),
        explicit_type_args: vec![],
        min_version: ErgoTreeVersion::V0
    };
    /// Coll.flatMap
    pub static ref FLATMAP_METHOD: SMethod = SMethod::new(STypeCompanion::Coll, FLATMAP_METHOD_DESC.clone());
}

lazy_static! {
    static ref ZIP_METHOD_DESC: SMethodDesc = SMethodDesc {
        method_id: ZIP_METHOD_ID,
        name: "zip",
        tpe: SFunc::new(
            vec![
                SType::SColl(SType::STypeVar(STypeVar::t()).into()),
                SType::SColl(SType::STypeVar(STypeVar::iv()).into())
            ],
            SType::SColl(SType::STuple(STuple::pair(
                STypeVar::t().into(), STypeVar::iv().into()
            ).into()).into())
        ),
        explicit_type_args: vec![],
        min_version: ErgoTreeVersion::V0
    };
    /// Coll.zip
    pub static ref ZIP_METHOD: SMethod = SMethod::new(STypeCompanion::Coll, ZIP_METHOD_DESC.clone());
}

lazy_static! {
    static ref INDICES_METHOD_DESC: SMethodDesc = SMethodDesc {
        method_id: INDICES_METHOD_ID,
        name: "indices",
        tpe: SFunc::new(
            vec![
                SType::SColl(SType::STypeVar(STypeVar::t()).into()),
            ],
            SType::SColl(SType::SInt.into())
        ),
        explicit_type_args: vec![],
        min_version: ErgoTreeVersion::V0
    };
    /// Coll.indices
    pub static ref INDICES_METHOD: SMethod = SMethod::new(STypeCompanion::Coll, INDICES_METHOD_DESC.clone());
}

lazy_static! {
    static ref PATCH_METHOD_DESC: SMethodDesc = SMethodDesc {
        method_id: PATCH_METHOD_ID,
        name: "patch",
        tpe: SFunc::new(
            vec![
                SType::SColl(SType::STypeVar(STypeVar::t()).into()),
                SType::SInt,
                SType::SColl(SType::STypeVar(STypeVar::t()).into()),
                SType::SInt,
            ],
            SType::SColl(SType::STypeVar(STypeVar::t()).into())
        ),
        explicit_type_args: vec![],
        min_version: ErgoTreeVersion::V0
    };
    /// Coll.patch
    pub static ref PATCH_METHOD: SMethod = SMethod::new(STypeCompanion::Coll, PATCH_METHOD_DESC.clone());
}

lazy_static! {
    static ref UPDATED_METHOD_DESC: SMethodDesc = SMethodDesc {
        method_id: UPDATED_METHOD_ID,
        name: "updated",
        tpe: SFunc::new(
            vec![
                SType::SColl(SType::STypeVar(STypeVar::t()).into()),
                SType::SInt,
                SType::STypeVar(STypeVar::t())

            ],
            SType::SColl(SType::STypeVar(STypeVar::t()).into())
        ),
        explicit_type_args: vec![],
        min_version: ErgoTreeVersion::V0
    };
    /// Coll.updated
    pub static ref UPDATED_METHOD: SMethod = SMethod::new(STypeCompanion::Coll, UPDATED_METHOD_DESC.clone());
}

lazy_static! {
    static ref UPDATE_MANY_METHOD_DESC: SMethodDesc = SMethodDesc {
        method_id: UPDATE_MANY_METHOD_ID,
        name: "updateMany",
        tpe: SFunc::new(
            vec![
                SType::SColl(SType::STypeVar(STypeVar::t()).into()),
                SType::SColl(SType::SInt.into()),
                SType::SColl(SType::STypeVar(STypeVar::t()).into())

            ],
            SType::SColl(SType::STypeVar(STypeVar::t()).into())
        ),
        explicit_type_args: vec![],
        min_version: ErgoTreeVersion::V0
    };
    /// Coll.updateMany
    pub static ref UPDATE_MANY_METHOD: SMethod = SMethod::new(STypeCompanion::Coll, UPDATE_MANY_METHOD_DESC.clone());
}

lazy_static! {
    static ref REVERSE_METHOD_DESC: SMethodDesc = SMethodDesc {
        method_id: REVERSE_METHOD_ID,
        name: "reverse",
        tpe: SFunc::new(
            vec![SType::SColl(SType::STypeVar(STypeVar::t()).into())],
            SType::SColl(SType::STypeVar(STypeVar::t()).into())
        ),
        explicit_type_args: vec![],
        min_version: ErgoTreeVersion::V3
    };
    /// Coll.reverse
    pub static ref REVERSE_METHOD: SMethod = SMethod::new(STypeCompanion::Coll, REVERSE_METHOD_DESC.clone());
}

lazy_static! {
    static ref STARTS_WITH_METHOD_DESC: SMethodDesc = SMethodDesc {
        method_id: STARTS_WITH_METHOD_ID,
        name: "startsWith",
        tpe: SFunc::new(
            vec![
                SType::SColl(SType::STypeVar(STypeVar::t()).into()),
                SType::SColl(SType::STypeVar(STypeVar::t()).into())
            ],
            SType::SBoolean
        ),
        explicit_type_args: vec![],
        min_version: ErgoTreeVersion::V3
    };
    /// Coll.startsWith
    pub static ref STARTS_WITH_METHOD: SMethod = SMethod::new(STypeCompanion::Coll, STARTS_WITH_METHOD_DESC.clone());
}

lazy_static! {
    static ref ENDS_WITH_METHOD_DESC: SMethodDesc = SMethodDesc {
        method_id: ENDS_WITH_METHOD_ID,
        name: "endsWith",
        tpe: SFunc::new(
            vec![
                SType::SColl(SType::STypeVar(STypeVar::t()).into()),
                SType::SColl(SType::STypeVar(STypeVar::t()).into())
            ],
            SType::SBoolean
        ),
        explicit_type_args: vec![],
        min_version: ErgoTreeVersion::V3
    };
    /// Coll.endsWith
    pub static ref ENDS_WITH_METHOD: SMethod = SMethod::new(STypeCompanion::Coll, ENDS_WITH_METHOD_DESC.clone());
}

lazy_static! {
    static ref GET_METHOD_DESC: SMethodDesc = SMethodDesc {
        method_id: GET_METHOD_ID,
        name: "get",
        tpe: SFunc::new(
            vec![
                SType::SColl(SType::STypeVar(STypeVar::t()).into()),
                SType::SInt
            ],
            SType::SOption(SType::STypeVar(STypeVar::t()).into())
        ),
        explicit_type_args: vec![],
        min_version: ErgoTreeVersion::V3
    };
    /// Coll.get
    pub static ref GET_METHOD: SMethod = SMethod::new(STypeCompanion::Coll, GET_METHOD_DESC.clone());
}

/// `IV`, a collection's item type (sigmastate's `tIV`)
fn iv() -> SType {
    STypeVar::iv().into()
}

/// `OV`, an output type (sigmastate's `tOV`)
fn ov() -> SType {
    STypeVar::ov().into()
}

/// `Coll[IV]`
fn coll_iv() -> SType {
    SType::SColl(iv().into())
}

/// One of the methods sigmastate's `SCollectionMethods` has at every tree version, with no
/// explicit type arguments (`methods.scala:821-945`)
fn v5_desc(
    method_id: MethodId,
    name: &'static str,
    t_dom: Vec<SType>,
    t_range: SType,
) -> SMethodDesc {
    SMethodDesc {
        method_id,
        name,
        tpe: SFunc::new(t_dom, t_range),
        explicit_type_args: vec![],
        min_version: ErgoTreeVersion::V0,
    }
}

lazy_static! {
    static ref SIZE_METHOD_DESC: SMethodDesc =
        v5_desc(SIZE_METHOD_ID, "size", vec![coll_iv()], SType::SInt);
    static ref GET_OR_ELSE_METHOD_DESC: SMethodDesc = v5_desc(
        GET_OR_ELSE_METHOD_ID,
        "getOrElse",
        vec![coll_iv(), SType::SInt, iv()],
        iv()
    );
    static ref MAP_METHOD_DESC: SMethodDesc = v5_desc(
        MAP_METHOD_ID,
        "map",
        vec![coll_iv(), SFunc::new(vec![iv()], ov()).into()],
        SType::SColl(ov().into())
    );
    static ref EXISTS_METHOD_DESC: SMethodDesc = v5_desc(
        EXISTS_METHOD_ID,
        "exists",
        vec![coll_iv(), SFunc::new(vec![iv()], SType::SBoolean).into()],
        SType::SBoolean
    );
    static ref FOLD_METHOD_DESC: SMethodDesc = v5_desc(
        FOLD_METHOD_ID,
        "fold",
        vec![coll_iv(), ov(), SFunc::new(vec![ov(), iv()], ov()).into()],
        ov()
    );
    static ref FORALL_METHOD_DESC: SMethodDesc = v5_desc(
        FORALL_METHOD_ID,
        "forall",
        vec![coll_iv(), SFunc::new(vec![iv()], SType::SBoolean).into()],
        SType::SBoolean
    );
    static ref SLICE_METHOD_DESC: SMethodDesc = v5_desc(
        SLICE_METHOD_ID,
        "slice",
        vec![coll_iv(), SType::SInt, SType::SInt],
        coll_iv()
    );
    static ref FILTER_METHOD_DESC: SMethodDesc = v5_desc(
        FILTER_METHOD_ID,
        "filter",
        vec![coll_iv(), SFunc::new(vec![iv()], SType::SBoolean).into()],
        coll_iv()
    );
    static ref APPEND_METHOD_DESC: SMethodDesc =
        v5_desc(APPEND_METHOD_ID, "append", vec![coll_iv(), coll_iv()], coll_iv());
    static ref APPLY_METHOD_DESC: SMethodDesc =
        v5_desc(APPLY_METHOD_ID, "apply", vec![coll_iv(), SType::SInt], iv());
    /// Coll.size
    pub static ref SIZE_METHOD: SMethod = SMethod::new(STypeCompanion::Coll, SIZE_METHOD_DESC.clone());
    /// Coll.getOrElse
    pub static ref GET_OR_ELSE_METHOD: SMethod =
        SMethod::new(STypeCompanion::Coll, GET_OR_ELSE_METHOD_DESC.clone());
    /// Coll.map
    pub static ref MAP_METHOD: SMethod = SMethod::new(STypeCompanion::Coll, MAP_METHOD_DESC.clone());
    /// Coll.exists
    pub static ref EXISTS_METHOD: SMethod = SMethod::new(STypeCompanion::Coll, EXISTS_METHOD_DESC.clone());
    /// Coll.fold
    pub static ref FOLD_METHOD: SMethod = SMethod::new(STypeCompanion::Coll, FOLD_METHOD_DESC.clone());
    /// Coll.forall
    pub static ref FORALL_METHOD: SMethod = SMethod::new(STypeCompanion::Coll, FORALL_METHOD_DESC.clone());
    /// Coll.slice
    pub static ref SLICE_METHOD: SMethod = SMethod::new(STypeCompanion::Coll, SLICE_METHOD_DESC.clone());
    /// Coll.filter
    pub static ref FILTER_METHOD: SMethod = SMethod::new(STypeCompanion::Coll, FILTER_METHOD_DESC.clone());
    /// Coll.append
    pub static ref APPEND_METHOD: SMethod = SMethod::new(STypeCompanion::Coll, APPEND_METHOD_DESC.clone());
    /// Coll.apply
    pub static ref APPLY_METHOD: SMethod = SMethod::new(STypeCompanion::Coll, APPLY_METHOD_DESC.clone());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_ids() {
        assert!(
            SMethod::from_ids(TYPE_CODE, INDEX_OF_METHOD_ID).map(|e| e.name()) == Ok("indexOf")
        );
        assert!(SMethod::from_ids(TYPE_CODE, FLATMAP_METHOD_ID).map(|e| e.name()) == Ok("flatMap"));
        assert!(SMethod::from_ids(TYPE_CODE, ZIP_METHOD_ID).map(|e| e.name()) == Ok("zip"));
        assert!(SMethod::from_ids(TYPE_CODE, INDICES_METHOD_ID).map(|e| e.name()) == Ok("indices"));
        assert!(SMethod::from_ids(TYPE_CODE, PATCH_METHOD_ID).map(|e| e.name()) == Ok("patch"));
        assert!(SMethod::from_ids(TYPE_CODE, UPDATED_METHOD_ID).map(|e| e.name()) == Ok("updated"));
        assert!(
            SMethod::from_ids(TYPE_CODE, UPDATE_MANY_METHOD_ID).map(|e| e.name())
                == Ok("updateMany")
        );
        assert!(SMethod::from_ids(TYPE_CODE, REVERSE_METHOD_ID).map(|e| e.name()) == Ok("reverse"));
        assert!(
            SMethod::from_ids(TYPE_CODE, STARTS_WITH_METHOD_ID).map(|e| e.name())
                == Ok("startsWith")
        );
        assert!(
            SMethod::from_ids(TYPE_CODE, ENDS_WITH_METHOD_ID).map(|e| e.name()) == Ok("endsWith")
        );
        assert!(SMethod::from_ids(TYPE_CODE, GET_METHOD_ID).map(|e| e.name()) == Ok("get"));
        for (id, name) in [
            (SIZE_METHOD_ID, "size"),
            (GET_OR_ELSE_METHOD_ID, "getOrElse"),
            (MAP_METHOD_ID, "map"),
            (EXISTS_METHOD_ID, "exists"),
            (FOLD_METHOD_ID, "fold"),
            (FORALL_METHOD_ID, "forall"),
            (SLICE_METHOD_ID, "slice"),
            (FILTER_METHOD_ID, "filter"),
            (APPEND_METHOD_ID, "append"),
            (APPLY_METHOD_ID, "apply"),
        ] {
            assert_eq!(SMethod::from_ids(TYPE_CODE, id).map(|e| e.name()), Ok(name));
        }
    }
}
