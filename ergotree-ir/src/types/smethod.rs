use alloc::format;
use alloc::vec;
use alloc::vec::Vec;

use crate::ergo_tree::ErgoTreeVersion;
use crate::serialization::sigma_byte_reader::SigmaByteRead;
use crate::serialization::sigma_byte_writer::SigmaByteWrite;
use crate::serialization::types::TypeCode;
use crate::serialization::SigmaParsingError;
use core::convert::TryFrom;
use hashbrown::HashMap;

use super::sfunc::SFunc;
use super::stype::SType;
use super::stype::TypeSubstError;
use super::stype_companion::STypeCompanion;
use super::stype_param::STypeVar;
use super::type_unify::unify_many;
use crate::serialization::SigmaParsingError::UnknownMethodId;

/// Method id unique among the methods of the same object
#[derive(PartialEq, Eq, Debug, Copy, Clone)]
pub struct MethodId(pub u8);

impl MethodId {
    pub(crate) fn sigma_serialize<W: SigmaByteWrite>(&self, w: &mut W) -> core3::io::Result<()> {
        w.put_u8(self.0)
    }

    pub(crate) fn sigma_parse<R: SigmaByteRead>(r: &mut R) -> core3::io::Result<Self> {
        Ok(Self(r.get_u8()?))
    }
}

/// Object method signature
#[derive(PartialEq, Eq, Debug, Clone)]
pub struct SMethod {
    /// Object type companion
    pub obj_type: STypeCompanion,
    pub(crate) method_raw: SMethodDesc,
}

impl SMethod {
    /// Create new SMethod
    pub const fn new(obj_type: STypeCompanion, method_raw: SMethodDesc) -> SMethod {
        SMethod {
            obj_type,
            method_raw,
        }
    }

    /// [`Self::from_ids`] over a type id as sigmastate reads it, a raw byte. `SMethod.fromIds`
    /// checks rule 1010 `CheckTypeWithMethods` first (v6.0.7 `SMethod.scala:344-349`), which a
    /// byte that is no type code fails as a type code without methods does.
    pub(crate) fn from_type_byte(
        type_id: u8,
        method_id: MethodId,
    ) -> Result<Self, SigmaParsingError> {
        match <TypeCode as num_traits::FromPrimitive>::from_u8(type_id) {
            Some(type_code) => Self::from_ids(type_code, method_id),
            None => Err(SigmaParsingError::NotImplementedYet(format!(
                "cannot find STypeCompanion for type id {type_id}"
            ))),
        }
    }

    /// Get method from type and method ids
    pub(crate) fn from_ids(
        type_id: TypeCode,
        method_id: MethodId,
    ) -> Result<Self, SigmaParsingError> {
        let obj_type = STypeCompanion::try_from(type_id)?;
        match obj_type.method_by_id(&method_id) {
            Some(m) => Ok(m),
            None => Err(UnknownMethodId(method_id, type_id.value())),
        }
    }

    /// Type
    pub fn tpe(&self) -> &SFunc {
        &self.method_raw.tpe
    }

    /// Returns method name
    pub fn name(&self) -> &'static str {
        self.method_raw.name
    }

    /// Returns method id
    pub fn method_id(&self) -> MethodId {
        self.method_raw.method_id
    }

    /// Return new SMethod with type variables substituted: sigmastate's `withConcreteTypes`
    /// (v6.0.7 `SMethod.scala:177-178`), which fails where its substitution does
    /// (`SType::with_subst`)
    pub fn with_concrete_types(
        self,
        subst: &HashMap<STypeVar, SType>,
    ) -> Result<Self, TypeSubstError> {
        let new_tpe = self.method_raw.tpe.with_subst(subst)?;
        Ok(Self {
            method_raw: self.method_raw.with_tpe(new_tpe),
            ..self
        })
    }

    /// Specializes this instance by creating a new [`SMethod`] instance where signature
    /// is specialized with respect to the given object and args types: sigmastate's
    /// `specializeFor` (v6.0.7 `SMethod.scala:193-199`). The method comes back as it is when
    /// the types do not unify or unify with no substitution; a substitution is applied with
    /// [`Self::with_concrete_types`], which can fail.
    pub fn specialize_for(
        self,
        obj_tpe: SType,
        args: Vec<SType>,
    ) -> Result<SMethod, TypeSubstError> {
        let mut items2 = vec![obj_tpe];
        let mut args = args;
        items2.append(args.as_mut());
        match unify_many(self.tpe().t_dom.clone(), items2) {
            Ok(subst) if !subst.is_empty() => self.with_concrete_types(&subst),
            _ => Ok(self),
        }
    }
}

/// Object method description
#[derive(PartialEq, Eq, Debug, Clone)]
pub struct SMethodDesc {
    pub(crate) name: &'static str,
    pub(crate) method_id: MethodId,
    pub(crate) tpe: SFunc,
    // Typevars that cannot be inferred from arguments. For example Box.getReg[T](4), T can not be inferred thus must be explicitly serialized
    pub(crate) explicit_type_args: Vec<STypeVar>,
    pub(crate) min_version: ErgoTreeVersion,
}

impl SMethodDesc {
    /// Initialize property method description
    pub fn property(
        obj_tpe: SType,
        name: &'static str,
        res_tpe: SType,
        id: MethodId,
    ) -> SMethodDesc {
        SMethodDesc {
            method_id: id,
            name,
            tpe: SFunc {
                t_dom: vec![obj_tpe],
                t_range: res_tpe.into(),
                tpe_params: vec![],
            },
            explicit_type_args: vec![],
            min_version: ErgoTreeVersion::V0,
        }
    }
    pub(crate) fn as_method(&self, obj_type: STypeCompanion) -> SMethod {
        SMethod {
            obj_type,
            method_raw: self.clone(),
        }
    }

    pub(crate) fn with_tpe(self, tpe: SFunc) -> Self {
        Self { tpe, ..self }
    }
}
