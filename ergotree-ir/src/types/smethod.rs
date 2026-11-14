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
use super::stype_param::STypeParam;
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

    /// The static type of a call of this (specialised) method: sigmastate's `MethodCall.tpe`
    /// (v6.0.7 `values.scala:1355-1358`), the method's range with the call's explicit type
    /// arguments substituted ONCE MORE. The method already holds them (the parser's
    /// `withConcreteTypes`), so the second pass changes the range only where the range still
    /// names the method's type variable `T`: where a type argument names it itself
    /// (`getReg[Coll[T]]` has the type `Option[Coll[Coll[T]]]`), or where `specializeFor`
    /// brought it in with the type of `Global.some`'s argument (`some[U]({(x: T) => 1})` has
    /// the type `Option[(U) => Int]`). A function type there whose type parameter is that
    /// variable loses the parameter or makes the substitution fail (`SType::with_subst`).
    /// sigmastate computes it when it builds the call, so a failure is a failure to parse
    /// (`try_call_tpe`); for a call put together field by field whose type
    /// sigmastate could not build, this is the method's range.
    pub fn call_tpe(&self, explicit_type_args: &HashMap<STypeVar, SType>) -> SType {
        self.try_call_tpe(explicit_type_args)
            .unwrap_or_else(|_| *self.tpe().t_range.clone())
    }

    /// [`Self::call_tpe`] where sigmastate builds the call: an error where its substitution
    /// throws
    pub(crate) fn try_call_tpe(
        &self,
        explicit_type_args: &HashMap<STypeVar, SType>,
    ) -> Result<SType, TypeSubstError> {
        let range = &*self.tpe().t_range;
        // `withSubstTypes` gives the type as it is for an empty substitution (v6.0.7
        // `sigma/ast/package.scala:166-169`): a call with no explicit type argument, which
        // is nearly every call, and its type is read whenever a node checks its value
        if explicit_type_args.is_empty() {
            return Ok(range.clone());
        }
        range.with_subst(explicit_type_args)
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
    /// The method's own type parameters: the third argument of the `SFunc` sigmastate gives a
    /// method (`paramIVSeq`, `Array(paramT)`, ...), in sigmastate's order. A substitution drops
    /// the ones it names (`SFunc::with_subst`), as sigmastate's `applySubst` does
    pub(crate) fn with_tpe_params(mut self, vars: Vec<STypeVar>) -> Self {
        self.tpe.tpe_params = vars.into_iter().map(|ident| STypeParam { ident }).collect();
        self
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

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod signature_tests {
    use super::*;
    use crate::serialization::sigma_byte_reader::from_bytes;
    use crate::serialization::SigmaSerializable;
    use alloc::string::String;

    // Every method sigmastate's parser resolves (`SMethod.fromIds`, v6.0.7 `SMethod.scala:344-349`)
    // with its whole signature, read from the 6.0.7 jars over every type byte, method byte and
    // tree version: the type id, the method id, the name, the first tree version that resolves
    // it, the type parameters, the domain's types and the range as the type serializer writes
    // them at tree version 3, and the explicit type arguments; the comment is sigmastate's text
    // of the signature. A method's signature does not depend on the version, only whether it
    // resolves. Generated by `Sigs4.java` of queue item 29's measurements.
    type Row = (
        u8,
        u8,
        &'static str,
        u8,
        &'static str,
        &'static str,
        &'static str,
        &'static str,
    );
    #[rustfmt::skip]
    const JVM: &[Row] = &[
        (2, 1, "toByte", 3, "", "02", "02", ""), // (SByte$) => SByte$
        (2, 2, "toShort", 3, "", "02", "03", ""), // (SByte$) => SShort$
        (2, 3, "toInt", 3, "", "02", "04", ""), // (SByte$) => SInt$
        (2, 4, "toLong", 3, "", "02", "05", ""), // (SByte$) => SLong$
        (2, 5, "toBigInt", 3, "", "02", "06", ""), // (SByte$) => SBigInt$
        (2, 6, "toBytes", 3, "", "02", "0e", ""), // (SByte$) => Coll[SByte$]
        (2, 7, "toBits", 3, "", "02", "0d", ""), // (SByte$) => Coll[SBoolean]
        (2, 8, "bitwiseInverse", 3, "", "02", "02", ""), // (SByte$) => SByte$
        (2, 9, "bitwiseOr", 3, "", "02 02", "02", ""), // (SByte$,SByte$) => SByte$
        (2, 10, "bitwiseAnd", 3, "", "02 02", "02", ""), // (SByte$,SByte$) => SByte$
        (2, 11, "bitwiseXor", 3, "", "02 02", "02", ""), // (SByte$,SByte$) => SByte$
        (2, 12, "shiftLeft", 3, "", "02 04", "02", ""), // (SByte$,SInt$) => SByte$
        (2, 13, "shiftRight", 3, "", "02 04", "02", ""), // (SByte$,SInt$) => SByte$
        (3, 1, "toByte", 3, "", "03", "02", ""), // (SShort$) => SByte$
        (3, 2, "toShort", 3, "", "03", "03", ""), // (SShort$) => SShort$
        (3, 3, "toInt", 3, "", "03", "04", ""), // (SShort$) => SInt$
        (3, 4, "toLong", 3, "", "03", "05", ""), // (SShort$) => SLong$
        (3, 5, "toBigInt", 3, "", "03", "06", ""), // (SShort$) => SBigInt$
        (3, 6, "toBytes", 3, "", "03", "0e", ""), // (SShort$) => Coll[SByte$]
        (3, 7, "toBits", 3, "", "03", "0d", ""), // (SShort$) => Coll[SBoolean]
        (3, 8, "bitwiseInverse", 3, "", "03", "03", ""), // (SShort$) => SShort$
        (3, 9, "bitwiseOr", 3, "", "03 03", "03", ""), // (SShort$,SShort$) => SShort$
        (3, 10, "bitwiseAnd", 3, "", "03 03", "03", ""), // (SShort$,SShort$) => SShort$
        (3, 11, "bitwiseXor", 3, "", "03 03", "03", ""), // (SShort$,SShort$) => SShort$
        (3, 12, "shiftLeft", 3, "", "03 04", "03", ""), // (SShort$,SInt$) => SShort$
        (3, 13, "shiftRight", 3, "", "03 04", "03", ""), // (SShort$,SInt$) => SShort$
        (4, 1, "toByte", 3, "", "04", "02", ""), // (SInt$) => SByte$
        (4, 2, "toShort", 3, "", "04", "03", ""), // (SInt$) => SShort$
        (4, 3, "toInt", 3, "", "04", "04", ""), // (SInt$) => SInt$
        (4, 4, "toLong", 3, "", "04", "05", ""), // (SInt$) => SLong$
        (4, 5, "toBigInt", 3, "", "04", "06", ""), // (SInt$) => SBigInt$
        (4, 6, "toBytes", 3, "", "04", "0e", ""), // (SInt$) => Coll[SByte$]
        (4, 7, "toBits", 3, "", "04", "0d", ""), // (SInt$) => Coll[SBoolean]
        (4, 8, "bitwiseInverse", 3, "", "04", "04", ""), // (SInt$) => SInt$
        (4, 9, "bitwiseOr", 3, "", "04 04", "04", ""), // (SInt$,SInt$) => SInt$
        (4, 10, "bitwiseAnd", 3, "", "04 04", "04", ""), // (SInt$,SInt$) => SInt$
        (4, 11, "bitwiseXor", 3, "", "04 04", "04", ""), // (SInt$,SInt$) => SInt$
        (4, 12, "shiftLeft", 3, "", "04 04", "04", ""), // (SInt$,SInt$) => SInt$
        (4, 13, "shiftRight", 3, "", "04 04", "04", ""), // (SInt$,SInt$) => SInt$
        (5, 1, "toByte", 3, "", "05", "02", ""), // (SLong$) => SByte$
        (5, 2, "toShort", 3, "", "05", "03", ""), // (SLong$) => SShort$
        (5, 3, "toInt", 3, "", "05", "04", ""), // (SLong$) => SInt$
        (5, 4, "toLong", 3, "", "05", "05", ""), // (SLong$) => SLong$
        (5, 5, "toBigInt", 3, "", "05", "06", ""), // (SLong$) => SBigInt$
        (5, 6, "toBytes", 3, "", "05", "0e", ""), // (SLong$) => Coll[SByte$]
        (5, 7, "toBits", 3, "", "05", "0d", ""), // (SLong$) => Coll[SBoolean]
        (5, 8, "bitwiseInverse", 3, "", "05", "05", ""), // (SLong$) => SLong$
        (5, 9, "bitwiseOr", 3, "", "05 05", "05", ""), // (SLong$,SLong$) => SLong$
        (5, 10, "bitwiseAnd", 3, "", "05 05", "05", ""), // (SLong$,SLong$) => SLong$
        (5, 11, "bitwiseXor", 3, "", "05 05", "05", ""), // (SLong$,SLong$) => SLong$
        (5, 12, "shiftLeft", 3, "", "05 04", "05", ""), // (SLong$,SInt$) => SLong$
        (5, 13, "shiftRight", 3, "", "05 04", "05", ""), // (SLong$,SInt$) => SLong$
        (6, 1, "toByte", 3, "", "06", "02", ""), // (SBigInt$) => SByte$
        (6, 2, "toShort", 3, "", "06", "03", ""), // (SBigInt$) => SShort$
        (6, 3, "toInt", 3, "", "06", "04", ""), // (SBigInt$) => SInt$
        (6, 4, "toLong", 3, "", "06", "05", ""), // (SBigInt$) => SLong$
        (6, 5, "toBigInt", 3, "", "06", "06", ""), // (SBigInt$) => SBigInt$
        (6, 6, "toBytes", 3, "", "06", "0e", ""), // (SBigInt$) => Coll[SByte$]
        (6, 7, "toBits", 3, "", "06", "0d", ""), // (SBigInt$) => Coll[SBoolean]
        (6, 8, "bitwiseInverse", 3, "", "06", "06", ""), // (SBigInt$) => SBigInt$
        (6, 9, "bitwiseOr", 3, "", "06 06", "06", ""), // (SBigInt$,SBigInt$) => SBigInt$
        (6, 10, "bitwiseAnd", 3, "", "06 06", "06", ""), // (SBigInt$,SBigInt$) => SBigInt$
        (6, 11, "bitwiseXor", 3, "", "06 06", "06", ""), // (SBigInt$,SBigInt$) => SBigInt$
        (6, 12, "shiftLeft", 3, "", "06 04", "06", ""), // (SBigInt$,SInt$) => SBigInt$
        (6, 13, "shiftRight", 3, "", "06 04", "06", ""), // (SBigInt$,SInt$) => SBigInt$
        (6, 14, "toUnsigned", 3, "", "06", "09", ""), // (SBigInt$) => SUnsignedBigInt$
        (6, 15, "toUnsignedMod", 3, "", "06 09", "09", ""), // (SBigInt$,SUnsignedBigInt$) => SUnsignedBigInt$
        (7, 2, "getEncoded", 0, "", "07", "0e", ""), // (SGroupElement) => Coll[SByte$]
        (7, 3, "exp", 0, "", "07 06", "07", ""), // (SGroupElement,SBigInt$) => SGroupElement
        (7, 4, "multiply", 0, "", "07 07", "07", ""), // (SGroupElement,SGroupElement) => SGroupElement
        (7, 5, "negate", 0, "", "07", "07", ""), // (SGroupElement) => SGroupElement
        (7, 6, "expUnsigned", 3, "", "07 09", "07", ""), // (SGroupElement,SUnsignedBigInt$) => SGroupElement
        (8, 1, "propBytes", 0, "", "08", "0e", ""), // (SSigmaProp) => Coll[SByte$]
        (8, 2, "isProven", 0, "", "08", "01", ""), // (SSigmaProp) => SBoolean
        (9, 1, "toByte", 3, "", "09", "02", ""), // (SUnsignedBigInt$) => SByte$
        (9, 2, "toShort", 3, "", "09", "03", ""), // (SUnsignedBigInt$) => SShort$
        (9, 3, "toInt", 3, "", "09", "04", ""), // (SUnsignedBigInt$) => SInt$
        (9, 4, "toLong", 3, "", "09", "05", ""), // (SUnsignedBigInt$) => SLong$
        (9, 5, "toBigInt", 3, "", "09", "06", ""), // (SUnsignedBigInt$) => SBigInt$
        (9, 6, "toBytes", 3, "", "09", "0e", ""), // (SUnsignedBigInt$) => Coll[SByte$]
        (9, 7, "toBits", 3, "", "09", "0d", ""), // (SUnsignedBigInt$) => Coll[SBoolean]
        (9, 8, "bitwiseInverse", 3, "", "09", "09", ""), // (SUnsignedBigInt$) => SUnsignedBigInt$
        (9, 9, "bitwiseOr", 3, "", "09 09", "09", ""), // (SUnsignedBigInt$,SUnsignedBigInt$) => SUnsignedBigInt$
        (9, 10, "bitwiseAnd", 3, "", "09 09", "09", ""), // (SUnsignedBigInt$,SUnsignedBigInt$) => SUnsignedBigInt$
        (9, 11, "bitwiseXor", 3, "", "09 09", "09", ""), // (SUnsignedBigInt$,SUnsignedBigInt$) => SUnsignedBigInt$
        (9, 12, "shiftLeft", 3, "", "09 04", "09", ""), // (SUnsignedBigInt$,SInt$) => SUnsignedBigInt$
        (9, 13, "shiftRight", 3, "", "09 04", "09", ""), // (SUnsignedBigInt$,SInt$) => SUnsignedBigInt$
        (9, 14, "modInverse", 3, "", "09 09", "09", ""), // (SUnsignedBigInt$,SUnsignedBigInt$) => SUnsignedBigInt$
        (9, 15, "plusMod", 3, "", "09 09 09", "09", ""), // (SUnsignedBigInt$,SUnsignedBigInt$,SUnsignedBigInt$) => SUnsignedBigInt$
        (9, 16, "subtractMod", 3, "", "09 09 09", "09", ""), // (SUnsignedBigInt$,SUnsignedBigInt$,SUnsignedBigInt$) => SUnsignedBigInt$
        (9, 17, "multiplyMod", 3, "", "09 09 09", "09", ""), // (SUnsignedBigInt$,SUnsignedBigInt$,SUnsignedBigInt$) => SUnsignedBigInt$
        (9, 18, "mod", 3, "", "09 09", "09", ""), // (SUnsignedBigInt$,SUnsignedBigInt$) => SUnsignedBigInt$
        (9, 19, "toSigned", 3, "", "09", "06", ""), // (SUnsignedBigInt$) => SBigInt$
        (12, 1, "size", 0, "", "0c67024956", "04", ""), // (Coll[IV]) => SInt$
        (12, 2, "getOrElse", 0, "IV", "0c67024956 04 67024956", "67024956", ""), // [IV](Coll[IV],SInt$,IV) => IV
        (12, 3, "map", 0, "IV,OV", "0c67024956 70016702495667024f5600", "0c67024f56", ""), // [IV,OV](Coll[IV],(IV) => OV) => Coll[OV]
        (12, 4, "exists", 0, "IV", "0c67024956 7001670249560100", "01", ""), // [IV](Coll[IV],(IV) => SBoolean) => SBoolean
        (12, 5, "fold", 0, "IV,OV", "0c67024956 67024f56 700267024f566702495667024f5600", "67024f56", ""), // [IV,OV](Coll[IV],OV,(OV,IV) => OV) => OV
        (12, 6, "forall", 0, "IV", "0c67024956 7001670249560100", "01", ""), // [IV](Coll[IV],(IV) => SBoolean) => SBoolean
        (12, 7, "slice", 0, "IV", "0c67024956 04 04", "0c67024956", ""), // [IV](Coll[IV],SInt$,SInt$) => Coll[IV]
        (12, 8, "filter", 0, "IV", "0c67024956 7001670249560100", "0c67024956", ""), // [IV](Coll[IV],(IV) => SBoolean) => Coll[IV]
        (12, 9, "append", 0, "IV", "0c67024956 0c67024956", "0c67024956", ""), // [IV](Coll[IV],Coll[IV]) => Coll[IV]
        (12, 10, "apply", 0, "IV", "0c67024956 04", "67024956", ""), // [IV](Coll[IV],SInt$) => IV
        (12, 14, "indices", 0, "", "0c67024956", "10", ""), // (Coll[IV]) => Coll[SInt$]
        (12, 15, "flatMap", 0, "IV,OV", "0c67024956 7001670249560c67024f5600", "0c67024f56", ""), // [IV,OV](Coll[IV],(IV) => Coll[OV]) => Coll[OV]
        (12, 19, "patch", 0, "IV", "0c67024956 04 0c67024956 04", "0c67024956", ""), // [IV](Coll[IV],SInt$,Coll[IV],SInt$) => Coll[IV]
        (12, 20, "updated", 0, "IV", "0c67024956 04 67024956", "0c67024956", ""), // [IV](Coll[IV],SInt$,IV) => Coll[IV]
        (12, 21, "updateMany", 0, "IV", "0c67024956 10 0c67024956", "0c67024956", ""), // [IV](Coll[IV],Coll[SInt$],Coll[IV]) => Coll[IV]
        (12, 26, "indexOf", 0, "IV", "0c67024956 67024956 04", "04", ""), // [IV](Coll[IV],IV,SInt$) => SInt$
        (12, 29, "zip", 0, "IV,OV", "0c67024956 0c67024f56", "0c3c6702495667024f56", ""), // [IV,OV](Coll[IV],Coll[OV]) => Coll[(IV,OV)]
        (12, 30, "reverse", 3, "IV", "0c67024956", "0c67024956", ""), // [IV](Coll[IV]) => Coll[IV]
        (12, 31, "startsWith", 3, "IV", "0c67024956 0c67024956", "01", ""), // [IV](Coll[IV],Coll[IV]) => SBoolean
        (12, 32, "endsWith", 3, "IV", "0c67024956 0c67024956", "01", ""), // [IV](Coll[IV],Coll[IV]) => SBoolean
        (12, 33, "get", 3, "IV", "0c67024956 04", "2467024956", ""), // [IV](Coll[IV],SInt$) => Option[IV]
        (36, 2, "isDefined", 0, "", "24670154", "01", ""), // (Option[T]) => SBoolean
        (36, 3, "get", 0, "", "24670154", "670154", ""), // (Option[T]) => T
        (36, 4, "getOrElse", 0, "T", "24670154 670154", "670154", ""), // [T](Option[T],T) => T
        (36, 7, "map", 0, "T,R", "24670154 700167015467015200", "24670152", ""), // [T,R](Option[T],(T) => R) => Option[R]
        (36, 8, "filter", 0, "T", "24670154 70016701540100", "24670154", ""), // [T](Option[T],(T) => SBoolean) => Option[T]
        (99, 1, "value", 0, "", "63", "05", ""), // (SBox) => SLong$
        (99, 2, "propositionBytes", 0, "", "63", "0e", ""), // (SBox) => Coll[SByte$]
        (99, 3, "bytes", 0, "", "63", "0e", ""), // (SBox) => Coll[SByte$]
        (99, 4, "bytesWithoutRef", 0, "", "63", "0e", ""), // (SBox) => Coll[SByte$]
        (99, 5, "id", 0, "", "63", "0e", ""), // (SBox) => Coll[SByte$]
        (99, 6, "creationInfo", 0, "", "63", "400e", ""), // (SBox) => (SInt$,Coll[SByte$])
        (99, 7, "getRegV5", 0, "T", "63 04", "24670154", ""), // [T](SBox,SInt$) => Option[T]
        (99, 8, "tokens", 0, "", "63", "0c4d0e", ""), // (SBox) => Coll[(Coll[SByte$],SLong$)]
        (99, 9, "R0", 0, "T", "63", "24670154", ""), // [T](SBox) => Option[T]
        (99, 10, "R1", 0, "T", "63", "24670154", ""), // [T](SBox) => Option[T]
        (99, 11, "R2", 0, "T", "63", "24670154", ""), // [T](SBox) => Option[T]
        (99, 12, "R3", 0, "T", "63", "24670154", ""), // [T](SBox) => Option[T]
        (99, 13, "R4", 0, "T", "63", "24670154", ""), // [T](SBox) => Option[T]
        (99, 14, "R5", 0, "T", "63", "24670154", ""), // [T](SBox) => Option[T]
        (99, 15, "R6", 0, "T", "63", "24670154", ""), // [T](SBox) => Option[T]
        (99, 16, "R7", 0, "T", "63", "24670154", ""), // [T](SBox) => Option[T]
        (99, 17, "R8", 0, "T", "63", "24670154", ""), // [T](SBox) => Option[T]
        (99, 18, "R9", 0, "T", "63", "24670154", ""), // [T](SBox) => Option[T]
        (99, 19, "getReg", 3, "T", "63 04", "24670154", "T"), // [T](SBox,SInt$) => Option[T]
        (100, 1, "digest", 0, "", "64", "0e", ""), // (SAvlTree) => Coll[SByte$]
        (100, 2, "enabledOperations", 0, "", "64", "02", ""), // (SAvlTree) => SByte$
        (100, 3, "keyLength", 0, "", "64", "04", ""), // (SAvlTree) => SInt$
        (100, 4, "valueLengthOpt", 0, "", "64", "28", ""), // (SAvlTree) => Option[SInt$]
        (100, 5, "isInsertAllowed", 0, "", "64", "01", ""), // (SAvlTree) => SBoolean
        (100, 6, "isUpdateAllowed", 0, "", "64", "01", ""), // (SAvlTree) => SBoolean
        (100, 7, "isRemoveAllowed", 0, "", "64", "01", ""), // (SAvlTree) => SBoolean
        (100, 8, "updateOperations", 0, "", "64 02", "64", ""), // (SAvlTree,SByte$) => SAvlTree
        (100, 9, "contains", 0, "", "64 0e 0e", "01", ""), // (SAvlTree,Coll[SByte$],Coll[SByte$]) => SBoolean
        (100, 10, "get", 0, "", "64 0e 0e", "32", ""), // (SAvlTree,Coll[SByte$],Coll[SByte$]) => Option[Coll[SByte$]]
        (100, 11, "getMany", 0, "", "64 1a 0e", "0c32", ""), // (SAvlTree,Coll[Coll[SByte$]],Coll[SByte$]) => Coll[Option[Coll[SByte$]]]
        (100, 12, "insert", 0, "", "64 0c3c0e0e 0e", "2464", ""), // (SAvlTree,Coll[(Coll[SByte$],Coll[SByte$])],Coll[SByte$]) => Option[SAvlTree]
        (100, 13, "update", 0, "", "64 0c3c0e0e 0e", "2464", ""), // (SAvlTree,Coll[(Coll[SByte$],Coll[SByte$])],Coll[SByte$]) => Option[SAvlTree]
        (100, 14, "remove", 0, "", "64 1a 0e", "2464", ""), // (SAvlTree,Coll[Coll[SByte$]],Coll[SByte$]) => Option[SAvlTree]
        (100, 15, "updateDigest", 0, "", "64 0e", "64", ""), // (SAvlTree,Coll[SByte$]) => SAvlTree
        (100, 16, "insertOrUpdate", 3, "", "64 0c3c0e0e 0e", "2464", ""), // (SAvlTree,Coll[(Coll[SByte$],Coll[SByte$])],Coll[SByte$]) => Option[SAvlTree]
        (101, 1, "dataInputs", 0, "", "65", "0c63", ""), // (SContext) => Coll[SBox]
        (101, 2, "headers", 0, "", "65", "0c68", ""), // (SContext) => Coll[SHeader]
        (101, 3, "preHeader", 0, "", "65", "69", ""), // (SContext) => SPreHeader
        (101, 4, "INPUTS", 0, "", "65", "0c63", ""), // (SContext) => Coll[SBox]
        (101, 5, "OUTPUTS", 0, "", "65", "0c63", ""), // (SContext) => Coll[SBox]
        (101, 6, "HEIGHT", 0, "", "65", "04", ""), // (SContext) => SInt$
        (101, 7, "SELF", 0, "", "65", "63", ""), // (SContext) => SBox
        (101, 8, "selfBoxIndex", 0, "", "65", "04", ""), // (SContext) => SInt$
        (101, 9, "LastBlockUtxoRootHash", 0, "", "65", "64", ""), // (SContext) => SAvlTree
        (101, 10, "minerPubKey", 0, "", "65", "0e", ""), // (SContext) => Coll[SByte$]
        (101, 11, "getVar", 0, "T", "65 02", "24670154", ""), // [T](SContext,SByte$) => Option[T]
        (101, 12, "getVarFromInput", 3, "T", "65 03 02", "24670154", "T"), // [T](SContext,SShort$,SByte$) => Option[T]
        (104, 1, "id", 0, "", "68", "0e", ""), // (SHeader) => Coll[SByte$]
        (104, 2, "version", 0, "", "68", "02", ""), // (SHeader) => SByte$
        (104, 3, "parentId", 0, "", "68", "0e", ""), // (SHeader) => Coll[SByte$]
        (104, 4, "ADProofsRoot", 0, "", "68", "0e", ""), // (SHeader) => Coll[SByte$]
        (104, 5, "stateRoot", 0, "", "68", "64", ""), // (SHeader) => SAvlTree
        (104, 6, "transactionsRoot", 0, "", "68", "0e", ""), // (SHeader) => Coll[SByte$]
        (104, 7, "timestamp", 0, "", "68", "05", ""), // (SHeader) => SLong$
        (104, 8, "nBits", 0, "", "68", "05", ""), // (SHeader) => SLong$
        (104, 9, "height", 0, "", "68", "04", ""), // (SHeader) => SInt$
        (104, 10, "extensionRoot", 0, "", "68", "0e", ""), // (SHeader) => Coll[SByte$]
        (104, 11, "minerPk", 0, "", "68", "07", ""), // (SHeader) => SGroupElement
        (104, 12, "powOnetimePk", 0, "", "68", "07", ""), // (SHeader) => SGroupElement
        (104, 13, "powNonce", 0, "", "68", "0e", ""), // (SHeader) => Coll[SByte$]
        (104, 14, "powDistance", 0, "", "68", "06", ""), // (SHeader) => SBigInt$
        (104, 15, "votes", 0, "", "68", "0e", ""), // (SHeader) => Coll[SByte$]
        (104, 16, "checkPow", 3, "", "68", "01", ""), // (SHeader) => SBoolean
        (105, 1, "version", 0, "", "69", "02", ""), // (SPreHeader) => SByte$
        (105, 2, "parentId", 0, "", "69", "0e", ""), // (SPreHeader) => Coll[SByte$]
        (105, 3, "timestamp", 0, "", "69", "05", ""), // (SPreHeader) => SLong$
        (105, 4, "nBits", 0, "", "69", "05", ""), // (SPreHeader) => SLong$
        (105, 5, "height", 0, "", "69", "04", ""), // (SPreHeader) => SInt$
        (105, 6, "minerPk", 0, "", "69", "07", ""), // (SPreHeader) => SGroupElement
        (105, 7, "votes", 0, "", "69", "0e", ""), // (SPreHeader) => Coll[SByte$]
        (106, 1, "groupGenerator", 0, "", "6a", "07", ""), // (SGlobal) => SGroupElement
        (106, 2, "xor", 0, "", "6a 0e 0e", "0e", ""), // (SGlobal,Coll[SByte$],Coll[SByte$]) => Coll[SByte$]
        (106, 3, "serialize", 3, "T", "6a 670154", "0e", ""), // [T](SGlobal,T) => Coll[SByte$]
        (106, 4, "deserializeTo", 3, "T", "6a 0e", "670154", "T"), // [T](SGlobal,Coll[SByte$]) => T
        (106, 5, "fromBigEndianBytes", 3, "T", "6a 0e", "670154", "T"), // [T](SGlobal,Coll[SByte$]) => T
        (106, 6, "encodeNbits", 3, "", "6a 06", "05", ""), // (SGlobal,SBigInt$) => SLong$
        (106, 7, "decodeNbits", 3, "", "6a 05", "06", ""), // (SGlobal,SLong$) => SBigInt$
        (106, 8, "powHit", 3, "", "6a 04 0e 0e 0e 04", "09", ""), // (SGlobal,SInt$,Coll[SByte$],Coll[SByte$],Coll[SByte$],SInt$) => SUnsignedBigInt$
        (106, 9, "some", 3, "T", "6a 670154", "24670154", "T"), // [T](SGlobal,T) => Option[T]
        (106, 10, "none", 3, "T", "6a", "24670154", "T"), // [T](SGlobal) => Option[T]
    ];

    fn written(tpe: &SType) -> String {
        let bytes = from_bytes(&[0u8; 0])
            .with_tree_version(ErgoTreeVersion::V3, |_| {
                Ok::<_, SigmaParsingError>(tpe.sigma_serialize_bytes().unwrap())
            })
            .unwrap();
        base16::encode_lower(&bytes)
    }

    fn names<'a>(vars: impl Iterator<Item = &'a STypeVar>) -> String {
        vars.map(|v| v.as_string()).collect::<Vec<_>>().join(",")
    }

    #[test]
    fn the_signatures_are_the_jvms() {
        assert_eq!(JVM.len(), 199);
        for (type_id, method_id, name, first, tpe_params, t_dom, t_range, explicit) in JVM {
            let at = alloc::format!("{type_id} {method_id} {name}");
            let m = SMethod::from_type_byte(*type_id, MethodId(*method_id)).unwrap();
            assert_eq!(m.name(), *name, "{at}");
            assert_eq!(m.obj_type.type_code().value(), *type_id, "{at}");
            let first = match first {
                0 => ErgoTreeVersion::V0,
                _ => ErgoTreeVersion::V3,
            };
            assert_eq!(m.method_raw.min_version, first, "{at}");
            let tpe = m.tpe();
            assert_eq!(
                names(tpe.tpe_params.iter().map(|p| &p.ident)),
                *tpe_params,
                "{at}"
            );
            let dom: Vec<String> = tpe.t_dom.iter().map(written).collect();
            assert_eq!(dom.join(" "), *t_dom, "{at}");
            assert_eq!(written(&tpe.t_range), *t_range, "{at}");
            assert_eq!(
                names(m.method_raw.explicit_type_args.iter()),
                *explicit,
                "{at}"
            );
        }
    }

    // The lookup resolves the table's methods and no other, at any type byte and method byte
    #[test]
    fn the_lookup_resolves_the_jvms_methods_only() {
        let mut resolved = 0;
        for type_id in 0..=255u8 {
            for method_id in 0..=255u8 {
                if SMethod::from_type_byte(type_id, MethodId(method_id)).is_ok() {
                    resolved += 1;
                    assert!(
                        JVM.iter().any(|r| r.0 == type_id && r.1 == method_id),
                        "{type_id} {method_id}"
                    );
                }
            }
        }
        assert_eq!(resolved, JVM.len());
    }
}
