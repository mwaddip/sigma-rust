use hashbrown::HashMap;

use alloc::boxed::Box;

use alloc::vec;

use crate::serialization::op_code::OpCode;
use crate::traversable::impl_traversable_expr;
use crate::types::smethod::SMethod;
use crate::types::stype::SType;
use crate::types::stype_param::STypeVar;

use super::expr::Expr;
use super::expr::InvalidArgumentError;
use crate::has_opcode::HasStaticOpCode;

/// Invocation of object's property
#[derive(PartialEq, Eq, Debug, Clone)]
pub struct PropertyCall {
    /// Object on which property will be invoked
    pub obj: Box<Expr>,
    /// Property to be invoked
    pub method: SMethod,
    /// Arguments that cannot be inferred from function signature, such as `Global.none[T]`
    pub explicit_type_args: HashMap<STypeVar, SType>,
}

impl PropertyCall {
    /// Create new object, returns an error if any of the requirements failed
    pub fn new(obj: Expr, method: SMethod) -> Result<Self, InvalidArgumentError> {
        PropertyCall::with_type_args(obj, method, Default::default())
    }

    /// Create new object with explicit type args, returns an error if any of the requirements
    /// failed. `method` is the method as its table has it, generic: the type arguments are
    /// substituted into it here, as the parser substitutes them
    pub fn with_type_args(
        obj: Expr,
        method: SMethod,
        type_args: HashMap<STypeVar, SType>,
    ) -> Result<Self, InvalidArgumentError> {
        if method.tpe().t_dom.len() != 1 {
            return Err(InvalidArgumentError(format!(
                "PropertyCall: expected method.t_dom to have 1 element, got {}",
                method.tpe().t_dom.len()
            )));
        }
        let expected_types = vec![obj.tpe()];
        if !method
            .tpe()
            .t_dom
            .iter()
            .zip(&expected_types)
            .all(|(expected, actual)| expected == actual)
        {
            return Err(InvalidArgumentError(format!(
                "PropertyCall: expected types {expected_types:?} do not match provided obj and args types {:?}",
                method.tpe().t_dom,
            )));
        }
        if method.method_raw.explicit_type_args.len() != type_args.len() {
            return Err(InvalidArgumentError(format!(
                "PropertyCall: expected explicit type args count {} does not match provided type args count {}",
                method.method_raw.explicit_type_args.len(),
                type_args.len()
            )));
        }
        if let Some(missing_tpe) = method
            .method_raw
            .explicit_type_args
            .iter()
            .find(|&tpe| !type_args.contains_key(tpe))
        {
            return Err(InvalidArgumentError(format!(
                "PropertyCall: explicit_type_args does not include substitution for STypeVar {missing_tpe:?}",
            )));
        }
        let method = method
            .with_concrete_types(&type_args)
            .map_err(|e| InvalidArgumentError(format!("PropertyCall: {e:?}")))?;
        // sigmastate computes the call's type when it builds the call (`val tpe`)
        method
            .try_call_tpe(&type_args)
            .map_err(|e| InvalidArgumentError(format!("PropertyCall: {e:?}")))?;
        Ok(Self {
            obj: obj.into(),
            method,
            explicit_type_args: type_args,
        })
    }

    /// Type: the call's, [`SMethod::call_tpe`]
    pub fn tpe(&self) -> SType {
        self.method.call_tpe(&self.explicit_type_args)
    }
}

impl HasStaticOpCode for PropertyCall {
    const OP_CODE: OpCode = OpCode::PROPERTY_CALL;
}

impl_traversable_expr!(PropertyCall, boxed obj);
