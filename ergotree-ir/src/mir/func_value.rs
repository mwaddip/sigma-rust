use alloc::boxed::Box;
use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;
use core::fmt;

use crate::has_opcode::HasStaticOpCode;
use crate::serialization::op_code::OpCode;
use crate::serialization::sigma_byte_reader::SigmaByteRead;
use crate::serialization::sigma_byte_writer::SigmaByteWrite;
use crate::serialization::SigmaParsingError;
use crate::serialization::SigmaSerializable;
use crate::serialization::SigmaSerializeResult;
use crate::traversable::Traversable;
use crate::types::sfunc::SFunc;
use crate::types::stype::SType;

use super::expr::Expr;
use super::val_def::ValId;
use super::value::Bindings;
use super::value::Lambda;

#[cfg(feature = "arbitrary")]
use proptest_derive::Arbitrary;

/// Argument parameter for the user-defined function [`FuncValue`]
#[derive(PartialEq, Eq, Debug, Clone)]
#[cfg_attr(feature = "arbitrary", derive(Arbitrary))]
pub struct FuncArg {
    /// Value id (defined with [`super::val_def::ValDef`])
    pub idx: ValId,
    /// Value type
    pub tpe: SType,
}

impl SigmaSerializable for FuncArg {
    fn sigma_serialize<W: SigmaByteWrite>(&self, w: &mut W) -> SigmaSerializeResult {
        self.idx.sigma_serialize(w)?;
        self.tpe.sigma_serialize(w)
    }

    fn sigma_parse<R: SigmaByteRead>(r: &mut R) -> Result<Self, SigmaParsingError> {
        let idx = ValId::sigma_parse(r)?;
        let tpe = SType::sigma_parse(r)?;
        Ok(FuncArg { idx, tpe })
    }
}

impl core::fmt::Display for FuncArg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "v{}: {}", self.idx, self.tpe)
    }
}

/// User-defined function
#[derive(PartialEq, Eq, Debug, Clone)]
pub struct FuncValue {
    args: Vec<FuncArg>,
    /// Shared with every lambda value evaluated from this node, as sigmastate's closure holds
    /// its `FuncValue`'s body
    body: Arc<Expr>,
    tpe: SType,
}

impl FuncValue {
    /// Create new object, returns an error if any of the requirements failed
    pub fn new(args: Vec<FuncArg>, body: Expr) -> Self {
        let t_dom = args.iter().map(|fa| fa.tpe.clone()).collect();
        let t_range = body.tpe();
        let tpe = SType::SFunc(
            SFunc {
                t_dom,
                t_range: Box::new(t_range),
                tpe_params: vec![],
            }
            .into(),
        );
        FuncValue {
            args,
            body: Arc::new(body),
            tpe,
        }
    }

    /// Recomputes the type from the arguments and the current body: sigmastate's `lazy val tpe`
    /// is computed afresh on a node it rebuilds, as the substitution at spend does
    pub(crate) fn rebuild(&mut self) {
        let body = core::mem::replace(&mut self.body, Arc::new(Expr::Const(true.into())));
        *self = FuncValue::new(core::mem::take(&mut self.args), Arc::unwrap_or_clone(body));
    }

    /// Function arguments
    pub fn args(&self) -> &[FuncArg] {
        self.args.as_ref()
    }

    /// Function body
    pub fn body(&self) -> &Expr {
        &self.body
    }

    /// Function body. A body a lambda value shares is copied first, so the lambda keeps its own
    pub fn body_mut(&mut self) -> &mut Expr {
        Arc::make_mut(&mut self.body)
    }

    /// The lambda value this function evaluates to over `captured`: it shares the body
    pub fn closure<'ctx>(&self, captured: Bindings<'ctx>) -> Lambda<'ctx> {
        Lambda {
            args: self.args.clone(),
            body: self.body.clone(),
            captured,
        }
    }

    /// Type
    pub fn tpe(&self) -> SType {
        self.tpe.clone()
    }
}

impl HasStaticOpCode for FuncValue {
    const OP_CODE: OpCode = OpCode::FUNC_VALUE;
}

impl SigmaSerializable for FuncValue {
    fn sigma_serialize<W: SigmaByteWrite>(&self, w: &mut W) -> SigmaSerializeResult {
        self.args.sigma_serialize(w)?;
        self.body.sigma_serialize(w)
    }

    fn sigma_parse<R: SigmaByteRead>(r: &mut R) -> Result<Self, SigmaParsingError> {
        let args = Vec::<FuncArg>::sigma_parse(r)?;
        args.iter()
            .for_each(|a| r.val_def_type_store().insert(a.idx, a.tpe.clone()));
        let body = Expr::sigma_parse(r)?;
        Ok(FuncValue::new(args, body))
    }
}

impl Traversable for FuncValue {
    type Item = Expr;

    fn children<'a>(&'a self) -> Box<dyn Iterator<Item = &'a Expr> + 'a> {
        Box::new(core::iter::once(&*self.body))
    }

    fn children_mut<'a>(&'a mut self) -> Box<dyn Iterator<Item = &'a mut Expr> + 'a> {
        Box::new(core::iter::once(self.body_mut()))
    }
}

#[cfg(test)]
#[cfg(feature = "arbitrary")]
#[allow(clippy::panic)]
mod tests {
    use crate::mir::expr::Expr;
    use crate::serialization::sigma_serialize_roundtrip;

    use proptest::collection::vec;
    use proptest::prelude::*;

    use super::*;

    #[test]
    fn rebuild_does_not_change_a_shared_body() {
        // a lambda value shares the node's body; changing a copy of the node must not reach it
        let fv = FuncValue::new(
            alloc::vec![FuncArg {
                idx: 1.into(),
                tpe: SType::SInt,
            }],
            Expr::Const(1i32.into()),
        );
        let mut changed = fv.clone();
        *changed.body_mut() = Expr::Const(2i32.into());
        let mut walked = fv.clone();
        if let Some(child) = crate::traversable::Traversable::children_mut(&mut walked).next() {
            *child = Expr::Const(3i32.into());
        }
        let mut rebuilt = fv.clone();
        rebuilt.rebuild();
        assert_eq!(fv.body(), &Expr::Const(1i32.into()));
        assert_eq!(changed.body(), &Expr::Const(2i32.into()));
        assert_eq!(walked.body(), &Expr::Const(3i32.into()));
        assert_eq!(rebuilt.body(), &Expr::Const(1i32.into()));
    }

    impl Arbitrary for FuncValue {
        type Strategy = BoxedStrategy<Self>;
        type Parameters = ();

        fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
            (any::<Expr>(), vec(any::<FuncArg>(), 1..10))
                .prop_map(|(body, args)| Self::new(args, body))
                .boxed()
        }
    }

    proptest! {

        #[test]
        fn ser_roundtrip(func_value in any::<FuncValue>()) {
            let e = Expr::FuncValue(func_value);
            prop_assert_eq![sigma_serialize_roundtrip(&e), e];
        }
    }
}
