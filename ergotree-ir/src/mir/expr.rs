//! IR expression

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;
use core::convert::Infallible;
use core::convert::TryFrom;
use core::convert::TryInto;

use crate::chain::context::Context;
use crate::chain::context::CostLimitExceeded;
use crate::chain::ergo_box::RegisterId;
use crate::chain::ergo_box::RegisterIdOutOfBounds;
use crate::chain::ergo_box::RegisterValueError;
use crate::pretty_printer::PosTrackingWriter;
use crate::pretty_printer::Print;
use crate::serialization::sigma_byte_reader;
use crate::serialization::sigma_byte_reader::SigmaByteRead;
use crate::serialization::SigmaParsingError;
use crate::serialization::SigmaSerializable;
use crate::source_span::Spanned;
use crate::traversable::Traversable;
use crate::types::stype::LiftIntoSType;
use crate::types::stype::SType;

use super::and::And;
use super::apply::Apply;
use super::bin_op::BinOp;
use super::bit_inversion::BitInversion;
use super::block::BlockValue;
use super::bool_to_sigma::BoolToSigmaProp;
use super::byte_array_to_long::ByteArrayToLong;
use super::calc_blake2b256::CalcBlake2b256;
use super::calc_sha256::CalcSha256;
use super::coll_append::Append;
use super::coll_by_index::ByIndex;
use super::coll_exists::Exists;
use super::coll_filter::Filter;
use super::coll_fold::Fold;
use super::coll_forall::ForAll;
use super::coll_map::Map;
use super::coll_size::SizeOf;
use super::coll_slice::Slice;
use super::collection::Collection;
use super::constant::Constant;
use super::constant::ConstantPlaceholder;
use super::constant::Literal;
use super::constant::TryExtractFrom;
use super::constant::TryExtractFromError;
use super::constant::TryExtractInto;
use super::create_avl_tree::CreateAvlTree;
use super::create_provedlog::CreateProveDlog;
use super::decode_point::DecodePoint;
use super::exponentiate::Exponentiate;
use super::extract_amount::ExtractAmount;
use super::extract_bytes::ExtractBytes;
use super::extract_bytes_with_no_ref::ExtractBytesWithNoRef;
use super::extract_creation_info::ExtractCreationInfo;
use super::extract_id::ExtractId;
use super::extract_reg_as::ExtractRegisterAs;
use super::extract_script_bytes::ExtractScriptBytes;
use super::func_value::FuncValue;
use super::global_vars::GlobalVars;
use super::if_op::If;
use super::logical_not::LogicalNot;
use super::long_to_byte_array::LongToByteArray;
use super::method_call::MethodCall;
use super::mod_q::{ModQ, ModQArithOp};
use super::multiply_group::MultiplyGroup;
use super::negation::Negation;
use super::option_get::OptionGet;
use super::option_get_or_else::OptionGetOrElse;
use super::option_is_defined::OptionIsDefined;
use super::or::Or;
use super::property_call::PropertyCall;
use super::select_field::SelectField;
use super::sigma_and::SigmaAnd;
use super::sigma_or::SigmaOr;
use super::sigma_prop_bytes::SigmaPropBytes;
use super::sigma_prop_is_proven::SigmaPropIsProven;
use super::subst_const::SubstConstants;
use super::tagged_variable::TaggedVariable;
use super::tree_lookup::TreeLookup;
use super::tuple::Tuple;
use super::upcast::Upcast;
use super::val_def::ValDef;
use super::val_use::ValUse;
use super::xor::Xor;

extern crate derive_more;
use crate::mir::atleast::Atleast;
use crate::mir::byte_array_to_bigint::ByteArrayToBigInt;
use crate::mir::create_prove_dh_tuple::CreateProveDhTuple;
use crate::mir::deserialize_context::DeserializeContext;
use crate::mir::deserialize_register::DeserializeRegister;
use crate::mir::downcast::Downcast;
use crate::mir::get_var::GetVar;
use crate::mir::xor_of::XorOf;
use bounded_vec::BoundedVecOutOfBounds;
use derive_more::From;
use derive_more::TryInto;
use thiserror::Error;

#[derive(PartialEq, Eq, Debug, Clone, From, TryInto)]
/// Expression in ErgoTree
pub enum Expr {
    /// Append - Concatenation of two collections
    Append(Spanned<Append>),
    /// Constant value
    Const(Constant),
    /// Placeholder for a constant
    ConstPlaceholder(ConstantPlaceholder),
    /// Substitute constants in serialized ergo tree
    SubstConstants(Spanned<SubstConstants>),
    /// Convert byte array to SLong
    ByteArrayToLong(Spanned<ByteArrayToLong>),
    /// Convert byte array to SLong
    ByteArrayToBigInt(Spanned<ByteArrayToBigInt>),
    /// Convert SLong to a byte array
    LongToByteArray(LongToByteArray),
    /// Collection declaration (array of expressions of the same type)
    Collection(Collection),
    /// Tuple declaration
    Tuple(Tuple),
    /// Predefined functions (global)
    /// Blake2b256 hash calculation
    CalcBlake2b256(CalcBlake2b256),
    /// Sha256 hash calculation
    CalcSha256(CalcSha256),
    /// Context variables (external)
    Context,
    /// Special global value which is used to define methods
    Global,
    /// Predefined global variables
    GlobalVars(GlobalVars),
    /// Function definition
    FuncValue(FuncValue),
    /// Function application
    Apply(Apply),
    /// Method call
    MethodCall(Spanned<MethodCall>),
    /// Property call
    PropertyCall(Spanned<PropertyCall>),
    /// Block (statements, followed by an expression)
    BlockValue(Spanned<BlockValue>),
    /// let-bound expression
    ValDef(Spanned<ValDef>),
    /// Reference to ValDef
    ValUse(ValUse),
    /// If, non-lazy - evaluate both branches
    If(If),
    /// Binary operation
    BinOp(Spanned<BinOp>),
    /// Logical AND
    And(Spanned<And>),
    /// Logical OR
    Or(Spanned<Or>),
    /// Byte-wise XOR
    Xor(Xor),
    /// THRESHOLD composition for sigma expressions
    Atleast(Atleast),
    /// LogicalNot
    LogicalNot(Spanned<LogicalNot>),
    /// Negation on numeric type
    Negation(Spanned<Negation>),
    /// Bit inversion on numeric type
    BitInversion(BitInversion),
    /// Option.get method
    OptionGet(Spanned<OptionGet>),
    /// Option.isDefined method
    OptionIsDefined(Spanned<OptionIsDefined>),
    /// Returns the option's value if the option is nonempty, otherwise return the result of evaluating `default`.
    OptionGetOrElse(Spanned<OptionGetOrElse>),
    /// Box monetary value
    ExtractAmount(ExtractAmount),
    /// Extract register's value (box.RX properties)
    ExtractRegisterAs(Spanned<ExtractRegisterAs>),
    /// Extract serialized box bytes
    ExtractBytes(ExtractBytes),
    /// Extract serialized box bytes excluding transaction_id & index
    ExtractBytesWithNoRef(ExtractBytesWithNoRef),
    /// Extract box's guarding script serialized to bytes
    ExtractScriptBytes(ExtractScriptBytes),
    /// Tuple of height when block got included into the blockchain and transaction identifier with
    /// box index in the transaction outputs serialized to the byte array.
    ExtractCreationInfo(ExtractCreationInfo),
    /// Box id, Blake2b256 hash of this box's content, basically equals to `blake2b256(bytes)`
    ExtractId(ExtractId),
    /// Collection, get element by index
    ByIndex(Spanned<ByIndex>),
    /// Collection size
    SizeOf(SizeOf),
    /// Collection slice
    Slice(Spanned<Slice>),
    /// Collection fold op
    Fold(Spanned<Fold>),
    /// Collection map op
    Map(Spanned<Map>),
    /// Collection filter op
    Filter(Spanned<Filter>),
    /// Tests whether a predicate holds for at least one element of this collection
    Exists(Spanned<Exists>),
    /// Tests whether a predicate holds for all elements of this collection.
    ForAll(Spanned<ForAll>),
    /// Tuple field access
    SelectField(Spanned<SelectField>),
    /// Bool to SigmaProp
    BoolToSigmaProp(BoolToSigmaProp),
    /// Upcast numeric value
    Upcast(Upcast),
    /// Downcast numeric value
    Downcast(Downcast),
    /// Create proveDlog from GroupElement(PK)
    CreateProveDlog(CreateProveDlog),
    /// Create proveDlog from GroupElement(PK)
    CreateProveDhTuple(CreateProveDhTuple),
    /// Extract serialized bytes of a SigmaProp value
    SigmaPropBytes(SigmaPropBytes),
    /// Decode byte array to EC point
    DecodePoint(DecodePoint),
    /// AND conjunction for sigma propositions
    SigmaAnd(SigmaAnd),
    /// OR conjunction for sigma propositions
    SigmaOr(SigmaOr),
    /// Extracts Context variable by id and type
    GetVar(Spanned<GetVar>),
    /// Extract register of SELF box as `Coll[Byte]`, deserialize it into Value and inline into
    /// the executing script.
    DeserializeRegister(DeserializeRegister),
    /// Extracts context variable as `Coll[Byte]`, deserializes it to script and then executes
    /// this script in the current context. The original `Coll[Byte]` of the script is
    /// available as `getVar[Coll[Byte]](id)` On evaluation returns the result of the
    /// script execution in the current context
    DeserializeContext(DeserializeContext),
    /// MultiplyGroup op for GroupElement
    MultiplyGroup(MultiplyGroup),
    /// Exponentiate op for GroupElement
    Exponentiate(Exponentiate),
    /// XOR for collection of booleans
    XorOf(XorOf),
    /// Perform a lookup by key in an AVL tree
    TreeLookup(Spanned<TreeLookup>),
    /// Create an AVL tree
    CreateAvlTree(CreateAvlTree),
    /// A `BigInt` modulo the group order (parsed, never evaluated)
    ModQ(ModQ),
    /// `PlusModQ` / `MinusModQ` (parsed, never evaluated)
    ModQArithOp(ModQArithOp),
    /// Whether a `SigmaProp` is proven (parsed, never evaluated)
    SigmaPropIsProven(SigmaPropIsProven),
    /// A context variable, `GetVar`'s predecessor (parsed, never evaluated)
    TaggedVariable(TaggedVariable),
}

impl Expr {
    /// Type of the expression
    pub fn tpe(&self) -> SType {
        match self {
            Expr::Append(ap) => ap.expr().tpe(),
            Expr::Const(v) => v.tpe.clone(),
            Expr::Collection(v) => v.tpe(),
            Expr::SubstConstants(v) => v.expr().tpe(),
            Expr::ByteArrayToLong(v) => v.expr().tpe(),
            Expr::ByteArrayToBigInt(v) => v.expr().tpe(),
            Expr::LongToByteArray(v) => v.tpe(),
            Expr::ConstPlaceholder(v) => v.tpe.clone(),
            Expr::CalcBlake2b256(v) => v.tpe(),
            Expr::CalcSha256(v) => v.tpe(),
            Expr::Global => SType::SGlobal,
            Expr::Context => SType::SContext,
            Expr::GlobalVars(v) => v.tpe(),
            Expr::FuncValue(v) => v.tpe(),
            Expr::Apply(v) => v.tpe(),
            Expr::MethodCall(v) => v.expr().tpe(),
            Expr::PropertyCall(v) => v.expr().tpe(),
            Expr::BlockValue(v) => v.expr().tpe(),
            Expr::ValDef(v) => v.expr().tpe(),
            Expr::ValUse(v) => v.tpe.clone(),
            Expr::BinOp(v) => v.expr().tpe(),
            Expr::OptionGet(v) => v.expr().tpe(),
            Expr::ExtractRegisterAs(v) => v.expr().tpe(),
            Expr::Fold(v) => v.expr().tpe(),
            Expr::SelectField(v) => v.expr().tpe(),
            Expr::ExtractAmount(v) => v.tpe(),
            Expr::And(v) => v.expr().tpe(),
            Expr::Or(v) => v.expr().tpe(),
            Expr::Xor(v) => v.tpe(),
            Expr::Atleast(v) => v.tpe(),
            Expr::LogicalNot(v) => v.expr().tpe(),
            Expr::Map(v) => v.expr().tpe(),
            Expr::Filter(v) => v.expr().tpe(),
            Expr::BoolToSigmaProp(v) => v.tpe(),
            Expr::Upcast(v) => v.tpe(),
            Expr::Downcast(v) => v.tpe(),
            Expr::If(v) => v.tpe(),
            Expr::ByIndex(v) => v.expr().tpe(),
            Expr::ExtractScriptBytes(v) => v.tpe(),
            Expr::SizeOf(v) => v.tpe(),
            Expr::Slice(v) => v.expr().tpe(),
            Expr::CreateProveDlog(v) => v.tpe(),
            Expr::CreateProveDhTuple(v) => v.tpe(),
            Expr::ExtractCreationInfo(v) => v.tpe(),
            Expr::Exists(v) => v.expr().tpe(),
            Expr::ExtractId(v) => v.tpe(),
            Expr::SigmaPropBytes(v) => v.tpe(),
            Expr::OptionIsDefined(v) => v.expr().tpe(),
            Expr::OptionGetOrElse(v) => v.expr().tpe(),
            Expr::Negation(v) => v.expr().tpe(),
            Expr::BitInversion(v) => v.tpe(),
            Expr::ForAll(v) => v.expr().tpe(),
            Expr::Tuple(v) => v.tpe(),
            Expr::DecodePoint(v) => v.tpe(),
            Expr::SigmaAnd(v) => v.tpe(),
            Expr::SigmaOr(v) => v.tpe(),
            Expr::DeserializeRegister(v) => v.tpe(),
            Expr::DeserializeContext(v) => v.tpe(),
            Expr::GetVar(v) => v.expr().tpe(),
            Expr::MultiplyGroup(v) => v.tpe(),
            Expr::Exponentiate(v) => v.tpe(),
            Expr::XorOf(v) => v.tpe(),
            Expr::ExtractBytes(v) => v.tpe(),
            Expr::ExtractBytesWithNoRef(v) => v.tpe(),
            Expr::TreeLookup(v) => v.expr().tpe(),
            Expr::CreateAvlTree(v) => v.tpe(),
            Expr::ModQ(v) => v.tpe(),
            Expr::ModQArithOp(v) => v.tpe(),
            Expr::SigmaPropIsProven(v) => v.tpe(),
            Expr::TaggedVariable(v) => v.tpe(),
        }
    }

    /// The type as sigmastate reads it, which throws where its `tpe` does: a `Filter` whose
    /// input is not a collection (its `def tpe` casts the input's type to `SCollection` on every
    /// read, v6.0.7 `transformers.scala:121`), and every node whose `tpe` is a `def` or `lazy
    /// val` over a child whose read throws: `If`, `ArithOp`, `BitOp`, `Negation`,
    /// `BitInversion`, `BlockValue`, `ValDef`, `Fold`, `Apply`, `FuncValue`, `Tuple`. Every
    /// other node's type is a `val` fixed when sigmastate built it, so reading it cannot
    /// throw. Where it does not throw it is [`Expr::tpe`]. Call it where sigmastate reads a
    /// type.
    pub(crate) fn try_tpe(&self) -> Result<SType, SigmaParsingError> {
        match self {
            Expr::Filter(v) => {
                let input_tpe = v.expr().input.try_tpe()?;
                if !matches!(input_tpe, SType::SColl(_) | SType::STuple(_)) {
                    return Err(SigmaParsingError::TypeReadFailed(format!(
                        "Filter of an input of type {input_tpe:?}"
                    )));
                }
            }
            Expr::If(v) => {
                v.true_branch.try_tpe()?;
            }
            Expr::BinOp(v)
                if matches!(
                    v.expr().kind,
                    super::bin_op::BinOpKind::Arith(_) | super::bin_op::BinOpKind::Bit(_)
                ) =>
            {
                v.expr().left.try_tpe()?;
            }
            Expr::Negation(v) => {
                v.expr().input.try_tpe()?;
            }
            Expr::BitInversion(v) => {
                v.input.try_tpe()?;
            }
            Expr::BlockValue(v) => {
                v.expr().result.try_tpe()?;
            }
            Expr::ValDef(v) => {
                v.expr().rhs.try_tpe()?;
            }
            Expr::Fold(v) => {
                v.expr().zero.try_tpe()?;
            }
            Expr::Apply(v) => {
                v.func.try_tpe()?;
            }
            Expr::FuncValue(v) => {
                v.body().try_tpe()?;
            }
            Expr::Tuple(v) => {
                for item in v.items.iter() {
                    item.try_tpe()?;
                }
            }
            _ => (),
        }
        Ok(self.tpe())
    }

    /// Type expected after the evaluation
    pub fn post_eval_tpe(&self) -> SType {
        match self.tpe() {
            SType::SFunc(sfunc) => *sfunc.t_range,
            tpe => tpe,
        }
    }

    /// Check that the expression's post-evaluation type is compatible with the
    /// expected type.
    ///
    /// Strict equality is the primary acceptance rule. As a parse-time leniency,
    /// `SOption(T)` is also accepted where `T` is expected — this matches the
    /// JVM v6.0.3 sigmastate-interpreter, whose
    /// `OneArgumentOperationSerializer.parse` uses an unchecked `asValue[T]`
    /// cast and performs no type validation when deserializing. Real on-chain
    /// scripts trigger this case by passing `getVar[T](id)` (post-eval type
    /// `SOption(T)`) directly into a combinator that statically expects `T` —
    /// most commonly `blake2b256` / `sha256` / `byteArrayToBigInt` /
    /// `byteArrayToLong` of `getVar[Coll[Byte]](id)` without an explicit
    /// `.get`. Such scripts are accepted by the JVM at parse time and would
    /// fail at evaluation time only if actually executed (our evaluator's
    /// `EvalError::UnexpectedValue` arm matches the JVM runtime failure).
    /// Without this leniency sigma-rust would reject txs the JVM accepts and
    /// ergo-node-rust would diverge from consensus.
    ///
    /// See: mainnet block 1,711,120, `tx[1]` `output[0]`, for a real-world reproducer
    /// (`parse_block_1711120_output0` in `serialization::expr::tests`).
    pub fn check_post_eval_tpe(
        &self,
        expected_tpe: &SType,
    ) -> Result<(), InvalidExprEvalTypeError> {
        let expr_tpe = self.post_eval_tpe();
        if &expr_tpe == expected_tpe {
            return Ok(());
        }
        if let SType::SOption(inner) = &expr_tpe {
            if inner.as_ref() == expected_tpe {
                return Ok(());
            }
        }
        #[cfg(feature = "std")]
        let backtrace = std::backtrace::Backtrace::capture();
        #[cfg(not(feature = "std"))]
        let backtrace = "Backtraces not supported without std";
        Err(InvalidExprEvalTypeError(format!(
            "expected: {0:?}, got: {1:?}\nBacktrace:\n{backtrace}",
            expected_tpe, expr_tpe
        )))
    }

    /// Rewrite tree bottom-up, matching nodes that satisfy `rule` and applying `subst` to them
    pub fn rewrite_bu(&self, rule: impl Fn(&Expr) -> bool, subst: impl Fn(&mut Expr)) -> Self {
        let mut new_root = self.clone();
        Self::rewrite_bu_inner(&mut new_root, &rule, &|node| {
            Ok::<_, Infallible>(subst(node))
        })
        .unwrap();
        new_root
    }

    /// Attempt to rewrite tree, returning early if `subst` returns `Err(E)`
    pub fn try_rewrite_bu<E>(
        mut self,
        rule: impl Fn(&Expr) -> bool,
        subst: impl Fn(&mut Expr) -> Result<(), E>,
    ) -> Result<Self, E> {
        Self::rewrite_bu_inner(&mut self, &rule, &subst).map(|_| self)
    }

    fn rewrite_bu_inner<E>(
        root: &mut Expr,
        rule: &impl Fn(&Expr) -> bool,
        subst: &impl Fn(&mut Expr) -> Result<(), E>,
    ) -> Result<(), E> {
        root.children_mut()
            .try_for_each(|expr| Self::rewrite_bu_inner(expr, rule, subst))?;
        if rule(root) {
            subst(root)?;
        }
        Ok(())
    }

    /// Return an iterator over the [`Expr`]
    /// The iterator performs a pre-order traversal
    pub fn iter(&self) -> impl Iterator<Item = &Expr> {
        struct ExprIterator<'a> {
            stack: Vec<&'a Expr>,
        }
        impl<'a> Iterator for ExprIterator<'a> {
            type Item = &'a Expr;

            fn next(&mut self) -> Option<Self::Item> {
                let cur = self.stack.pop()?;
                for node in cur.children() {
                    self.stack.push(node);
                }
                Some(cur)
            }
        }
        ExprIterator { stack: vec![&self] }
    }

    /// Returns true if the [`Expr`] has deserialize nodes, see: [`DeserializeContext`] and [`DeserializeRegister`]
    pub fn has_deserialize(&self) -> bool {
        self.iter().any(|c| {
            matches!(
                c,
                Expr::DeserializeContext(_) | Expr::DeserializeRegister(_)
            )
        })
    }

    /// Rewrite expr, replacing [`DeserializeContext`] and [`DeserializeRegister`] nodes with
    /// the scripts their context variable or register holds, as sigmastate's
    /// `applyDeserializeContextJITC` does (v6.0.7 `Interpreter.scala:149-157`), through Kiama's
    /// `everywherebu(strategy(rule))`. A `ClassCastException` raised inside the rule leaves the
    /// node where it is (`strategy`, `Rewriter.scala:180-191`); it errors only if the live path
    /// evaluates it. Any other failure fails the substitution, and so does a failure to rebuild
    /// an ancestor of a replaced node.
    pub fn substitute_deserialize(mut self, ctx: &Context) -> Result<Self, SubstDeserializeError> {
        subst_bu(&mut self, ctx)?;
        Ok(self)
    }

    /// Substitute [`ConstantPlaceholder`] nodes in `self` with [`Constant`]
    /// Errors if a constant can not be found in `constants`
    pub fn substitute_constants(self, constants: &[Constant]) -> Result<Self, SigmaParsingError> {
        self.try_rewrite_bu::<SigmaParsingError>(
            |expr| matches!(expr, Expr::ConstPlaceholder(_)),
            |expr| {
                if let Expr::ConstPlaceholder(ConstantPlaceholder { id, tpe: _ }) = expr {
                    *expr = constants
                        .get(*id as usize)
                        .cloned()
                        .map(Expr::from)
                        .ok_or(SigmaParsingError::ConstantForPlaceholderNotFound(*id))?;
                }
                Ok(())
            },
        )
    }

    /// Prints the tree with newlines
    pub fn debug_tree(&self) -> String {
        let tree = format!("{:#?}", self);
        tree
    }

    /// Pretty prints the tree
    pub fn to_string_pretty(&self) -> String {
        let mut printer = PosTrackingWriter::new();
        #[allow(clippy::unwrap_used)] // it only fail due to formatting errors
        let _spanned_expr = self.print(&mut printer).unwrap();
        printer.as_string()
    }
}

impl Traversable for Expr {
    type Item = Expr;

    fn children<'a>(&'a self) -> Box<dyn Iterator<Item = &'a Expr> + 'a> {
        match self {
            Expr::Const(_) => Box::new(core::iter::empty()),
            Expr::SubstConstants(op) => op.children(),
            Expr::ByteArrayToLong(op) => op.children(),
            Expr::ByteArrayToBigInt(op) => op.children(),
            Expr::LongToByteArray(op) => op.children(),
            Expr::CalcBlake2b256(op) => op.children(),
            Expr::CalcSha256(op) => op.children(),
            Expr::Fold(op) => op.children(),
            Expr::ExtractRegisterAs(op) => op.children(),
            Expr::GlobalVars(op) => op.children(),
            Expr::MethodCall(op) => op.children(),
            Expr::PropertyCall(op) => op.children(),
            Expr::BinOp(op) => op.children(),
            Expr::Global => Box::new(core::iter::empty()),
            Expr::Context => Box::new(core::iter::empty()),
            Expr::OptionGet(v) => v.children(),
            Expr::Apply(op) => op.children(),
            Expr::FuncValue(op) => op.children(),
            Expr::ValUse(op) => op.children(),
            Expr::BlockValue(op) => op.children(),
            Expr::SelectField(op) => op.children(),
            Expr::ExtractAmount(op) => op.children(),
            Expr::ConstPlaceholder(_) => Box::new(core::iter::empty()),
            Expr::Collection(op) => op.children(),
            Expr::ValDef(op) => op.children(),
            Expr::And(op) => op.children(),
            Expr::Or(op) => op.children(),
            Expr::Xor(op) => op.children(),
            Expr::Atleast(op) => op.children(),
            Expr::LogicalNot(op) => op.children(),
            Expr::Map(op) => op.children(),
            Expr::Filter(op) => op.children(),
            Expr::BoolToSigmaProp(op) => op.children(),
            Expr::Upcast(op) => op.children(),
            Expr::Downcast(op) => op.children(),
            Expr::If(op) => op.children(),
            Expr::Append(op) => op.children(),
            Expr::ByIndex(op) => op.children(),
            Expr::ExtractScriptBytes(op) => op.children(),
            Expr::SizeOf(op) => op.children(),
            Expr::Slice(op) => op.children(),
            Expr::CreateProveDlog(op) => op.children(),
            Expr::CreateProveDhTuple(op) => op.children(),
            Expr::ExtractCreationInfo(op) => op.children(),
            Expr::Exists(op) => op.children(),
            Expr::ExtractId(op) => op.children(),
            Expr::SigmaPropBytes(op) => op.children(),
            Expr::OptionIsDefined(op) => op.children(),
            Expr::OptionGetOrElse(op) => op.children(),
            Expr::Negation(op) => op.children(),
            Expr::BitInversion(op) => op.children(),
            Expr::ForAll(op) => op.children(),
            Expr::Tuple(op) => op.children(),
            Expr::DecodePoint(op) => op.children(),
            Expr::SigmaAnd(op) => op.children(),
            Expr::SigmaOr(op) => op.children(),
            Expr::DeserializeRegister(op) => op.children(),
            Expr::DeserializeContext(op) => op.children(),
            Expr::GetVar(op) => op.children(),
            Expr::MultiplyGroup(op) => op.children(),
            Expr::Exponentiate(op) => op.children(),
            Expr::XorOf(op) => op.children(),
            Expr::ExtractBytes(op) => op.children(),
            Expr::ExtractBytesWithNoRef(op) => op.children(),
            Expr::TreeLookup(op) => op.children(),
            Expr::CreateAvlTree(op) => op.children(),
            Expr::ModQ(op) => op.children(),
            Expr::ModQArithOp(op) => op.children(),
            Expr::SigmaPropIsProven(op) => op.children(),
            Expr::TaggedVariable(op) => op.children(),
        }
    }
    fn children_mut<'a>(&'a mut self) -> Box<dyn Iterator<Item = &'a mut Expr> + 'a> {
        match self {
            Expr::Const(_) => Box::new(core::iter::empty()),
            Expr::SubstConstants(op) => op.children_mut(),
            Expr::ByteArrayToLong(op) => op.children_mut(),
            Expr::ByteArrayToBigInt(op) => op.children_mut(),
            Expr::LongToByteArray(op) => op.children_mut(),
            Expr::CalcBlake2b256(op) => op.children_mut(),
            Expr::CalcSha256(op) => op.children_mut(),
            Expr::Fold(op) => op.children_mut(),
            Expr::ExtractRegisterAs(op) => op.children_mut(),
            Expr::GlobalVars(op) => op.children_mut(),
            Expr::MethodCall(op) => op.children_mut(),
            Expr::PropertyCall(op) => op.children_mut(),
            Expr::BinOp(op) => op.children_mut(),
            Expr::Global => Box::new(core::iter::empty()),
            Expr::Context => Box::new(core::iter::empty()),
            Expr::OptionGet(v) => v.children_mut(),
            Expr::Apply(op) => op.children_mut(),
            Expr::FuncValue(op) => op.children_mut(),
            Expr::ValUse(op) => op.children_mut(),
            Expr::BlockValue(op) => op.children_mut(),
            Expr::SelectField(op) => op.children_mut(),
            Expr::ExtractAmount(op) => op.children_mut(),
            Expr::ConstPlaceholder(_) => Box::new(core::iter::empty()),
            Expr::Collection(op) => op.children_mut(),
            Expr::ValDef(op) => op.children_mut(),
            Expr::And(op) => op.children_mut(),
            Expr::Or(op) => op.children_mut(),
            Expr::Xor(op) => op.children_mut(),
            Expr::Atleast(op) => op.children_mut(),
            Expr::LogicalNot(op) => op.children_mut(),
            Expr::Map(op) => op.children_mut(),
            Expr::Filter(op) => op.children_mut(),
            Expr::BoolToSigmaProp(op) => op.children_mut(),
            Expr::Upcast(op) => op.children_mut(),
            Expr::Downcast(op) => op.children_mut(),
            Expr::If(op) => op.children_mut(),
            Expr::Append(op) => op.children_mut(),
            Expr::ByIndex(op) => op.children_mut(),
            Expr::ExtractScriptBytes(op) => op.children_mut(),
            Expr::SizeOf(op) => op.children_mut(),
            Expr::Slice(op) => op.children_mut(),
            Expr::CreateProveDlog(op) => op.children_mut(),
            Expr::CreateProveDhTuple(op) => op.children_mut(),
            Expr::ExtractCreationInfo(op) => op.children_mut(),
            Expr::Exists(op) => op.children_mut(),
            Expr::ExtractId(op) => op.children_mut(),
            Expr::SigmaPropBytes(op) => op.children_mut(),
            Expr::OptionIsDefined(op) => op.children_mut(),
            Expr::OptionGetOrElse(op) => op.children_mut(),
            Expr::Negation(op) => op.children_mut(),
            Expr::BitInversion(op) => op.children_mut(),
            Expr::ForAll(op) => op.children_mut(),
            Expr::Tuple(op) => op.children_mut(),
            Expr::DecodePoint(op) => op.children_mut(),
            Expr::SigmaAnd(op) => op.children_mut(),
            Expr::SigmaOr(op) => op.children_mut(),
            Expr::DeserializeRegister(op) => op.children_mut(),
            Expr::DeserializeContext(op) => op.children_mut(),
            Expr::GetVar(op) => op.children_mut(),
            Expr::MultiplyGroup(op) => op.children_mut(),
            Expr::Exponentiate(op) => op.children_mut(),
            Expr::XorOf(op) => op.children_mut(),
            Expr::ExtractBytes(op) => op.children_mut(),
            Expr::ExtractBytesWithNoRef(op) => op.children_mut(),
            Expr::TreeLookup(op) => op.children_mut(),
            Expr::CreateAvlTree(op) => op.children_mut(),
            Expr::ModQ(op) => op.children_mut(),
            Expr::ModQArithOp(op) => op.children_mut(),
            Expr::SigmaPropIsProven(op) => op.children_mut(),
            Expr::TaggedVariable(op) => op.children_mut(),
        }
    }
}

impl<T: Into<Literal> + LiftIntoSType> From<T> for Expr {
    fn from(t: T) -> Self {
        Expr::Const(Constant {
            tpe: T::stype(),
            v: t.into(),
        })
    }
}

/// Unexpected argument on node construction (i.e non-Option input in OptionGet)
#[derive(Error, PartialEq, Eq, Debug, Clone)]
#[error("InvalidArgumentError: {0}")]
pub struct InvalidArgumentError(pub String);

/// Invalid (unexpected) expr type
#[derive(PartialEq, Eq, Debug, Clone, Error)]
#[error("InvalidExprEvalTypeError: {0}")]
pub struct InvalidExprEvalTypeError(pub String);

impl From<InvalidExprEvalTypeError> for InvalidArgumentError {
    fn from(e: InvalidExprEvalTypeError) -> Self {
        InvalidArgumentError(format!("InvalidExprEvalTypeError: {0}", e))
    }
}

impl From<BoundedVecOutOfBounds> for InvalidArgumentError {
    fn from(e: BoundedVecOutOfBounds) -> Self {
        InvalidArgumentError(format!("BoundedVecOutOfBounds: {0}", e))
    }
}

/// `everywherebu`'s walk (v6.0.7 `Rewriter.scala:841`, `:812`, `bottomup(s) = all(bottomup(s))
/// <* s`): the children first; then the node is rebuilt through its constructor if a child
/// changed (`all`'s `allProduct`, which builds a new node only then); then the rule is tried
/// on it, and its replacement is not traversed. Returns whether the subtree changed.
fn subst_bu(node: &mut Expr, ctx: &Context) -> Result<bool, SubstDeserializeError> {
    let mut changed = false;
    for child in node.children_mut() {
        changed |= subst_bu(child, ctx)?;
    }
    if changed {
        crate::serialization::check_build::rebuild(node)
            .map_err(SubstDeserializeError::AncestorRebuild)?;
    }
    if matches!(
        node,
        Expr::DeserializeContext(_) | Expr::DeserializeRegister(_)
    ) {
        match subst_deserialize(node, ctx) {
            Ok(Some(script)) => {
                *node = script;
                return Ok(true);
            }
            Ok(None) => (),
            Err(e) if e.is_class_cast() => (),
            Err(e) => return Err(e),
        }
    }
    Ok(changed)
}

/// sigmastate's rule for one node (v6.0.7 `Interpreter.scala:110-127`,
/// `ErgoLikeInterpreter.scala:17-38`): the script to put in its place, or `None` to leave it
// TODO: soft fork. sigmastate runs the substitution inside `trySoftForkable`
// (`Interpreter.scala:254-262`): a `ValidationException` under a soft-fork condition makes the
// reduction true; sigma-rust has no such handling yet.
fn subst_deserialize(expr: &Expr, ctx: &Context) -> Result<Option<Expr>, SubstDeserializeError> {
    match expr {
        Expr::DeserializeContext(DeserializeContext { tpe, id }) => {
            let value = match ctx.extension.values.get(id) {
                Some(value) => value,
                // Absent context variable: leave the DeserializeContext node unchanged,
                // mirroring the JVM `Interpreter.substDeserialize` `else None` (and the
                // DeserializeRegister branch below). A leftover node errors only if the *live*
                // reduction path evaluates it (eval/expr.rs: "DeserializeContext cannot be
                // evaluated"); on a dead branch it is harmless, which is how the JVM accepts
                // testnet block 111,927.
                None => return Ok(None),
            };
            // Present but not a `Coll[Byte]`: leave the node unchanged too, mirroring the JVM
            // `Interpreter.substDeserialize` inner `case _ => None` (a non-SByteArray extension
            // value is not substituted; the leftover node errors only on the live eval path).
            // The type decides before the data is read (`eba.tpe == SByteArray`, then
            // `eba.value`, `Interpreter.scala:114-115`).
            if value.tpe() != Some(SType::SColl(SType::SByte.into())) {
                return Ok(None);
            }
            // For a `Coll[Byte]`, its constant form is its data as the script reads it
            let bytes = value.to_constant()?.try_extract_into::<Vec<u8>>()?;
            let script = deserialize_measured(&bytes, ctx)?;
            // rule 1000 `CheckDeserializedScriptType` (`Interpreter.scala:119`)
            check_script_tpe(&script, tpe)?;
            Ok(Some(script))
        }
        Expr::DeserializeRegister(DeserializeRegister { reg, tpe, default }) => {
            match register_script_bytes(*reg, ctx)? {
                Some(bytes) => {
                    let script = deserialize_measured(&bytes, ctx)?;
                    // a mismatch is a `sys.error` (`:25-27`)
                    check_script_tpe(&script, tpe)?;
                    Ok(Some(script))
                }
                // An absent register substitutes the default, whose type nothing checks
                // (`.orElse(d.default)`, `:36`); with no default the node stays, and errors if
                // it is evaluated
                None => Ok(default.as_deref().cloned()),
            }
        }
        _ => Ok(None),
    }
}

/// A register as `eba.value.toArray` reads it (v6.0.7 `ErgoLikeInterpreter.scala:19-22`): `None`
/// for a register the box does not have, else the bytes of a `Coll[Byte]`. The value is read as
/// a script reads it, its data (`EvaluatedValue::script_data`): a read that fails (an item that
/// is not a value, an `AssertionError`; a pair's tuple expression in a collection, an
/// `ArrayStoreException`) fails the spend. The arm matching an `EvaluatedValue[SByteArray]` is
/// erased, so data that is not a byte array fails the cast of `toArray`'s result, a
/// `ClassCastException`.
fn register_script_bytes(
    reg: RegisterId,
    ctx: &Context,
) -> Result<Option<Vec<u8>>, SubstDeserializeError> {
    use crate::chain::ergo_box::RegisterValue;
    use crate::mir::constant::Literal;
    use crate::mir::value::{CollKind, NativeColl};
    let data = match reg {
        RegisterId::NonMandatoryRegisterId(id) => match ctx.self_box.additional_registers.get(id) {
            None => return Ok(None),
            Some(RegisterValue::Parsed(c)) => c.v.clone(),
            Some(RegisterValue::ParsedExpr(e)) => e.value().script_data()?,
            Some(RegisterValue::Invalid { error_msg, .. }) => {
                return Err(RegisterValueError::Invalid(error_msg.clone()).into())
            }
        },
        RegisterId::MandatoryRegisterId(_) => match ctx.self_box.get_register(reg)? {
            Some(c) => c.v,
            None => return Ok(None),
        },
    };
    match data {
        Literal::Coll(CollKind::NativeColl(NativeColl::CollByte(bytes))) => {
            Ok(Some(bytes.iter().map(|b| *b as u8).collect()))
        }
        other => Err(SubstDeserializeError::RegisterNotBytes(format!(
            "{other:?}"
        ))),
    }
}

/// `deserializeMeasured` (v6.0.7 `Interpreter.scala:99-107`): decodes the script, then charges
/// `CostPerByteDeserialized`, 2 block cost (20 jit) per byte, so a decode that throws charges
/// nothing. sigmastate decodes inside the spend's `VersionContext` (`Interpreter.scala:366`): a
/// tree in the script is compared with the context's activated version.
fn deserialize_measured(bytes: &[u8], ctx: &Context) -> Result<Expr, SubstDeserializeError> {
    let script = sigma_byte_reader::from_bytes(bytes).with_versions(
        ctx.activated_script_version(),
        ctx.tree_version(),
        Expr::sigma_parse,
    )?;
    ctx.add_jit_cost(bytes.len() as u64 * 20)?;
    Ok(script)
}

/// The decoded script's type against the declared one; reading it can throw, a cast
/// ([`Expr::try_tpe`])
fn check_script_tpe(script: &Expr, tpe: &SType) -> Result<(), SubstDeserializeError> {
    let actual = script.try_tpe()?;
    if actual != *tpe {
        return Err(SubstDeserializeError::ExprTpeError {
            expected: tpe.clone(),
            actual,
        });
    }
    Ok(())
}

#[derive(Error, Debug, Clone, PartialEq, Eq)]
#[allow(missing_docs)]
pub enum SubstDeserializeError {
    #[error("TryExtractFromError: {0}")]
    TryExtractFromError(#[from] TryExtractFromError),
    #[error("executeFromReg: Register out of bounds {0}")]
    InvalidRegister(#[from] RegisterIdOutOfBounds),
    #[error("Register {0} does not exist")]
    RegisterNotFound(RegisterId),
    #[error("RegisterValueError: {0}")]
    RegisterValueError(#[from] RegisterValueError),
    #[error("Error while parsing Expr from bytes: {0}")]
    ExprParsingError(#[from] SigmaParsingError),
    #[error("Expected tpe {expected}, found {actual}")]
    ExprTpeError { expected: SType, actual: SType },
    #[error("Cost limit exceeded during deserialize substitution: {0}")]
    CostLimitExceeded(CostLimitExceeded),
    #[error("a register that is not a Coll[Byte]: {0}")]
    RegisterNotBytes(String),
    /// A substituted node's ancestor failed to rebuild: `everywherebu`'s `all` rebuilds it
    /// through its constructor outside the rule, with no catch (v6.0.7 `Rewriter.scala:427`,
    /// `:260`), so any failure there fails the spend
    #[error("rebuilding an ancestor of a substituted node: {0}")]
    AncestorRebuild(SigmaParsingError),
}

impl SubstDeserializeError {
    /// What sigmastate raises as a `ClassCastException` inside the substitution's rule, which
    /// Kiama's `strategy` swallows (v6.0.7 `Rewriter.scala:180-191`): a decode or a type read
    /// that casts ([`SigmaParsingError::jvm_failure`]), and a register read as bytes that is not
    pub(crate) fn is_class_cast(&self) -> bool {
        match self {
            SubstDeserializeError::ExprParsingError(e) => {
                e.jvm_failure() == crate::serialization::JvmFailure::ClassCast
            }
            SubstDeserializeError::RegisterNotBytes(_) => true,
            SubstDeserializeError::TryExtractFromError(_)
            | SubstDeserializeError::InvalidRegister(_)
            | SubstDeserializeError::RegisterNotFound(_)
            | SubstDeserializeError::RegisterValueError(_)
            | SubstDeserializeError::ExprTpeError { .. }
            | SubstDeserializeError::CostLimitExceeded(_)
            | SubstDeserializeError::AncestorRebuild(_) => false,
        }
    }
}

impl From<CostLimitExceeded> for SubstDeserializeError {
    fn from(e: CostLimitExceeded) -> Self {
        SubstDeserializeError::CostLimitExceeded(e)
    }
}

impl<T: TryFrom<Expr>> TryExtractFrom<Expr> for T {
    fn try_extract_from(v: Expr) -> Result<Self, TryExtractFromError> {
        let res: Result<Self, TryExtractFromError> = v.clone().try_into().map_err(|_| {
            TryExtractFromError(format!(
                "Cannot extract {0:?} from {1:?}",
                core::any::type_name::<T>(),
                v
            ))
        });
        res
    }
}

#[cfg(feature = "arbitrary")]
#[allow(clippy::unwrap_used)]
#[allow(clippy::panic)]
#[allow(clippy::todo)]
/// Arbitrary impl
pub(crate) mod arbitrary {
    use super::*;
    use crate::mir::func_value::FuncArg;
    use crate::sigma_protocol::sigma_boolean::ProveDlog;
    use crate::types::sfunc::SFunc;
    use alloc::sync::Arc;
    use alloc::vec;
    use proptest::collection::*;
    use proptest::prelude::*;

    /// Parameters for arbitrary Expr generation
    #[derive(PartialEq, Eq, Debug, Clone)]
    pub struct ArbExprParams {
        /// Expr type
        pub tpe: SType,
        /// Expr tree depth (levels)
        pub depth: usize,
    }

    impl Default for ArbExprParams {
        fn default() -> Self {
            ArbExprParams {
                tpe: SType::SBoolean,
                depth: 1,
            }
        }
    }

    fn numeric_nested_expr(depth: usize, elem_tpe: &SType) -> BoxedStrategy<Expr> {
        prop_oneof![any_with::<BinOp>(ArbExprParams {
            tpe: elem_tpe.clone(),
            depth
        })
        .prop_map_into(),]
        .boxed()
    }

    fn bool_nested_expr(depth: usize) -> BoxedStrategy<Expr> {
        prop_oneof![
            any_with::<BinOp>(ArbExprParams {
                tpe: SType::SBoolean,
                depth
            })
            .prop_map_into(),
            any_with::<And>(depth).prop_map_into(),
            any_with::<Or>(depth).prop_map_into(),
            any_with::<LogicalNot>(depth).prop_map_into(),
        ]
        .boxed()
    }

    fn coll_nested_numeric(depth: usize, elem_tpe: &SType) -> BoxedStrategy<Expr> {
        let ty = elem_tpe.clone();
        vec(numeric_nested_expr(depth, elem_tpe), 0..10)
            .prop_map(move |items| Collection::new(ty.clone(), items).unwrap())
            .prop_map_into()
            .boxed()
    }

    fn sigma_prop_nested_expr(_depth: usize) -> BoxedStrategy<Expr> {
        prop_oneof![
            any::<ProveDlog>().prop_map(|pk| Expr::Const(pk.into())),
            any::<SigmaAnd>().prop_map_into(),
            any::<SigmaOr>().prop_map_into(),
        ]
        .boxed()
    }

    fn coll_nested_expr(depth: usize, elem_tpe: &SType) -> BoxedStrategy<Expr> {
        match elem_tpe {
            SType::SBoolean => vec(bool_nested_expr(depth), 0..10)
                .prop_map(|items| Collection::new(SType::SBoolean, items).unwrap())
                .prop_map_into()
                .boxed(),
            SType::SByte => coll_nested_numeric(depth, elem_tpe),
            SType::SShort => coll_nested_numeric(depth, elem_tpe),
            SType::SInt => coll_nested_numeric(depth, elem_tpe),
            SType::SLong => coll_nested_numeric(depth, elem_tpe),
            SType::SBigInt => coll_nested_numeric(depth, elem_tpe),

            SType::STypeVar(_) => prop_oneof![
                vec(bool_nested_expr(depth), 0..10).prop_map(|items| Collection::new(
                    SType::SBoolean,
                    items
                )
                .unwrap()),
                vec(numeric_nested_expr(depth, &SType::SInt), 0..10)
                    .prop_map(|items| Collection::new(SType::SInt, items).unwrap())
            ]
            .prop_map_into()
            .boxed(),
            SType::SSigmaProp => vec(sigma_prop_nested_expr(depth), 0..10)
                .prop_map(|items| Collection::new(SType::SSigmaProp, items).unwrap())
                .prop_map_into()
                .boxed(),

            _ => panic!("Nested expression not implemented for {:?}", &elem_tpe),
        }
    }

    fn any_nested_expr(depth: usize) -> BoxedStrategy<Expr> {
        prop_oneof![
            bool_nested_expr(depth),
            numeric_nested_expr(depth, &SType::SByte),
            numeric_nested_expr(depth, &SType::SShort),
            numeric_nested_expr(depth, &SType::SInt),
            numeric_nested_expr(depth, &SType::SLong),
            numeric_nested_expr(depth, &SType::SBigInt),
        ]
        .boxed()
    }

    fn nested_expr(tpe: SType, depth: usize) -> BoxedStrategy<Expr> {
        match tpe {
            SType::SAny => any_nested_expr(depth),
            SType::SBoolean => bool_nested_expr(depth),
            SType::SByte => numeric_nested_expr(depth, &tpe),
            SType::SShort => numeric_nested_expr(depth, &tpe),
            SType::SInt => numeric_nested_expr(depth, &tpe),
            SType::SLong => numeric_nested_expr(depth, &tpe),
            SType::SBigInt => numeric_nested_expr(depth, &tpe),
            SType::SColl(elem_type) => coll_nested_expr(depth, elem_type.as_ref()),
            SType::SSigmaProp => sigma_prop_nested_expr(depth),
            _ => todo!("nested expr is not implemented for type: {:?}", tpe),
        }
        .boxed()
    }

    fn int_non_nested_expr() -> BoxedStrategy<Expr> {
        prop_oneof![Just(GlobalVars::Height.into()),].boxed()
    }

    fn constant(tpe: &SType) -> BoxedStrategy<Expr> {
        any_with::<Constant>(tpe.clone().into())
            .prop_map_into()
            .boxed()
    }

    fn bool_non_nested_expr() -> BoxedStrategy<Expr> {
        prop_oneof![any_with::<Constant>(SType::SBoolean.into()).prop_map_into()].boxed()
    }

    fn any_non_nested_expr() -> BoxedStrategy<Expr> {
        prop_oneof![int_non_nested_expr(), bool_non_nested_expr()].boxed()
    }

    fn coll_non_nested_expr(elem_tpe: &SType) -> BoxedStrategy<Expr> {
        match elem_tpe {
            SType::SBoolean => any_with::<Constant>(SType::SColl(Arc::new(SType::SBoolean)).into())
                .prop_map(Expr::Const)
                .boxed(),
            SType::SByte => any_with::<Constant>(SType::SColl(Arc::new(SType::SByte)).into())
                .prop_map(Expr::Const)
                .boxed(),
            SType::SShort => any_with::<Constant>(SType::SColl(Arc::new(SType::SShort)).into())
                .prop_map(Expr::Const)
                .boxed(),
            SType::SInt => any_with::<Constant>(SType::SColl(Arc::new(SType::SInt)).into())
                .prop_map(Expr::Const)
                .boxed(),
            SType::SLong => any_with::<Constant>(SType::SColl(Arc::new(SType::SLong)).into())
                .prop_map(Expr::Const)
                .boxed(),
            _ => todo!("Collection of {0:?} is not yet implemented", elem_tpe),
        }
    }

    fn non_nested_expr(tpe: &SType) -> BoxedStrategy<Expr> {
        match tpe {
            SType::SAny => any_non_nested_expr(),
            SType::SInt => int_non_nested_expr(),
            SType::SBoolean => bool_non_nested_expr(),
            SType::SColl(elem_type) => coll_non_nested_expr(elem_type),
            t => constant(t),
        }
    }

    fn sfunc_expr(sfunc: SFunc) -> BoxedStrategy<Expr> {
        match (sfunc.t_dom.first().unwrap(), *sfunc.t_range) {
            (SType::SBoolean, SType::SBoolean) => any_with::<Expr>(ArbExprParams {
                tpe: SType::SBoolean,
                depth: 2,
            })
            .prop_map(|expr| {
                Expr::FuncValue(FuncValue::new(
                    vec![FuncArg {
                        idx: 1.into(),
                        tpe: SType::SBoolean,
                    }],
                    expr,
                ))
            })
            .boxed(),
            _ => todo!(),
        }
    }

    impl Arbitrary for Expr {
        type Parameters = ArbExprParams;
        type Strategy = BoxedStrategy<Self>;

        fn arbitrary_with(args: Self::Parameters) -> Self::Strategy {
            if args.depth == 0 {
                match args.tpe {
                    SType::SFunc(sfunc) => sfunc_expr(sfunc),
                    _ => prop_oneof![
                        any_with::<Constant>(args.tpe.clone().into())
                            .prop_map(Expr::Const)
                            .boxed(),
                        non_nested_expr(&args.tpe)
                    ]
                    .boxed(),
                }
            } else {
                nested_expr(args.tpe, args.depth - 1)
            }
        }
    }
}

#[cfg(test)]
mod tests {}

#[cfg(test)]
#[cfg(feature = "arbitrary")]
#[allow(clippy::unwrap_used)]
mod substitute_deserialize_tests {
    //! JVM parity: sigmastate reduces a tree under
    //! `withVersions(context.activatedScriptVersion, ergoTree.version)` (v6.0.6
    //! `Interpreter.scala:366`), so a tree deserialized while a script is decoded is compared
    //! with the context's activated version (`ErgoTreeSerializer.scala:150-154`). Its
    //! `SerializerException` is no class cast, so the substitution does not swallow it and the
    //! spend fails. SANTA `tree-version-above-activated-eval` #0 to #5 are the spends (replayed
    //! in ergo-lib's `tx_context`).
    use super::*;
    use crate::chain::context_extension::ContextExtension;
    use crate::chain::ergo_box::ErgoBox;
    use crate::chain::ergo_box::NonMandatoryRegisterId;
    use sigma_test_util::force_any_val;

    /// A Box constant as an expression: type code 99, then SANTA's box
    /// (`Box.tree_version_above_activated` #0) with the tree `header 02 08 d3`
    fn box_constant(header: u8) -> Vec<u8> {
        let hex = format!(
            "63c0843d{header:02x}0208d3010000\
             1d823ee9ea823cc80232a19181efad41d66849c33ed5d0d6c5750b8d60f1d66400"
        );
        base16::decode(hex.as_bytes()).unwrap()
    }

    /// (the box's tree header, the activated version, the tree version the JVM refuses)
    const CASES: [(u8, u8, Option<u8>); 4] = [
        (0x0c, 3, Some(4)),
        (0x0b, 3, None),
        (0x0b, 2, Some(3)),
        (0x0c, 1, None),
    ];

    fn check(res: Result<Expr, SubstDeserializeError>, activated: u8, refused: Option<u8>) {
        match refused {
            Some(version) => assert_eq!(
                res,
                Err(SubstDeserializeError::ExprParsingError(
                    SigmaParsingError::TreeVersionAboveActivated(version, activated)
                ))
            ),
            None => assert!(matches!(res, Ok(Expr::Const(_))), "{res:?}"),
        }
    }

    #[test]
    fn a_script_decoded_from_a_context_variable_is_read_under_the_activated_version() {
        let expr = Expr::from(DeserializeContext {
            tpe: SType::SBox,
            id: 1,
        });
        for (header, activated, refused) in CASES {
            let extension = ContextExtension {
                values: [(1u8, box_constant(header).into())].into_iter().collect(),
            };
            let mut ctx = force_any_val::<Context>().with_extension(&extension);
            ctx.activated_script_version_byte = activated as i8;
            check(
                expr.clone().substitute_deserialize(&ctx),
                activated,
                refused,
            );
        }
    }

    #[test]
    fn a_script_decoded_from_a_register_is_read_under_the_activated_version() {
        let expr = Expr::from(DeserializeRegister {
            reg: NonMandatoryRegisterId::R4.into(),
            tpe: SType::SBox,
            default: None,
        });
        for (header, activated, refused) in CASES {
            let script: Constant = box_constant(header).into();
            let self_box = force_any_val::<ErgoBox>()
                .with_additional_registers(vec![script].try_into().unwrap());
            let mut ctx = Context {
                self_box: &self_box,
                ..force_any_val::<Context>()
            };
            ctx.activated_script_version_byte = activated as i8;
            check(
                expr.clone().substitute_deserialize(&ctx),
                activated,
                refused,
            );
        }
    }
}

#[cfg(test)]
#[cfg(feature = "arbitrary")]
#[allow(clippy::unwrap_used, clippy::panic)]
mod substitution_rule_tests {
    //! sigmastate's rule (v6.0.7 `Interpreter.scala:110-127`, `ErgoLikeInterpreter.scala:17-38`)
    //! decodes, then charges (`deserializeMeasured`, `:99-107`), then checks the decoded
    //! script's type. Kiama's `strategy` leaves the node where it is on a `ClassCastException`
    //! raised inside the rule (`Rewriter.scala:180-191`), and only on that.
    use super::type_read_tests::{bi, filter};
    use super::*;
    use crate::chain::context_extension::ContextExtension;
    use crate::chain::ergo_box::ErgoBox;
    use crate::chain::ergo_box::NonMandatoryRegisterId;
    use crate::mir::option_get::OptionGet;
    use crate::serialization::SigmaSerializable;
    use alloc::boxed::Box;
    use sigma_test_util::force_any_val;

    fn register(tpe: SType, default: Option<Expr>) -> Expr {
        DeserializeRegister {
            reg: NonMandatoryRegisterId::R4.into(),
            tpe,
            default: default.map(Box::new),
        }
        .into()
    }

    fn context_var(tpe: SType) -> Expr {
        DeserializeContext { tpe, id: 1 }.into()
    }

    /// Substitutes `expr` in a context whose R4 and context variable 1 both hold `value`,
    /// returning the result and the cost it charged
    fn substitute(expr: &Expr, value: Constant) -> (Result<Expr, SubstDeserializeError>, u64) {
        let self_box = force_any_val::<ErgoBox>()
            .with_additional_registers(vec![value.clone()].try_into().unwrap());
        let extension = ContextExtension {
            values: [(1u8, value.into())].into_iter().collect(),
        };
        let ctx = Context {
            self_box: &self_box,
            ..force_any_val::<Context>()
        }
        .with_extension(&extension);
        let before = ctx.jit_cost_value();
        let res = expr.clone().substitute_deserialize(&ctx);
        (res, ctx.jit_cost_value() - before)
    }

    fn bytes(e: Expr) -> Constant {
        e.sigma_serialize_bytes().unwrap().into()
    }

    #[test]
    fn a_cast_in_the_decode_leaves_the_node_and_charges_nothing() {
        // SANTA s3j, s16b: the script OptionGet(BI) throws as it is built, before the charge
        let script = bytes(
            OptionGet {
                input: Box::new(bi()),
            }
            .into(),
        );
        for expr in [register(SType::SAny, None), context_var(SType::SAny)] {
            assert_eq!(substitute(&expr, script.clone()), (Ok(expr.clone()), 0));
        }
    }

    #[test]
    fn a_cast_in_the_type_check_leaves_the_node_and_keeps_the_charge() {
        // SANTA s3, s16: F(BI) decodes and is charged 20 jit per byte; its type read throws
        let script = bytes(filter(bi()));
        let len = script
            .v
            .clone()
            .try_extract_into::<Vec<i8>>()
            .unwrap()
            .len() as u64;
        for expr in [register(SType::SAny, None), context_var(SType::SAny)] {
            assert_eq!(
                substitute(&expr, script.clone()),
                (Ok(expr.clone()), len * 20)
            );
        }
    }

    #[test]
    fn a_register_that_is_not_bytes_leaves_the_node() {
        // SANTA s15: `eba.value.toArray` on an Int is a cast, so the default is not reached
        let expr = register(SType::SInt, Some(2i32.into()));
        assert_eq!(substitute(&expr, 1i32.into()), (Ok(expr.clone()), 0));
    }

    /// Substitutes `expr` in a context whose self box has no registers and whose extension is
    /// empty
    fn substitute_empty(expr: &Expr) -> Result<Expr, SubstDeserializeError> {
        let self_box = force_any_val::<ErgoBox>()
            .with_additional_registers(crate::chain::ergo_box::NonMandatoryRegisters::empty());
        let extension = ContextExtension::empty();
        let ctx = Context {
            self_box: &self_box,
            ..force_any_val::<Context>()
        }
        .with_extension(&extension);
        expr.clone().substitute_deserialize(&ctx)
    }

    /// sigmaProp(If(false, c, true)), whose `c` is never evaluated
    fn dead(c: Expr) -> Expr {
        crate::mir::bool_to_sigma::BoolToSigmaProp {
            input: Box::new(
                crate::mir::if_op::If {
                    condition: Box::new(false.into()),
                    true_branch: Box::new(c),
                    false_branch: Box::new(true.into()),
                }
                .into(),
            ),
        }
        .into()
    }

    fn bin(kind: crate::mir::bin_op::BinOpKind, left: Expr, right: Expr) -> Expr {
        crate::mir::bin_op::BinOp {
            kind,
            left: Box::new(left),
            right: Box::new(right),
        }
        .into()
    }

    #[test]
    fn a_default_is_substituted_untyped() {
        // `.orElse(d.default)` checks nothing (v6.0.7 `ErgoLikeInterpreter.scala:36`; SANTA s2)
        assert_eq!(
            substitute_empty(&register(SType::SAny, Some(1i32.into()))),
            Ok(1i32.into())
        );
    }

    #[test]
    fn a_rebuilt_if_reads_its_branch_types() {
        // SANTA s8: dead(EQ(If(true, DR(R4, SAny, F(BI)), BI), BI)), R4 absent. `all` rebuilds
        // each ancestor through its constructor, with no catch (`Rewriter.scala:427`, `:260`),
        // and the rebuilt If reads F(BI)'s type (`trees.scala:1313`): the spend fails although
        // the branch is dead
        use crate::mir::bin_op::RelationOp;
        let if_: Expr = crate::mir::if_op::If {
            condition: Box::new(true.into()),
            true_branch: Box::new(register(SType::SAny, Some(filter(bi())))),
            false_branch: Box::new(bi()),
        }
        .into();
        assert!(matches!(
            substitute_empty(&dead(bin(RelationOp::Eq.into(), if_, bi()))),
            Err(SubstDeserializeError::AncestorRebuild(
                SigmaParsingError::TypeReadFailed(_)
            ))
        ));
    }

    #[test]
    fn a_rebuilt_plus_reads_its_operand_types() {
        // SANTA s9: dead(GT(Plus(DR(R4, Int, F(BI)), 0), 0)), R4 absent: the rebuilt Plus reads
        // both operands' types (`trees.scala:707-708`)
        use crate::mir::bin_op::{ArithOp, RelationOp};
        let plus = bin(
            ArithOp::Plus.into(),
            register(SType::SInt, Some(filter(bi()))),
            0i32.into(),
        );
        assert!(matches!(
            substitute_empty(&dead(bin(RelationOp::Gt.into(), plus, 0i32.into()))),
            Err(SubstDeserializeError::AncestorRebuild(
                SigmaParsingError::TypeReadFailed(_)
            ))
        ));
    }

    #[test]
    fn an_unchanged_sibling_is_not_rebuilt() {
        // `allProduct` rebuilds a node only when a child changed: If(true, F(BI), F(BI)) would
        // fail a rebuild, and as the substituted node's sibling it is left as it is
        use crate::mir::bin_op::RelationOp;
        let sibling: Expr = crate::mir::if_op::If {
            condition: Box::new(true.into()),
            true_branch: Box::new(filter(bi())),
            false_branch: Box::new(filter(bi())),
        }
        .into();
        let expr = dead(bin(
            RelationOp::Eq.into(),
            register(SType::SAny, Some(bi())),
            sibling.clone(),
        ));
        assert_eq!(
            substitute_empty(&expr),
            Ok(dead(bin(RelationOp::Eq.into(), bi(), sibling)))
        );
    }

    #[test]
    fn a_decoded_script_is_not_traversed_again() {
        // bottom-up, the rule runs after the children and the replacement is not traversed:
        // R4 holds DeserializeContext(Int, 1), and variable 1 holds Int 5
        let script: Expr = DeserializeContext {
            tpe: SType::SInt,
            id: 1,
        }
        .into();
        let self_box = force_any_val::<ErgoBox>()
            .with_additional_registers(vec![bytes(script.clone())].try_into().unwrap());
        let extension = ContextExtension {
            values: [(1u8, bytes(5i32.into()).into())].into_iter().collect(),
        };
        let ctx = Context {
            self_box: &self_box,
            ..force_any_val::<Context>()
        }
        .with_extension(&extension);
        assert_eq!(
            register(SType::SInt, None).substitute_deserialize(&ctx),
            Ok(script)
        );
    }

    #[test]
    fn a_rebuilt_node_recomputes_its_cached_type() {
        // Exists(DR(R4, Coll[Int], default Coll[Long](2)), (x: Long) => true), R4 absent: the
        // default goes in untyped, and the rebuilt Exists takes its element type from it
        use crate::mir::coll_exists::Exists;
        use crate::mir::func_value::{FuncArg, FuncValue};
        let exists: Expr = Exists {
            input: Box::new(register(
                SType::SColl(SType::SInt.into()),
                Some(vec![2i64].into()),
            )),
            condition: Box::new(
                FuncValue::new(
                    vec![FuncArg {
                        idx: 1.into(),
                        tpe: SType::SLong,
                    }],
                    true.into(),
                )
                .into(),
            ),
            elem_tpe: SType::SInt.into(),
        }
        .into();
        match substitute_empty(&exists).unwrap() {
            Expr::Exists(e) => assert_eq!(*e.expr().elem_tpe, SType::SLong),
            e => panic!("{e:?}"),
        }
    }

    /// Substitutes `expr` in a context whose R4 holds the register value `value`
    fn substitute_register(expr: &Expr, value: Expr) -> Result<Expr, SubstDeserializeError> {
        substitute_register_bytes(expr, &value.sigma_serialize_bytes().unwrap())
    }

    /// Substitutes `expr` in a context whose R4 holds the register value serialized as `value`
    fn substitute_register_bytes(expr: &Expr, value: &[u8]) -> Result<Expr, SubstDeserializeError> {
        let mut bytes = vec![1u8];
        bytes.extend(value);
        let registers =
            crate::chain::ergo_box::NonMandatoryRegisters::sigma_parse_bytes(&bytes).unwrap();
        let self_box = force_any_val::<ErgoBox>().with_additional_registers(registers);
        let ctx = Context {
            self_box: &self_box,
            ..force_any_val::<Context>()
        };
        expr.clone().substitute_deserialize(&ctx)
    }

    #[test]
    fn a_register_whose_data_read_fails_fails_the_spend() {
        // R4 = Coll[(Int, Int)] holding the tuple expression (1, 2): `ConcreteCollection.value`
        // stores its data, a collection, in an array of Tuple2, an `ArrayStoreException`
        // (`values.scala:882-884`), which is no cast
        use crate::mir::collection::Collection;
        use crate::mir::tuple::Tuple;
        use crate::types::stuple::STuple;
        let pairs: Expr = Collection::Exprs {
            elem_tpe: SType::STuple(STuple::pair(SType::SInt, SType::SInt)),
            items: vec![Tuple::new(vec![1i32.into(), 2i32.into()]).unwrap().into()],
        }
        .into();
        let expr = register(SType::SInt, Some(2i32.into()));
        assert!(matches!(
            substitute_register(&expr, pairs),
            Err(SubstDeserializeError::TryExtractFromError(_))
        ));
    }

    #[test]
    fn a_register_holding_a_tuple_leaves_the_node() {
        // R4 = the 1-item tuple (Int 1): `Tuple.value` is a `Coll[Any]`, and reading it as bytes
        // is a cast, so the node stays and the default is not reached (`values.scala:818-822`)
        // the tuple opcode, 1 item, Int 1: sigma-rust's Tuple node holds 2 items at least, so
        // written by hand
        let one = base16::decode(b"86010402").unwrap();
        let expr = register(SType::SInt, Some(2i32.into()));
        assert_eq!(substitute_register_bytes(&expr, &one), Ok(expr.clone()));
    }

    #[test]
    fn a_type_mismatch_still_fails_the_spend() {
        // a decoded Int where a Long is declared: `sys.error` for a register, rule 1000 for a
        // context variable; neither is a cast
        let script = bytes(1i32.into());
        for expr in [register(SType::SLong, None), context_var(SType::SLong)] {
            assert!(
                matches!(
                    substitute(&expr, script.clone()).0,
                    Err(SubstDeserializeError::ExprTpeError { .. })
                ),
                "{expr:?}"
            );
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
pub(crate) mod type_read_tests {
    use super::*;
    use crate::mir::bin_op::{ArithOp, BinOp};
    use crate::mir::coll_by_index::ByIndex;
    use crate::mir::coll_filter::Filter;
    use crate::mir::coll_size::SizeOf;
    use crate::mir::func_value::{FuncArg, FuncValue};
    use crate::mir::if_op::If;
    use crate::mir::tuple::Tuple;
    use alloc::boxed::Box;
    use alloc::vec;

    /// `ByIndex(Tuple(0, 0), 0)`, typed SAny (a tuple is a collection of SAny)
    pub(crate) fn bi() -> Expr {
        ByIndex {
            input: Box::new(Tuple::new(vec![0i32.into(), 0i32.into()]).unwrap().into()),
            index: Box::new(0i32.into()),
            default: None,
        }
        .into()
    }

    /// `Filter(input, (i: Int) => true)`, built without the constructor's checks
    pub(crate) fn filter(input: Expr) -> Expr {
        Filter {
            input: Box::new(input),
            condition: Box::new(
                FuncValue::new(
                    vec![FuncArg {
                        idx: 1.into(),
                        tpe: SType::SInt,
                    }],
                    Expr::Const(true.into()),
                )
                .into(),
            ),
            elem_tpe: SType::SAny.into(),
        }
        .into()
    }

    #[test]
    fn a_filter_over_a_non_collection_throws_when_its_type_is_read() {
        // sigmastate's `Filter.tpe` casts its input's type to `SCollection` on every read
        // (v6.0.7 `transformers.scala:121`); `If.tpe` and `ArithOp.tpe` read a child's type
        // (`trees.scala:1351`, `:707-708`), `SizeOf`'s is `SInt`
        let throws = |e: Expr| matches!(e.try_tpe(), Err(SigmaParsingError::TypeReadFailed(_)));
        assert!(throws(filter(bi())));
        assert!(throws(
            If {
                condition: Box::new(Expr::Const(true.into())),
                true_branch: Box::new(filter(bi())),
                false_branch: Box::new(filter(bi())),
            }
            .into()
        ));
        assert!(throws(
            BinOp {
                kind: ArithOp::Plus.into(),
                left: Box::new(filter(bi())),
                right: Box::new(filter(bi())),
            }
            .into()
        ));
        let size: Expr = SizeOf {
            input: Box::new(filter(bi())),
        }
        .into();
        assert_eq!(size.try_tpe().unwrap(), SType::SInt);
    }

    #[test]
    fn a_filter_over_a_tuple_is_typed_as_the_tuple() {
        // a tuple passes the cast to `SCollection` (`SType.scala:838-841`), so `Filter.tpe` is
        // the tuple's type
        let tuple: Expr = Tuple::new(vec![0i32.into(), 0i32.into()]).unwrap().into();
        let tpe = tuple.tpe();
        let f = filter(tuple);
        assert_eq!(f.tpe(), tpe);
        assert_eq!(f.try_tpe().unwrap(), tpe);
    }
}

#[cfg(test)]
#[cfg(feature = "arbitrary")]
mod type_read_proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn try_tpe_equals_tpe_without_a_throwing_filter(e in any::<Expr>()) {
            prop_assert_eq!(e.try_tpe(), Ok(e.tpe()));
        }
    }
}
