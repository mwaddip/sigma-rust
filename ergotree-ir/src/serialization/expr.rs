use super::bin_op::bin_op_sigma_parse;
use super::bin_op::bin_op_sigma_serialize;
use super::check_build::check_build;
use super::{op_code::OpCode, sigma_byte_writer::SigmaByteWrite};
use crate::ergo_tree::ErgoTreeVersion;
use crate::has_opcode::HasOpCode;
use crate::has_opcode::HasStaticOpCode;
use crate::mir::and::And;
use crate::mir::apply::Apply;
use crate::mir::atleast::Atleast;
use crate::mir::bin_op::ArithOp;
use crate::mir::bin_op::BitOp;
use crate::mir::bin_op::LogicalOp;
use crate::mir::bin_op::RelationOp;
use crate::mir::bit_inversion::BitInversion;
use crate::mir::block::BlockValue;
use crate::mir::bool_to_sigma::BoolToSigmaProp;
use crate::mir::byte_array_to_bigint::ByteArrayToBigInt;
use crate::mir::byte_array_to_long::ByteArrayToLong;
use crate::mir::calc_blake2b256::CalcBlake2b256;
use crate::mir::calc_sha256::CalcSha256;
use crate::mir::coll_append::Append;
use crate::mir::coll_by_index::ByIndex;
use crate::mir::coll_exists::Exists;
use crate::mir::coll_filter::Filter;
use crate::mir::coll_fold::Fold;
use crate::mir::coll_forall::ForAll;
use crate::mir::coll_map::Map;
use crate::mir::coll_size::SizeOf;
use crate::mir::coll_slice::Slice;
use crate::mir::collection::bool_const_coll_sigma_parse;
use crate::mir::collection::coll_sigma_parse;
use crate::mir::collection::coll_sigma_serialize;
use crate::mir::constant::Constant;
use crate::mir::constant::ConstantPlaceholder;
use crate::mir::create_avl_tree::CreateAvlTree;
use crate::mir::create_prove_dh_tuple::CreateProveDhTuple;
use crate::mir::create_provedlog::CreateProveDlog;
use crate::mir::decode_point::DecodePoint;
use crate::mir::deserialize_context::DeserializeContext;
use crate::mir::deserialize_register::DeserializeRegister;
use crate::mir::downcast::Downcast;
use crate::mir::exponentiate::Exponentiate;
use crate::mir::expr::Expr;
use crate::mir::extract_amount::ExtractAmount;
use crate::mir::extract_bytes::ExtractBytes;
use crate::mir::extract_bytes_with_no_ref::ExtractBytesWithNoRef;
use crate::mir::extract_creation_info::ExtractCreationInfo;
use crate::mir::extract_id::ExtractId;
use crate::mir::extract_reg_as::ExtractRegisterAs;
use crate::mir::extract_script_bytes::ExtractScriptBytes;
use crate::mir::func_value::FuncValue;
use crate::mir::get_var::GetVar;
use crate::mir::global_vars::GlobalVars;
use crate::mir::if_op::If;
use crate::mir::logical_not::LogicalNot;
use crate::mir::long_to_byte_array::LongToByteArray;
use crate::mir::method_call::MethodCall;
use crate::mir::mod_q::{ModQ, ModQArithOp, ModQArithOpKind};
use crate::mir::multiply_group::MultiplyGroup;
use crate::mir::negation::Negation;
use crate::mir::option_get::OptionGet;
use crate::mir::option_get_or_else::OptionGetOrElse;
use crate::mir::option_is_defined::OptionIsDefined;
use crate::mir::or::Or;
use crate::mir::property_call::PropertyCall;
use crate::mir::select_field::SelectField;
use crate::mir::sigma_and::SigmaAnd;
use crate::mir::sigma_or::SigmaOr;
use crate::mir::sigma_prop_bytes::SigmaPropBytes;
use crate::mir::sigma_prop_is_proven::SigmaPropIsProven;
use crate::mir::subst_const::SubstConstants;
use crate::mir::tagged_variable::TaggedVariable;
use crate::mir::tree_lookup::TreeLookup;
use crate::mir::tuple::Tuple;
use crate::mir::upcast::Upcast;
use crate::mir::val_def::ValDef;
use crate::mir::val_use::ValUse;
use crate::mir::xor::Xor;
use crate::serialization::SigmaSerializeResult;
use crate::serialization::{
    sigma_byte_reader::SigmaByteRead, SigmaParsingError, SigmaSerializable,
};

use crate::mir::xor_of::XorOf;
use crate::source_span::Spanned;

impl Expr {
    /// Parse expression from byte stream. This function should be used instead of
    /// `sigma_parse` when tag byte is already read for look-ahead
    pub fn parse_with_tag<R: SigmaByteRead>(r: &mut R, tag: u8) -> Result<Self, SigmaParsingError> {
        // `ValueSerializer.deserialize` (sigmastate v6.0.6 `ValueSerializer.scala:396-409`):
        // every value is one nesting level, released only on success (`r.level =
        // r.level - 1`, no `finally`), so a level a failed nested parse leaves behind
        // stays on the reader.
        let depth = r.level();
        r.set_level(depth + 1)?;
        let expr = Self::parse_tagged(r, tag)?;
        r.set_level(r.level().saturating_sub(1))?;
        Ok(expr)
    }

    /// The value after its first byte `tag`, without its nesting level
    pub(crate) fn parse_tagged<R: SigmaByteRead>(
        r: &mut R,
        tag: u8,
    ) -> Result<Self, SigmaParsingError> {
        if tag <= OpCode::LAST_CONSTANT_CODE.value() {
            let constant = Constant::parse_with_tag(r, tag)?;
            Ok(Expr::Const(constant))
        } else {
            let op_code = OpCode::parse(tag);
            let expr = match op_code {
                OpCode::APPEND => Ok(Append::sigma_parse(r)?.into()),
                OpCode::FOLD => Ok(Fold::sigma_parse(r)?.into()),
                ConstantPlaceholder::OP_CODE => {
                    let cp = ConstantPlaceholder::sigma_parse(r)?;
                    if r.substitute_placeholders() {
                        // ConstantPlaceholder itself can be created only if a corresponding
                        // constant is in the constant_store, thus unwrap() is safe here
                        #[allow(clippy::unwrap_used)]
                        let c = r.constant_store().get(cp.id).unwrap();
                        Ok(Expr::Const(c.clone()))
                    } else {
                        Ok(Expr::ConstPlaceholder(cp))
                    }
                }
                // sigmastate v6.0.6 reads `TrueLeaf` and `FalseLeaf` under their own opcodes
                // (`ValueSerializer.scala:79-80`), but both are the Boolean constant
                // (`values.scala:771-790`), written back through the constant path
                // (`ValueSerializer.scala:362-370`): `7f` as `01 01`, `80` as `01 00`
                OpCode::TRUE => Ok(Expr::Const(true.into())),
                OpCode::FALSE => Ok(Expr::Const(false.into())),
                OpCode::HEIGHT => Ok(Expr::GlobalVars(GlobalVars::Height)),
                OpCode::SELF_BOX => Ok(Expr::GlobalVars(GlobalVars::SelfBox)),
                OpCode::INPUTS => Ok(Expr::GlobalVars(GlobalVars::Inputs)),
                OpCode::OUTPUTS => Ok(Expr::GlobalVars(GlobalVars::Outputs)),
                OpCode::MINER_PUBKEY => Ok(Expr::GlobalVars(GlobalVars::MinerPubKey)),
                OpCode::GROUP_GENERATOR => Ok(Expr::GlobalVars(GlobalVars::GroupGenerator)),
                OpCode::LAST_BLOCK_UTXO_ROOT_HASH => {
                    Ok(Expr::GlobalVars(GlobalVars::LastBlockUtxoRootHash))
                }
                OpCode::GLOBAL => Ok(Expr::Global),
                OpCode::PROPERTY_CALL => {
                    Ok(Expr::PropertyCall(PropertyCall::sigma_parse(r)?.into()))
                }
                OpCode::METHOD_CALL => {
                    let mc = MethodCall::sigma_parse(r)?;
                    // A MethodCall with no arguments, which parses below tree version 3, has
                    // PropertyCall as its companion and is written as one (v6.0.7
                    // `values.scala:1351`)
                    if mc.args.is_empty() {
                        Ok(Expr::PropertyCall(
                            PropertyCall {
                                obj: mc.obj,
                                method: mc.method,
                                explicit_type_args: mc.explicit_type_args,
                            }
                            .into(),
                        ))
                    } else {
                        Ok(Expr::MethodCall(mc.into()))
                    }
                }
                OpCode::CONTEXT => Ok(Expr::Context),
                OptionGet::OP_CODE => Ok(OptionGet::sigma_parse(r)?.into()),
                OptionIsDefined::OP_CODE => Ok(OptionIsDefined::sigma_parse(r)?.into()),
                OptionGetOrElse::OP_CODE => Ok(OptionGetOrElse::sigma_parse(r)?.into()),
                ExtractRegisterAs::OP_CODE => Ok(ExtractRegisterAs::sigma_parse(r)?.into()),
                ExtractScriptBytes::OP_CODE => Ok(ExtractScriptBytes::sigma_parse(r)?.into()),
                ExtractBytes::OP_CODE => Ok(ExtractBytes::sigma_parse(r)?.into()),
                ExtractBytesWithNoRef::OP_CODE => Ok(ExtractBytesWithNoRef::sigma_parse(r)?.into()),
                ExtractCreationInfo::OP_CODE => Ok(ExtractCreationInfo::sigma_parse(r)?.into()),
                ExtractId::OP_CODE => Ok(ExtractId::sigma_parse(r)?.into()),
                OpCode::EQ => Ok(bin_op_sigma_parse(RelationOp::Eq.into(), r)?),
                OpCode::NEQ => Ok(bin_op_sigma_parse(RelationOp::NEq.into(), r)?),
                Negation::OP_CODE => Ok(Negation::sigma_parse(r)?.into()),
                BitInversion::OP_CODE => Ok(BitInversion::sigma_parse(r)?.into()),
                OpCode::LOGICAL_NOT => Ok(LogicalNot::sigma_parse(r)?.into()),
                OpCode::BIN_AND => Ok(bin_op_sigma_parse(LogicalOp::And.into(), r)?),
                OpCode::BIN_OR => Ok(bin_op_sigma_parse(LogicalOp::Or.into(), r)?),
                OpCode::BIN_XOR => Ok(bin_op_sigma_parse(LogicalOp::Xor.into(), r)?),
                OpCode::GT => Ok(bin_op_sigma_parse(RelationOp::Gt.into(), r)?),
                OpCode::LT => Ok(bin_op_sigma_parse(RelationOp::Lt.into(), r)?),
                OpCode::GE => Ok(bin_op_sigma_parse(RelationOp::Ge.into(), r)?),
                OpCode::LE => Ok(bin_op_sigma_parse(RelationOp::Le.into(), r)?),
                OpCode::PLUS => Ok(bin_op_sigma_parse(ArithOp::Plus.into(), r)?),
                OpCode::MINUS => Ok(bin_op_sigma_parse(ArithOp::Minus.into(), r)?),
                OpCode::MULTIPLY => Ok(bin_op_sigma_parse(ArithOp::Multiply.into(), r)?),
                OpCode::DIVISION => Ok(bin_op_sigma_parse(ArithOp::Divide.into(), r)?),
                OpCode::MAX => Ok(bin_op_sigma_parse(ArithOp::Max.into(), r)?),
                OpCode::MIN => Ok(bin_op_sigma_parse(ArithOp::Min.into(), r)?),
                OpCode::MODULO => Ok(bin_op_sigma_parse(ArithOp::Modulo.into(), r)?),
                OpCode::BIT_OR => Ok(bin_op_sigma_parse(BitOp::BitOr.into(), r)?),
                OpCode::BIT_AND => Ok(bin_op_sigma_parse(BitOp::BitAnd.into(), r)?),
                OpCode::BIT_XOR => Ok(bin_op_sigma_parse(BitOp::BitXor.into(), r)?),
                OpCode::BIT_SHIFT_RIGHT => Ok(bin_op_sigma_parse(BitOp::BitShiftRight.into(), r)?),
                OpCode::BIT_SHIFT_LEFT => Ok(bin_op_sigma_parse(BitOp::BitShiftLeft.into(), r)?),
                OpCode::BIT_SHIFT_RIGHT_ZEROED => {
                    Ok(bin_op_sigma_parse(BitOp::BitShiftRightZeroed.into(), r)?)
                }
                OpCode::BLOCK_VALUE => Ok(Expr::BlockValue(BlockValue::sigma_parse(r)?.into())),
                OpCode::FUNC_VALUE => Ok(Expr::FuncValue(FuncValue::sigma_parse(r)?)),
                OpCode::APPLY => Ok(Expr::Apply(Apply::sigma_parse(r)?)),
                OpCode::VAL_DEF => Ok(Expr::ValDef(ValDef::sigma_parse(r)?.into())),
                OpCode::FUN_DEF => Ok(Expr::ValDef(ValDef::sigma_parse_fun_def(r)?.into())),
                OpCode::VAL_USE => Ok(Expr::ValUse(ValUse::sigma_parse(r)?)),
                ExtractAmount::OP_CODE => Ok(Expr::ExtractAmount(ExtractAmount::sigma_parse(r)?)),
                OpCode::SELECT_FIELD => Ok(Expr::SelectField(SelectField::sigma_parse(r)?.into())),
                OpCode::CALC_BLAKE2B256 => Ok(CalcBlake2b256::sigma_parse(r)?.into()),
                CalcSha256::OP_CODE => Ok(CalcSha256::sigma_parse(r)?.into()),
                And::OP_CODE => Ok(And::sigma_parse(r)?.into()),
                Or::OP_CODE => Ok(Or::sigma_parse(r)?.into()),
                Xor::OP_CODE => Ok(Xor::sigma_parse(r)?.into()),
                Atleast::OP_CODE => Ok(Atleast::sigma_parse(r)?.into()),
                OpCode::COLL => Ok(coll_sigma_parse(r)?.into()),
                OpCode::COLL_OF_BOOL_CONST => Ok(bool_const_coll_sigma_parse(r)?.into()),
                Map::OP_CODE => Ok(Map::sigma_parse(r)?.into()),
                Filter::OP_CODE => Ok(Filter::sigma_parse(r)?.into()),
                Exists::OP_CODE => Ok(Exists::sigma_parse(r)?.into()),
                ForAll::OP_CODE => Ok(ForAll::sigma_parse(r)?.into()),
                BoolToSigmaProp::OP_CODE => Ok(BoolToSigmaProp::sigma_parse(r)?.into()),
                Upcast::OP_CODE => Ok(Upcast::sigma_parse(r)?.into()),
                Downcast::OP_CODE => Ok(Downcast::sigma_parse(r)?.into()),
                If::OP_CODE => Ok(If::sigma_parse(r)?.into()),
                ByIndex::OP_CODE => Ok(ByIndex::sigma_parse(r)?.into()),
                SizeOf::OP_CODE => Ok(SizeOf::sigma_parse(r)?.into()),
                Slice::OP_CODE => Ok(Slice::sigma_parse(r)?.into()),
                CreateProveDlog::OP_CODE => Ok(CreateProveDlog::sigma_parse(r)?.into()),
                CreateProveDhTuple::OP_CODE => Ok(CreateProveDhTuple::sigma_parse(r)?.into()),
                SigmaPropBytes::OP_CODE => Ok(SigmaPropBytes::sigma_parse(r)?.into()),
                Tuple::OP_CODE => Ok(Tuple::sigma_parse(r)?.into()),
                DecodePoint::OP_CODE => Ok(DecodePoint::sigma_parse(r)?.into()),
                SubstConstants::OP_CODE => Ok(SubstConstants::sigma_parse(r)?.into()),
                ByteArrayToLong::OP_CODE => Ok(ByteArrayToLong::sigma_parse(r)?.into()),
                ByteArrayToBigInt::OP_CODE => Ok(ByteArrayToBigInt::sigma_parse(r)?.into()),
                LongToByteArray::OP_CODE => Ok(LongToByteArray::sigma_parse(r)?.into()),
                SigmaAnd::OP_CODE => Ok(SigmaAnd::sigma_parse(r)?.into()),
                SigmaOr::OP_CODE => Ok(SigmaOr::sigma_parse(r)?.into()),
                GetVar::OP_CODE => Ok(GetVar::sigma_parse(r)?.into()),
                DeserializeRegister::OP_CODE => Ok(DeserializeRegister::sigma_parse(r)?.into()),
                DeserializeContext::OP_CODE => Ok(DeserializeContext::sigma_parse(r)?.into()),
                MultiplyGroup::OP_CODE => Ok(MultiplyGroup::sigma_parse(r)?.into()),
                Exponentiate::OP_CODE => Ok(Exponentiate::sigma_parse(r)?.into()),
                XorOf::OP_CODE => Ok(XorOf::sigma_parse(r)?.into()),
                TreeLookup::OP_CODE => Ok(TreeLookup::sigma_parse(r)?.into()),
                CreateAvlTree::OP_CODE => Ok(CreateAvlTree::sigma_parse(r)?.into()),
                ModQ::OP_CODE => Ok(ModQ::sigma_parse(r)?.into()),
                SigmaPropIsProven::OP_CODE => Ok(SigmaPropIsProven::sigma_parse(r)?.into()),
                TaggedVariable::OP_CODE => Ok(TaggedVariable::sigma_parse(r)?.into()),
                OpCode::PLUS_MOD_Q => {
                    Ok(ModQArithOp::sigma_parse_operands(ModQArithOpKind::PlusModQ, r)?.into())
                }
                OpCode::MINUS_MOD_Q => {
                    Ok(ModQArithOp::sigma_parse_operands(ModQArithOpKind::MinusModQ, r)?.into())
                }
                o => Err(SigmaParsingError::NotImplementedOpCode(format!(
                    "{0}(shift {1})",
                    o.value(),
                    o.shift()
                ))),
            }?;
            // sigmastate builds the node now, its bytes read (`check_build`'s module doc)
            check_build(expr, r.tree_version())
        }
    }
}

trait SigmaSerializableWithOpCode: SigmaSerializable + HasOpCode {
    fn sigma_serialize_w_opcode<W: SigmaByteWrite>(&self, w: &mut W) -> SigmaSerializeResult {
        self.op_code().sigma_serialize(w)?;
        self.sigma_serialize(w)
    }
}

impl<T: SigmaSerializable + HasOpCode> SigmaSerializableWithOpCode for T {}

impl SigmaSerializable for Expr {
    fn sigma_serialize<W: SigmaByteWrite>(&self, w: &mut W) -> SigmaSerializeResult {
        match self {
            Expr::Const(c) => match w.constant_store_mut_ref() {
                Some(cs) => {
                    let ph = (*cs).put(c.clone());
                    ph.op_code().sigma_serialize(w)?;
                    ph.sigma_serialize(w)
                }
                None => c.sigma_serialize(w),
            },
            Expr::Append(ap) => ap.expr().sigma_serialize_w_opcode(w),
            Expr::Fold(op) => op.expr().sigma_serialize_w_opcode(w),
            Expr::ConstPlaceholder(cp) => cp.sigma_serialize_w_opcode(w),
            Expr::SubstConstants(s) => s.expr().sigma_serialize_w_opcode(w),
            Expr::ByteArrayToLong(s) => s.expr().sigma_serialize_w_opcode(w),
            Expr::ByteArrayToBigInt(s) => s.expr().sigma_serialize_w_opcode(w),
            Expr::LongToByteArray(s) => s.sigma_serialize_w_opcode(w),
            Expr::GlobalVars(op) => op.op_code().sigma_serialize(w),
            // sigmastate writes a MethodCall without arguments as a PropertyCall, its companion
            // then (v6.0.7 `values.scala:1351`)
            Expr::MethodCall(mc) if mc.expr().args.is_empty() => {
                let mc = mc.expr();
                OpCode::PROPERTY_CALL.sigma_serialize(w)?;
                super::property_call::serialize_property_call_body(
                    &mc.method,
                    &mc.obj,
                    &mc.explicit_type_args,
                    w,
                )
            }
            Expr::MethodCall(mc) => mc.expr().sigma_serialize_w_opcode(w),
            Expr::PropertyCall(pc) => pc.expr().sigma_serialize_w_opcode(w),
            Expr::Global => OpCode::GLOBAL.sigma_serialize(w),
            Expr::Context => OpCode::CONTEXT.sigma_serialize(w),
            Expr::OptionGet(v) => v.expr().sigma_serialize_w_opcode(w),
            Expr::ExtractRegisterAs(v) => v.expr().sigma_serialize_w_opcode(w),
            Expr::BinOp(op) => {
                op.expr().op_code().sigma_serialize(w)?;
                bin_op_sigma_serialize(op.expr(), w)
            }
            Expr::BlockValue(op) => op.expr().sigma_serialize_w_opcode(w),
            Expr::ValUse(op) => op.sigma_serialize_w_opcode(w),
            Expr::ValDef(op) => op.expr().sigma_serialize_w_opcode(w),
            Expr::FuncValue(op) => op.sigma_serialize_w_opcode(w),
            Expr::Apply(op) => op.sigma_serialize_w_opcode(w),
            Expr::ExtractAmount(op) => op.sigma_serialize_w_opcode(w),
            Expr::SelectField(op) => op.sigma_serialize_w_opcode(w),
            Expr::CalcBlake2b256(op) => op.sigma_serialize_w_opcode(w),
            Expr::CalcSha256(op) => op.sigma_serialize_w_opcode(w),
            Expr::Collection(op) => {
                op.op_code().sigma_serialize(w)?;
                coll_sigma_serialize(op, w)
            }
            Expr::And(op) => op.sigma_serialize_w_opcode(w),
            Expr::Or(op) => op.sigma_serialize_w_opcode(w),
            Expr::Xor(op) => op.sigma_serialize_w_opcode(w),
            Expr::Atleast(op) => op.sigma_serialize_w_opcode(w),
            Expr::LogicalNot(op) => op.sigma_serialize_w_opcode(w),
            Expr::Map(op) => op.sigma_serialize_w_opcode(w),
            Expr::Filter(op) => op.sigma_serialize_w_opcode(w),
            Expr::BoolToSigmaProp(op) => op.sigma_serialize_w_opcode(w),
            // Below tree version 3, `ValueSerializer.serialize` writes an `Upcast` whose input is
            // a constant as that constant, through the constant path (v6.0.6
            // `ValueSerializer.scala:157-169`, `:362-372`)
            Expr::Upcast(op) => match &*op.input {
                input @ Expr::Const(_) if w.tree_version() < ErgoTreeVersion::V3 => {
                    input.sigma_serialize(w)
                }
                _ => op.sigma_serialize_w_opcode(w),
            },
            Expr::Downcast(op) => op.sigma_serialize_w_opcode(w),
            Expr::If(op) => op.sigma_serialize_w_opcode(w),
            Expr::ByIndex(op) => op.expr().sigma_serialize_w_opcode(w),
            Expr::ExtractScriptBytes(op) => op.sigma_serialize_w_opcode(w),
            Expr::SizeOf(op) => op.sigma_serialize_w_opcode(w),
            Expr::Slice(op) => op.expr().sigma_serialize_w_opcode(w),
            Expr::CreateProveDlog(op) => op.sigma_serialize_w_opcode(w),
            Expr::CreateProveDhTuple(op) => op.sigma_serialize_w_opcode(w),
            Expr::ExtractCreationInfo(op) => op.sigma_serialize_w_opcode(w),
            Expr::Exists(op) => op.sigma_serialize_w_opcode(w),
            Expr::ExtractId(op) => op.sigma_serialize_w_opcode(w),
            Expr::SigmaPropBytes(op) => op.sigma_serialize_w_opcode(w),
            Expr::OptionIsDefined(op) => op.expr().sigma_serialize_w_opcode(w),
            Expr::OptionGetOrElse(op) => op.expr().sigma_serialize_w_opcode(w),
            Expr::Negation(op) => op.expr().sigma_serialize_w_opcode(w),
            Expr::BitInversion(op) => op.sigma_serialize_w_opcode(w),
            Expr::ForAll(op) => op.sigma_serialize_w_opcode(w),
            Expr::Tuple(op) => op.sigma_serialize_w_opcode(w),
            Expr::DecodePoint(op) => op.sigma_serialize_w_opcode(w),
            Expr::SigmaAnd(op) => op.sigma_serialize_w_opcode(w),
            Expr::SigmaOr(op) => op.sigma_serialize_w_opcode(w),
            Expr::GetVar(op) => op.sigma_serialize_w_opcode(w),
            Expr::DeserializeRegister(op) => op.sigma_serialize_w_opcode(w),
            Expr::DeserializeContext(op) => op.sigma_serialize_w_opcode(w),
            Expr::MultiplyGroup(op) => op.sigma_serialize_w_opcode(w),
            Expr::Exponentiate(op) => op.sigma_serialize_w_opcode(w),
            Expr::XorOf(op) => op.sigma_serialize_w_opcode(w),
            Expr::ExtractBytes(op) => op.sigma_serialize_w_opcode(w),
            Expr::ExtractBytesWithNoRef(op) => op.sigma_serialize_w_opcode(w),
            Expr::TreeLookup(op) => op.sigma_serialize_w_opcode(w),
            Expr::CreateAvlTree(op) => op.sigma_serialize_w_opcode(w),
            Expr::ModQ(op) => op.sigma_serialize_w_opcode(w),
            Expr::SigmaPropIsProven(op) => op.sigma_serialize_w_opcode(w),
            Expr::TaggedVariable(op) => op.sigma_serialize_w_opcode(w),
            // `ModQArithOpSerializer.serialize`: the opcode, then the two operands
            Expr::ModQArithOp(op) => {
                op.op_code().sigma_serialize(w)?;
                op.left.sigma_serialize(w)?;
                op.right.sigma_serialize(w)
            }
        }
    }

    fn sigma_parse<R: SigmaByteRead>(r: &mut R) -> Result<Self, SigmaParsingError> {
        // `ValueSerializer.deserialize` (`ValueSerializer.scala:396-409`) takes the value's
        // level, then peeks its first byte without the position check
        // (`CoreByteReader.scala:41`): past the end of the input that is a hard error, not
        // the soft rule-1014 one, and only the read after the peek checks the position.
        let depth = r.level();
        r.set_level(depth + 1)?;
        r.peek_u8()?;
        let tag = r.get_u8()?;
        let expr = Self::parse_tagged(r, tag)?;
        r.set_level(r.level().saturating_sub(1))?;
        Ok(expr)
    }
}

impl<T: SigmaSerializable> SigmaSerializable for Spanned<T> {
    fn sigma_serialize<W: SigmaByteWrite>(&self, w: &mut W) -> SigmaSerializeResult {
        self.expr.sigma_serialize(w)
    }

    fn sigma_parse<R: SigmaByteRead>(r: &mut R) -> Result<Self, SigmaParsingError> {
        T::sigma_parse(r).map(Into::into)
    }
}

impl<T: HasOpCode> HasOpCode for Spanned<T> {
    fn op_code(&self) -> OpCode {
        self.expr.op_code()
    }
}

#[cfg(test)]
#[cfg(feature = "arbitrary")]
#[allow(clippy::unwrap_used)]
#[allow(clippy::panic)]
mod tests {
    use crate::chain::address::AddressEncoder;
    use crate::chain::address::NetworkPrefix;

    use super::*;
    use crate::serialization::sigma_serialize_roundtrip;
    use proptest::prelude::*;

    #[test]
    fn ser_global() {
        let e = Expr::Global;
        assert_eq!(sigma_serialize_roundtrip(&e), e);
    }

    /// Regression test for ergo-node-rust mainnet sync failure at block 1,711,120 tx[1]
    /// (`dce90de114a7d36cb2dd724c04ccc11b461a02bd2a783ef820234046cb47ff2c`,
    /// block id `bcfb89f0d7c3c6c3f13d885e037c79e349a1ffa37301cb899a1a24c8f47ba49f`).
    /// `output[0]` uses `blake2b256(getVar[Coll[Byte]](1))` directly (no `.get`),
    /// which the JVM v6.0.3 reference node parses cleanly because its
    /// `OneArgumentOperationSerializer` does an unchecked `asValue[T]` cast at parse
    /// time. sigma-rust must match that leniency at parse time; runtime behavior
    /// (`EvalError::UnexpectedValue`) stays the same.
    #[test]
    fn parse_block_1711120_output0() {
        use crate::ergo_tree::ErgoTree;
        let hex = "00ea02d193b4cbe3010e040004300e183092aafc2966e1c791f619b14b084b26d41825661c6c040bd40801";
        let bytes = base16::decode(hex).unwrap();
        let tree = ErgoTree::sigma_parse_bytes(&bytes).unwrap();
        // Round-trip: serialized bytes must match the input exactly.
        let reser = tree.sigma_serialize_bytes().unwrap();
        assert_eq!(reser, bytes, "round-trip bytes must be identical");
    }

    /// Companion regression for `output[2]` of the same tx. This box has the
    /// constant-segregation header (`0x10`) and uses `0xea` (SIGMA_AND) deeper in the
    /// tree body. Already passes today, but pinned here so a future regression that
    /// only breaks `output[0]` is caught alongside one that breaks `output[2]`.
    #[test]
    fn parse_block_1711120_output2() {
        use crate::ergo_tree::ErgoTree;
        let hex = "1005040004000e36100204a00b08cd0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798ea02d192a39a8cc7a701730073011001020402d19683030193a38cc7b2a57300000193c2b2a57301007473027303830108cdeeac93b1a57304";
        let bytes = base16::decode(hex).unwrap();
        let tree = ErgoTree::sigma_parse_bytes(&bytes).unwrap();
        let reser = tree.sigma_serialize_bytes().unwrap();
        assert_eq!(reser, bytes, "round-trip bytes must be identical");
    }

    proptest! {

        #[test]
        fn ser_roundtrip(v in any::<Expr>()) {
            prop_assert_eq![sigma_serialize_roundtrip(&v), v];
        }
    }

    #[test]
    fn full_age_usd_bank_contract() {
        // almost full version of
        // https://github.com/Emurgo/age-usd/tree/main/ageusd-smart-contracts/v0.4
        /*
            {

              val rcDefaultPrice = 1000000L

              val minStorageRent = 10000000L

              val feePercent = 1

              val HEIGHT = 377771

              val coolingOffHeight: Int = 377770
              val INF = 1000000000L

              val longMax = 9223372036854775807L

              val minReserveRatioPercent = 400L // percent
              val defaultMaxReserveRatioPercent = 800L // percent

              val isExchange = if (CONTEXT.dataInputs.size > 0) {
                val dataInput = CONTEXT.dataInputs(0)
                //val validDataInput = dataInput.tokens(0)._1 == oraclePoolNFT
                val validDataInput = true

                val bankBoxIn = SELF
                val bankBoxOut = OUTPUTS(0)

                val rateBox = dataInput
                val receiptBox = OUTPUTS(1)

                val rate = rateBox.R4[Long].get / 100
                // val rate = 100000000 / 100

                val scCircIn = bankBoxIn.R4[Long].get
                // val scCircIn = 100L
                val rcCircIn = bankBoxIn.R5[Long].get
                // val rcCircIn = 100L
                val bcReserveIn = bankBoxIn.value
                // val bcReserveIn = 100000000L

                val scTokensIn = bankBoxIn.tokens(0)._2
                // val scTokensIn = 100
                val rcTokensIn = bankBoxIn.tokens(1)._2
                // val rcTokensIn = 100

                val scCircOut = bankBoxOut.R4[Long].get
                //val scCircOut = 100
                val rcCircOut = bankBoxOut.R5[Long].get
                //val rcCircOut = 101

                val scTokensOut = bankBoxOut.tokens(0)._2
                //val scTokensOut = 100
                val rcTokensOut = bankBoxOut.tokens(1)._2
                //val rcTokensOut = 99

                val totalScIn = scTokensIn + scCircIn
                val totalScOut = scTokensOut + scCircOut

                val totalRcIn = rcTokensIn + rcCircIn
                val totalRcOut = rcTokensOut + rcCircOut

                val rcExchange = rcTokensIn != rcTokensOut
                val scExchange = scTokensIn != scTokensOut

                val rcExchangeXorScExchange = (rcExchange || scExchange) && !(rcExchange && scExchange)

                val circDelta = receiptBox.R4[Long].get
                //val circDelta = 1L
                val bcReserveDelta = receiptBox.R5[Long].get
                //val bcReserveDelta = 1010000L

                val bcReserveOut = bankBoxOut.value
                //val bcReserveOut = 100000000L + 1010000L

                val rcCircDelta = if (rcExchange) circDelta else 0L
                val scCircDelta = if (rcExchange) 0L else circDelta

                val validDeltas = (scCircIn + scCircDelta == scCircOut) &&
                                  (rcCircIn + rcCircDelta == rcCircOut) &&
                                  (bcReserveIn + bcReserveDelta == bcReserveOut)

                val coinsConserved = totalRcIn == totalRcOut && totalScIn == totalScOut

                val tokenIdsConserved = bankBoxOut.tokens(0)._1 == bankBoxIn.tokens(0)._1 && // also ensures that at least one token exists
                                         bankBoxOut.tokens(1)._1 == bankBoxIn.tokens(1)._1 && // also ensures that at least one token exists
                                         bankBoxOut.tokens(2)._1 == bankBoxIn.tokens(2)._1    // also ensures that at least one token exists

                //val tokenIdsConserved = true

                //val mandatoryRateConditions = rateBox.tokens(0)._1 == oraclePoolNFT
                val mandatoryRateConditions = true
                val mandatoryBankConditions = bankBoxOut.value >= minStorageRent &&
                                              rcExchangeXorScExchange &&
                                              coinsConserved &&
                                              validDeltas &&
                                              tokenIdsConserved

                // exchange equations
                val bcReserveNeededOut = scCircOut * rate
                val bcReserveNeededIn = scCircIn * rate
                val liabilitiesIn = max(min(bcReserveIn, bcReserveNeededIn), 0)


                val maxReserveRatioPercent = if (HEIGHT > coolingOffHeight) defaultMaxReserveRatioPercent else INF

                val reserveRatioPercentOut =
                    if (bcReserveNeededOut == 0) maxReserveRatioPercent else bcReserveOut * 100 / bcReserveNeededOut

                val validReserveRatio = if (scExchange) {
                  if (scCircDelta > 0) {
                    reserveRatioPercentOut >= minReserveRatioPercent
                  } else true
                } else {
                  if (rcCircDelta > 0) {
                    reserveRatioPercentOut <= maxReserveRatioPercent
                  } else {
                    reserveRatioPercentOut >= minReserveRatioPercent
                  }
                }

                val brDeltaExpected = if (scExchange) { // sc
                  val liableRate = if (scCircIn == 0) longMax else liabilitiesIn / scCircIn
                  val scNominalPrice = min(rate, liableRate)
                  scNominalPrice * scCircDelta
                } else { // rc
                  val equityIn = bcReserveIn - liabilitiesIn
                  val equityRate = if (rcCircIn == 0) rcDefaultPrice else equityIn / rcCircIn
                  val rcNominalPrice = if (equityIn == 0) rcDefaultPrice else equityRate
                  rcNominalPrice * rcCircDelta
                }

                val fee = brDeltaExpected * feePercent / 100

                val actualFee = if (fee < 0) {fee * -1} else fee
                // actualFee is always positive, irrespective of brDeltaExpected

                val brDeltaExpectedWithFee = brDeltaExpected + actualFee

                mandatoryRateConditions &&
                mandatoryBankConditions &&
                bcReserveDelta == brDeltaExpectedWithFee &&
                validReserveRatio &&
                validDataInput
            } else false

            sigmaProp(isExchange || // INPUTS(0).tokens(0)._1 == updateNFT &&
                CONTEXT.dataInputs.size == 0)
        }
        */
        let p2s_addr_str = "HfdbQC2Zwr5vfAUxdmjmX6b3TxQbq5w764pwsz9LLKyZVhv7SpifLB22PieCgvzSaFLomv8HNr9dxxQSSYaQg6ZyFL37nPfuVib3hVL8h42jajp754NXGqv1s4eKcbPsKkBMeTmYVSSGrpnZHzjqvcT4oN8rqKGUtLVXHs4QKyBwwNQKS5KNC8DLkdvHUQRNv5r8pCJ6ehTi31h1rfLVTsaMhAeDcYCs1uS7YMXk3msfH36krAskv8TgApoFJ1DarszwiacTuE1o4N6o4PJJifAgJ1WH4XuGRieYE1k3fo631benRDQw9nQ49p4oqAda5aXTNmabAsfCgAR8jbmUzzi3UCyYJgRUtXp7ijaGfr6o3hXd5VHDZe4gM6Vw4Ly3s881WZX2WWNedrXNqKKMVXKk55jbgn3ZmFpZiLtvPHSBCG7ULyARrTz2rAUC16StdYBqPuhHpRKEx3QYeFTYJGcMbsMGompAkCxG37X7ZVs7m7xCpPuP3AqxWtWdxkTzw5FCHALsu6ZD334n8mFgn9kiif4tbShpBo1AJu6dP22XvPU3S93q5LuNaXx6d7u5VFrpQKSN6WnhkU4LUfh3t8YU1ZBATrQDGRkaji59pqoNDuwVSfn7g1UhcMWdMnwzrCNNq1jsX2KrkX7o81aS7LEmz6xAySdyvubGh51oXNd2cmgbJ9at2Tp3hNi9FwWG5iEk882AZ7gby6QktknAwyaw9CL5qdodeh4t659H42SoqK2ATtfrZgjU5b5pYAzNp9EjFHCKkYxTo7t5G1vHHZUXjTbkzc22ggJdH3BvZYEcdQtUCLbEFJSCiMp2RjxEmyh";
        let encoder = AddressEncoder::new(NetworkPrefix::Mainnet);
        let addr = encoder.parse_address_from_str(p2s_addr_str).unwrap();
        addr.script().unwrap().proposition().unwrap();
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod depth_limit_tests {
    //! JVM parity: every value is one nesting level (sigmastate v6.0.6
    //! `ValueSerializer.deserialize`, `ValueSerializer.scala:396-409`), up to
    //! `MaxTreeDepth` = 110 per reader.
    use super::*;
    use alloc::vec::Vec;

    /// `LogicalNot^k(true)`: k values, then the constant's value and its data level.
    fn logical_not_chain(k: usize) -> Vec<u8> {
        let mut e: Expr = Expr::Const(true.into());
        for _ in 0..k {
            e = LogicalNot { input: e.into() }.into();
        }
        e.sigma_serialize_bytes().unwrap()
    }

    #[test]
    fn nested_values_reach_exactly_max_tree_depth() {
        // A debug build takes ~40 KB of stack per expression level here (a release
        // build ~2 KB), more than the default test thread holds at this depth.
        std::thread::Builder::new()
            .stack_size(32 * 1024 * 1024)
            .spawn(|| {
                assert!(Expr::sigma_parse_bytes(&logical_not_chain(108)).is_ok());
                assert!(matches!(
                    Expr::sigma_parse_bytes(&logical_not_chain(109)),
                    Err(SigmaParsingError::DeserializeCallDepthExceeded(111))
                ));
            })
            .unwrap()
            .join()
            .unwrap();
    }
}

/// The never-evaluated node kinds as sigmastate 6.0.7 writes and judges them: SANTA `b246bf4`,
/// `vectors/wire/v6/authored/Box.unevaluated_node_kinds.json`, every box with its verdict
/// (true: the box is valid, false: refused). Tree versions 0 (unsized) and 3 (size-flagged).
#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod jvm_authored_unevaluated_nodes {
    use crate::chain::ergo_box::ErgoBox;
    use crate::ergo_tree::ErgoTree;
    use crate::ergo_tree::ErgoTreeError;
    use crate::serialization::SigmaParsingError;
    use crate::serialization::SigmaSerializable;

    const BOXES: &[(&str, &str, bool)] = &[
        (
            "bit_shift_left-v0",
            "c0843d00d193f7040204040408010000593042c0147b743fb62109874978602615004cead9aac8bd\
             7a2a368fbbbb82d600",
            true,
        ),
        (
            "bit_shift_right-v0",
            "c0843d00d193f6040204040408010000ae356a9b955875373603f303cc73f63b9f26927c11861624\
             f410b0ce748062b200",
            true,
        ),
        (
            "bit_shift_right_zeroed-v0",
            "c0843d00d193f80402040404080100004c14899c32b0115a2a531cd1d52c8c7df707638e1eb18cd9\
             69b3aceaf15a69ab00",
            true,
        ),
        (
            "mod_q-v0",
            "c0843d00d193e70601010601010100000159370056b1a948a6bffb29ab1cb6af1fcc3e6d734e20d6\
             f9c7a866e6ca6ba100",
            true,
        ),
        (
            "plus_mod_q-v0",
            "c0843d00d193e8060101060101060101010000594463d6495789527daf206bbaf45e0b24887662fa\
             07a63a838925b9817ee24900",
            true,
        ),
        (
            "minus_mod_q-v0",
            "c0843d00d193e9060101060101060101010000c217d900d1652c6f75c2aa442d6362f4b7430379e9\
             f61e3eed3b6e31e21cbd2800",
            true,
        ),
        (
            "sigma_prop_is_proven-v0",
            "c0843d00d1cfd10101010000d678068f06b667bd37990aeee59a2bf7c0ebf78252c98477f57f4f8c\
             4ffe7fed00",
            true,
        ),
        (
            "tagged_variable-v0",
            "c0843d00d17101010100009a05596ddd73409db51d3904cc346501ff5a412e0f7437ae46441687dc\
             e34a6400",
            true,
        ),
        (
            "bit_or-v0",
            "c0843d00d193f2040a040604020100005a7e4732c1acfaadf79ca84685e1e3b4fdb26387d2989c55\
             f9b75dbd1752cd0c00",
            true,
        ),
        (
            "bit_and-v0",
            "c0843d00d193f3040a04060402010000109bdfc04c0f181827dfb0e70499ef13d9069bd5991b25cd\
             73a61edafe38c2d100",
            true,
        ),
        (
            "bit_xor-v0",
            "c0843d00d193f5040a04060402010000237c22eabb1194a82058aef439ef8539809e4df922119e69\
             7608dd28ea5517b500",
            true,
        ),
        (
            "bit_inversion-v0",
            "c0843d00d193f1040a040b0100002f71886c59d9aa9fa14b80cc11de4dcc65edbbb09213865a41b9\
             a051d05a906f00",
            true,
        ),
        (
            "create_avl_tree_four_value_getvar-v0",
            "c0843d00d193b602070e210000000000000000000000000000000000000000000000000000000000\
             000000000440e30104b602070e210000000000000000000000000000000000000000000000000000\
             000000000000000440e30104010000761cb8a5bb9f4c14b64c8c0b7fe6ff10e495910fff3e57047c\
             c62a90a34c83fc00",
            true,
        ),
        (
            "bit_shift_left-v3",
            "c0843d0b09d193f7040204040408010000593042c0147b743fb62109874978602615004cead9aac8\
             bd7a2a368fbbbb82d600",
            true,
        ),
        (
            "bit_shift_right-v3",
            "c0843d0b09d193f6040204040408010000ae356a9b955875373603f303cc73f63b9f26927c118616\
             24f410b0ce748062b200",
            true,
        ),
        (
            "bit_shift_right_zeroed-v3",
            "c0843d0b09d193f80402040404080100004c14899c32b0115a2a531cd1d52c8c7df707638e1eb18c\
             d969b3aceaf15a69ab00",
            true,
        ),
        (
            "mod_q-v3",
            "c0843d0b09d193e70601010601010100000159370056b1a948a6bffb29ab1cb6af1fcc3e6d734e20\
             d6f9c7a866e6ca6ba100",
            true,
        ),
        (
            "plus_mod_q-v3",
            "c0843d0b0cd193e8060101060101060101010000594463d6495789527daf206bbaf45e0b24887662\
             fa07a63a838925b9817ee24900",
            true,
        ),
        (
            "minus_mod_q-v3",
            "c0843d0b0cd193e9060101060101060101010000c217d900d1652c6f75c2aa442d6362f4b7430379\
             e9f61e3eed3b6e31e21cbd2800",
            true,
        ),
        (
            "sigma_prop_is_proven-v3",
            "c0843d0b05d1cfd10101010000d678068f06b667bd37990aeee59a2bf7c0ebf78252c98477f57f4f\
             8c4ffe7fed00",
            true,
        ),
        (
            "tagged_variable-v3",
            "c0843d0b04d17101010100009a05596ddd73409db51d3904cc346501ff5a412e0f7437ae46441687\
             dce34a6400",
            true,
        ),
        (
            "create_avl_tree_four_value-v3",
            "c0843d0b56d193b602070e2100000000000000000000000000000000000000000000000000000000\
             000000000004402800b602070e210000000000000000000000000000000000000000000000000000\
             0000000000000004402800010000c376e1d477900aa96d8602642fda29a09f1e42d7a77bfc31e279\
             cbb6320feb6600",
            true,
        ),
        (
            "bit_or-v3",
            "c0843d0b09d193f2040a040604020100005a7e4732c1acfaadf79ca84685e1e3b4fdb26387d2989c\
             55f9b75dbd1752cd0c00",
            true,
        ),
        (
            "bit_and-v3",
            "c0843d0b09d193f3040a04060402010000109bdfc04c0f181827dfb0e70499ef13d9069bd5991b25\
             cd73a61edafe38c2d100",
            true,
        ),
        (
            "bit_xor-v3",
            "c0843d0b09d193f5040a04060402010000237c22eabb1194a82058aef439ef8539809e4df922119e\
             697608dd28ea5517b500",
            true,
        ),
        (
            "bit_inversion-v3",
            "c0843d0b07d193f1040a040b0100002f71886c59d9aa9fa14b80cc11de4dcc65edbbb09213865a41\
             b9a051d05a906f00",
            true,
        ),
        (
            "create-avl-tree-v0-reject",
            "c0843d00d193b602070e210000000000000000000000000000000000000000000000000000000000\
             0000000004402800b602070e21000000000000000000000000000000000000000000000000000000\
             000000000000044028000100000d352e729de58ad8bb90801cdaebb573108a32cd8506f4905dd66c\
             24e97ccc9700",
            false,
        ),
        (
            "old-layout-v0-reject",
            "c0843d00d193b602070e210000000000000000000000000000000000000000000000000000000000\
             00000000044000b602070e2100000000000000000000000000000000000000000000000000000000\
             00000000000440000100005f8fe4e6a17346b93e93f9b7e139d1e6c062f19743ed9e51f8ba2cd7a6\
             f4730200",
            false,
        ),
        (
            "old-layout-v3-reject",
            "c0843d0b54d193b602070e2100000000000000000000000000000000000000000000000000000000\
             0000000000044000b602070e21000000000000000000000000000000000000000000000000000000\
             0000000000000440000100002c1a9623e0bc8f82321c6aaabc5371a1418334387329142bc93c708d\
             13613d7a00",
            false,
        ),
        (
            "bool-operand-v0-reject",
            "c0843d00d193f7010104040408010000da6f46bf4dccb0771283d92acbe21915252a738315cb6f69\
             0a5b3661383c803b00",
            false,
        ),
        (
            "bool-operand-v3-reject",
            "c0843d0b09d193f701010404040801000072d7090bb7a7d35d3af2e58926c276e99524f484535d82\
             f235551d19d523b7a200",
            false,
        ),
        (
            "spip-filter-v0-reject",
            "c0843d00d1cfb50402d901010401010100004bba7134e7ab1b1aaa3d34a791c0fe694607ab998622\
             e3f60f7e369d78632e9200",
            false,
        ),
    ];

    /// The class of error a refused box of `BOXES` is refused with, as the box reader reports
    /// it: a row that fails for any other reason (a damaged hex string, say) is not the row.
    /// A new refused row has no class until it is given one here
    fn refusal(name: &str) -> fn(&SigmaParsingError) -> bool {
        match name {
            // an unsized tree whose body does not parse: an `Option` constant's data
            "create-avl-tree-v0-reject" => |e| {
                matches!(
                    e,
                    SigmaParsingError::UnsizedTreeValidationError(inner)
                        if matches!(
                            **inner,
                            ErgoTreeError::SigmaParsingError(SigmaParsingError::NotSupported(_))
                        )
                )
            },
            "old-layout-v0-reject" | "old-layout-v3-reject" => {
                |e| matches!(e, SigmaParsingError::InvalidTypePrefix)
            }
            "bool-operand-v0-reject" | "bool-operand-v3-reject" => {
                |e| matches!(e, SigmaParsingError::BitOpOperandsNotNumeric(_))
            }
            "spip-filter-v0-reject" => |e| matches!(e, SigmaParsingError::TypeReadFailed(_)),
            _ => panic!("{name}: a refused box with no error class"),
        }
    }

    /// A box the JVM accepts parses with its tree `Parsed` (asked directly: an `Unparsed` tree
    /// round-trips the same bytes) and is written back to the JVM's bytes; a box it refuses is
    /// refused, with the error of its own row
    #[test]
    fn jvm_authored_boxes_parse_as_sigmastate_judges_them() {
        for (name, hex, accept) in BOXES {
            let bytes = base16::decode(hex).unwrap();
            let parsed = ErgoBox::sigma_parse_bytes(&bytes);
            if *accept {
                let b = parsed.unwrap_or_else(|e| panic!("{name}: {e:?}"));
                assert!(matches!(b.ergo_tree, ErgoTree::Parsed(_)), "{name}");
                assert_eq!(b.sigma_serialize_bytes().unwrap(), bytes, "{name}");
            } else {
                let err = parsed
                    .err()
                    .unwrap_or_else(|| panic!("{name}: the box parses"));
                assert!(refusal(name)(&err), "{name}: {err:?}");
            }
        }
    }
}
