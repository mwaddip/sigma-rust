use alloc::boxed::Box;
use alloc::format;
use ergo_chain_types::Header;
use sigma_ser::ScorexSerializable;

use alloc::string::ToString;
use alloc::vec;
use alloc::vec::Vec;
use sigma_util::AsVecU8;

use crate::bigint256::BigInt256;
use crate::chain::ergo_box::ErgoBox;
use crate::ergo_tree::ErgoTreeVersion;
use crate::mir::avl_tree_data::AvlTreeData;
use crate::mir::coll_value::CollValue;
use crate::mir::constant::Literal;
use crate::mir::constant::TryExtractFrom;
use crate::mir::constant::TryExtractFromError;
use crate::mir::constant::TryExtractInto;
use crate::mir::value::debug_start;
use crate::mir::value::CollKind;
use crate::mir::value::NativeColl;
use crate::mir::value::Value;
use crate::serialization::SigmaSerializationError;
use crate::serialization::SigmaSerializeResult;
use crate::serialization::{
    sigma_byte_reader::SigmaByteRead, SigmaParsingError, SigmaSerializable,
};
use crate::sigma_protocol::{sigma_boolean::SigmaBoolean, sigma_boolean::SigmaProp};
use crate::types::stype::SType;
use crate::unsignedbigint256::UnsignedBigInt;
use ergo_chain_types::EcPoint;
use num_traits::Zero;

use super::sigma_byte_writer::SigmaByteWrite;
use super::types::TypeCode;
use crate::serialization::jvm_string::jvm_string;
use alloc::sync::Arc;

/// Used to serialize and parse `Literal` and `Value`.
pub struct DataSerializer {}

/// Writes a collection's length. sigmastate writes it with `putUShort`, which refuses a count
/// above 65535 (v6.0.6 `CoreDataSerializer.scala:51-52`). A constant read from bytes cannot
/// hold more; a collection built at evaluation can (by `Append`, `patch`, `flatMap`, or as the
/// bytes `Global.serialize` and `propBytes` return), and a script hands it to this writer
/// through `Global.serialize` and through the new values of `substConstants`.
///
/// `putUShort` adds its cost before it looks at the count (`SigmaByteWriter.scala:83-86`), so a
/// refused count has been charged.
fn put_coll_len<W: SigmaByteWrite>(w: &mut W, len: usize) -> SigmaSerializeResult {
    w.add_put_numeric_cost()?;
    let count = u16::try_from(len).map_err(|_| {
        SigmaSerializationError::NotSupported(format!(
            "a collection of {len} items is out of unsigned short range"
        ))
    })?;
    w.put_u16(count)?;
    Ok(())
}

/// A type whose data takes no bytes: `SUnit`, a collection of such elements, or a tuple of
/// such items (sigmastate 6.0.7 `CoreDataSerializer.isZeroWidth`, `CoreDataSerializer.scala:159-163`)
fn is_zero_width(tpe: &SType) -> bool {
    match tpe {
        SType::SUnit => true,
        SType::SColl(elem_tpe) => is_zero_width(elem_tpe),
        SType::STuple(tuple) => tuple.items.iter().all(is_zero_width),
        _ => false,
    }
}

/// A `String`'s data: the length of its UTF-8 bytes, then the bytes (v6.0.6
/// `CoreDataSerializer.scala:29-32`). The length is free: sigmastate writes it with
/// `putUInt(length)`, which the cost-counting writer hands to the inner writer without a cost
/// (`SigmaByteWriter.scala:105-107`)
fn put_string<W: SigmaByteWrite>(w: &mut W, s: &str) -> SigmaSerializeResult {
    w.put_usize_as_u32_unwrapped(s.len())?;
    w.add_put_chunk_cost(s.len())?;
    w.write_all(s.as_bytes())?;
    Ok(())
}

/// A group element's data. `EcPoint::scorex_serialize` writes exactly `GROUP_SIZE` (33) bytes as
/// one block; Scala meters it as `putBytes(33)` = `PutChunkCost`. (`EcPoint` is in
/// ergo-chain-types and can't reach the cost sink, so the cost is added here.)
fn put_group_element<W: SigmaByteWrite>(w: &mut W, ecp: &EcPoint) -> SigmaSerializeResult {
    w.add_put_chunk_cost(EcPoint::GROUP_SIZE)?;
    ecp.sigma_serialize(w)
}

/// A header's data (tree version 3)
fn put_header<W: SigmaByteWrite>(w: &mut W, h: &Header) -> SigmaSerializeResult {
    // Header is in ergo-chain-types and can't reach the cost sink, so mirror the
    // JVM's metered put sequence here (as for EcPoint/AvlTree), each cost before
    // the one statement that writes all the fields. Order matches
    // `ErgoHeader.sigmaSerializer` = `HeaderWithoutPowSerializer.serialize` +
    // `AutolykosSolution.sigmaSerializerV{1,2}` (sigma
    // `org/ergoplatform/{ErgoHeader,HeaderWithoutPow}.scala`); keep in sync with
    // `Header::serialize_without_pow` / `AutolykosSolution::serialize_bytes`. All
    // puts are no-info, so `putUInt(height)` is unmetered (0). Blessed v6
    // `Global.serialize[Header]` = 333 (= 89 + put_cost 244, v2/empty-unparsed).
    w.add_put_byte_cost()?; // put(version)
    w.add_put_chunk_cost(32)?; // putBytes(parentId: Digest32)
    w.add_put_chunk_cost(32)?; // putBytes(ADProofsRoot: Digest32)
    w.add_put_chunk_cost(32)?; // putBytes(transactionsRoot: Digest32)
    w.add_put_chunk_cost(33)?; // putBytes(stateRoot: ADDigest)
    w.add_put_numeric_cost()?; // putULong(timestamp)
    w.add_put_chunk_cost(32)?; // putBytes(extensionRoot: Digest32)
    w.add_put_chunk_cost(4)?; // DifficultySerializer: putBytes(nBits, 4 BE)
                              // putUInt(height): no-info, unmetered (0)
    w.add_put_chunk_cost(3)?; // putBytes(votes)
    if h.version > 1 {
        w.add_put_byte_cost()?; // putUByte(unparsedBytes.len)
        w.add_put_chunk_cost(h.unparsed_bytes.len())?; // putBytes(unparsedBytes)
    }
    // AutolykosSolution.sigmaSerializerV{1,2}
    w.add_put_chunk_cost(EcPoint::GROUP_SIZE)?; // GroupElementSerializer(pk)
    if h.version == 1 {
        // GroupElementSerializer(w) — one-time pk, v1 only
        w.add_put_chunk_cost(EcPoint::GROUP_SIZE)?;
    }
    w.add_put_chunk_cost(8)?; // putBytes(nonce)
    if h.version == 1 {
        w.add_put_byte_cost()?; // putUByte(dBytes.len) — v1 only
                                // putBytes(dBytes): length per sigma `asUnsignedByteArray` — empty for
                                // zero, else minimal big-endian (matches `fix/autolykos-v1-zero-distance`).
        let d_len = h.autolykos_solution.pow_distance.as_ref().map_or(0, |d| {
            if d.is_zero() {
                0
            } else {
                d.to_bytes_be().len()
            }
        });
        w.add_put_chunk_cost(d_len)?;
    }
    h.scorex_serialize(w)?;
    Ok(())
}

/// The error of a value that is not of its static type's class, where sigmastate's cast throws
fn another_class(tpe: &SType, value: &dyn core::fmt::Debug) -> SigmaSerializationError {
    TryExtractFromError(format!(
        "a value of type {} cannot be {}",
        debug_start(tpe),
        debug_start(value)
    ))
    .into()
}

/// The items of a collection whose array sigmastate casts to a primitive array, read one at a
/// time into the buffer the put takes
fn primitive_items<'ctx, T: TryExtractFrom<Value<'ctx>>>(
    coll: &CollValue<'ctx>,
) -> Result<Vec<T>, TryExtractFromError> {
    (0..coll.len())
        .filter_map(|i| coll.get_val(i))
        .map(|item| item.try_extract_into::<T>())
        .collect()
}

impl DataSerializer {
    /// Serialize Literal without typecode serialized
    pub fn sigma_serialize<W: SigmaByteWrite>(c: &Literal, w: &mut W) -> SigmaSerializeResult {
        // for reference see http://github.com/ScorexFoundation/sigmastate-interpreter/blob/25251c1313b0131835f92099f02cef8a5d932b5e/sigmastate/src/main/scala/sigmastate/serialization/DataSerializer.scala#L26-L26
        Ok(match c {
            Literal::Unit => (),
            Literal::Boolean(v) => {
                w.add_put_byte_cost()?;
                w.put_u8(u8::from(*v))?;
            }
            Literal::Byte(v) => {
                w.add_put_byte_cost()?;
                w.put_i8(*v)?;
            }
            Literal::Short(v) => {
                w.add_put_numeric_cost()?;
                w.put_i16(*v)?;
            }
            Literal::Int(v) => {
                w.add_put_numeric_cost()?;
                w.put_i32(*v)?;
            }
            Literal::Long(v) => {
                w.add_put_numeric_cost()?;
                w.put_i64(*v)?;
            }
            Literal::BigInt(v) => {
                v.sigma_serialize(w)?;
            }
            Literal::String(s) => put_string(w, s)?,
            Literal::GroupElement(ecp) => put_group_element(w, ecp)?,
            Literal::SigmaProp(s) => s.value().sigma_serialize(w)?,
            Literal::UnsignedBigInt(v) if w.tree_version() >= ErgoTreeVersion::V3 => {
                v.sigma_serialize(w)?
            }
            Literal::UnsignedBigInt(_) => {
                return Err(SigmaSerializationError::NotSupported(
                    "Can't serialize UnsignedBigInt with tree version < 3".into(),
                ))
            }
            Literal::AvlTree(a) => a.sigma_serialize(w)?,
            Literal::CBox(b) => b.sigma_serialize(w)?,
            Literal::Coll(ct) => match ct {
                CollKind::NativeColl(NativeColl::CollByte(b)) => {
                    put_coll_len(w, b.len())?;
                    w.add_put_chunk_cost(b.len())?;
                    w.write_all(b.clone().as_vec_u8().as_slice())?;
                }
                CollKind::WrappedColl {
                    elem_tpe: SType::SBoolean,
                    items: v,
                } => {
                    put_coll_len(w, v.len())?;
                    let maybe_bools: Result<Vec<bool>, TryExtractFromError> = v
                        .clone()
                        .iter()
                        .cloned()
                        .map(|i| i.try_extract_into::<bool>())
                        .collect();
                    // the array is cast before `putBits` adds its cost
                    let bools = maybe_bools?;
                    w.add_put_chunk_cost(v.len())?;
                    w.put_bits(bools.as_slice())?;
                }
                // sigmastate refuses to write a collection of zero-width elements before its
                // length (6.0.7 `CoreDataSerializer.scala:50-55`)
                CollKind::WrappedColl { elem_tpe, items: _ } if is_zero_width(elem_tpe) => {
                    return Err(SigmaSerializationError::NotSupported(format!(
                        "collection with zero-width element type {elem_tpe:?} cannot be serialized"
                    )))
                }
                CollKind::WrappedColl {
                    elem_tpe: _,
                    items: v,
                } => {
                    put_coll_len(w, v.len())?;
                    v.iter()
                        .try_for_each(|e| DataSerializer::sigma_serialize(e, w))?
                }
            },
            Literal::Tup(items) => items
                .iter()
                .try_for_each(|i| DataSerializer::sigma_serialize(i, w))?,
            Literal::Header(h) if w.tree_version() >= ErgoTreeVersion::V3 => put_header(w, h)?,
            Literal::Opt(opt) if w.tree_version() >= ErgoTreeVersion::V3 => {
                // the tag's cost before the tag (`SigmaByteWriter.scala:171-174`)
                w.add_put_byte_cost()?;
                w.put_option(Option::as_ref(opt), |w, v| {
                    DataSerializer::sigma_serialize(v, w)
                })?;
            }
            // unsupported, see
            // https://github.com/ScorexFoundation/sigmastate-interpreter/issues/659
            Literal::Opt(_) => {
                return Err(SigmaSerializationError::NotSupported(
                    "Option serialization is not supported".to_string(),
                ));
            }
            Literal::Header(_) => {
                return Err(SigmaSerializationError::NotSupported(
                    "Header serialization is not supported".to_string(),
                ));
            }
        })
    }

    /// Writes a runtime value as data of type `tpe`, without the type: sigmastate's
    /// `DataSerializer.serialize(value, tpe, w)` (v6.0.6 `CoreDataSerializer.scala:22-87`,
    /// `DataSerializer.scala:15-24`), which `Global.serialize` calls with its argument's static
    /// type (`methods.scala:1982`).
    ///
    /// The value is read by `tpe` and cast to the type's class where the type's arm reads it, so
    /// whatever comes before a value of another class has been written. A tuple type takes a
    /// pair, or a collection of `Any` holding as many items as the type (`fromDslTuple`,
    /// `Evaluation.scala:91-96`, and the length assert, `CoreDataSerializer.scala:69`). A
    /// collection type takes any collection and writes its items by the item type; for
    /// `Coll[Byte]` and `Coll[Boolean]` the JVM casts the collection's array, after the length
    /// (`CoreDataSerializer.scala:54-57`). `Unit` writes nothing and reads no value (`:23`). A
    /// type with no arm (`Any`, a function, a type variable, `Global`, `Context`, `PreHeader`)
    /// is an error (`:84-86`).
    ///
    /// Each put's cost goes to the writer's cost sink BEFORE the put
    /// ([`SigmaByteWriter::with_cost_sink`](super::sigma_byte_writer::SigmaByteWriter::with_cost_sink)).
    /// A value built at evaluation can hold one collection many times over, so it can be far
    /// larger written than it cost to build: the costs, added as the write goes, are what bounds
    /// the work. That is why this writer walks the value and does not convert it to data first:
    /// a collection's items are read one at a time, and the bytes or bits of a `Coll[Byte]` or
    /// `Coll[Boolean]` are gathered after their cost. What is copied before its own cost is one
    /// item at a time (the collections inside it are shared, not copied) and a tree's digest.
    pub fn sigma_serialize_value<W: SigmaByteWrite>(
        value: &Value<'_>,
        tpe: &SType,
        w: &mut W,
    ) -> SigmaSerializeResult {
        let v3 = w.tree_version() >= ErgoTreeVersion::V3;
        match (tpe, value) {
            (SType::SUnit, _) => {}
            (SType::SBoolean, Value::Boolean(v)) => {
                w.add_put_byte_cost()?;
                w.put_u8(u8::from(*v))?;
            }
            (SType::SByte, Value::Byte(v)) => {
                w.add_put_byte_cost()?;
                w.put_i8(*v)?;
            }
            (SType::SShort, Value::Short(v)) => {
                w.add_put_numeric_cost()?;
                w.put_i16(*v)?;
            }
            (SType::SInt, Value::Int(v)) => {
                w.add_put_numeric_cost()?;
                w.put_i32(*v)?;
            }
            (SType::SLong, Value::Long(v)) => {
                w.add_put_numeric_cost()?;
                w.put_i64(*v)?;
            }
            (SType::SString, Value::String(s)) => put_string(w, s)?,
            // data is 256-bit: `require(bi.fitsIn256Bits)`, before the length
            // (`CoreDataSerializer.scala:35`)
            (SType::SBigInt, Value::BigInt(v)) => BigInt256::try_from(v.clone())
                .map_err(TryExtractFromError)?
                .sigma_serialize(w)?,
            (SType::SUnsignedBigInt, Value::UnsignedBigInt(v)) if v3 => v.sigma_serialize(w)?,
            (SType::SGroupElement, Value::GroupElement(ecp)) => put_group_element(w, ecp)?,
            (SType::SSigmaProp, Value::SigmaProp(s)) => s.value().sigma_serialize(w)?,
            // the digest's array is cast to bytes before anything of the tree is written
            // (`AvlTreeData.scala:75`)
            (SType::SAvlTree, Value::AvlTree(tree)) => {
                if tree.digest.elem_tpe() != SType::SByte {
                    return Err(another_class(tpe, &tree.digest));
                }
                AvlTreeData::try_from((**tree).clone())?.sigma_serialize(w)?
            }
            (SType::SBox, Value::CBox(b)) => b.sigma_serialize(w)?,
            (SType::SHeader, Value::Header(h)) if v3 => put_header(w, h)?,
            (SType::SColl(elem_tpe), Value::Coll(coll)) => {
                // sigmastate refuses to write a collection of zero-width elements before its
                // length (6.0.7 `CoreDataSerializer.scala:50-55`)
                if is_zero_width(elem_tpe) {
                    return Err(SigmaSerializationError::NotSupported(format!(
                        "collection with zero-width element type {} cannot be serialized",
                        debug_start(&**elem_tpe)
                    )));
                }
                put_coll_len(w, coll.len())?;
                match **elem_tpe {
                    // the cast of the collection's array, after the length and before the
                    // items' cost
                    SType::SByte | SType::SBoolean if coll.elem_tpe() != **elem_tpe => {
                        return Err(another_class(tpe, coll))
                    }
                    SType::SByte => {
                        w.add_put_chunk_cost(coll.len())?;
                        let bytes: Vec<u8> = primitive_items::<i8>(coll)?.as_vec_u8();
                        w.write_all(&bytes)?;
                    }
                    SType::SBoolean => {
                        w.add_put_chunk_cost(coll.len())?;
                        w.put_bits(&primitive_items::<bool>(coll)?)?;
                    }
                    _ => {
                        for item in (0..coll.len()).filter_map(|i| coll.get_val(i)) {
                            DataSerializer::sigma_serialize_value(&item, elem_tpe, w)?;
                        }
                    }
                }
            }
            (SType::STuple(t), Value::Pair(pair)) if t.items.len() == 2 => {
                for (item, item_tpe) in [&pair.0, &pair.1].into_iter().zip(t.items.iter()) {
                    DataSerializer::sigma_serialize_value(item, item_tpe, w)?;
                }
            }
            (SType::STuple(t), Value::Coll(coll))
                if coll.elem_tpe() == SType::SAny && coll.len() == t.items.len() =>
            {
                for (i, item_tpe) in t.items.iter().enumerate() {
                    if let Some(item) = coll.get_val(i) {
                        DataSerializer::sigma_serialize_value(&item, item_tpe, w)?;
                    }
                }
            }
            (SType::SOption(elem_tpe), Value::Opt(opt)) if v3 => {
                // the tag's cost before the tag (`SigmaByteWriter.scala:171-174`)
                w.add_put_byte_cost()?;
                w.put_option(opt.as_deref(), |w, v| {
                    DataSerializer::sigma_serialize_value(v, elem_tpe, w)
                })?;
            }
            _ => return Err(another_class(tpe, value)),
        }
        Ok(())
    }

    /// Parse sigma-serialized literal
    pub fn sigma_parse<R: SigmaByteRead>(
        tpe: &SType,
        r: &mut R,
    ) -> Result<Literal, SigmaParsingError> {
        // for reference see http://github.com/ScorexFoundation/sigmastate-interpreter/blob/25251c1313b0131835f92099f02cef8a5d932b5e/sigmastate/src/main/scala/sigmastate/serialization/DataSerializer.scala#L84-L84
        // Every data value is one nesting level, released only on success
        // (sigmastate v6.0.6 `CoreDataSerializer.deserialize`,
        // `CoreDataSerializer.scala:94-148`, and the Box/Header arms of
        // `DataSerializer.deserialize`, `DataSerializer.scala:31-49`).
        let depth = r.level();
        r.set_level(depth + 1)?;
        use SType::*;
        let literal = match tpe {
            // NoType has no wire form: type code 0 is rejected as a hard InvalidTypePrefix
            // before a value is ever read, so this arm is unreachable from the wire
            SNoType => {
                return Err(SigmaParsingError::NotSupported(
                    "NoType has no serialized data form",
                ))
            }
            SBoolean => Literal::Boolean(r.get_u8()? != 0),
            SByte => Literal::Byte(r.get_i8()?),
            SShort => Literal::Short(r.get_i16()?),
            SInt => Literal::Int(r.get_i32()?),
            SLong => Literal::Long(r.get_i64()?),
            SString => {
                let len = r.get_u32()?;
                r.check_remaining(len as usize)?;
                let mut buf = vec![0; len as usize];
                r.get_bytes_into(&mut buf)?;
                Literal::String(jvm_string(&buf).into())
            }
            SBigInt => Literal::BigInt(BigInt256::sigma_parse(r)?),
            SUnit => Literal::Unit,
            SGroupElement => Literal::GroupElement(Arc::new(EcPoint::sigma_parse(r)?)),
            SSigmaProp => {
                Literal::SigmaProp(Box::new(SigmaProp::new(SigmaBoolean::sigma_parse(r)?)))
            }
            SUnsignedBigInt if r.tree_version() >= ErgoTreeVersion::V3 => {
                Literal::UnsignedBigInt(UnsignedBigInt::sigma_parse(r)?)
            }
            SColl(elem_type) if **elem_type == SByte => {
                let len = r.get_u16()? as usize;
                let mut buf = vec![0u8; len];
                r.get_bytes_into(&mut buf)?;
                Literal::Coll(CollKind::NativeColl(NativeColl::CollByte(
                    buf.into_iter().map(|v| v as i8).collect(),
                )))
            }
            SColl(elem_type) if **elem_type == SBoolean => {
                let len = r.get_u16()? as usize;
                let bools = r.get_bits(len)?;
                Literal::Coll(CollKind::WrappedColl {
                    elem_tpe: (**elem_type).clone(),
                    items: bools.into_iter().map(|b| b.into()).collect(),
                })
            }
            SColl(elem_type) => {
                let len = r.get_u16()? as usize;
                // Rule 1020, after the length and before any element (6.0.7
                // `CoreDataSerializer.scala:136-138`, `:166-169`)
                if is_zero_width(elem_type) {
                    return Err(SigmaParsingError::ZeroWidthCollection(format!(
                        "{elem_type:?}"
                    )));
                }
                // The item's runtime type is resolved next, before any item is read and also
                // for none: a pair's and any other type's by `stypeToRType`, which throws where
                // there is none; a tuple of another size is read as `Coll[Any]` and resolves
                // nothing (6.0.7 `CoreDataSerializer.deserializeColl`)
                let read_as_any = matches!(&**elem_type, STuple(t) if t.items.len() != 2);
                if !read_as_any && !elem_type.has_runtime_type() {
                    return Err(SigmaParsingError::NoRuntimeType(format!("{elem_type:?}")));
                }
                let elems = (0..len)
                    .map(|_| DataSerializer::sigma_parse(elem_type, r))
                    .collect::<Result<Arc<[_]>, SigmaParsingError>>()?;
                Literal::Coll(CollKind::WrappedColl {
                    elem_tpe: (**elem_type).clone(),
                    items: elems,
                })
            }
            STuple(tuple) => {
                let types = &tuple.items;
                // an item's data for each item type, any count (v6.0.7
                // `CoreDataSerializer.scala:139-144`)
                Literal::Tup(types.try_mapped_ref(|tpe| DataSerializer::sigma_parse(tpe, r))?)
            }
            SUnsignedBigInt => {
                return Err(SigmaParsingError::NotSupported(
                    "UnsignedBigInt can't be serialized on tree versions < 3",
                ))
            }
            SOption(inner_tpe) if r.tree_version() >= ErgoTreeVersion::V3 => {
                let res = r.get_option(|r| DataSerializer::sigma_parse(inner_tpe, r))?;
                Literal::Opt(res.map(Box::new))
            }
            SBox => Literal::CBox(Arc::new(ErgoBox::sigma_parse(r)?).into()),
            SAvlTree => Literal::AvlTree(Box::new(AvlTreeData::sigma_parse(r)?)),
            SHeader if r.tree_version() >= ErgoTreeVersion::V3 => {
                Literal::Header(Box::new(Header::scorex_parse(r)?))
            }
            // Non-serializable constant data types: mirror sigma-state's
            // `CoreDataSerializer` fallback + rule 1009 (`CheckSerializableTypeCode`).
            // A type code that is neither `OptionTypeCode` (36) nor `> LastDataType`
            // (111) is NOT soft-forkable — the JVM throws a hard `SerializerException`
            // that escapes `deserializeErgoTree`'s `UnparsedErgoTree` fallback, so a
            // size-flagged tree carrying one is rejected (`NonSerializableTypeCode`).
            // `SOption` (36) and `SFunc` (112 > 111) ARE soft-forkable and keep the
            // degradable `NotSupported` (a size-flagged tree carrying one degrades to
            // `Unparsed`, matching the JVM soft-fork).
            STypeVar(_) => {
                return Err(SigmaParsingError::NonSerializableTypeCode(
                    TypeCode::STYPE_VAR.value(),
                ))
            }
            SAny => {
                return Err(SigmaParsingError::NonSerializableTypeCode(
                    TypeCode::SANY.value(),
                ))
            }
            SOption(_) => return Err(SigmaParsingError::NotSupported("SOption data")),
            SFunc(_) => return Err(SigmaParsingError::NotSupported("SFunc data")),
            SContext => {
                return Err(SigmaParsingError::NonSerializableTypeCode(
                    TypeCode::SCONTEXT.value(),
                ))
            }
            SHeader => {
                return Err(SigmaParsingError::NonSerializableTypeCode(
                    TypeCode::SHEADER.value(),
                ))
            }
            SPreHeader => {
                return Err(SigmaParsingError::NonSerializableTypeCode(
                    TypeCode::SPRE_HEADER.value(),
                ))
            }
            SGlobal => {
                return Err(SigmaParsingError::NonSerializableTypeCode(
                    TypeCode::SGLOBAL.value(),
                ))
            }
        };
        r.set_level(r.level().saturating_sub(1))?;
        Ok(literal)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod depth_limit_tests {
    //! JVM parity: every data value is one nesting level (sigmastate v6.0.6
    //! `CoreDataSerializer.deserialize`, `CoreDataSerializer.scala:94-148`), up to
    //! `MaxTreeDepth` = 110 per reader. A constant read on its own, as a segregated
    //! tree constant is, counts data levels only.
    use crate::mir::constant::Constant;
    use crate::serialization::sigma_byte_reader::{from_bytes, SigmaByteRead};
    use crate::serialization::{SigmaParsingError, SigmaSerializable};
    use alloc::vec::Vec;

    /// `Coll^n[Byte]` constant (n ≥ 2): type `0c`×(n−2) `1a`, data `01`×(n−1) `00`.
    fn coll_n_byte(n: usize) -> Vec<u8> {
        let mut bytes = vec![0x0c; n - 2];
        bytes.push(0x1a);
        bytes.extend(vec![0x01; n - 1]);
        bytes.push(0x00);
        bytes
    }

    /// `Coll^n[Int]` constant (n ≥ 2) whose innermost `Coll[Int]` is empty or holds
    /// one `Int` 0: type `0c`×(n−2) `1c`, data `01`×(n−1) then the innermost.
    fn coll_n_int(n: usize, innermost_holds_an_int: bool) -> Vec<u8> {
        let mut bytes = vec![0x0c; n - 2];
        bytes.push(0x1c);
        bytes.extend(vec![0x01; n - 1]);
        if innermost_holds_an_int {
            bytes.extend([0x01, 0x00]);
        } else {
            bytes.push(0x00);
        }
        bytes
    }

    fn depth_exceeded(r: Result<Constant, SigmaParsingError>) -> bool {
        matches!(r, Err(SigmaParsingError::DeserializeCallDepthExceeded(111)))
    }

    /// Parse `bytes` as a constant on a reader already `level` deep. A type nests at most
    /// `MaxTypeDepth` (8) deep, so a collection alone cannot reach `MaxTreeDepth` (110) from
    /// level 0: start near it instead.
    fn parse_at_level(level: usize, bytes: &[u8]) -> Result<Constant, SigmaParsingError> {
        let mut r = from_bytes(bytes);
        r.set_level(level)?;
        Constant::sigma_parse(&mut r)
    }

    #[test]
    fn nested_coll_of_bytes_reaches_exactly_max_tree_depth() {
        // Coll[Byte] reads its bytes in bulk, so Coll^n[Byte] takes n levels.
        assert!(parse_at_level(102, &coll_n_byte(8)).is_ok());
        assert!(depth_exceeded(parse_at_level(102, &coll_n_byte(9))));
    }

    #[test]
    fn each_element_of_a_non_byte_coll_is_one_more_level() {
        // Coll[Int] reads each element as a data value: one level past the collection.
        assert!(parse_at_level(101, &coll_n_int(8, true)).is_ok());
        assert!(depth_exceeded(parse_at_level(101, &coll_n_int(9, true))));
        // An empty innermost Coll[Int] reads no element.
        assert!(parse_at_level(101, &coll_n_int(9, false)).is_ok());
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod runtime_type_tests {
    //! Collection data whose item type has no runtime type is refused after its length and
    //! before any item, on the type and not on the count ([`SType::has_runtime_type`]).
    //! Encodings and verdicts measured on the sigma-state 6.0.7 jars (`ValueSerializer` for a
    //! constant, `ErgoTreeSerializer` for a tree; the round's `measure/`).
    use crate::mir::constant::Constant;
    use crate::serialization::{SigmaParsingError, SigmaSerializable};
    use alloc::format;
    use alloc::string::String;
    use alloc::vec;
    use alloc::vec::Vec;

    const T: &str = "670154";

    fn parse(hex: &str) -> Result<Constant, SigmaParsingError> {
        Constant::sigma_parse_bytes(&base16::decode(hex.as_bytes()).unwrap())
    }

    fn no_runtime_type(hex: &str) -> bool {
        matches!(parse(hex), Err(SigmaParsingError::NoRuntimeType(_)))
    }

    /// Item types with no runtime type that the reader resolves
    fn items_without() -> Vec<String> {
        vec![
            T.into(),                  // T
            format!("0c{T}"),          // Coll[T]
            format!("24{T}"),          // Option[T]
            format!("3c{T}04"),        // (T, Int)
            format!("40{T}"),          // (Int, T)
            format!("0c480404{T}"),    // Coll[(Int, Int, T)]
            "700204040400".into(),     // (Int, Int) => Int
            "70000400".into(),         // () => Int
            "7001040401670154".into(), // [T](Int) => Int
            format!("7001{T}0400"),    // T => Int
            "0c700204040400".into(),   // Coll[(Int, Int) => Int]
        ]
    }

    #[test]
    fn a_collection_of_items_with_no_runtime_type_is_refused_whatever_its_length() {
        for item in items_without() {
            // no items; one item, whose byte is never read; an inner collection of none
            assert!(no_runtime_type(&format!("0c{item}00")), "{item}");
            assert!(no_runtime_type(&format!("0c{item}0100")), "{item}");
            assert!(no_runtime_type(&format!("0c0c{item}0100")), "{item}");
        }
    }

    #[test]
    fn a_collection_of_tuples_of_other_than_two_items_resolves_no_runtime_type() {
        // `Coll[(Int, Int, T)]()`: such an item is read as `Coll[Any]`, whatever its items are
        assert!(parse(&format!("0c480404{T}00")).is_ok());
        // and so are tuples of four and of five items
        assert!(parse(&format!("0c54040404{T}00")).is_ok());
        assert!(parse(&format!("0c600504040404{T}00")).is_ok());
        // one level up the item is a collection, and its type is resolved
        assert!(no_runtime_type(&format!("0c0c480404{T}0100")));
        // with an item, the tuple's own items are read: `T` has no data
        assert!(matches!(
            parse(&format!("0c480404{T}0102020400")),
            Err(SigmaParsingError::NonSerializableTypeCode(103))
        ));
    }

    #[test]
    fn data_that_is_no_collection_resolves_no_runtime_type() {
        // `None: Option[T]`, and `(1, None: Option[T])`
        assert!(parse(&format!("24{T}00")).is_ok());
        assert!(parse(&format!("4024{T}0200")).is_ok());
    }

    #[test]
    fn a_collection_of_items_with_a_runtime_type_is_read() {
        // Coll[Int](), Coll[Any](), Coll[Option[Int]](), Coll[(Int, Int, Int)](),
        // Coll[Box](), Coll[Int => Int]()
        for item in ["04", "61", "28", "48040404", "63", "7001040400"] {
            let hex = if item == "04" {
                "1000".into()
            } else {
                format!("0c{item}00")
            };
            assert!(parse(&hex).is_ok(), "{item}");
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod zero_width_tests {
    //! Rule 1020 `CheckZeroWidthCollection` (sigmastate 6.0.7): collection data whose
    //! element type takes no bytes is refused, on the type and not on the count. Encodings
    //! and verdicts from SANTA's measurement on sigma-state 6.0.7
    //! (`sigma-rust-zero-width-collection-reply.md`).
    use crate::mir::constant::{Constant, Literal};
    use crate::mir::value::CollKind;
    use crate::serialization::{SigmaParsingError, SigmaSerializable};
    use crate::types::stype::SType;

    fn zero_width(bytes: &[u8]) -> bool {
        matches!(
            Constant::sigma_parse_bytes(bytes),
            Err(SigmaParsingError::ZeroWidthCollection(_))
        )
    }

    #[test]
    fn a_collection_of_zero_width_elements_is_refused_whatever_its_length() {
        assert!(zero_width(&[0x0c, 0x62, 0x03])); // Coll[Unit] of 3
        assert!(zero_width(&[0x0c, 0x62, 0x00])); // Coll[Unit]()
        assert!(zero_width(&[0x0c, 0x0c, 0x62, 0x01])); // Coll[Coll[Unit]]
        assert!(zero_width(&[0x0c, 0x3c, 0x62, 0x62, 0x01])); // Coll[(Unit, Unit)]
    }

    #[test]
    fn a_tuple_with_one_sized_item_and_a_bare_unit_are_not_zero_width() {
        // Coll[(Unit, Int)](((), 1)) and Unit
        assert!(Constant::sigma_parse_bytes(&[0x0c, 0x4c, 0x62, 0x01, 0x02]).is_ok());
        assert_eq!(
            Constant::sigma_parse_bytes(&[0x62]).unwrap(),
            Constant {
                tpe: SType::SUnit,
                v: Literal::Unit
            }
        );
    }

    #[test]
    fn a_collection_of_zero_width_elements_is_not_written() {
        let c = Constant {
            tpe: SType::SColl(SType::SUnit.into()),
            v: Literal::Coll(CollKind::WrappedColl {
                elem_tpe: SType::SUnit,
                items: vec![Literal::Unit; 3].into(),
            }),
        };
        assert!(c.sigma_serialize_bytes().is_err());
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod small_tuple_constant_tests {
    use crate::mir::constant::Constant;
    use crate::mir::constant::Literal;
    use crate::serialization::SigmaParsingError;
    use crate::serialization::SigmaSerializable;
    use crate::types::stuple::STuple;
    use crate::types::stuple::TupleItems;
    use crate::types::stype::SType;
    use alloc::vec;

    #[test]
    fn small_tuple_constants_parse() {
        // their data reads item by item (v6.0.7 `CoreDataSerializer.scala:139-144`); their type
        // cannot be written back (SANTA cases 1, 2, 6a)
        let one = Constant::sigma_parse_bytes(&[0x60, 0x01, 0x04, 0x02]).unwrap();
        assert_eq!(
            one.tpe,
            SType::STuple(
                STuple {
                    items: TupleItems::try_from(vec![SType::SInt]).unwrap()
                }
                .into()
            )
        );
        assert_eq!(
            one.v,
            Literal::Tup(TupleItems::try_from(vec![Literal::Int(1)]).unwrap())
        );
        let empty = Constant::sigma_parse_bytes(&[0x60, 0x00]).unwrap();
        assert_eq!(empty.v, Literal::Tup(TupleItems::try_from(vec![]).unwrap()));
        assert!(one.sigma_serialize_bytes().is_err());
        assert!(empty.sigma_serialize_bytes().is_err());
    }

    #[test]
    fn a_collection_of_empty_tuples_is_rule_1020() {
        // `Coll[()]` of 5: an empty tuple is zero-width (`CoreDataSerializer.scala:162`)
        assert!(matches!(
            Constant::sigma_parse_bytes(&[0x0c, 0x60, 0x00, 0x05]),
            Err(SigmaParsingError::ZeroWidthCollection(_))
        ));
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod collection_length_tests {
    //! A collection's length is an unsigned short: sigmastate's `putUShort` throws `Value N is
    //! out of unsigned short range` above 65535 (v6.0.6 `CoreDataSerializer.scala:51-52`;
    //! measured on the sigma-state 6.0.7 jars, the round's `measure/rows/eval2.txt`). No reader
    //! gives such a collection; a script builds one.
    use crate::mir::constant::Constant;
    use crate::serialization::SigmaSerializable;
    use crate::serialization::SigmaSerializationError;
    use alloc::vec;
    use alloc::vec::Vec;

    fn refused(c: &Constant) -> bool {
        matches!(
            c.sigma_serialize_bytes(),
            Err(SigmaSerializationError::NotSupported(_))
        )
    }

    #[test]
    fn a_collection_of_65535_items_is_written() {
        // the type, the length `ff ff 03`, the items
        let rows: [(Constant, usize); 3] = [
            (vec![1i8; 65535].into(), 65535),
            (vec![true; 65535].into(), 8192),
            (vec![1i32; 65535].into(), 65535),
        ];
        for (constant, items_len) in rows {
            let bytes = constant.sigma_serialize_bytes().unwrap();
            assert_eq!(bytes.len(), 1 + 3 + items_len, "{:?}", constant.tpe);
            assert_eq!(bytes[1..4], [0xff, 0xff, 0x03], "{:?}", constant.tpe);
        }
    }

    #[test]
    fn a_collection_of_more_than_65535_items_is_refused() {
        for n in [65536usize, 70000] {
            // each of the writer's three collection arms
            assert!(refused(&vec![1i8; n].into()), "bytes {n}");
            assert!(refused(&vec![true; n].into()), "booleans {n}");
            assert!(refused(&vec![1i32; n].into()), "ints {n}");
        }
        // an inner collection's length is written the same way, in each arm
        let bytes: Vec<Vec<i8>> = vec![vec![1i8; 65536]];
        assert!(refused(&bytes.into()), "inner bytes");
        let bools: Vec<Vec<bool>> = vec![vec![true; 65536]];
        assert!(refused(&bools.into()), "inner booleans");
        let ints: Vec<Vec<i32>> = vec![vec![1i32; 65536]];
        assert!(refused(&ints.into()), "inner ints");
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod value_writer_tests {
    use super::*;
    use crate::chain::context::CostLimitExceeded;
    use crate::mir::avl_tree_data::AvlTreeFlags;
    use crate::mir::avl_tree_data::AvlTreeValue;
    use crate::serialization::sigma_byte_writer::SigmaByteWriter;
    use crate::sigma_protocol::sigma_boolean::ProveDlog;
    use crate::types::stuple::STuple;
    use core::cell::Cell;
    use ergo_chain_types::ec_point::generator;
    use num_bigint::BigInt;

    use ErgoTreeVersion as Version;

    /// What a write at a tree version gave, and the sum of the costs it added, also when it failed
    fn written_at(
        version: Version,
        value: &Value<'_>,
        tpe: &SType,
    ) -> (Result<Vec<u8>, SigmaSerializationError>, u64) {
        let mut buf = Vec::new();
        let cost = Cell::new(0u64);
        let sink = |c: u64| {
            cost.set(cost.get() + c);
            Ok(())
        };
        let res = {
            let mut w = SigmaByteWriter::with_cost_sink(&mut buf, None, &sink);
            w.with_tree_version(version, |w| {
                DataSerializer::sigma_serialize_value(value, tpe, w)
            })
        };
        (res.map(|()| buf), cost.get())
    }

    fn written(value: &Value<'_>, tpe: &SType) -> (Result<Vec<u8>, SigmaSerializationError>, u64) {
        written_at(Version::V3, value, tpe)
    }

    /// Data as the data writer writes it at tree version 3, and the sum of its costs
    fn data(lit: &Literal) -> (Result<Vec<u8>, SigmaSerializationError>, u64) {
        let mut buf = Vec::new();
        let cost = Cell::new(0u64);
        let sink = |c: u64| {
            cost.set(cost.get() + c);
            Ok(())
        };
        {
            let mut w = SigmaByteWriter::with_cost_sink(&mut buf, None, &sink);
            w.with_tree_version(Version::V3, |w| DataSerializer::sigma_serialize(lit, w))
                .unwrap();
        }
        (Ok(buf), cost.get())
    }

    /// The error of a value that is not of its type's class, and the costs added before it
    fn refused(value: &Value<'_>, tpe: &SType) -> Option<u64> {
        let (res, cost) = written(value, tpe);
        matches!(res, Err(SigmaSerializationError::UnexpectedValue(_))).then_some(cost)
    }

    fn coll(t: SType) -> SType {
        SType::SColl(t.into())
    }

    fn int_pair() -> SType {
        SType::STuple(STuple::pair(SType::SInt, SType::SInt).into())
    }

    fn int_triple() -> SType {
        SType::STuple(STuple::triple(SType::SInt, SType::SInt, SType::SInt).into())
    }

    fn ints(xs: &[i32]) -> Vec<Value<'static>> {
        xs.iter().map(|x| Value::Int(*x)).collect()
    }

    /// sigmastate's value of a tuple of other than two items
    fn any_coll(items: Vec<Value<'static>>) -> Value<'static> {
        Value::Coll(CollValue::generic(SType::SAny, items))
    }

    fn int_tuple(xs: &[i32]) -> Literal {
        let items: Vec<Literal> = xs.iter().map(|x| Literal::Int(*x)).collect();
        Literal::Tup(items.try_into().unwrap())
    }

    fn items(elem_tpe: SType, items: Vec<Literal>) -> Literal {
        Literal::Coll(CollKind::WrappedColl {
            elem_tpe,
            items: items.into(),
        })
    }

    fn bytes(xs: &[i8]) -> Literal {
        Literal::Coll(CollKind::NativeColl(NativeColl::CollByte(xs.into())))
    }

    // every arm that needs no context: a constant's value is written as the constant's data is,
    // bytes and costs
    #[test]
    fn a_value_is_written_as_its_data() {
        let tree = AvlTreeData {
            digest: vec![1; 33],
            tree_flags: AvlTreeFlags::new(true, true, true),
            key_length: 32,
            value_length_opt: None,
        };
        let dlog = SigmaProp::new(ProveDlog::new(generator()).into());
        let rows: Vec<(Literal, SType)> = vec![
            (Literal::Unit, SType::SUnit),
            (Literal::Boolean(true), SType::SBoolean),
            (Literal::Byte(-5), SType::SByte),
            (Literal::Short(-300), SType::SShort),
            (Literal::Int(70000), SType::SInt),
            (Literal::Long(-1), SType::SLong),
            (Literal::String("abc".into()), SType::SString),
            (Literal::String("".into()), SType::SString),
            (Literal::BigInt(BigInt256::from(-70000i32)), SType::SBigInt),
            (
                Literal::UnsignedBigInt(UnsignedBigInt::from(70000u32)),
                SType::SUnsignedBigInt,
            ),
            (
                Literal::GroupElement(Arc::new(generator())),
                SType::SGroupElement,
            ),
            (Literal::SigmaProp(Box::new(dlog)), SType::SSigmaProp),
            (Literal::AvlTree(Box::new(tree)), SType::SAvlTree),
            (bytes(&[1, -2, 3]), coll(SType::SByte)),
            (bytes(&[]), coll(SType::SByte)),
            (
                items(
                    SType::SBoolean,
                    vec![Literal::Boolean(true), Literal::Boolean(false)],
                ),
                coll(SType::SBoolean),
            ),
            (
                items(SType::SInt, vec![Literal::Int(1), Literal::Int(2)]),
                coll(SType::SInt),
            ),
            (
                items(coll(SType::SByte), vec![bytes(&[1]), bytes(&[2, 3])]),
                coll(coll(SType::SByte)),
            ),
            (
                items(int_pair(), vec![int_tuple(&[1, 2]), int_tuple(&[3, 4])]),
                coll(int_pair()),
            ),
            (
                items(int_triple(), vec![int_tuple(&[1, 2, 3])]),
                coll(int_triple()),
            ),
            (int_tuple(&[1, 2]), int_pair()),
            (int_tuple(&[1, 2, 3]), int_triple()),
            (
                Literal::Opt(Some(Box::new(Literal::Int(1)))),
                SType::SOption(SType::SInt.into()),
            ),
            (Literal::Opt(None), SType::SOption(SType::SInt.into())),
        ];
        for (lit, tpe) in rows {
            assert_eq!(
                written(&Value::from(lit.clone()), &tpe),
                data(&lit),
                "{tpe:?}"
            );
        }
    }

    // the two arms that need a context to have a value: a box and a header
    #[cfg(feature = "arbitrary")]
    #[test]
    fn a_box_and_a_header_are_written_as_their_data() {
        use sigma_test_util::force_any_val;
        for _ in 0..8 {
            let b = Literal::CBox(Arc::new(force_any_val::<ErgoBox>()).into());
            assert_eq!(written(&Value::from(b.clone()), &SType::SBox), data(&b));
            let h = Literal::Header(Box::new(force_any_val::<Header>()));
            assert_eq!(written(&Value::from(h.clone()), &SType::SHeader), data(&h));
        }
    }

    // A runtime `BigInt` is as wide as sigmastate's `BigInteger`; data is 256-bit, and the JVM's
    // writer requires `fitsIn256Bits` before it writes the length (v6.0.6
    // `CoreDataSerializer.scala:35`): `-2^255 ..= 2^255 - 1`
    #[test]
    fn a_runtime_bigint_outside_256_bits_is_not_written() {
        let p: BigInt = BigInt::from(1) << 255usize;
        for v in [p.clone(), -p.clone() - 1, p.clone() << 2000usize] {
            assert_eq!(
                refused(&Value::BigInt(v.clone()), &SType::SBigInt),
                Some(0),
                "{v}"
            );
        }
        for v in [-p.clone(), p.clone() - 1, BigInt::from(0)] {
            let lit = Literal::BigInt(BigInt256::try_from(v.clone()).unwrap());
            assert_eq!(
                written(&Value::BigInt(v.clone()), &SType::SBigInt),
                data(&lit),
                "{v}"
            );
        }
    }

    #[test]
    fn a_collection_of_any_is_a_tuple_of_its_length() {
        assert_eq!(
            written(&any_coll(ints(&[1, 2, 3])), &int_triple()),
            data(&int_tuple(&[1, 2, 3]))
        );
        // `fromDslTuple` takes a collection of `Any` for a pair type too
        assert_eq!(
            written(&any_coll(ints(&[1, 2])), &int_pair()),
            data(&int_tuple(&[1, 2]))
        );
        // the length assert, before anything is written
        assert_eq!(refused(&any_coll(ints(&[1, 2])), &int_triple()), Some(0));
        assert_eq!(refused(&any_coll(ints(&[1, 2, 3])), &int_pair()), Some(0));
        // `fromDslTuple` refuses a collection of another item type
        let coll_of_ints = Value::Coll(CollValue::generic(SType::SInt, ints(&[1, 2, 3])));
        assert_eq!(refused(&coll_of_ints, &int_triple()), Some(0));
    }

    #[test]
    fn a_pair_is_a_tuple_of_two() {
        let pair: Value = (1i32, 2i32).into();
        assert_eq!(written(&pair, &int_pair()), data(&int_tuple(&[1, 2])));
        assert_eq!(refused(&pair, &int_triple()), Some(0));
        // the first item is written before the second one's class is looked at
        let long_second: Value = (1i32, 2i64).into();
        assert_eq!(refused(&long_second, &int_pair()), Some(3));
    }

    #[test]
    fn a_collection_writes_its_items_by_the_item_type() {
        // a data `Coll[(Int, Int, Int)]` as sigmastate holds it: a `Coll[Coll[Any]]`
        let triples = Value::Coll(CollValue::generic(
            coll(SType::SAny),
            vec![any_coll(ints(&[1, 2, 3]))],
        ));
        assert_eq!(
            written(&triples, &coll(int_triple())),
            data(&items(int_triple(), vec![int_tuple(&[1, 2, 3])]))
        );
        // the collection arm takes any collection: a tuple of ints is written as a `Coll[Int]`
        assert_eq!(
            written(&any_coll(ints(&[1, 2])), &coll(SType::SInt)),
            data(&items(SType::SInt, vec![Literal::Int(1), Literal::Int(2)]))
        );
        // an item of another class is met after the length and the items before it
        let second_is_long = any_coll(vec![Value::Int(1), Value::Long(2)]);
        assert_eq!(refused(&second_is_long, &coll(SType::SInt)), Some(6));
    }

    #[test]
    fn a_value_of_another_class_is_refused() {
        // by reading: `DataSerializer.serialize` has no arm for `Any`, and casts a value to its
        // type's class
        assert_eq!(refused(&Value::Int(1), &SType::SAny), Some(0));
        assert_eq!(refused(&Value::Long(1), &SType::SInt), Some(0));
        assert_eq!(refused(&Value::Int(1), &coll(SType::SInt)), Some(0));
        assert_eq!(refused(&Value::Global, &SType::SGlobal), Some(0));
        assert_eq!(
            written(&Value::Int(1), &SType::SInt),
            data(&Literal::Int(1))
        );
    }

    #[test]
    fn byte_and_boolean_collections_need_their_array() {
        let coll_byte = coll(SType::SByte);
        let two = vec![Value::Byte(1), Value::Byte(2)];
        for value in [
            Value::Coll(CollValue::Bytes(vec![1i8, 2].into())),
            Value::Coll(CollValue::generic(SType::SByte, two.clone())),
        ] {
            assert_eq!(written(&value, &coll_byte), data(&bytes(&[1, 2])));
        }
        // a tuple of bytes is an `Object[]`: the cast to `byte[]` throws, after the length
        assert_eq!(refused(&any_coll(two), &coll_byte), Some(3));
        let coll_bool = coll(SType::SBoolean);
        let bools = items(
            SType::SBoolean,
            vec![Literal::Boolean(true), Literal::Boolean(false)],
        );
        for value in [
            Value::Coll(CollValue::Booleans(vec![true, false].into())),
            Value::Coll(CollValue::generic(
                SType::SBoolean,
                vec![Value::Boolean(true), Value::Boolean(false)],
            )),
        ] {
            assert_eq!(written(&value, &coll_bool), data(&bools));
        }
        assert_eq!(
            refused(&any_coll(vec![Value::Boolean(true)]), &coll_bool),
            Some(3)
        );
    }

    #[test]
    fn unit_reads_no_value_and_an_option_writes_its_value() {
        assert_eq!(written(&Value::Int(1), &SType::SUnit), (Ok(vec![]), 0));
        let opt_triple = SType::SOption(int_triple().into());
        assert_eq!(
            written(
                &Value::Opt(Some(Box::new(any_coll(ints(&[1, 2, 3]))))),
                &opt_triple
            ),
            data(&Literal::Opt(Some(Box::new(int_tuple(&[1, 2, 3])))))
        );
        // a `None` casts nothing
        assert_eq!(written(&Value::Opt(None), &opt_triple), (Ok(vec![0]), 1));
        // the tag is written before the value's class is looked at
        assert_eq!(
            refused(&Value::Opt(Some(Box::new(Value::Long(1)))), &opt_triple),
            Some(1)
        );
    }

    // sigmastate refuses a collection of zero-width items before it writes the length (6.0.7
    // `CoreDataSerializer.scala:50-55`)
    #[test]
    fn a_collection_of_zero_width_items_is_refused_before_its_length() {
        let units = Value::Coll(CollValue::generic(SType::SUnit, vec![Value::Unit]));
        let (res, cost) = written(&units, &coll(SType::SUnit));
        assert!(
            matches!(res, Err(SigmaSerializationError::NotSupported(_))),
            "{res:?}"
        );
        assert_eq!(cost, 0);
    }

    // an option and an unsigned big integer have data from tree version 3 only
    // (`isV3OrLaterErgoTreeVersion`, `CoreDataSerializer.scala:78`, `:39`)
    #[test]
    fn an_option_and_an_unsigned_big_integer_have_no_arm_below_tree_version_3() {
        let rows = [
            (
                Value::Opt(Some(Box::new(Value::Int(1)))),
                SType::SOption(SType::SInt.into()),
            ),
            (
                Value::UnsignedBigInt(UnsignedBigInt::from(1u32)),
                SType::SUnsignedBigInt,
            ),
        ];
        for (value, tpe) in rows {
            for version in [Version::V0, Version::V1, Version::V2] {
                let (res, cost) = written_at(version, &value, &tpe);
                assert!(res.is_err(), "{tpe:?}");
                assert_eq!(cost, 0);
            }
            assert!(written(&value, &tpe).0.is_ok(), "{tpe:?}");
        }
    }

    // and so has a header (`DataSerializer.scala:19`)
    #[cfg(feature = "arbitrary")]
    #[test]
    fn a_header_has_no_arm_below_tree_version_3() {
        use sigma_test_util::force_any_val;
        let header = Value::Header(Box::new(force_any_val::<Header>()));
        for version in [Version::V0, Version::V1, Version::V2] {
            let (res, cost) = written_at(version, &header, &SType::SHeader);
            assert!(res.is_err(), "{version:?}");
            assert_eq!(cost, 0);
        }
        assert!(written(&header, &SType::SHeader).0.is_ok());
    }

    /// 50 collections that are all ONE collection of 50 collections that are all one collection
    /// of 100 bytes: three allocations, 252,551 bytes when written
    fn shared() -> (Value<'static>, SType) {
        let inner = Value::Coll(CollValue::Bytes(vec![1i8; 100].into()));
        let middle = Value::Coll(CollValue::generic(coll(SType::SByte), vec![inner; 50]));
        let top = Value::Coll(CollValue::generic(
            coll(coll(SType::SByte)),
            vec![middle; 50],
        ));
        (top, coll(coll(coll(SType::SByte))))
    }

    // The bound, as a count: a sink that refuses its k-th cost has been called k times, and
    // the writer has written the bytes of the puts before the k-th and no more. The costs are,
    // in order: the three lengths (one byte each), the first 100 bytes, the next length, ...
    #[test]
    fn a_shared_collection_is_written_only_as_far_as_its_costs_are_accepted() {
        let (value, tpe) = shared();
        assert_eq!(
            written(&value, &tpe).0.map(|bytes| bytes.len()),
            Ok(1 + 50 * (1 + 50 * (1 + 100)))
        );
        assert_eq!(written(&value, &tpe).1, 3 + 50 * (3 + 50 * (3 + 103)));
        for (refused, bytes_written) in [(1, 0), (2, 1), (3, 2), (4, 3), (5, 103), (6, 104)] {
            let mut buf = Vec::new();
            let calls = Cell::new(0);
            let sink = |_: u64| {
                calls.set(calls.get() + 1);
                if calls.get() == refused {
                    Err(CostLimitExceeded(7))
                } else {
                    Ok(())
                }
            };
            let res = {
                let mut w = SigmaByteWriter::with_cost_sink(&mut buf, None, &sink);
                w.with_tree_version(Version::V3, |w| {
                    DataSerializer::sigma_serialize_value(&value, &tpe, w)
                })
            };
            assert_eq!(
                res,
                Err(SigmaSerializationError::CostLimitExceeded(
                    CostLimitExceeded(7)
                ))
            );
            assert_eq!((calls.get(), buf.len()), (refused, bytes_written));
        }
    }

    // an error names the start of the value and of the type, never the whole of either: the
    // `Debug` text of this value is over half a megabyte
    #[test]
    fn an_error_does_not_print_a_whole_value() {
        let (value, tpe) = shared();
        // a tree whose digest is that value: the digest's class is looked at before the digest
        // is read as bytes, which would print it
        let tree = match &value {
            Value::Coll(digest) => Value::AvlTree(Box::new(AvlTreeValue {
                digest: digest.clone(),
                tree_flags: AvlTreeFlags::new(true, true, true),
                key_length: 32,
                value_length_opt: None,
            })),
            _ => Value::Unit,
        };
        let trees = Value::Coll(CollValue::generic(SType::SAvlTree, vec![tree.clone()]));
        for (value, tpe) in [
            (value.clone(), SType::SLong),
            (value.clone(), coll(SType::SByte)),
            (Value::Long(1), tpe),
            (tree, SType::SAvlTree),
            (trees, coll(SType::SAvlTree)),
        ] {
            let text = match written(&value, &tpe).0 {
                Err(SigmaSerializationError::UnexpectedValue(e)) => e.0,
                _ => alloc::string::String::new(),
            };
            assert!(!text.is_empty() && text.len() < 600, "{}", text.len());
        }
        assert!(alloc::format!("{value:?}").len() > 500_000);
    }
}
