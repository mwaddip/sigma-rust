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
use crate::mir::constant::Literal;
use crate::mir::constant::TryExtractFromError;
use crate::mir::constant::TryExtractInto;
use crate::mir::value::CollKind;
use crate::mir::value::NativeColl;
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

impl DataSerializer {
    /// Serialize Literal without typecode serialized
    pub fn sigma_serialize<W: SigmaByteWrite>(c: &Literal, w: &mut W) -> SigmaSerializeResult {
        // for reference see http://github.com/ScorexFoundation/sigmastate-interpreter/blob/25251c1313b0131835f92099f02cef8a5d932b5e/sigmastate/src/main/scala/sigmastate/serialization/DataSerializer.scala#L26-L26
        Ok(match c {
            Literal::Unit => (),
            Literal::Boolean(v) => {
                w.put_u8(u8::from(*v))?;
                w.add_put_byte_cost();
            }
            Literal::Byte(v) => {
                w.put_i8(*v)?;
                w.add_put_byte_cost();
            }
            Literal::Short(v) => {
                w.put_i16(*v)?;
                w.add_put_numeric_cost();
            }
            Literal::Int(v) => {
                w.put_i32(*v)?;
                w.add_put_numeric_cost();
            }
            Literal::Long(v) => {
                w.put_i64(*v)?;
                w.add_put_numeric_cost();
            }
            Literal::BigInt(v) => {
                v.sigma_serialize(w)?;
            }
            Literal::String(s) => {
                // the length is free: sigmastate writes it with `putUInt(length)`
                // (`CoreDataSerializer.scala:31`), which the cost-counting writer hands to the
                // inner writer without a cost (`SigmaByteWriter.scala:105-107`)
                w.put_usize_as_u32_unwrapped(s.len())?;
                w.write_all(s.as_bytes())?;
                w.add_put_chunk_cost(s.len());
            }
            Literal::GroupElement(ecp) => {
                // EcPoint::scorex_serialize writes exactly GROUP_SIZE (33) bytes as one block;
                // Scala meters it as putBytes(33) = PutChunkCost. (EcPoint is in ergo-chain-types
                // and can't reach the cost sink, so record it here at the delegating site.)
                ecp.sigma_serialize(w)?;
                w.add_put_chunk_cost(EcPoint::GROUP_SIZE);
            }
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
                    w.put_usize_as_u16_unwrapped(b.len())?;
                    w.add_put_numeric_cost();
                    w.write_all(b.clone().as_vec_u8().as_slice())?;
                    w.add_put_chunk_cost(b.len());
                }
                CollKind::WrappedColl {
                    elem_tpe: SType::SBoolean,
                    items: v,
                } => {
                    w.put_usize_as_u16_unwrapped(v.len())?;
                    w.add_put_numeric_cost();
                    let maybe_bools: Result<Vec<bool>, TryExtractFromError> = v
                        .clone()
                        .iter()
                        .cloned()
                        .map(|i| i.try_extract_into::<bool>())
                        .collect();
                    w.put_bits(maybe_bools?.as_slice())?;
                    w.add_put_chunk_cost(v.len());
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
                    w.put_usize_as_u16_unwrapped(v.len())?;
                    w.add_put_numeric_cost();
                    v.iter()
                        .try_for_each(|e| DataSerializer::sigma_serialize(e, w))?
                }
            },
            Literal::Tup(items) => items
                .iter()
                .try_for_each(|i| DataSerializer::sigma_serialize(i, w))?,
            Literal::Header(h) if w.tree_version() >= ErgoTreeVersion::V3 => {
                h.scorex_serialize(w)?;
                // Header is in ergo-chain-types and can't reach the cost sink, so mirror the
                // JVM's metered put sequence here (as for EcPoint/AvlTree). Order matches
                // `ErgoHeader.sigmaSerializer` = `HeaderWithoutPowSerializer.serialize` +
                // `AutolykosSolution.sigmaSerializerV{1,2}` (sigma
                // `org/ergoplatform/{ErgoHeader,HeaderWithoutPow}.scala`); keep in sync with
                // `Header::serialize_without_pow` / `AutolykosSolution::serialize_bytes`. All
                // puts are no-info, so `putUInt(height)` is unmetered (0). Blessed v6
                // `Global.serialize[Header]` = 333 (= 89 + put_cost 244, v2/empty-unparsed).
                w.add_put_byte_cost(); // put(version)
                w.add_put_chunk_cost(32); // putBytes(parentId: Digest32)
                w.add_put_chunk_cost(32); // putBytes(ADProofsRoot: Digest32)
                w.add_put_chunk_cost(32); // putBytes(transactionsRoot: Digest32)
                w.add_put_chunk_cost(33); // putBytes(stateRoot: ADDigest)
                w.add_put_numeric_cost(); // putULong(timestamp)
                w.add_put_chunk_cost(32); // putBytes(extensionRoot: Digest32)
                w.add_put_chunk_cost(4); // DifficultySerializer: putBytes(nBits, 4 BE)
                                         // putUInt(height): no-info, unmetered (0)
                w.add_put_chunk_cost(3); // putBytes(votes)
                if h.version > 1 {
                    w.add_put_byte_cost(); // putUByte(unparsedBytes.len)
                    w.add_put_chunk_cost(h.unparsed_bytes.len()); // putBytes(unparsedBytes)
                }
                // AutolykosSolution.sigmaSerializerV{1,2}
                w.add_put_chunk_cost(EcPoint::GROUP_SIZE); // GroupElementSerializer(pk)
                if h.version == 1 {
                    // GroupElementSerializer(w) — one-time pk, v1 only
                    w.add_put_chunk_cost(EcPoint::GROUP_SIZE);
                }
                w.add_put_chunk_cost(8); // putBytes(nonce)
                if h.version == 1 {
                    w.add_put_byte_cost(); // putUByte(dBytes.len) — v1 only
                                           // putBytes(dBytes): length per sigma `asUnsignedByteArray` — empty for
                                           // zero, else minimal big-endian (matches `fix/autolykos-v1-zero-distance`).
                    let d_len = h.autolykos_solution.pow_distance.as_ref().map_or(0, |d| {
                        if d.is_zero() {
                            0
                        } else {
                            d.to_bytes_be().len()
                        }
                    });
                    w.add_put_chunk_cost(d_len);
                }
            }
            Literal::Opt(opt) if w.tree_version() >= ErgoTreeVersion::V3 => {
                w.put_option(Option::as_ref(opt), |w, v| {
                    DataSerializer::sigma_serialize(v, w)
                })?;
                w.add_put_byte_cost();
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
