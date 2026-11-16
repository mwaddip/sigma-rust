//! Sigma byte stream writer
use crate::chain::context::CostLimitExceeded;
use crate::ergo_tree::ErgoTreeVersion;

use super::constant_store::ConstantStore;
use super::SigmaSerializeResult;
use core3::io::Write;
use sigma_ser::vlq_encode::WriteSigmaVlqExt;

// Scala `Global.serialize` cost constants (sigma-state `SigmaByteWriter.scala:235-262`).
// `serialize_eval` adds these to the evaluator before each writer `put` while it writes the
// value; a `SigmaByteWriter` made by `with_cost_sink` hands them to its sink the same way
// (see `SERIALIZE_EVAL_FN`). All are `JitCost` units (1:1 with the accumulator).
/// `PutByteCost = FixedCost(JitCost(1))` — `put`/`putBoolean`/option tag byte.
const PUT_BYTE_COST: u64 = 1;
/// `Put{Signed,Unsigned}NumericCost = FixedCost(JitCost(3))` — `putShort/Int/Long`, `putU*`.
const PUT_NUMERIC_COST: u64 = 3;
/// `PutChunkCost = PerItemCost(JitCost(3), JitCost(1), 1)` ⇒ `cost(n) = 3 + n` —
/// `putBytes`/`putBits`/`putChunk` over `n` items.
const PUT_CHUNK_BASE_COST: u64 = 3;

/// What a costed writer charges: it is called with the cost of a `put` before the put is made,
/// and fails when that cost crosses the limit of whoever counts it
pub type CostSink<'a> = &'a dyn Fn(u64) -> Result<(), CostLimitExceeded>;

/// Implementation for SigmaByteWrite
pub struct SigmaByteWriter<'a, W> {
    inner: &'a mut W,
    tree_version: ErgoTreeVersion,
    /// Constant store where constants (swapped for placeholders) are stored
    pub constant_store: Option<ConstantStore>,
    /// Where the cost of each `put` goes before the put is made (`Global.serialize` hands the
    /// evaluator's cost counter). `None` for every other writer, which writes uncharged
    cost_sink: Option<CostSink<'a>>,
}

impl<'a, W: Write> SigmaByteWriter<'a, W> {
    /// Make a new writer with underlying Write and optional constant store
    pub fn new(w: &'a mut W, constant_store: Option<ConstantStore>) -> SigmaByteWriter<'a, W> {
        SigmaByteWriter {
            inner: w,
            tree_version: ErgoTreeVersion::V0,
            constant_store,
            cost_sink: None,
        }
    }

    /// A writer that hands the cost of each `put` to `cost_sink` before it makes the put, and
    /// stops at the put whose cost the sink refuses: sigmastate's writer with its two cost
    /// callbacks, which `Global.serialize` alone makes (v6.0.6 `methods.scala:1968-1983`,
    /// `SigmaByteWriter.scala:27-36`)
    pub fn with_cost_sink(
        w: &'a mut W,
        constant_store: Option<ConstantStore>,
        cost_sink: CostSink<'a>,
    ) -> SigmaByteWriter<'a, W> {
        SigmaByteWriter {
            inner: w,
            tree_version: ErgoTreeVersion::V0,
            constant_store,
            cost_sink: Some(cost_sink),
        }
    }

    /// `cost` to the sink, if the writer has one
    fn charge(&self, cost: u64) -> SigmaSerializeResult {
        match self.cost_sink {
            Some(sink) => Ok(sink(cost)?),
            None => Ok(()),
        }
    }
}

/// Sigma byte writer trait with a store for constant segregation
pub trait SigmaByteWrite: WriteSigmaVlqExt {
    /// Constant store (if any) attached to the writer to collect segregated constants
    fn constant_store_mut_ref(&mut self) -> Option<&mut ConstantStore>;
    /// ErgoTree Version
    fn tree_version(&self) -> ErgoTreeVersion;
    /// Execute `f` with ErgoTree version set to `version`
    fn with_tree_version<T>(
        &mut self,
        version: ErgoTreeVersion,
        f: impl FnOnce(&mut Self) -> T,
    ) -> T;

    /// Add the cost of a fixed-cost byte `put` (Scala `PutByteCost`). A serializer calls it
    /// BEFORE the put, as sigmastate's writer adds a put's cost before it writes
    /// (`SigmaByteWriter.scala:45-48`). Nothing unless the writer has a cost sink; an error
    /// when the sink refuses the cost, and then the put is not made.
    fn add_put_byte_cost(&mut self) -> SigmaSerializeResult {
        Ok(())
    }
    /// Add the cost of a numeric `put` (Scala `Put{Signed,Unsigned}NumericCost`), before the
    /// put. Nothing unless the writer has a cost sink; an error when the sink refuses the cost.
    fn add_put_numeric_cost(&mut self) -> SigmaSerializeResult {
        Ok(())
    }
    /// Add the cost of a chunk `put` of `n` items (Scala `PutChunkCost`, `cost(n) = 3 + n`),
    /// before the put. Nothing unless the writer has a cost sink; an error when the sink
    /// refuses the cost.
    fn add_put_chunk_cost(&mut self, _n: usize) -> SigmaSerializeResult {
        Ok(())
    }
}

impl<'a, W: Write> Write for SigmaByteWriter<'a, W> {
    fn write(&mut self, buf: &[u8]) -> core3::io::Result<usize> {
        self.inner.write(buf)
    }

    fn flush(&mut self) -> core3::io::Result<()> {
        self.inner.flush()
    }
}

impl<'a, W: Write> SigmaByteWrite for SigmaByteWriter<'a, W> {
    fn constant_store_mut_ref(&mut self) -> Option<&mut ConstantStore> {
        self.constant_store.as_mut()
    }
    fn tree_version(&self) -> ErgoTreeVersion {
        self.tree_version
    }
    fn with_tree_version<T>(
        &mut self,
        version: ErgoTreeVersion,
        f: impl FnOnce(&mut Self) -> T,
    ) -> T {
        let tmp = self.tree_version;
        self.tree_version = version;
        let res = f(self);
        self.tree_version = tmp;
        res
    }

    fn add_put_byte_cost(&mut self) -> SigmaSerializeResult {
        self.charge(PUT_BYTE_COST)
    }
    fn add_put_numeric_cost(&mut self) -> SigmaSerializeResult {
        self.charge(PUT_NUMERIC_COST)
    }
    fn add_put_chunk_cost(&mut self, n: usize) -> SigmaSerializeResult {
        self.charge(PUT_CHUNK_BASE_COST.saturating_add(n as u64))
    }
}

/// `value` as a writer at ErgoTree `version` writes it: below version 3 an `Upcast` of a
/// constant is written as the constant, and a value with no encoding there fails
#[cfg(feature = "json")]
pub(crate) fn bytes_at<T: super::SigmaSerializable>(
    value: &T,
    version: ErgoTreeVersion,
) -> Result<alloc::vec::Vec<u8>, super::SigmaSerializationError> {
    let mut data = alloc::vec::Vec::new();
    let mut w = SigmaByteWriter::new(&mut data, None);
    w.with_tree_version(version, |w| value.sigma_serialize(w))?;
    Ok(data)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod cost_sink_tests {
    use super::*;
    use crate::bigint256::BigInt256;
    use crate::chain::context::CostLimitExceeded;
    use crate::chain::ergo_box::box_value::BoxValue;
    use crate::chain::ergo_box::serialize_box_with_indexed_digests;
    use crate::chain::ergo_box::ErgoBox;
    use crate::chain::ergo_box::NonMandatoryRegisters;
    use crate::chain::evaluated_value::EvaluatedExpr;
    use crate::chain::evaluated_value::EvaluatedValue;
    use crate::chain::token::Token;
    use crate::chain::token::TokenAmount;
    use crate::chain::token::TokenId;
    use crate::chain::tx_id::TxId;
    use crate::ergo_tree::ErgoTree;
    use crate::mir::avl_tree_data::AvlTreeData;
    use crate::mir::avl_tree_data::AvlTreeFlags;
    use crate::mir::constant::Constant;
    use crate::mir::constant::Literal;
    use crate::mir::expr::Expr;
    use crate::mir::tuple::Tuple;
    use crate::mir::value::CollKind;
    use crate::mir::value::NativeColl;
    use crate::mir::value::Value;
    use crate::serialization::data::DataSerializer;
    use crate::serialization::op_code::OpCode;
    use crate::serialization::SigmaSerializable;
    use crate::serialization::SigmaSerializationError;
    use crate::sigma_protocol::sigma_boolean::cand::Cand;
    use crate::sigma_protocol::sigma_boolean::cor::Cor;
    use crate::sigma_protocol::sigma_boolean::cthreshold::Cthreshold;
    use crate::sigma_protocol::sigma_boolean::ProveDhTuple;
    use crate::sigma_protocol::sigma_boolean::ProveDlog;
    use crate::sigma_protocol::sigma_boolean::SigmaBoolean;
    use crate::sigma_protocol::sigma_boolean::SigmaProp;
    use crate::types::sfunc::SFunc;
    use crate::types::stuple::STuple;
    use crate::types::stype::SType;
    use crate::types::stype_param::STypeVar;
    use crate::unsignedbigint256::UnsignedBigInt;
    use alloc::boxed::Box;
    use alloc::sync::Arc;
    use alloc::vec;
    use alloc::vec::Vec;
    use core::cell::Cell;
    use core::cell::RefCell;
    use ergo_chain_types::ec_point::generator;
    use ergo_chain_types::Digest32;

    /// A buffer the test reads while a writer holds it
    struct Shared<'b>(&'b RefCell<Vec<u8>>);

    impl Write for Shared<'_> {
        fn write(&mut self, buf: &[u8]) -> core3::io::Result<usize> {
            self.0.borrow_mut().extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> core3::io::Result<()> {
            Ok(())
        }
    }

    /// Something a writer writes
    type Put =
        Box<dyn for<'a, 'b> Fn(&mut SigmaByteWriter<'a, Shared<'b>>) -> SigmaSerializeResult>;

    /// A writer's name, what it writes, and each cost it adds with the bytes written before it
    type Row = (&'static str, Put, Vec<(u64, usize)>);

    fn put<T: SigmaSerializable + 'static>(x: T) -> Put {
        Box::new(move |w| x.sigma_serialize(w))
    }

    fn data(x: Literal) -> Put {
        Box::new(move |w| DataSerializer::sigma_serialize(&x, w))
    }

    /// Each cost a write hands its sink, with the number of bytes written before it, and the bytes
    fn costs(write: &Put) -> (Vec<(u64, usize)>, Vec<u8>) {
        let buf = RefCell::new(Vec::new());
        let seen = RefCell::new(Vec::new());
        let sink = |cost: u64| {
            seen.borrow_mut().push((cost, buf.borrow().len()));
            Ok(())
        };
        let mut inner = Shared(&buf);
        let mut w = SigmaByteWriter::with_cost_sink(&mut inner, None, &sink);
        w.with_tree_version(ErgoTreeVersion::V3, |w| write(w))
            .unwrap();
        (seen.into_inner(), buf.into_inner())
    }

    fn dlog() -> SigmaBoolean {
        ProveDlog::new(generator()).into()
    }

    fn tree() -> ErgoTree {
        ErgoTree::sigma_parse_bytes(&[0x00, 0x08, 0xd3]).unwrap()
    }

    fn ints(xs: &[i32]) -> Literal {
        Literal::Coll(CollKind::WrappedColl {
            elem_tpe: SType::SInt,
            items: xs.iter().map(|x| Literal::Int(*x)).collect(),
        })
    }

    /// Every writer that adds a cost, with each cost and the byte position it is added at: the
    /// position its put starts at, as sigmastate's writer adds a put's cost and then writes
    /// (`SigmaByteWriter.scala:45-174`). A byte put is 1, a numeric put 3, a chunk of n items
    /// 3 + n; a `putUInt` adds nothing, so the byte after it starts one further
    fn rows() -> Vec<Row> {
        let g = generator();
        let coll = |t: SType| SType::SColl(t.into());
        let pair = |a: SType, b: SType| SType::STuple(STuple::pair(a, b).into());
        let int = |x: i32| Expr::Const(x.into());
        let token = Token {
            token_id: TokenId::from(Digest32::zero()),
            amount: TokenAmount::try_from(1u64).unwrap(),
        };
        let tokens = Some(vec![token].try_into().unwrap());
        let value = BoxValue::try_from(1_000_000u64).unwrap();
        let body_tree = tree();
        vec![
            ("Boolean", data(Literal::Boolean(true)), vec![(1, 0)]),
            ("Byte", data(Literal::Byte(5)), vec![(1, 0)]),
            ("Short", data(Literal::Short(1)), vec![(3, 0)]),
            ("Int", data(Literal::Int(1)), vec![(3, 0)]),
            ("Long", data(Literal::Long(1)), vec![(3, 0)]),
            // the length is a `putUInt`
            ("String", data(Literal::String("abc".into())), vec![(6, 1)]),
            (
                "BigInt",
                data(Literal::BigInt(BigInt256::from(1i32))),
                vec![(3, 0), (4, 1)],
            ),
            (
                "UnsignedBigInt",
                data(Literal::UnsignedBigInt(UnsignedBigInt::from(1u32))),
                vec![(3, 0), (4, 1)],
            ),
            (
                "GroupElement",
                data(Literal::GroupElement(Arc::new(g))),
                vec![(36, 0)],
            ),
            (
                "SigmaProp",
                data(Literal::SigmaProp(Box::new(SigmaProp::new(dlog())))),
                vec![(1, 0), (36, 1)],
            ),
            (
                "ProveDhTuple",
                put(SigmaBoolean::from(ProveDhTuple::new(g, g, g, g))),
                vec![(1, 0), (36, 1), (36, 34), (36, 67), (36, 100)],
            ),
            (
                "Cand",
                put(SigmaBoolean::from(Cand {
                    items: vec![dlog(), dlog()],
                })),
                vec![(1, 0), (3, 1), (1, 2), (36, 3), (1, 36), (36, 37)],
            ),
            (
                "Cor",
                put(SigmaBoolean::from(Cor {
                    items: vec![dlog(), dlog()],
                })),
                vec![(1, 0), (3, 1), (1, 2), (36, 3), (1, 36), (36, 37)],
            ),
            (
                "Cthreshold",
                put(SigmaBoolean::from(Cthreshold {
                    k: 1,
                    children: vec![dlog(), dlog()].try_into().unwrap(),
                })),
                vec![(1, 0), (3, 1), (3, 2), (1, 3), (36, 4), (1, 37), (36, 38)],
            ),
            // the key length is a `putUInt`
            (
                "AvlTree",
                data(Literal::AvlTree(Box::new(AvlTreeData {
                    digest: vec![1; 33],
                    tree_flags: AvlTreeFlags::new(true, true, true),
                    key_length: 32,
                    value_length_opt: None,
                }))),
                vec![(36, 0), (1, 33), (1, 35)],
            ),
            (
                "Coll[Byte]",
                data(Literal::Coll(CollKind::NativeColl(NativeColl::CollByte(
                    vec![1i8, 2, 3].into(),
                )))),
                vec![(3, 0), (6, 1)],
            ),
            (
                "Coll[Boolean]",
                data(Literal::Coll(CollKind::WrappedColl {
                    elem_tpe: SType::SBoolean,
                    items: vec![Literal::Boolean(true); 3].into(),
                })),
                vec![(3, 0), (6, 1)],
            ),
            (
                "Coll[Int]",
                data(ints(&[1, 2])),
                vec![(3, 0), (3, 1), (3, 2)],
            ),
            (
                "Option",
                data(Literal::Opt(Some(Box::new(Literal::Int(1))))),
                vec![(1, 0), (3, 1)],
            ),
            ("TxId", put(TxId::zero()), vec![(35, 0)]),
            ("BoxValue", put(value), vec![(3, 0)]),
            ("OpCode", put(OpCode::TUPLE), vec![(1, 0)]),
            // the types: one code, a collection's and a pair's combined code, the two-code pairs,
            // a tuple's count, a type variable's name, a function type's two counts
            ("SInt", put(SType::SInt), vec![(1, 0)]),
            ("Coll[Byte] type", put(coll(SType::SByte)), vec![(1, 0)]),
            (
                "Coll[Coll[Byte]] type",
                put(coll(coll(SType::SByte))),
                vec![(1, 0)],
            ),
            (
                "(Int, Int) type",
                put(pair(SType::SInt, SType::SInt)),
                vec![(1, 0)],
            ),
            (
                "(Int, Long) type",
                put(pair(SType::SInt, SType::SLong)),
                vec![(1, 0), (1, 1)],
            ),
            (
                "(Coll[Byte], Int) type",
                put(pair(coll(SType::SByte), SType::SInt)),
                vec![(1, 0), (1, 1)],
            ),
            (
                "a tuple type of five",
                put(SType::STuple(
                    STuple {
                        items: vec![SType::SInt; 5].try_into().unwrap(),
                    }
                    .into(),
                )),
                vec![(1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6)],
            ),
            (
                "a type variable",
                put(SType::STypeVar(STypeVar::t())),
                vec![(1, 0), (1, 1), (4, 2)],
            ),
            (
                "a function type",
                put(SType::SFunc(
                    SFunc::new(vec![SType::SInt], SType::SInt).into(),
                )),
                vec![(1, 0), (1, 1), (1, 2), (1, 3), (1, 4)],
            ),
            // a Tuple node: its count, then each constant's type and value
            (
                "Tuple node",
                put(Tuple::new(vec![int(1), int(2)]).unwrap()),
                vec![(1, 0), (1, 1), (3, 2), (1, 3), (3, 4)],
            ),
            // a register's tuple expression: the opcode, the count, the items
            (
                "a tuple in a register",
                put(EvaluatedValue::Expr(EvaluatedExpr::Tuple(
                    vec![int(1), int(2)].try_into().unwrap(),
                ))),
                vec![(1, 0), (1, 1), (1, 2), (3, 3), (1, 4), (3, 5)],
            ),
            (
                "registers",
                put(NonMandatoryRegisters::try_from(vec![Constant::from(1i32)]).unwrap()),
                vec![(1, 0), (1, 1), (3, 2)],
            ),
            // a box's body: the value (3 bytes), the tree's 3 bytes, the creation height (a
            // `putUInt` of 2 bytes), the token count, a token's id and amount, the register count
            (
                "a box's body",
                Box::new(move |w| {
                    serialize_box_with_indexed_digests(
                        &value,
                        &body_tree,
                        &tokens,
                        &NonMandatoryRegisters::empty(),
                        12345,
                        None,
                        w,
                    )
                }),
                vec![(3, 0), (6, 3), (1, 8), (35, 9), (3, 41), (1, 42)],
            ),
            // a box: its body (the creation height 0 is one byte), the transaction id, the index
            (
                "a box",
                put(ErgoBox::new(
                    value,
                    tree(),
                    None,
                    NonMandatoryRegisters::empty(),
                    0,
                    TxId::zero(),
                    0,
                )
                .unwrap()),
                vec![(3, 0), (6, 3), (1, 7), (1, 8), (35, 9), (3, 41)],
            ),
        ]
    }

    /// Data, its type, and each cost its write adds with the bytes written before it
    type ValueRow = (Literal, SType, Vec<(u64, usize)>);

    fn value(x: Literal, tpe: SType) -> Put {
        let v = Value::from(x);
        Box::new(move |w| DataSerializer::sigma_serialize_value(&v, &tpe, w))
    }

    // the writer of a runtime value adds each cost where the data writer adds it: every arm of
    // its own that puts (a box and a header go to the writers of the table above)
    #[test]
    fn a_values_costs_are_added_before_its_puts() {
        let coll = |t: SType| SType::SColl(t.into());
        let rows: Vec<ValueRow> = vec![
            (Literal::Boolean(true), SType::SBoolean, vec![(1, 0)]),
            (Literal::Byte(5), SType::SByte, vec![(1, 0)]),
            (Literal::Short(1), SType::SShort, vec![(3, 0)]),
            (Literal::Int(1), SType::SInt, vec![(3, 0)]),
            (Literal::Long(1), SType::SLong, vec![(3, 0)]),
            (Literal::String("abc".into()), SType::SString, vec![(6, 1)]),
            (
                Literal::BigInt(BigInt256::from(1i32)),
                SType::SBigInt,
                vec![(3, 0), (4, 1)],
            ),
            (
                Literal::UnsignedBigInt(UnsignedBigInt::from(1u32)),
                SType::SUnsignedBigInt,
                vec![(3, 0), (4, 1)],
            ),
            (
                Literal::GroupElement(Arc::new(generator())),
                SType::SGroupElement,
                vec![(36, 0)],
            ),
            (
                Literal::SigmaProp(Box::new(SigmaProp::new(dlog()))),
                SType::SSigmaProp,
                vec![(1, 0), (36, 1)],
            ),
            (
                Literal::AvlTree(Box::new(AvlTreeData {
                    digest: vec![1; 33],
                    tree_flags: AvlTreeFlags::new(true, true, true),
                    key_length: 32,
                    value_length_opt: None,
                })),
                SType::SAvlTree,
                vec![(36, 0), (1, 33), (1, 35)],
            ),
            (
                Literal::Coll(CollKind::NativeColl(NativeColl::CollByte(
                    vec![1i8, 2, 3].into(),
                ))),
                coll(SType::SByte),
                vec![(3, 0), (6, 1)],
            ),
            (
                Literal::Coll(CollKind::WrappedColl {
                    elem_tpe: SType::SBoolean,
                    items: vec![Literal::Boolean(true); 3].into(),
                }),
                coll(SType::SBoolean),
                vec![(3, 0), (6, 1)],
            ),
            (
                ints(&[1, 2]),
                coll(SType::SInt),
                vec![(3, 0), (3, 1), (3, 2)],
            ),
            (
                Literal::Opt(Some(Box::new(Literal::Int(1)))),
                SType::SOption(SType::SInt.into()),
                vec![(1, 0), (3, 1)],
            ),
            (
                Literal::Tup([Literal::Int(1), Literal::Long(1)].into()),
                SType::STuple(STuple::pair(SType::SInt, SType::SLong).into()),
                vec![(3, 0), (3, 1)],
            ),
        ];
        for (lit, tpe, expected) in rows {
            assert_eq!(costs(&data(lit.clone())).0, expected, "{tpe:?}");
            assert_eq!(costs(&value(lit, tpe.clone())).0, expected, "{tpe:?}");
        }
    }

    #[test]
    fn each_cost_is_added_before_its_put() {
        for (name, write, expected) in rows() {
            assert_eq!(costs(&write).0, expected, "{name}");
        }
    }

    #[test]
    fn a_writer_with_no_sink_writes_the_same_bytes() {
        for (name, write, _) in rows() {
            let buf = RefCell::new(Vec::new());
            let mut inner = Shared(&buf);
            let mut w = SigmaByteWriter::new(&mut inner, None);
            w.with_tree_version(ErgoTreeVersion::V3, |w| write(w))
                .unwrap();
            assert_eq!(buf.into_inner(), costs(&write).1, "{name}");
        }
    }

    // a header's fields are written by one statement of another crate: all its costs go first
    #[cfg(feature = "arbitrary")]
    #[test]
    fn a_headers_costs_are_added_before_anything_of_it_is_written() {
        use sigma_test_util::force_any_val;
        for _ in 0..8 {
            let header = force_any_val::<ergo_chain_types::Header>();
            let (seen, bytes) = costs(&data(Literal::Header(Box::new(header))));
            assert!(seen.len() >= 11, "{seen:?}");
            assert!(seen.iter().all(|(_, written)| *written == 0), "{seen:?}");
            assert!(!bytes.is_empty());
        }
    }

    // `Coll[Int](1, 2)` adds three costs: its length's, at no byte, and each item's, one byte
    // further. A cost the sink refuses is the write's error, and its put is not made
    #[test]
    fn a_refused_cost_stops_the_write_before_its_put() {
        let write = data(ints(&[1, 2]));
        for (refused, written) in [(1, 0), (2, 1), (3, 2)] {
            let buf = RefCell::new(Vec::new());
            let calls = Cell::new(0);
            let sink = |_: u64| {
                calls.set(calls.get() + 1);
                if calls.get() == refused {
                    Err(CostLimitExceeded(7))
                } else {
                    Ok(())
                }
            };
            let mut inner = Shared(&buf);
            let mut w = SigmaByteWriter::with_cost_sink(&mut inner, None, &sink);
            assert_eq!(
                write(&mut w),
                Err(SigmaSerializationError::CostLimitExceeded(
                    CostLimitExceeded(7)
                )),
                "{refused}"
            );
            assert_eq!((calls.get(), buf.borrow().len()), (refused, written));
        }
    }

    // sigmastate writes a box's value, then serializes its tree and puts the tree's bytes (v6.0.6
    // `ErgoBoxCandidate.scala:141-142`): a tree that cannot be written (its type variable's name
    // is 86 malformed bytes, a text of 258) stops the box after the value's cost and its bytes
    #[test]
    fn a_box_whose_tree_is_not_written_has_written_its_value() {
        let unwritable = alloc::format!("00d193b183006756{}0400", "ff".repeat(86));
        let tree =
            ErgoTree::sigma_parse_bytes(&base16::decode(unwritable.as_bytes()).unwrap()).unwrap();
        assert!(tree.sigma_serialize_bytes().is_err());
        let value = BoxValue::try_from(1_000_000u64).unwrap();
        let buf = RefCell::new(Vec::new());
        let seen = RefCell::new(Vec::new());
        let sink = |cost: u64| {
            seen.borrow_mut().push((cost, buf.borrow().len()));
            Ok(())
        };
        let mut inner = Shared(&buf);
        let mut w = SigmaByteWriter::with_cost_sink(&mut inner, None, &sink);
        let res = serialize_box_with_indexed_digests(
            &value,
            &tree,
            &None,
            &NonMandatoryRegisters::empty(),
            0,
            None,
            &mut w,
        );
        assert!(
            matches!(&res, Err(SigmaSerializationError::NotSupported(text)) if text.contains("out of unsigned byte range")),
            "{res:?}"
        );
        assert_eq!((seen.into_inner(), buf.borrow().len()), (vec![(3, 0)], 3));
    }

    // sigmastate's `putUShort` adds its cost and then refuses a count above 65535
    // (`SigmaByteWriter.scala:83-86`): a refused count has been charged, and nothing is written
    #[test]
    fn a_count_out_of_range_is_refused_after_its_cost() {
        let long = Literal::Coll(CollKind::WrappedColl {
            elem_tpe: SType::SInt,
            items: vec![Literal::Int(1); 65536].into(),
        });
        let conjunction: Put = put(SigmaBoolean::from(Cand {
            items: vec![dlog(); 65536],
        }));
        for (name, write, expected) in [
            ("a collection", data(long), vec![(3, 0)]),
            ("a conjunction", conjunction, vec![(1, 0), (3, 1)]),
        ] {
            let buf = RefCell::new(Vec::new());
            let seen = RefCell::new(Vec::new());
            let sink = |cost: u64| {
                seen.borrow_mut().push((cost, buf.borrow().len()));
                Ok(())
            };
            let mut inner = Shared(&buf);
            let mut w = SigmaByteWriter::with_cost_sink(&mut inner, None, &sink);
            assert!(
                matches!(write(&mut w), Err(SigmaSerializationError::NotSupported(_))),
                "{name}"
            );
            assert_eq!(seen.into_inner(), expected, "{name}");
        }
    }
}
