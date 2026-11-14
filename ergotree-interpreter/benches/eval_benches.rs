use std::str::FromStr;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ergo_chain_types::ec_point::{exponentiate, generator};
use ergotree_interpreter::eval::test_util::eval_out_wo_ctx;
use ergotree_ir::{
    bigint256::BigInt256,
    ergo_tree::{ErgoTree, ErgoTreeHeader},
    mir::{
        bin_op::{BinOp, BinOpKind, RelationOp},
        calc_blake2b256::CalcBlake2b256,
        calc_sha256::CalcSha256,
        coll_append::Append,
        coll_filter::Filter,
        constant::Constant,
        exponentiate::Exponentiate,
        expr::Expr,
        func_value::{FuncArg, FuncValue},
        method_call::MethodCall,
        sigma_and::SigmaAnd,
        subst_const::SubstConstants,
        val_use::ValUse,
        value::Value,
    },
    serialization::SigmaSerializable,
    sigma_protocol::sigma_boolean::{SigmaBoolean, SigmaProp},
    types::{scoll::FLATMAP_METHOD, stype::SType, stype_param::STypeVar},
};

fn bench_blake2b256(c: &mut Criterion) {
    let mut group = c.benchmark_group("blake2b256");
    for size in (1..1024).step_by(200) {
        let arr = vec![0u8; size];
        let expr: Expr = CalcBlake2b256 {
            input: Box::new(Constant::from(arr).into()),
        }
        .into();
        group.bench_function(BenchmarkId::from_parameter(format!("{size:4}")), |b| {
            b.iter(|| eval_out_wo_ctx::<Vec<i8>>(&expr))
        });
    }
}

fn bench_sha256(c: &mut Criterion) {
    let mut group = c.benchmark_group("sha256");
    for size in (1..1024).step_by(200) {
        let arr = vec![0u8; size];
        let expr: Expr = CalcSha256 {
            input: Box::new(Constant::from(arr).into()),
        }
        .into();
        group.bench_function(BenchmarkId::from_parameter(format!("{size:4}")), |b| {
            b.iter(|| eval_out_wo_ctx::<Vec<i8>>(&expr))
        });
    }
}

fn bench_append(c: &mut Criterion) {
    let mut group = c.benchmark_group("Coll.append");
    for size in (1..1024).step_by(200) {
        let arr = vec![0u8; size];
        let expr: Expr = Append::new(
            Constant::from(arr.clone()).into(),
            Constant::from(arr).into(),
        )
        .unwrap()
        .into();
        group.bench_function(BenchmarkId::from_parameter(format!("{size:4}")), |b| {
            b.iter(|| eval_out_wo_ctx::<Value<'_>>(&expr));
        });
    }
}

fn bench_substitute_constants(c: &mut Criterion) {
    let mut group = c.benchmark_group("substituteConstants");
    for size in (1..=101).step_by(10) {
        let tree_expr: Expr =
            SigmaAnd::new(vec![Constant::from(SigmaBoolean::from(true)).into(); size])
                .unwrap()
                .into();
        let tree_bytes = ErgoTree::new(ErgoTreeHeader::v1(true), &tree_expr)
            .unwrap()
            .sigma_serialize_bytes()
            .unwrap();
        let subst_expr = SubstConstants::new(
            tree_bytes.into(),
            Constant::from((0..size as i32).collect::<Vec<_>>()).into(),
            Constant::from(vec![SigmaProp::new(false.into()); size]).into(),
        )
        .unwrap()
        .into();
        group.bench_function(BenchmarkId::from_parameter(format!("{size:3}")), |b| {
            b.iter(|| eval_out_wo_ctx::<Value<'_>>(&subst_expr))
        });
    }
}

fn bench_exponentiate(c: &mut Criterion) {
    c.bench_function("exponentiate(generator, a)", |b| {
        let a = BigInt256::from_str("1000").unwrap();
        let expr: Expr = Exponentiate::new(generator().into(), a.into())
            .unwrap()
            .into();
        b.iter(|| eval_out_wo_ctx::<Value<'_>>(&expr))
    });
    c.bench_function("exponentiate(point, a)", |b| {
        let a = BigInt256::from_str("1000").unwrap();
        let expr: Expr =
            Exponentiate::new((exponentiate(&generator(), &2u32.into())).into(), a.into())
                .unwrap()
                .into();
        b.iter(|| eval_out_wo_ctx::<Value<'_>>(&expr))
    });
}

fn bench_filter(c: &mut Criterion) {
    let mut group = c.benchmark_group("filter");
    // v % 2 == 0
    let filter_lambda = BinOp {
        kind: BinOpKind::Relation(RelationOp::Eq),
        left: Box::new(
            BinOp {
                kind: BinOpKind::Arith(ergotree_ir::mir::bin_op::ArithOp::Modulo),
                left: Box::new(
                    ValUse {
                        val_id: 1.into(),
                        tpe: SType::SInt,
                    }
                    .into(),
                ),
                right: Box::new(Constant::from(2i32).into()),
            }
            .into(),
        ),
        right: Box::new(Constant::from(0i32).into()),
    };
    for size in (1..=1001i32).step_by(100) {
        let arr: Expr = Constant::from((0..size).collect::<Vec<_>>()).into();
        let expr: Expr = Filter::new(
            arr,
            FuncValue::new(
                vec![FuncArg {
                    idx: 1.into(),
                    tpe: SType::SInt,
                }],
                filter_lambda.clone().into(),
            )
            .into(),
        )
        .unwrap()
        .into();
        group.bench_function(BenchmarkId::from_parameter(format!("{size:4}")), |b| {
            b.iter(|| eval_out_wo_ctx::<Value<'_>>(&expr))
        });
    }
}

fn bench_flatmap(c: &mut Criterion) {
    let mut group = c.benchmark_group("flatMap");
    // flatten Coll[Coll[Byte]] into Coll[Byte]
    let flatmap_lambda: Expr = ValUse {
        val_id: 1.into(),
        tpe: SType::SColl(SType::SByte.into()),
    }
    .into();
    for size in (1..=1001).step_by(100) {
        let obj: Expr = Constant::from(vec![vec![0u8; 100]; size]).into();
        let mc: Expr = MethodCall::new(
            obj,
            FLATMAP_METHOD
                .clone()
                .with_concrete_types(
                    &[
                        (STypeVar::iv(), SType::SColl(SType::SByte.into())),
                        (STypeVar::ov(), SType::SByte),
                    ]
                    .into_iter()
                    .collect(),
                )
                .unwrap(),
            vec![FuncValue::new(
                vec![FuncArg {
                    idx: 1.into(),
                    tpe: SType::SColl(SType::SByte.into()),
                }],
                flatmap_lambda.clone(),
            )
            .into()],
        )
        .unwrap()
        .into();
        group.bench_function(BenchmarkId::from_parameter(format!("{size:4}")), |b| {
            b.iter(|| assert_eq!(eval_out_wo_ctx::<Vec<i8>>(&mc).len(), 100 * size))
        });
    }
}

criterion_group!(
    eval_benches,
    bench_blake2b256,
    bench_sha256,
    bench_append,
    bench_substitute_constants,
    bench_exponentiate,
    bench_filter,
    bench_flatmap
);

criterion_main!(eval_benches);
