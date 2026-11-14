//! The JVM's fixed method costs.

use ergotree_ir::types::smethod::SMethod;
use ergotree_ir::types::stype_companion::STypeCompanion;

/// sigmastate's `method.costKind` where it is a `FixedCost`. `MethodCall.eval` adds such a cost
/// itself, after the arguments are evaluated and before it invokes the method
/// (`values.scala:1364-1380`), so the cost is the call's and not the evaluator's: a method
/// has its row here whether or not it has an evaluator. A method of any other cost kind (per
/// item, dynamic, `powHit`'s own, or none at all) adds its costs in its body.
///
/// The rows are sigma-state 6.0.7's, read from the jars (`MethodsContainer.methods`, each
/// `SMethod.costKind`); the test `the_fixed_costs_are_the_jvms` holds that census.
pub(crate) fn fixed_method_cost(method: &SMethod) -> Option<u64> {
    let id = method.method_id().0;
    match method.obj_type {
        STypeCompanion::SByte
        | STypeCompanion::SShort
        | STypeCompanion::SInt
        | STypeCompanion::SLong => handler_cost(id),
        STypeCompanion::SBigInt => match id {
            14 => Some(5),  // toUnsigned
            15 => Some(15), // toUnsignedMod
            _ => handler_cost(id),
        },
        STypeCompanion::SUnsignedBigInt => match id {
            14 => Some(150), // modInverse
            15 => Some(30),  // plusMod
            16 => Some(30),  // subtractMod
            17 => Some(40),  // multiplyMod
            18 => Some(20),  // mod
            19 => Some(10),  // toSigned
            _ => handler_cost(id),
        },
        STypeCompanion::GroupElem => match id {
            2 => Some(250), // getEncoded
            3 => Some(900), // exp
            4 => Some(40),  // multiply
            5 => Some(45),  // negate
            6 => Some(900), // expUnsigned
            _ => None,
        },
        // propBytes is per item; isProven has no cost kind
        STypeCompanion::SigmaProp => None,
        STypeCompanion::Coll => match id {
            1 => Some(14),  // size
            10 => Some(30), // apply
            33 => Some(30), // get
            _ => None,
        },
        STypeCompanion::Option => match id {
            2 => Some(10), // isDefined
            3 => Some(15), // get
            4 => Some(20), // getOrElse
            7 => Some(20), // map
            8 => Some(20), // filter
            _ => None,
        },
        STypeCompanion::Box => match id {
            1 => Some(8),       // value
            2 => Some(10),      // propositionBytes
            3 => Some(12),      // bytes
            4 => Some(12),      // bytesWithoutRef
            5 => Some(12),      // id
            6 => Some(16),      // creationInfo
            7 => Some(50),      // getReg (v5)
            8 => Some(15),      // tokens
            9..=18 => Some(50), // R0 .. R9
            19 => Some(50),     // getReg
            _ => None,
        },
        STypeCompanion::AvlTree => match id {
            // digest, enabledOperations, keyLength, valueLengthOpt, isInsertAllowed,
            // isUpdateAllowed, isRemoveAllowed
            1..=7 => Some(15),
            8 => Some(45),  // updateOperations
            15 => Some(40), // updateDigest
            _ => None,
        },
        STypeCompanion::Context => match id {
            1 => Some(15),  // dataInputs
            2 => Some(15),  // headers
            3 => Some(15),  // preHeader
            4 => Some(10),  // INPUTS
            5 => Some(10),  // OUTPUTS
            6 => Some(26),  // HEIGHT
            7 => Some(10),  // SELF
            8 => Some(20),  // selfBoxIndex
            9 => Some(15),  // LastBlockUtxoRootHash
            10 => Some(20), // minerPubKey
            11 => Some(10), // getVar
            12 => Some(10), // getVarFromInput
            _ => None,
        },
        STypeCompanion::Header => match id {
            // id, version, parentId, ADProofsRoot, stateRoot, transactionsRoot, timestamp,
            // nBits, height, extensionRoot, minerPk, powOnetimePk, powNonce, powDistance, votes
            1..=15 => Some(10),
            16 => Some(700), // checkPow
            _ => None,
        },
        STypeCompanion::PreHeader => match id {
            // version, parentId, timestamp, nBits, height, minerPk, votes
            1..=7 => Some(10),
            _ => None,
        },
        STypeCompanion::Global => match id {
            1 => Some(10), // groupGenerator
            5 => Some(10), // fromBigEndianBytes
            6 => Some(25), // encodeNbits
            7 => Some(50), // decodeNbits
            9 => Some(5),  // some
            10 => Some(5), // none
            _ => None,
        },
    }
}

/// The eight numeric methods sigmastate serves with a `userDefinedInvoke` handler, on each of
/// the six numeric owners: `toBytes`, `toBits`, `bitwiseInverse`, `bitwiseOr`, `bitwiseAnd`,
/// `bitwiseXor`, `shiftLeft`, `shiftRight` (ids 6 to 13), `FixedCost(5)` each. The casts
/// (ids 1 to 5) have no cost kind.
fn handler_cost(id: u8) -> Option<u64> {
    match id {
        6..=13 => Some(5),
        _ => None,
    }
}

#[cfg(test)]
#[cfg(feature = "arbitrary")]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::fixed_method_cost;
    use crate::eval::test_util::{eval_body, eval_body_err};
    use crate::eval::EvalError;
    use crate::eval::{smethod_eval_fn, Env, MethodCallSite};
    use alloc::boxed::Box;
    use ergo_chain_types::ec_point::generator;
    use ergo_chain_types::{Header, PreHeader};
    use ergotree_ir::chain::context::Context;
    use ergotree_ir::chain::ergo_box::ErgoBox;
    use ergotree_ir::ergo_tree::ErgoTreeVersion;
    use ergotree_ir::mir::avl_tree_data::AvlTreeData;
    use ergotree_ir::mir::coll_value::CollValue;
    use ergotree_ir::mir::value::Value;
    use ergotree_ir::reference::Ref;
    use ergotree_ir::sigma_protocol::sigma_boolean::SigmaProp;
    use ergotree_ir::types::smethod::MethodId;
    use ergotree_ir::types::stype_companion::STypeCompanion as C;
    use ergotree_ir::unsignedbigint256::UnsignedBigInt;
    use sigma_test_util::force_any_val;

    /// Every method of sigma-state 6.0.7's tables at tree version 3: owner, method id, and its
    /// `FixedCost` or `None` for any other cost kind. Generated from the jars' census
    /// (`MethodsContainer.methods`, each `SMethod.costKind`).
    const JVM: &[(C, u8, Option<u64>)] = &[
        (C::SByte, 1, None),                 // toByte
        (C::SByte, 2, None),                 // toShort
        (C::SByte, 3, None),                 // toInt
        (C::SByte, 4, None),                 // toLong
        (C::SByte, 5, None),                 // toBigInt
        (C::SByte, 6, Some(5)),              // toBytes
        (C::SByte, 7, Some(5)),              // toBits
        (C::SByte, 8, Some(5)),              // bitwiseInverse
        (C::SByte, 9, Some(5)),              // bitwiseOr
        (C::SByte, 10, Some(5)),             // bitwiseAnd
        (C::SByte, 11, Some(5)),             // bitwiseXor
        (C::SByte, 12, Some(5)),             // shiftLeft
        (C::SByte, 13, Some(5)),             // shiftRight
        (C::SShort, 1, None),                // toByte
        (C::SShort, 2, None),                // toShort
        (C::SShort, 3, None),                // toInt
        (C::SShort, 4, None),                // toLong
        (C::SShort, 5, None),                // toBigInt
        (C::SShort, 6, Some(5)),             // toBytes
        (C::SShort, 7, Some(5)),             // toBits
        (C::SShort, 8, Some(5)),             // bitwiseInverse
        (C::SShort, 9, Some(5)),             // bitwiseOr
        (C::SShort, 10, Some(5)),            // bitwiseAnd
        (C::SShort, 11, Some(5)),            // bitwiseXor
        (C::SShort, 12, Some(5)),            // shiftLeft
        (C::SShort, 13, Some(5)),            // shiftRight
        (C::SInt, 1, None),                  // toByte
        (C::SInt, 2, None),                  // toShort
        (C::SInt, 3, None),                  // toInt
        (C::SInt, 4, None),                  // toLong
        (C::SInt, 5, None),                  // toBigInt
        (C::SInt, 6, Some(5)),               // toBytes
        (C::SInt, 7, Some(5)),               // toBits
        (C::SInt, 8, Some(5)),               // bitwiseInverse
        (C::SInt, 9, Some(5)),               // bitwiseOr
        (C::SInt, 10, Some(5)),              // bitwiseAnd
        (C::SInt, 11, Some(5)),              // bitwiseXor
        (C::SInt, 12, Some(5)),              // shiftLeft
        (C::SInt, 13, Some(5)),              // shiftRight
        (C::SLong, 1, None),                 // toByte
        (C::SLong, 2, None),                 // toShort
        (C::SLong, 3, None),                 // toInt
        (C::SLong, 4, None),                 // toLong
        (C::SLong, 5, None),                 // toBigInt
        (C::SLong, 6, Some(5)),              // toBytes
        (C::SLong, 7, Some(5)),              // toBits
        (C::SLong, 8, Some(5)),              // bitwiseInverse
        (C::SLong, 9, Some(5)),              // bitwiseOr
        (C::SLong, 10, Some(5)),             // bitwiseAnd
        (C::SLong, 11, Some(5)),             // bitwiseXor
        (C::SLong, 12, Some(5)),             // shiftLeft
        (C::SLong, 13, Some(5)),             // shiftRight
        (C::SBigInt, 1, None),               // toByte
        (C::SBigInt, 2, None),               // toShort
        (C::SBigInt, 3, None),               // toInt
        (C::SBigInt, 4, None),               // toLong
        (C::SBigInt, 5, None),               // toBigInt
        (C::SBigInt, 6, Some(5)),            // toBytes
        (C::SBigInt, 7, Some(5)),            // toBits
        (C::SBigInt, 8, Some(5)),            // bitwiseInverse
        (C::SBigInt, 9, Some(5)),            // bitwiseOr
        (C::SBigInt, 10, Some(5)),           // bitwiseAnd
        (C::SBigInt, 11, Some(5)),           // bitwiseXor
        (C::SBigInt, 12, Some(5)),           // shiftLeft
        (C::SBigInt, 13, Some(5)),           // shiftRight
        (C::SBigInt, 14, Some(5)),           // toUnsigned
        (C::SBigInt, 15, Some(15)),          // toUnsignedMod
        (C::GroupElem, 2, Some(250)),        // getEncoded
        (C::GroupElem, 3, Some(900)),        // exp
        (C::GroupElem, 4, Some(40)),         // multiply
        (C::GroupElem, 5, Some(45)),         // negate
        (C::GroupElem, 6, Some(900)),        // expUnsigned
        (C::SigmaProp, 1, None),             // propBytes
        (C::SigmaProp, 2, None),             // isProven
        (C::SUnsignedBigInt, 1, None),       // toByte
        (C::SUnsignedBigInt, 2, None),       // toShort
        (C::SUnsignedBigInt, 3, None),       // toInt
        (C::SUnsignedBigInt, 4, None),       // toLong
        (C::SUnsignedBigInt, 5, None),       // toBigInt
        (C::SUnsignedBigInt, 6, Some(5)),    // toBytes
        (C::SUnsignedBigInt, 7, Some(5)),    // toBits
        (C::SUnsignedBigInt, 8, Some(5)),    // bitwiseInverse
        (C::SUnsignedBigInt, 9, Some(5)),    // bitwiseOr
        (C::SUnsignedBigInt, 10, Some(5)),   // bitwiseAnd
        (C::SUnsignedBigInt, 11, Some(5)),   // bitwiseXor
        (C::SUnsignedBigInt, 12, Some(5)),   // shiftLeft
        (C::SUnsignedBigInt, 13, Some(5)),   // shiftRight
        (C::SUnsignedBigInt, 14, Some(150)), // modInverse
        (C::SUnsignedBigInt, 15, Some(30)),  // plusMod
        (C::SUnsignedBigInt, 16, Some(30)),  // subtractMod
        (C::SUnsignedBigInt, 17, Some(40)),  // multiplyMod
        (C::SUnsignedBigInt, 18, Some(20)),  // mod
        (C::SUnsignedBigInt, 19, Some(10)),  // toSigned
        (C::Coll, 1, Some(14)),              // size
        (C::Coll, 2, None),                  // getOrElse
        (C::Coll, 3, None),                  // map
        (C::Coll, 4, None),                  // exists
        (C::Coll, 5, None),                  // fold
        (C::Coll, 6, None),                  // forall
        (C::Coll, 7, None),                  // slice
        (C::Coll, 8, None),                  // filter
        (C::Coll, 9, None),                  // append
        (C::Coll, 10, Some(30)),             // apply
        (C::Coll, 14, None),                 // indices
        (C::Coll, 15, None),                 // flatMap
        (C::Coll, 19, None),                 // patch
        (C::Coll, 20, None),                 // updated
        (C::Coll, 21, None),                 // updateMany
        (C::Coll, 26, None),                 // indexOf
        (C::Coll, 29, None),                 // zip
        (C::Coll, 30, None),                 // reverse
        (C::Coll, 31, None),                 // startsWith
        (C::Coll, 32, None),                 // endsWith
        (C::Coll, 33, Some(30)),             // get
        (C::Option, 2, Some(10)),            // isDefined
        (C::Option, 3, Some(15)),            // get
        (C::Option, 4, Some(20)),            // getOrElse
        (C::Option, 7, Some(20)),            // map
        (C::Option, 8, Some(20)),            // filter
        (C::Box, 1, Some(8)),                // value
        (C::Box, 2, Some(10)),               // propositionBytes
        (C::Box, 3, Some(12)),               // bytes
        (C::Box, 4, Some(12)),               // bytesWithoutRef
        (C::Box, 5, Some(12)),               // id
        (C::Box, 6, Some(16)),               // creationInfo
        (C::Box, 8, Some(15)),               // tokens
        (C::Box, 7, Some(50)),               // getRegV5
        (C::Box, 9, Some(50)),               // R0
        (C::Box, 10, Some(50)),              // R1
        (C::Box, 11, Some(50)),              // R2
        (C::Box, 12, Some(50)),              // R3
        (C::Box, 13, Some(50)),              // R4
        (C::Box, 14, Some(50)),              // R5
        (C::Box, 15, Some(50)),              // R6
        (C::Box, 16, Some(50)),              // R7
        (C::Box, 17, Some(50)),              // R8
        (C::Box, 18, Some(50)),              // R9
        (C::Box, 19, Some(50)),              // getReg
        (C::AvlTree, 1, Some(15)),           // digest
        (C::AvlTree, 2, Some(15)),           // enabledOperations
        (C::AvlTree, 3, Some(15)),           // keyLength
        (C::AvlTree, 4, Some(15)),           // valueLengthOpt
        (C::AvlTree, 5, Some(15)),           // isInsertAllowed
        (C::AvlTree, 6, Some(15)),           // isUpdateAllowed
        (C::AvlTree, 7, Some(15)),           // isRemoveAllowed
        (C::AvlTree, 8, Some(45)),           // updateOperations
        (C::AvlTree, 9, None),               // contains
        (C::AvlTree, 10, None),              // get
        (C::AvlTree, 11, None),              // getMany
        (C::AvlTree, 12, None),              // insert
        (C::AvlTree, 13, None),              // update
        (C::AvlTree, 14, None),              // remove
        (C::AvlTree, 15, Some(40)),          // updateDigest
        (C::AvlTree, 16, None),              // insertOrUpdate
        (C::Context, 1, Some(15)),           // dataInputs
        (C::Context, 2, Some(15)),           // headers
        (C::Context, 3, Some(15)),           // preHeader
        (C::Context, 4, Some(10)),           // INPUTS
        (C::Context, 5, Some(10)),           // OUTPUTS
        (C::Context, 6, Some(26)),           // HEIGHT
        (C::Context, 7, Some(10)),           // SELF
        (C::Context, 8, Some(20)),           // selfBoxIndex
        (C::Context, 9, Some(15)),           // LastBlockUtxoRootHash
        (C::Context, 10, Some(20)),          // minerPubKey
        (C::Context, 11, Some(10)),          // getVar
        (C::Context, 12, Some(10)),          // getVarFromInput
        (C::Header, 1, Some(10)),            // id
        (C::Header, 2, Some(10)),            // version
        (C::Header, 3, Some(10)),            // parentId
        (C::Header, 4, Some(10)),            // ADProofsRoot
        (C::Header, 5, Some(10)),            // stateRoot
        (C::Header, 6, Some(10)),            // transactionsRoot
        (C::Header, 7, Some(10)),            // timestamp
        (C::Header, 8, Some(10)),            // nBits
        (C::Header, 9, Some(10)),            // height
        (C::Header, 10, Some(10)),           // extensionRoot
        (C::Header, 11, Some(10)),           // minerPk
        (C::Header, 12, Some(10)),           // powOnetimePk
        (C::Header, 13, Some(10)),           // powNonce
        (C::Header, 14, Some(10)),           // powDistance
        (C::Header, 15, Some(10)),           // votes
        (C::Header, 16, Some(700)),          // checkPow
        (C::PreHeader, 1, Some(10)),         // version
        (C::PreHeader, 2, Some(10)),         // parentId
        (C::PreHeader, 3, Some(10)),         // timestamp
        (C::PreHeader, 4, Some(10)),         // nBits
        (C::PreHeader, 5, Some(10)),         // height
        (C::PreHeader, 6, Some(10)),         // minerPk
        (C::PreHeader, 7, Some(10)),         // votes
        (C::Global, 1, Some(10)),            // groupGenerator
        (C::Global, 2, None),                // xor
        (C::Global, 3, None),                // serialize
        (C::Global, 8, None),                // powHit
        (C::Global, 4, None),                // deserializeTo
        (C::Global, 6, Some(25)),            // encodeNbits
        (C::Global, 7, Some(50)),            // decodeNbits
        (C::Global, 5, Some(10)),            // fromBigEndianBytes
        (C::Global, 9, Some(5)),             // some
        (C::Global, 10, Some(5)),            // none
    ];

    #[test]
    fn the_fixed_costs_are_the_jvms() {
        assert_eq!(JVM.len(), 199);
        assert_eq!(JVM.iter().filter(|r| r.2.is_some()).count(), 138);
        for (owner, id, cost) in JVM {
            let method = owner.method_by_id(&MethodId(*id)).unwrap();
            assert_eq!(fixed_method_cost(&method), *cost, "{owner:?} {id}");
        }
    }

    #[derive(Debug)]
    enum Refusal {
        UnexpectedValue,
        Misc,
        NotFound,
    }

    fn assert_stops(rows: &[(&str, Refusal, u64)]) {
        for (body, refusal, cost) in rows {
            let (e, c) = eval_body_err(body);
            let named = match refusal {
                Refusal::UnexpectedValue => matches!(e, EvalError::UnexpectedValue(_)),
                Refusal::Misc => matches!(e, EvalError::Misc(_)),
                Refusal::NotFound => matches!(e, EvalError::NotFound(_)),
            };
            assert!(named, "{body}: expected {refusal:?}, got {e:?}");
            assert_eq!(c, *cost, "{body}");
        }
    }

    // sigmastate adds a fixed-cost method's cost and then invokes it: a receiver of another
    // class, another argument count and a method with no Java method are refused by that
    // invoke, after the cost (`values.scala:1364-1380`). Measured on the jars.
    #[test]
    fn a_refused_call_of_a_fixed_cost_method_has_paid_for_it() {
        use Refusal::*;
        assert_stops(&[
            ("db0c010101", UnexpectedValue, 23), // B1 true.size
            ("db0c0195010010030002040101", UnexpectedValue, 38), // B2 If(false, coll, true).size
            ("dc040c0101010402", UnexpectedValue, 19), // B3 true.shiftLeft(1) as Int's
            ("dc0703040201060102", UnexpectedValue, 914), // B4 1.exp(2) as GroupElement's
            ("db63010101", UnexpectedValue, 17), // B5 true.value as Box's
            // B6 true.R4 as Box's. The cost is the JVM's; its refusal there is the missing Java
            // method of `R4`, while Rust's first refusal is the receiver's
            ("db630d0101", UnexpectedValue, 59),
            ("dc0c0a10030002040204020402", Misc, 49), // C1 coll.apply(1, 1)
            ("dc0c011003000204010402", Misc, 28),     // C2 coll.size with an argument
            ("dc090f09010101090101", Misc, 44),       // C3 1.plusMod(1)
            ("dc240428010201040e", NotFound, 34),     // E1 Some(1).getOrElse(7): no evaluator
        ]);
        // the totals of the calls that evaluate; C4: a handler ignores an extra argument
        for (body, value, cost) in [
            ("dc040c04020204020402", Value::Int(2), 24), // C4 1.shiftLeft(1, 1)
            ("db0c011003000204", Value::Int(3), 23),     // coll.size
            ("dc0c0a1003000204010402", Value::Int(1), 44), // coll.apply(1)
        ] {
            let (res, c) = eval_body(body, ErgoTreeVersion::V3);
            assert_eq!(res.unwrap(), value, "{body}");
            assert_eq!(c, cost, "{body}");
        }
    }

    // any other method is invoked directly and refuses before anything of its own is charged
    // (`values.scala:1381-1391`)
    #[test]
    fn a_refused_call_of_another_method_has_paid_nothing_of_it() {
        use Refusal::*;
        assert_stops(&[
            ("dc0c1d01010110010e", UnexpectedValue, 14), // true.zip(..)
            ("dc0c1d10030002040210010e0402", Misc, 19),  // coll.zip(.., 1)
            ("db04010402", NotFound, 9),                 // 1.toByte: no cost kind, no evaluator
        ]);
    }

    /// A value of the owner's class; no wildcard, so a new owner does not compile until it
    /// has one
    fn receiver_of(owner: C) -> Value<'static> {
        match owner {
            C::SByte => Value::Byte(1),
            C::SShort => Value::Short(1),
            C::SInt => Value::Int(1),
            C::SLong => Value::Long(1),
            C::SBigInt => Value::BigInt(num_bigint::BigInt::from(1)),
            C::SUnsignedBigInt => Value::UnsignedBigInt(UnsignedBigInt::from(1u32)),
            C::Context => Value::Context,
            C::Box => Value::CBox(Ref::from(force_any_val::<ErgoBox>())),
            C::Coll => Value::Coll(CollValue::Ints(alloc::vec![1, 2].into())),
            C::GroupElem => Value::from(generator()),
            C::SigmaProp => Value::SigmaProp(Box::new(force_any_val::<SigmaProp>())),
            C::Global => Value::Global,
            C::Header => Value::Header(Box::new(force_any_val::<Header>())),
            C::PreHeader => Value::PreHeader(Box::new(force_any_val::<PreHeader>())),
            C::Option => Value::Opt(Some(Box::new(Value::Int(1)))),
            C::AvlTree => Value::from(force_any_val::<AvlTreeData>()),
        }
    }

    // The call adds a fixed-cost method's cost, so its evaluator must add none of it: a
    // charge left in an evaluator would be paid twice by every script that calls the method,
    // and the JVM's body of such a method adds nothing. Each of the 125 fixed-cost methods
    // that have an evaluator is run on a receiver of its owner's class with NO arguments: a
    // method without parameters evaluates in full, one with parameters stops where it reads
    // its first argument, and neither may have moved the counter. (A charge placed after
    // the first argument read is beyond this test.)
    #[test]
    fn a_fixed_cost_evaluator_adds_nothing_itself() {
        let mut run = 0;
        for (owner, id, cost) in JVM {
            if cost.is_none() {
                continue;
            }
            let method = owner.method_by_id(&MethodId(*id)).unwrap();
            let Ok(eval_fn) = smethod_eval_fn(&method) else {
                continue;
            };
            let mut ctx = force_any_val::<Context>();
            ctx.activated_script_version_byte = 3;
            ctx.tree_version.set(ErgoTreeVersion::V3);
            ctx.jit_cost.set(0);
            let site = MethodCallSite {
                method: &method,
                args: &[],
                explicit_type_args: &hashbrown::HashMap::new(),
            };
            let _ = eval_fn(
                &site,
                &mut Env::empty(),
                &ctx,
                receiver_of(*owner),
                alloc::vec![],
            );
            assert_eq!(ctx.jit_cost_value(), 0, "{owner:?} {id} {}", method.name());
            run += 1;
        }
        assert_eq!(run, 125);
    }
}
