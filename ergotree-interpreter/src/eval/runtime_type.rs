//! The JVM's runtime type of a static type, where the evaluator resolves one.
//!
//! sigmastate turns a script's static type into the runtime type of its values with
//! `Evaluation.stypeToRType`, which throws a `RuntimeException` for a type that has none
//! ([`SType::has_runtime_type`]). A node or a method that resolves a runtime type therefore
//! fails for such a type whatever it was about to do with it: a collection of no items, a
//! register or a variable that is not there, a method that would not read it. The sites
//! (sigma-state 6.0.7, every caller of `stypeToRType` an evaluation reaches):
//!
//! - the `ConcreteCollection` node, for its item type, after its cost;
//! - the `GetVar` node, for the variable's type, before its cost;
//! - the `ExtractRegisterAs` node, for the register's type, after its box and its cost;
//! - `MapCollection`, `Coll.map` and `Coll.flatMap`, for the result's item type;
//! - every register of a box that is not a constant when one is read, and every self context
//!   variable before a tree is evaluated (`EvaluatedValue::script_type`).
//!
//! A new node or method that sigmastate gives an `RType` built from a static type takes the
//! check at the place sigmastate resolves it.

use alloc::format;

use ergotree_ir::types::stype::SType;

use super::EvalError;

/// sigmastate's `stypeToRType` at a site that resolves the runtime type of `tpe`: an error
/// for a type that has none. `site` names the caller in the error.
pub(crate) fn check_runtime_type(tpe: &SType, site: &str) -> Result<(), EvalError> {
    if tpe.has_runtime_type() {
        Ok(())
    } else {
        Err(EvalError::Misc(format!(
            "{site}: the type {tpe:?} has no runtime type (a type variable, or a function of \
             other than one argument or with type parameters)"
        )))
    }
}

#[cfg(test)]
#[cfg(feature = "arbitrary")]
#[allow(clippy::unwrap_used)]
mod tests {
    //! Rows measured on the sigma-state 6.0.7 jars (the round's `measure/`): each body
    //! evaluated at tree versions 0 and 3, activated version 3, with the cost counter read at
    //! the throw. SANTA measured `Coll[T]().size` (its third reference-store reply, X13).
    use super::*;
    use crate::eval::test_util::{eval_body, innermost};
    use alloc::string::String;
    use alloc::vec;
    use alloc::vec::Vec;
    use ergotree_ir::ergo_tree::ErgoTreeVersion;
    use ergotree_ir::mir::value::Value;

    /// The type variable `T`
    const T: &str = "670154";
    /// `Coll[Box]()(0)`: throws, 40 on its own
    const THROWING_BOX: &str = "b20c6300040000";

    /// A box constant: value 1000000, a P2PK tree, no tokens, the given register section
    fn box_with(registers: &str) -> String {
        let tree = "0008cd0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798";
        format!("63c0843d{tree}0000{registers}{}00", "00".repeat(32))
    }

    /// Types with no runtime type, and the lowest tree version whose parser reads each (a
    /// function type is read from version 3)
    fn types_without() -> Vec<(&'static str, String, u8)> {
        vec![
            ("T", T.into(), 0),
            ("Coll[T]", format!("0c{T}"), 0),
            ("Coll[Coll[T]]", format!("0c0c{T}"), 0),
            ("Option[T]", format!("24{T}"), 0),
            ("(T, Int)", format!("3c{T}04"), 0),
            ("(Int, T)", format!("40{T}"), 0),
            ("(Int, Int, T)", format!("480404{T}"), 0),
            ("Coll[(Int, Int, T)]", format!("0c480404{T}"), 0),
            ("(Int, Int) => Int", "700204040400".into(), 3),
            ("() => Int", "70000400".into(), 3),
            ("[U](Int) => Int", "7001040401670155".into(), 3),
            ("T => Int", format!("7001{T}0400"), 3),
            ("Int => T", format!("700104{T}00"), 3),
            ("Coll[(Int, Int) => Int]", "0c700204040400".into(), 3),
        ]
    }

    /// A site: its name, its body for a type, the tree versions that have it, and the cost
    /// counter where sigmastate throws for a type with no runtime type
    type Site = (&'static str, fn(&str) -> String, &'static [u8], u64);

    fn sites() -> Vec<Site> {
        vec![
            // the node's 20, then the item type, before any item
            ("Coll[ty]()", |ty| format!("8300{ty}"), &[0, 3], 20),
            // the type before the node's 10 and before the lookup
            ("getVar[ty](1)", |ty| format!("e301{ty}"), &[0, 3], 0),
            // the box (5), the node's 50, then the type: a register that is not there
            (
                "box.R4[ty], no R4",
                |ty| format!("c6{}04{ty}", box_with("00")),
                &[0, 3],
                55,
            ),
            // and one that is
            (
                "box.R4[ty], R4 = 1",
                |ty| format!("c6{}04{ty}", box_with("010402")),
                &[0, 3],
                55,
            ),
        ]
    }

    fn version(v: u8) -> ErgoTreeVersion {
        if v == 0 {
            ErgoTreeVersion::V0
        } else {
            ErgoTreeVersion::V3
        }
    }

    /// The error of a body that must not evaluate, under the span wrapper each node adds, and
    /// the cost counter where the evaluation stopped
    fn stopped(body: &str, v: u8) -> (EvalError, u64) {
        let (res, cost) = eval_body(body, version(v));
        assert!(res.is_err(), "{body} at v{v}: {res:?}");
        (innermost(&res.unwrap_err()).clone(), cost)
    }

    #[test]
    fn a_type_with_no_runtime_type_stops_every_site_at_the_jvms_counter() {
        let mut rows = 0;
        for (site, body, versions, stops_at) in sites() {
            for (tpe, bytes, from) in types_without() {
                for v in versions.iter().filter(|v| **v >= from) {
                    let (err, cost) = stopped(&body(&bytes), *v);
                    assert!(
                        matches!(err, EvalError::Misc(_)),
                        "{site} with {tpe} at v{v}: {err:?}"
                    );
                    assert_eq!(cost, stops_at, "{site} with {tpe} at v{v}");
                    rows += 1;
                }
            }
        }
        // 4 node sites at both versions (8 types at v0, 14 at v3)
        assert_eq!(rows, 4 * (8 + 14));
    }

    #[test]
    fn the_register_node_adds_its_cost_after_its_box() {
        // an input that is no box fails the cast, before the node's cost (the jars: the
        // constant `1` stops at its own 5, for any type)
        for tpe in ["04", T] {
            for v in [0, 3] {
                assert_eq!(
                    stopped(&format!("c60402{}{tpe}", "04"), v).1,
                    5,
                    "{tpe} at v{v}"
                );
            }
        }
        // `ExtractRegisterAs.eval`: the box, the node's cost, the type. A box expression that
        // throws stops at its own 40, for any type (the jars)
        for tpe in ["04", T] {
            for v in [0, 3] {
                let cost = stopped(&format!("c6{THROWING_BOX}04{tpe}"), v).1;
                assert_eq!(cost, 40, "{tpe} at v{v}");
            }
        }
        // the value and the cost of a read that succeeds are where they were
        let r4 = format!("c6{}0404", box_with("010402"));
        let some_1 = Value::Opt(Some(alloc::boxed::Box::new(Value::Int(1))));
        for v in [0, 3] {
            assert_eq!(eval_body(&r4, version(v)), (Ok(some_1.clone()), 55), "v{v}");
        }
    }
}
