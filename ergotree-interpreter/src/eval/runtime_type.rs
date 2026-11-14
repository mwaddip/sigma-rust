//! The JVM's runtime type of a static type, where the evaluator resolves one.
//!
//! sigmastate turns a script's static type into the runtime type of its values with
//! `Evaluation.stypeToRType`, which throws a `RuntimeException` for a type that has none
//! ([`SType::has_runtime_type`]). A node or a method that resolves a runtime type therefore
//! fails for such a type whatever it was about to do with it: a collection of no items, a
//! register or a variable that is not there, a method that would not read it. The sites
//! (sigma-state 6.0.7), each a caller of `stypeToRType` with a type a script chooses:
//!
//! - the `ConcreteCollection` node, for its item type, after its cost;
//! - the `GetVar` node, for the variable's type, before its cost;
//! - the `ExtractRegisterAs` node, for the register's type, after its box and its cost;
//! - `MapCollection`, `Coll.map` and `Coll.flatMap`, for the result's item type;
//! - a fixed-cost method that is handed a type descriptor ([`descriptor_type`]), at the call,
//!   before its arguments;
//! - `Global.deserializeTo`, for the call's type, after its arguments and before its cost;
//! - every register of a box when a script reads one, or the box's tokens or creation info
//!   (`sbox::convert_registers`);
//! - every self context variable before a tree is evaluated, and the variable
//!   `getVarFromInput` reads (`EvaluatedValue::script_type`);
//! - the data reader, for a collection's item type, at parse (`ergotree-ir`,
//!   `SigmaParsingError::NoRuntimeType`).
//!
//! One more caller takes a script's type and needs nothing here: the lookup of a fixed-cost
//! method's Java method resolves its parameter types (`SMethod.javaMethod`), and the one
//! method whose parameter follows the script's types is `Option.getOrElse`, which that lookup
//! fails for every type (it has no evaluator here either).
//!
//! A new node or method that sigmastate gives an `RType` built from a static type takes the
//! check at the place sigmastate resolves it.

use alloc::format;

use ergotree_ir::types::smethod::SMethod;
use ergotree_ir::types::stype::SType;
use ergotree_ir::types::stype_companion::STypeCompanion;

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

/// The type whose runtime type decides whether a fixed-cost method's type descriptor
/// resolves: sigmastate's `invokeDescsBuilder` (`SMethod.extraDescriptors`,
/// `SMethod.scala:108-116`), read from the SPECIALISED method's signature before the arguments
/// are evaluated (`values.scala:1366`). The five rows are sigma-state 6.0.7's, read from the
/// jars (every method of `MethodsContainer` with `irInfo.invokeDescsBuilder` and a
/// `FixedCost`). Each is answered with the method's range: sigmastate builds the descriptor
/// from the range for the three `Global` methods and from the item of the `Option` range for
/// `getReg` and `getVarFromInput`, and an option has a runtime type exactly when its item has.
/// `Coll.flatMap` has a builder too and is per item: its body resolves the type
/// (`flatMap_eval`), as `deserializeTo`'s does.
pub(crate) fn descriptor_type(method: &SMethod) -> Option<&SType> {
    match (method.obj_type, method.method_id().0) {
        (STypeCompanion::Box, 19)          // getReg[T]
        | (STypeCompanion::Context, 12)    // getVarFromInput[T]
        | (STypeCompanion::Global, 5)      // fromBigEndianBytes[T]
        | (STypeCompanion::Global, 9)      // some[T]
        | (STypeCompanion::Global, 10) => Some(&*method.tpe().t_range), // none[T]
        _ => None,
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
    /// `Coll[Int]()(0)`: throws, 40 on its own
    const THROWS: &str = "b21000040000";
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

    /// Types with a runtime type, the same way
    fn types_with() -> Vec<(&'static str, &'static str, u8)> {
        vec![
            ("Int", "04", 0),
            ("Coll[Int]", "10", 0),
            ("Any", "61", 0),
            ("Option[Int]", "28", 0),
            ("(Int, Int, Int)", "48040404", 0),
            ("Box", "63", 0),
            ("Int => Int", "7001040400", 3),
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
            // the receiver (5) and the call's 4, before the arguments
            (
                "box.getReg[ty](4)",
                |ty| format!("dc6313{}010408{ty}", box_with("00")),
                &[3],
                9,
            ),
            (
                "box.getReg[ty](throws)",
                |ty| format!("dc6313{}01{THROWS}{ty}", box_with("00")),
                &[3],
                9,
            ),
            // `CONTEXT` (1) and the call's 4
            (
                "CONTEXT.getVarFromInput[ty](0, 1)",
                |ty| format!("dc650cfe0203000201{ty}"),
                &[3],
                5,
            ),
            ("Global.none[ty]", |ty| format!("db6a0add{ty}"), &[3], 9),
            (
                "Global.some[ty](1)",
                |ty| format!("dc6a09dd010402{ty}"),
                &[3],
                9,
            ),
            (
                "Global.some[ty](throws)",
                |ty| format!("dc6a09dd01{THROWS}{ty}"),
                &[3],
                9,
            ),
            (
                "Global.fromBigEndianBytes[ty](Coll(1))",
                |ty| format!("dc6a05dd010e0101{ty}"),
                &[3],
                9,
            ),
            // `Global` (5), the call's 4, the argument (5): before the method's own cost
            (
                "Global.deserializeTo[ty](Coll(0))",
                |ty| format!("dc6a04dd010e0100{ty}"),
                &[3],
                14,
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

    /// `some[T](x)`: the call is specialised by its argument's type, so the bare type variable
    /// is the argument's type by the time the descriptor is read (the jars: a value at 19, and
    /// the argument's own error at 49)
    fn specialised_by_its_argument(site: &str, tpe: &str) -> bool {
        site.starts_with("Global.some") && tpe == "T"
    }

    #[test]
    fn a_type_with_no_runtime_type_stops_every_site_at_the_jvms_counter() {
        let mut rows = 0;
        for (site, body, versions, stops_at) in sites() {
            for (tpe, bytes, from) in types_without() {
                for v in versions.iter().filter(|v| **v >= from) {
                    if specialised_by_its_argument(site, tpe) {
                        continue;
                    }
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
        // 4 node sites at both versions (8 types at v0, 14 at v3) and 8 method sites at v3,
        // less the two `some[T]` rows
        assert_eq!(rows, 4 * (8 + 14) + 8 * 14 - 2);
    }

    #[test]
    fn a_type_with_a_runtime_type_passes_every_site() {
        // (site, cost of a value) for every type with a runtime type; `None`: the site's own
        // refusal, at that counter (a register of another type, bytes of the wrong length,
        // data that does not read as the type)
        let passes: &[(&str, u64)] = &[
            ("Coll[ty]()", 20),
            ("box.R4[ty], no R4", 55),
            ("box.getReg[ty](4)", 64),
            ("Global.none[ty]", 14),
            ("Global.some[ty](1)", 19),
        ];
        for (site, body, versions, _) in sites() {
            let Some((_, cost)) = passes.iter().find(|(name, _)| *name == site) else {
                continue;
            };
            for (tpe, bytes, from) in types_with() {
                for v in versions.iter().filter(|v| **v >= from) {
                    let (res, c) = eval_body(&body(bytes), version(*v));
                    assert!(res.is_ok(), "{site} with {tpe} at v{v}: {res:?}");
                    assert_eq!(c, *cost, "{site} with {tpe} at v{v}");
                }
            }
        }
        let v3 = ErgoTreeVersion::V3;
        // the sites whose outcome depends on the type: the value where there is one
        let int_1 = Value::Opt(Some(alloc::boxed::Box::new(Value::Int(1))));
        let site = |name: &str| sites().into_iter().find(|s| s.0 == name).unwrap().1;
        assert_eq!(
            eval_body(&site("box.R4[ty], R4 = 1")("04"), v3),
            (Ok(int_1.clone()), 55)
        );
        assert_eq!(
            eval_body(&site("Global.some[ty](1)")(T), v3),
            (Ok(int_1), 19)
        );
        assert_eq!(
            eval_body(&site("Global.deserializeTo[ty](Coll(0))")("04"), v3),
            (Ok(Value::Int(0)), 146)
        );
        // and the site's own refusal once the type passed, at the JVM's counter: past the
        // counter of the type's refusal, or another error where the two counters are one
        for (name, tpe, stops_at) in [
            ("box.getReg[ty](throws)", "04", 49),
            ("Global.some[ty](throws)", "04", 49),
            ("Global.some[ty](throws)", T, 49),
            ("Global.fromBigEndianBytes[ty](Coll(1))", "04", 24),
            ("Global.fromBigEndianBytes[ty](Coll(1))", "7001040400", 24),
            ("Global.deserializeTo[ty](Coll(0))", "61", 146),
            ("Global.deserializeTo[ty](Coll(0))", "7001040400", 146),
        ] {
            assert_eq!(
                stopped(&site(name)(tpe), 3).1,
                stops_at,
                "{name} with {tpe}"
            );
        }
        let (err, cost) = stopped(&site("box.R4[ty], R4 = 1")("10"), 3);
        assert!(matches!(err, EvalError::UnexpectedValue(_)), "{err:?}");
        assert_eq!(cost, 55);
    }

    #[test]
    fn deserialize_to_with_a_function_type_that_has_type_parameters() {
        // The jars, each row. sigmastate's `mc.tpe` is the method's range with the type
        // argument substituted a second time, which drops a function type's own parameter
        // `T`. One type is left out here because of it, `[T](Int) => Int`: the JVM takes it
        // for `Int => Int` and stops at 146, this stops at 14 (queue candidate 24)
        let call = |tpe: &str| format!("dc6a04dd010e0100{tpe}");
        let u = "670155";
        for (name, tpe, stops_at) in [
            // a function type with a runtime type: the data reader refuses a function
            ("Int => Int", "7001040400".into(), 146),
            // a parameter that is not the method's stays, and the type has no runtime type
            ("[U](Int) => Int", format!("7001040401{u}"), 14),
            ("[T, U](Int) => Int", format!("7001040402{T}{u}"), 14),
            ("[U, T](Int) => Int", format!("7001040402{u}{T}"), 14),
            // and so does a `T` that the function type uses, or a second argument
            ("[T](T) => Int", format!("7001{T}0401{T}"), 14),
            ("[T](Int) => T", format!("700104{T}01{T}"), 14),
            ("[T](Int, Int) => Int", format!("700204040401{T}"), 14),
        ] {
            assert_eq!(stopped(&call(&tpe), 3).1, stops_at, "{name}");
        }
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

    #[test]
    fn a_register_whose_type_has_no_runtime_type_fails_every_register_read_of_its_box() {
        // `CBox.registers` converts every register of the box on the first read. An empty
        // collection of tuples of other than two items is data the JVM reads whatever the
        // items are, so a CONSTANT register can have such a type: R4 = `Coll[(Int, Int, T)]()`
        let b = box_with(&format!("010c480404{T}00"));
        for v in [0, 3] {
            // R5[Int], which is not there; R4 at its own type
            for body in [format!("c6{b}0504"), format!("c6{b}040c480404{T}")] {
                let (err, cost) = stopped(&body, v);
                assert!(matches!(err, EvalError::Misc(_)), "{err:?}");
                assert_eq!(cost, 55, "v{v}");
            }
            // a read that is not of a register does not convert them
            assert_eq!(
                eval_body(&format!("c1{b}"), version(v)),
                (Ok(Value::Long(1000000)), 13)
            );
        }
        // `getReg[Int](5)`: the method's cost is added before the registers are read
        assert_eq!(stopped(&format!("dc6313{b}01040a04"), 3).1, 64);
    }

    #[test]
    fn every_read_that_goes_through_the_registers_converts_them() {
        // sigmastate's `CBox.tokens` and `CBox.creationInfo` are `getReg(2)` and `getReg(3)`,
        // and `getReg` reads `registers.length` for every index that is not negative
        // (`CBox.scala:33`, `:46-58`): each converts every register of the box. The jars, each
        // row, with an R4 that does not convert: the constant `Coll[(Int, Int, T)]()` and the
        // collection expression `Coll[T]()`
        let reads: [(&str, &[u8], u64); 5] = [
            ("db6308{b}", &[0, 3], 24),        // box.tokens
            ("c7{b}", &[0, 3], 21),            // the creation info node
            ("db6306{b}", &[0, 3], 25),        // box.creationInfo
            ("dc6313{b}01041404", &[3], 64),   // box.getReg[Int](10)
            ("dc6313{b}0104900304", &[3], 64), // box.getReg[Int](200)
        ];
        for r4 in [format!("010c480404{T}00"), format!("018300{T}")] {
            let b = box_with(&r4);
            for (body, versions, stops_at) in reads {
                for v in versions {
                    let (err, cost) = stopped(&body.replace("{b}", &b), *v);
                    assert!(
                        matches!(err, EvalError::Misc(_) | EvalError::TryExtractFrom(_)),
                        "{body} with R4 = {r4}: {err:?}"
                    );
                    assert_eq!(cost, stops_at, "{body} with R4 = {r4} at v{v}");
                }
            }
            // a negative index answers before the registers are read
            let (res, cost) = eval_body(&format!("dc6313{b}01040104"), ErgoTreeVersion::V3);
            assert_eq!((res, cost), (Ok(Value::Opt(None)), 64), "R4 = {r4}");
        }
        // with registers that convert, each read gives its value at the same counter
        let b = box_with("010402");
        for (body, versions, cost) in reads {
            for v in versions {
                let (res, c) = eval_body(&body.replace("{b}", &b), version(*v));
                assert!(res.is_ok(), "{body} at v{v}: {res:?}");
                assert_eq!(c, cost, "{body} at v{v}");
            }
        }
    }

    #[test]
    fn the_creation_info_node_adds_its_cost_after_its_box() {
        // `ExtractCreationInfo.eval`: the box, the node's cost, the read. A box expression that
        // throws stops at its own 40, and an input that is no box at its own 5 (the jars)
        for v in [0, 3] {
            assert_eq!(stopped(&format!("c7{THROWING_BOX}"), v).1, 40, "v{v}");
            assert_eq!(stopped("c70402", v).1, 5, "v{v}");
        }
    }

    #[test]
    fn the_five_descriptor_methods_and_no_other() {
        use crate::eval::method_call::test_owners;
        use ergotree_ir::types::smethod::MethodId;
        let mut found = Vec::new();
        for owner in test_owners::all() {
            for id in 0u8..=60 {
                if let Some(m) = owner.method_by_id(&MethodId(id)) {
                    if descriptor_type(&m).is_some() {
                        found.push(format!("{owner:?}.{}", m.name()));
                    }
                }
            }
        }
        assert_eq!(
            found,
            [
                "Context.getVarFromInput",
                "Box.getReg",
                "Global.fromBigEndianBytes",
                "Global.some",
                "Global.none"
            ]
        );
    }
}
