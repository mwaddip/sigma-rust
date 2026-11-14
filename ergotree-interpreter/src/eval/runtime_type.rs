//! The JVM's runtime type of a static type, where the evaluator resolves one.
//!
//! sigmastate turns a script's static type into the runtime type of its values with
//! `Evaluation.stypeToRType`, which throws a `RuntimeException` for a type that has none
//! ([`SType::has_runtime_type`]). A node or a method that resolves a runtime type therefore
//! fails for such a type whatever it was about to do with it: a collection of no items, a
//! register or a variable that is not there, a method that would not read it. The sites
//! (sigma-state 6.0.7, every caller of `stypeToRType` an evaluation reaches):
//!
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
