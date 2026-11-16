use core::fmt::Display;

use alloc::sync::Arc;
use alloc::vec::Vec;
use ergotree_ir::mir::val_def::ValId;
use ergotree_ir::mir::value::bindings_to_static;
use ergotree_ir::mir::value::debug_start;
use ergotree_ir::mir::value::display_start;
use ergotree_ir::mir::value::Bindings;
use ergotree_ir::mir::value::Value;

/// Environment for the interpreter. Copy-on-write: a lambda captures it by sharing its bindings
/// (`FuncValue.eval` closes over the JVM's immutable `DataEnv`), and a later change copies them
/// first, so a captured environment never changes
#[derive(PartialEq, Eq, Clone)]
pub struct Env<'ctx> {
    store: Bindings<'ctx>,
}

impl<'ctx> Env<'ctx> {
    /// Empty environment
    pub fn empty() -> Env<'ctx> {
        Env {
            store: Default::default(),
        }
    }

    /// The environment of these bindings: a lambda's captured ones
    pub(crate) fn from_bindings(store: Bindings<'ctx>) -> Env<'ctx> {
        Env { store }
    }

    /// Returns `true` if the environment is empty
    pub fn is_empty(&self) -> bool {
        self.store.is_empty()
    }

    /// Extend this environment (create new) with added element
    pub fn extend(&self, idx: ValId, v: Value<'ctx>) -> Env<'ctx> {
        let mut new_env = self.clone();
        new_env.insert(idx, v);
        new_env
    }

    /// Insert a Value for the given ValId
    pub fn insert(&mut self, idx: ValId, v: Value<'ctx>) {
        Arc::make_mut(&mut self.store).insert(idx, v);
    }

    /// Remove a Value for the given ValId
    pub fn remove(&mut self, idx: &ValId) {
        Arc::make_mut(&mut self.store).remove(idx);
    }

    /// Get an element
    pub fn get(&self, idx: ValId) -> Option<&Value<'ctx>> {
        self.store.get(&idx)
    }

    /// The current bindings, shared: what a `FuncValue` captures when it evaluates to a lambda
    /// value
    pub(crate) fn shared(&self) -> Bindings<'ctx> {
        self.store.clone()
    }

    /// Convert borrowed data to Arc
    pub(crate) fn to_static(&'ctx self) -> Env<'static> {
        Env {
            store: bindings_to_static(&self.store),
        }
    }
}

impl<'ctx> Env<'ctx> {
    /// The bindings in the order of their ids
    fn sorted(&self) -> Vec<(&ValId, &Value<'ctx>)> {
        let mut bindings: Vec<_> = self.store.iter().collect();
        bindings.sort_by_key(|(id, _)| **id);
        bindings
    }
}

/// Each binding with the START of its value's text. An environment travels inside every error
/// the evaluator returns, and whoever reports the error prints it; a value built at evaluation
/// can hold one collection many times over, so its whole text is work without a bound
impl Display for Env<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        for (id, value) in self.sorted() {
            writeln!(f, "v{}: {}", id, display_start(value))?;
        }
        Ok(())
    }
}

/// Each binding with the start of its value's `Debug` text (see the `Display`)
impl core::fmt::Debug for Env<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut map = f.debug_map();
        for (id, value) in self.sorted() {
            map.entry(
                &format_args!("v{id}"),
                &format_args!("{}", debug_start(value)),
            );
        }
        map.finish()
    }
}
