//! On-chain types

pub mod address;
pub mod context;
pub mod context_extension;
pub mod ergo_box;
pub mod evaluated_value;
#[cfg(feature = "json")]
pub mod json;
pub mod token;
pub mod tx_id;

/// Index Map
pub type IndexMap<K, V> = indexmap::IndexMap<K, V, foldhash::fast::RandomState>;
/// Index Set
pub type IndexSet<T> = indexmap::IndexSet<T, foldhash::fast::RandomState>;
