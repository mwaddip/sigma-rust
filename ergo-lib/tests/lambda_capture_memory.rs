//! Mainnet block 1,379,778, transaction 1 (`41140f6c…`), one of the block's 23 spends of a
//! contract that defines lambdas over lambdas. A lambda value copied its whole environment, an
//! earlier lambda in it included, so each nested definition doubled what the next held: on fork
//! `5268baa6` this spend took 5 GB and 5 s to validate, and the block 56 GB in parallel. A lambda
//! now shares its environment and its body, as sigmastate's closure does.
//!
//! The fixture holds the transaction, its 3 input and 2 data-input boxes, the block's header and
//! its 9 predecessors, and the epoch's parameters.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use ergo_chain_types::{Header, PreHeader};
use ergo_lib::chain::ergo_state_context::{ErgoStateContext, Headers};
use ergo_lib::chain::parameters::{Parameter, Parameters};
use ergo_lib::chain::transaction::Transaction;
use ergo_lib::wallet::tx_context::TransactionContext;
use ergotree_ir::chain::ergo_box::ErgoBox;
use ergotree_ir::serialization::SigmaSerializable;

/// Counts the bytes allocated and not yet freed, and their peak
struct Counting;

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            let live = LIVE.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
            PEAK.fetch_max(live, Ordering::Relaxed);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// At most this much memory above what was live before `validate`: the spend now peaks under
/// 2 MB, and took 5 GB with copied environments
const PEAK_BOUND: usize = 128 << 20;

#[test]
#[allow(clippy::unwrap_used)]
fn a_spend_of_nested_lambdas_validates_in_bounded_memory() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("test-vectors/mainnet_1379778_tx1.json")).unwrap();
    let bytes = |v: &serde_json::Value| base16::decode(v.as_str().unwrap()).unwrap();
    let boxes = |key: &str| -> Vec<ErgoBox> {
        fixture[key]
            .as_array()
            .unwrap()
            .iter()
            .map(|b| ErgoBox::sigma_parse_bytes(&bytes(b)).unwrap())
            .collect()
    };
    let tx = Transaction::sigma_parse_bytes(&bytes(&fixture["tx"])).unwrap();
    let mut headers: Vec<Header> = serde_json::from_value(fixture["headers"].clone()).unwrap();
    let header = headers.remove(0);
    let mut parameters = Parameters::default();
    for (name, parameter) in [
        ("StorageFeeFactor", Parameter::StorageFeeFactor),
        ("MinValuePerByte", Parameter::MinValuePerByte),
        ("MaxBlockSize", Parameter::MaxBlockSize),
        ("MaxBlockCost", Parameter::MaxBlockCost),
        ("TokenAccessCost", Parameter::TokenAccessCost),
        ("InputCost", Parameter::InputCost),
        ("DataInputCost", Parameter::DataInputCost),
        ("OutputCost", Parameter::OutputCost),
        ("BlockVersion", Parameter::BlockVersion),
    ] {
        let value = fixture["parameters"][name].as_i64().unwrap();
        parameters
            .parameters_table
            .insert(parameter, i32::try_from(value).unwrap());
    }
    let state_context = ErgoStateContext::new(
        PreHeader::from(header),
        Headers::from_vec(headers).unwrap(),
        parameters,
    );
    let tx_context = TransactionContext::new(tx, boxes("inputs"), boxes("data_inputs")).unwrap();

    let before = LIVE.load(Ordering::Relaxed);
    PEAK.store(before, Ordering::Relaxed);
    let cost = tx_context.validate(&state_context).unwrap();
    let peak = PEAK.load(Ordering::Relaxed) - before;

    assert_eq!(cost, 23_225);
    assert!(
        peak < PEAK_BOUND,
        "validation peaked {} MiB above the baseline",
        peak >> 20
    );
}
