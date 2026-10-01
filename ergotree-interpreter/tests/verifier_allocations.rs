//! Verifying a proof must cost heap in proportion to the proposition, as it does on sigmastate,
//! which computes the commitments and writes the Fiat-Shamir bytes in one pass over the tree
//! (`Interpreter.scala:388-409`, `UnprovenTree.scala:268-287`). A conjecture without children
//! is 2 bytes and 15 JitCost, so a deep and wide tree is cheap to send: a pass that copies each
//! subtree at every level turns a transaction of 100 KB into a gigabyte.
//!
//! These tests count heap bytes. They compare a wide tree with the same tree under a chain of
//! nodes: the chain may add the cost of its own nodes, and no copy of the tree below them.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use ergotree_interpreter::sigma_protocol::sig_serializer::parse_sig_compute_challenges;
use ergotree_interpreter::sigma_protocol::verifier::{compute_commitments, verify_signature};
use ergotree_ir::serialization::SigmaSerializable;
use ergotree_ir::sigma_protocol::sigma_boolean::SigmaBoolean;

struct Counting;

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
static TOTAL: AtomicUsize = AtomicUsize::new(0);

// SAFETY: every call goes to `System` unchanged; the counters only add up the sizes.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = System.alloc(layout);
        if !p.is_null() {
            let live = LIVE.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
            PEAK.fetch_max(live, Ordering::Relaxed);
            TOTAL.fetch_add(layout.size(), Ordering::Relaxed);
        }
        p
    }

    unsafe fn dealloc(&self, p: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
        System.dealloc(p, layout)
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// The counters are the process's: one measurement at a time
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

fn one_at_a_time() -> std::sync::MutexGuard<'static, ()> {
    // a test that failed while it measured has poisoned nothing worth keeping
    ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// What a call cost in heap bytes
#[derive(Debug)]
struct Heap {
    /// the most that was live above the level the call started at
    peak: usize,
    /// everything it allocated
    total: usize,
}

fn heap_of<T>(f: impl FnOnce() -> T) -> (T, Heap) {
    let start = LIVE.load(Ordering::Relaxed);
    PEAK.store(start, Ordering::Relaxed);
    let total = TOTAL.load(Ordering::Relaxed);
    let res = f();
    let heap = Heap {
        peak: PEAK.load(Ordering::Relaxed) - start,
        total: TOTAL.load(Ordering::Relaxed) - total,
    };
    (res, heap)
}

const MESSAGE: &[u8] = b"a message";
const WIDTH: usize = 5000;
const DEPTH: usize = 60;
/// What the chain's own nodes may add, with room to spare
const CHAIN: usize = DEPTH * 1024;

/// `CAND(5000 × CAND())`: the count is the VLQ `88 27`
fn wide() -> Vec<u8> {
    [&[0x96, 0x88, 0x27][..], &[0x96, 0x00].repeat(WIDTH)].concat()
}

/// `inner` under a chain of 60 ANDs of one child each
fn under_ands(inner: &[u8]) -> Vec<u8> {
    [&[0x96, 0x01].repeat(DEPTH)[..], inner].concat()
}

/// `inner` as the first child of 60 nested ORs, each with `CAND()` as its last child
fn under_ors(inner: &[u8]) -> Vec<u8> {
    [
        &[0x97, 0x02].repeat(DEPTH)[..],
        inner,
        &[0x96, 0x00].repeat(DEPTH)[..],
    ]
    .concat()
}

/// A proof of all zeros: long enough to be read, and never valid
fn zeros(len: usize) -> Vec<u8> {
    vec![0u8; len]
}

#[allow(clippy::unwrap_used)]
fn commitments_heap(proposition: &[u8]) -> Heap {
    let proposition = SigmaBoolean::sigma_parse_bytes(proposition).unwrap();
    // a tree of ANDs without leaves reads its root challenge and nothing else
    let tree = parse_sig_compute_challenges(&proposition, zeros(24)).unwrap();
    heap_of(|| compute_commitments(tree)).1
}

#[test]
fn computing_commitments_copies_no_subtree() {
    let _one = one_at_a_time();
    let alone = commitments_heap(&wide());
    let under_a_chain = commitments_heap(&under_ands(&wide()));
    assert!(
        under_a_chain.peak <= alone.peak + CHAIN,
        "peak: {under_a_chain:?} under a chain, {alone:?} alone"
    );
    assert!(
        under_a_chain.total <= alone.total + CHAIN,
        "total: {under_a_chain:?} under a chain, {alone:?} alone"
    );
}

/// The heap that verifying `proof` costs. The proposition is given, so its own bytes do not
/// count.
#[allow(clippy::unwrap_used)]
fn verification_heap(proposition: &[u8], proof: &[u8]) -> Heap {
    let proposition = SigmaBoolean::sigma_parse_bytes(proposition).unwrap();
    let (valid, heap) = heap_of(|| verify_signature(proposition, MESSAGE, proof).unwrap());
    assert!(!valid);
    heap
}

#[test]
fn verifying_under_a_chain_of_ands_copies_no_subtree() {
    let _one = one_at_a_time();
    let alone = verification_heap(&wide(), &zeros(24));
    let under_a_chain = verification_heap(&under_ands(&wide()), &zeros(24));
    assert!(
        under_a_chain.peak <= alone.peak + CHAIN,
        "peak: {under_a_chain:?} under a chain, {alone:?} alone"
    );
    assert!(
        under_a_chain.total <= alone.total + CHAIN,
        "total: {under_a_chain:?} under a chain, {alone:?} alone"
    );
}

#[test]
fn verifying_under_a_chain_of_ors_copies_no_subtree() {
    let _one = one_at_a_time();
    let alone = verification_heap(&wide(), &zeros(24));
    // an OR reads a challenge for each child but its last: one a level, after the root's
    let under_a_chain = verification_heap(&under_ors(&wide()), &zeros(24 * (DEPTH + 1)));
    assert!(
        under_a_chain.peak <= alone.peak + CHAIN,
        "peak: {under_a_chain:?} under a chain, {alone:?} alone"
    );
    assert!(
        under_a_chain.total <= alone.total + CHAIN,
        "total: {under_a_chain:?} under a chain, {alone:?} alone"
    );
}
