//! Cost-oracle binding for the ergots mainnet-validate harness.
//!
//! Exposes `compute_tx_oracle_costs` (the production cost-oracle entry
//! point) plus a `CostOracleResult` wrapper and five `_test_only_*`
//! constructors (see "Test helpers" section below for the rationale on
//! why they live in the production module rather than behind cfg(test)).
//!
//! Given a parsed Transaction + per-input box list + data-input box list
//! + state context, `compute_tx_oracle_costs` runs `reduce_to_crypto`
//! for each input and returns raw `ctx.jit_cost_value()` (NOT
//! `ReductionResult.cost`, which is `jit_cost / 10` per
//! `ergotree-interpreter/src/eval.rs:174`). Mirrors
//! `tools/mainnet-validate/shim/src/cost_oracle.rs` semantics exactly.
//!
//! ## Why this binding exists
//!
//! The harness needs a cost-equivalence channel to validate the TS
//! evaluator against sigma-rust's `Context::jit_cost`. The shim used to
//! provide this by linking sigma-rust directly. With the REST refactor
//! the shim goes away; this WASM binding takes over the same role,
//! invoked from Node.js as `WasmCostOracle.computeTxOracleCosts(...)`.
//!
//! ## Invariants (must match shim's cost_oracle.rs:82-164)
//!
//! 1. Read `ctx.jit_cost_value()` AFTER `reduce_to_crypto`. Do NOT use
//!    `ReductionResult.cost` — that's block cost (`jit_cost / 10`).
//! 2. Override `ctx.tree_version` from the spent box's ergo_tree header
//!    version. `make_context` defaults this to `V0`, which is wrong for
//!    V1+ trees.
//! 3. Set `ctx.jit_cost_limit = Some(max_block_cost * 10)`. sigma-rust
//!    tracks raw JitCost internally while parameters expose block cost;
//!    the `* 10` converts. See `ergo-lib/src/wallet/tx_context.rs:202`.
//! 4. On any per-input error, return `is_ok=false` with `cost` still set
//!    to whatever was accumulated before the throw — sigma-rust preserves
//!    `ctx.jit_cost` across error returns.

use std::cell::Cell;

use ergo_lib::ergotree_interpreter::eval::reduce_to_crypto;
use ergo_lib::ergotree_ir::chain::ergo_box::ErgoBox as RsErgoBox;
use ergo_lib::wallet::signing::make_context;
use ergo_lib::wallet::tx_context::TransactionContext;
use wasm_bindgen::prelude::*;

use crate::box_coll::ErgoBoxes;
#[cfg(feature = "cost-oracle-tests")]
use crate::ergo_box::ErgoBox;
use crate::ergo_state_ctx::ErgoStateContext;
#[cfg(feature = "cost-oracle-tests")]
use crate::ergo_tree::ErgoTree;
use crate::parameters::Parameters;
use crate::transaction::Transaction;

/// Per-input cost-oracle result. Matches the shim's `CostOracleResult`
/// field-for-field. Returned as an element of the array produced by
/// [`compute_tx_oracle_costs`].
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct CostOracleResult {
    cost: u64,
    is_ok: bool,
    error_msg: Option<String>,
}

#[wasm_bindgen]
impl CostOracleResult {
    /// Raw JitCost accumulated on `ctx.jit_cost` post-eval. NOT block
    /// cost (which divides by 10). Use this directly for cost-equivalence
    /// comparison against the TS evaluator's `ctx.jitCost`.
    pub fn cost(&self) -> u64 {
        self.cost
    }
    /// `true` when `reduce_to_crypto` returned Ok; `false` when it errored.
    /// Note: a trivial-false SigmaProp short-circuits successfully (Ok),
    /// so `is_ok` reflects evaluator-Ok, not sigma-prop truth.
    pub fn is_ok(&self) -> bool {
        self.is_ok
    }
    /// Stringified `EvalError` when `is_ok == false`; None on success.
    pub fn error_msg(&self) -> Option<String> {
        self.error_msg.clone()
    }
}

/// Compute sigma-rust's per-input JIT cost for every input in `tx`.
///
/// Returns one [`CostOracleResult`] per input, in order. Per-input
/// errors do NOT short-circuit the loop — every input gets a result.
///
/// `spent_boxes` must satisfy `TransactionContext::new`'s ordering rule
/// (every input's `box_id` must appear in `spent_boxes` — order need
/// not match `tx.inputs` order; `TransactionContext` builds an internal
/// box-id index).
///
/// `data_boxes` is the data-input set; may be empty.
///
/// `state_ctx` carries pre_header / headers / parameters needed by
/// `make_context` and `jit_cost_limit` derivation.
#[wasm_bindgen]
pub fn compute_tx_oracle_costs(
    tx: &Transaction,
    spent_boxes: &ErgoBoxes,
    data_boxes: &ErgoBoxes,
    state_ctx: &ErgoStateContext,
) -> Result<Vec<CostOracleResult>, JsValue> {
    // Convert WASM wrappers to inner sigma-rust types. The
    // `clone().into()` pattern matches existing bindings (e.g.
    // verify_tx_input_proof in transaction.rs:419-429).
    let rs_spent: Vec<RsErgoBox> = spent_boxes.clone().into();
    let rs_data: Vec<RsErgoBox> = data_boxes.clone().into();
    // Transaction's inner field is pub(crate) (mirrors the existing
    // UnsignedTransaction pattern), so we read it directly. This
    // avoids adding a Clone derive + From impl to transaction.rs.
    let rs_tx: ergo_lib::chain::transaction::Transaction = tx.0.clone();

    let tx_ctx = TransactionContext::new(rs_tx, rs_spent, rs_data)
        .map_err(|e| JsValue::from_str(&format!("TransactionContext::new: {e}")))?;
    let rs_state: ergo_lib::chain::ergo_state_context::ErgoStateContext =
        state_ctx.clone().into();
    // Block-cost units → raw JitCost units. Mirrors shim line 144.
    let max_block_cost = rs_state.parameters.max_block_cost() as u64;

    // BoundedVec::len returns a BoundedVecLen wrapper that converts to
    // usize via Into. Matches shim's pattern (cost_oracle.rs:117).
    let input_count: usize = tx_ctx.spending_tx.inputs.len().into();
    let mut results = Vec::with_capacity(input_count);

    for input_index in 0..input_count {
        // Build a fresh Context per input. `make_context` returns a
        // Context with jit_cost: Cell::new(0), jit_cost_limit: None,
        // tree_version: Default::default() — we override those next.
        let mut ctx = match make_context(&rs_state, &tx_ctx, input_index) {
            Ok(c) => c,
            Err(e) => {
                results.push(CostOracleResult {
                    cost: 0,
                    is_ok: false,
                    error_msg: Some(format!("make_context failed: {e}")),
                });
                continue;
            }
        };

        // Look up the spent box for this input — needed to derive
        // tree_version from its ergo_tree header.
        let input_box = match tx_ctx
            .spending_tx
            .inputs
            .get(input_index)
            .and_then(|i| tx_ctx.get_input_box(&i.box_id))
        {
            Some(b) => b,
            None => {
                results.push(CostOracleResult {
                    cost: 0,
                    is_ok: false,
                    error_msg: Some(format!(
                        "input box not found for input_index={input_index}"
                    )),
                });
                continue;
            }
        };

        // Override tree_version from the spent box's actual tree header.
        // Mirrors shim line 128-138.
        let tree_version = match input_box.ergo_tree.header() {
            Ok(h) => h.version(),
            Err(e) => {
                results.push(CostOracleResult {
                    cost: ctx.jit_cost_value(),
                    is_ok: false,
                    error_msg: Some(format!("ergo_tree.header() failed: {e}")),
                });
                continue;
            }
        };
        ctx.tree_version = Cell::new(tree_version);

        // Set jit_cost_limit. * 10 converts block cost → raw JitCost.
        // Mirrors shim line 144.
        ctx.jit_cost_limit = Some(max_block_cost * 10);

        // Reduce-to-crypto. ctx.jit_cost mutates in place; we read it
        // after the call (success or error). NEVER use
        // ReductionResult.cost — that's jit_cost / 10.
        let outcome = reduce_to_crypto(&input_box.ergo_tree, &ctx);
        let cost = ctx.jit_cost_value();
        results.push(match outcome {
            Ok(_) => CostOracleResult {
                cost,
                is_ok: true,
                error_msg: None,
            },
            Err(e) => CostOracleResult {
                cost,
                is_ok: false,
                error_msg: Some(format!("{e}")),
            },
        });
    }
    Ok(results)
}

// =============================================================================
// Test helpers + production Parameters constructor.
//
// Four `_test_only_*` helpers below (trivial_sigma_prop_tree,
// ergo_box_with_tree, one_input_one_output_tx, synthetic_state_context)
// are gated behind `#[cfg(feature = "cost-oracle-tests")]`. They exist
// solely to let the smoke tests in `tests/test_cost_oracle.js`
// reproduce shim's cost_oracle.rs tests bit-equivalently. Production
// builds (npm run build-nodejs) skip the feature; these helpers are
// NOT in the shipped pkg-nodejs/.
//
// `parameters_new` (below, NOT feature-gated) is a production gap-filler:
// the WASM Parameters wrapper only exposes `default_parameters()`. The
// ergots harness needs per-field control (specifically `max_block_cost`
// override) to derive `ctx.jit_cost_limit = max_block_cost * 10` per
// the cost-oracle contract. Keeping this constructor in the production
// surface is intentional.
// =============================================================================

/// Build a bare `Const(SigmaProp(TrivialProp(true|false)))` ergo-tree.
/// Mirrors `tools/mainnet-validate/shim/src/cost_oracle.rs:229-240`
/// (the shim's `trivial_true_tree` test helper). This is the dominant
/// mainnet shape (>90% of inputs are P2PK-or-similar trivial sigma-prop
/// constants); sigma-rust short-circuits it via `trivial_reduce` at exactly
/// 50 JitCost (`EVAL_SIGMA_PROP_CONSTANT`).
#[cfg(feature = "cost-oracle-tests")]
#[wasm_bindgen]
pub fn _test_only_trivial_sigma_prop_tree(value: bool) -> Result<ErgoTree, JsValue> {
    use ergo_lib::ergotree_ir::ergo_tree::{ErgoTree as RsErgoTree, ErgoTreeHeader};
    use ergo_lib::ergotree_ir::mir::constant::{Constant, Literal};
    use ergo_lib::ergotree_ir::mir::expr::Expr;
    use ergo_lib::ergotree_ir::sigma_protocol::sigma_boolean::{SigmaBoolean, SigmaProp};
    use ergo_lib::ergotree_ir::types::stype::SType;

    let expr = Expr::Const(Constant {
        tpe: SType::SSigmaProp,
        v: Literal::SigmaProp(Box::new(SigmaProp::new(SigmaBoolean::TrivialProp(value)))),
    });
    RsErgoTree::new(ErgoTreeHeader::v0(false), &expr)
        .map(ErgoTree::from)
        .map_err(|e| JsValue::from_str(&format!("_test_only_trivial_sigma_prop_tree: {e}")))
}

/// Build an ErgoBox holding `tree` as its proposition. tx_id is the
/// all-zeros TxId; index is 0. Only `ergo_tree` is read by the cost
/// oracle, so the other fields are filler.
///
/// Mirrors shim's `box_with_tree` helper (cost_oracle.rs:244-258).
#[cfg(feature = "cost-oracle-tests")]
#[wasm_bindgen]
pub fn _test_only_ergo_box_with_tree(tree: &ErgoTree, value: u64) -> Result<ErgoBox, JsValue> {
    use ergo_lib::chain::transaction::TxId;
    use ergo_lib::ergotree_ir::chain::ergo_box::box_value::BoxValue;
    use ergo_lib::ergotree_ir::chain::ergo_box::{
        ErgoBox as RsErgoBox2, ErgoBoxCandidate, NonMandatoryRegisters,
    };

    let rs_tree: ergo_lib::ergotree_ir::ergo_tree::ErgoTree = tree.clone().into();
    let candidate = ErgoBoxCandidate {
        value: BoxValue::new(value).map_err(|e| JsValue::from_str(&format!("BoxValue: {e}")))?,
        ergo_tree: rs_tree,
        tokens: None,
        additional_registers: NonMandatoryRegisters::empty(),
        creation_height: 0,
    };
    RsErgoBox2::from_box_candidate(&candidate, TxId::zero(), 0)
        .map(ErgoBox::from)
        .map_err(|e| JsValue::from_str(&format!("_test_only_ergo_box_with_tree: {e}")))
}

/// Build a 1-input, 1-output Transaction spending `input_box`. The output
/// is a self-spend (same value + tree). Mirrors shim's
/// `one_input_one_output_tx` helper (cost_oracle.rs:261-282).
#[cfg(feature = "cost-oracle-tests")]
#[wasm_bindgen]
pub fn _test_only_one_input_one_output_tx(input_box: &ErgoBox) -> Result<Transaction, JsValue> {
    use ergo_lib::chain::transaction::input::Input;
    use ergo_lib::chain::transaction::prover_result::ProverResult;
    use ergo_lib::chain::transaction::{Transaction as RsTransaction, TxIoVec};
    use ergo_lib::ergotree_interpreter::sigma_protocol::prover::ProofBytes;
    use ergo_lib::ergotree_ir::chain::context_extension::ContextExtension;
    use ergo_lib::ergotree_ir::chain::ergo_box::{ErgoBoxCandidate, NonMandatoryRegisters};

    let rs_input_box: RsErgoBox = input_box.clone().into();
    let output_candidate = ErgoBoxCandidate {
        value: rs_input_box.value,
        ergo_tree: rs_input_box.ergo_tree.clone(),
        tokens: None,
        additional_registers: NonMandatoryRegisters::empty(),
        creation_height: 0,
    };
    let inputs = TxIoVec::from_vec(vec![Input {
        box_id: rs_input_box.box_id(),
        spending_proof: ProverResult {
            proof: ProofBytes::Empty,
            extension: ContextExtension::empty(),
        },
    }])
    .map_err(|e| JsValue::from_str(&format!("inputs vec: {e}")))?;
    let outputs = TxIoVec::from_vec(vec![output_candidate])
        .map_err(|e| JsValue::from_str(&format!("outputs vec: {e}")))?;
    RsTransaction::new(inputs, None, outputs)
        .map(Transaction::from)
        .map_err(|e| JsValue::from_str(&format!("Transaction::new: {e}")))
}

/// Build a `Parameters` instance with all nine positional fields. This
/// is the test-side counterpart to the shim's `Parameters::new(...)`
/// call (cost_oracle.rs:354-361). The WASM wrapper only exposes
/// `Parameters::default_parameters()`; we need fine-grained control to
/// build tight-max-block-cost parameters for the cost-limit-exceeded test.
///
/// Field order matches `ergo-lib/src/chain/parameters.rs:128-151`:
///   1. `block_version`        — block protocol version
///   2. `storage_fee_factor`   — fee per byte of long-lived storage
///   3. `min_value_per_byte`   — minimum nanoErg per output byte
///   4. `max_block_size`       — block-bytes ceiling
///   5. `max_block_cost`       — block-cost ceiling (raw JitCost = this × 10)
///   6. `token_access_cost`    — per-token-access cost
///   7. `input_cost`           — per-input base cost
///   8. `data_input_cost`      — per-data-input base cost
///   9. `output_cost`          — per-output base cost
#[wasm_bindgen]
pub fn parameters_new(
    block_version: i32,
    storage_fee_factor: i32,
    min_value_per_byte: i32,
    max_block_size: i32,
    max_block_cost: i32,
    token_access_cost: i32,
    input_cost: i32,
    data_input_cost: i32,
    output_cost: i32,
) -> Parameters {
    ergo_lib::chain::parameters::Parameters::new(
        block_version,
        storage_fee_factor,
        min_value_per_byte,
        max_block_size,
        max_block_cost,
        token_access_cost,
        input_cost,
        data_input_cost,
        output_cost,
    )
    .into()
}

/// Build an `ErgoStateContext` with the given parameters and otherwise
/// arbitrary state (default pre-header + ten dummy headers). This mirrors
/// the shim's `synthetic_state_context` helper which uses
/// `force_any_val::<ErgoStateContext>()`. Cost-oracle behavior depends
/// only on `parameters.max_block_cost` and `pre_header.height` (latter is
/// arbitrary for these tests).
///
/// We build a hard-coded default-shape Header (`synth_header()` below)
/// and round-trip it into a `PreHeader`, replicating it ten times for the
/// `Headers` array. This avoids needing the `arbitrary` feature in the
/// production WASM build.
#[cfg(feature = "cost-oracle-tests")]
#[wasm_bindgen]
pub fn _test_only_synthetic_state_context(
    parameters: &Parameters,
) -> Result<ErgoStateContext, JsValue> {
    use ergo_lib::chain::ergo_state_context::{ErgoStateContext as RsErgoStateContext, Headers};
    use ergo_lib::ergo_chain_types::{Header, PreHeader};

    let header = synth_header();
    let pre_header = PreHeader::from(header.clone());
    let headers_vec: Vec<Header> = (0..10).map(|_| header.clone()).collect();
    let headers: Headers = headers_vec.try_into().map_err(|_| {
        JsValue::from_str("_test_only_synthetic_state_context: failed to build [Header; 10]")
    })?;
    let rs_params: ergo_lib::chain::parameters::Parameters = parameters.clone().into();
    Ok(RsErgoStateContext::new(pre_header, headers, rs_params).into())
}

/// Construct a placeholder Header for tests. All zero-bytes for digest
/// fields; height=1; version=1. The cost oracle does not read most of
/// these fields (pre_header.height is what gets surfaced into Context).
#[cfg(feature = "cost-oracle-tests")]
fn synth_header() -> ergo_lib::ergo_chain_types::Header {
    use ergo_lib::ergo_chain_types::ec_point::generator;
    use ergo_lib::ergo_chain_types::{
        ADDigest, AutolykosSolution, BlockId, Digest32, Header, Votes,
    };

    Header {
        version: 1,
        id: BlockId(Digest32::zero()),
        parent_id: BlockId(Digest32::zero()),
        ad_proofs_root: Digest32::zero(),
        state_root: ADDigest::zero(),
        transaction_root: Digest32::zero(),
        timestamp: 0,
        n_bits: 0,
        height: 1,
        extension_root: Digest32::zero(),
        autolykos_solution: AutolykosSolution {
            miner_pk: Box::new(generator()),
            pow_onetime_pk: None,
            nonce: vec![],
            pow_distance: None,
        },
        votes: Votes([0u8; 3]),
        unparsed_bytes: Box::new([]),
    }
}
