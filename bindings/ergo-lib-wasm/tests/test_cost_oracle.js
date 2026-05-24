// Smoke tests for the cost-oracle WASM binding.
//
// Reproduces the Rust-side tests in
// `tools/mainnet-validate/shim/src/cost_oracle.rs:294-396` through the
// WASM API surface. Bit-equivalence with those tests is the gate for
// PLAN-2j-rest.md Task 1 — see
// `docs/specs/2026-05-24-ergoscript-2j-rest-design.md` §3.3.
//
// What we assert:
//   1. trivial_true tree → 50 raw JitCost, is_ok = true
//   2. trivial_false tree → 50 raw JitCost, is_ok = true (reduce_to_crypto
//      returns Ok(TrivialProp(false)); the verifier short-circuits, not
//      the oracle's evaluator)
//   3. tight Parameters (max_block_cost=1 → jit_cost_limit=10) → is_ok =
//      false with an error_msg (structural failure; partial cost may be
//      0 or >0 depending on sigma-rust's check order)
//   4. tree_version derived from spent box (V0 — mirrors shim
//      cost_oracle.rs:370-396 weak V0 coverage; exercises the
//      `ctx.tree_version` override path at src/cost_oracle.rs:160-171)

import { assert } from "chai";

import * as ergo from "..";

let ergo_wasm;
beforeEach(async () => {
  ergo_wasm = await ergo;
});

describe("cost_oracle WASM binding", () => {
  it("trivial_true charges 50 raw JitCost", async () => {
    const tree = ergo_wasm._test_only_trivial_sigma_prop_tree(true);
    const input_box = ergo_wasm._test_only_ergo_box_with_tree(tree, BigInt(1_000_000));
    const tx = ergo_wasm._test_only_one_input_one_output_tx(input_box);
    const spent_boxes = new ergo_wasm.ErgoBoxes(input_box);
    const data_boxes = ergo_wasm.ErgoBoxes.empty();
    const params = ergo_wasm.Parameters.default_parameters();
    const state_ctx = ergo_wasm._test_only_synthetic_state_context(params);

    const results = ergo_wasm.compute_tx_oracle_costs(
      tx,
      spent_boxes,
      data_boxes,
      state_ctx
    );

    assert.strictEqual(
      results.length,
      1,
      "expected exactly one per-input result"
    );
    const r = results[0];
    assert.strictEqual(
      r.is_ok(),
      true,
      `trivial-true should succeed: error_msg=${r.error_msg()}`
    );
    assert.strictEqual(
      r.cost(),
      BigInt(50),
      "EVAL_SIGMA_PROP_CONSTANT short-circuit charges exactly 50 raw JitCost"
    );
    assert.strictEqual(r.error_msg(), undefined);
  });

  it("trivial_false charges 50 raw JitCost and succeeds (no verifier-side short-circuit)", async () => {
    // Per shim cost_oracle.rs:313-339: sigma-rust returns
    // Ok(ReductionResult { sigma_prop: TrivialProp(false), cost: 50 })
    // for a trivial-false tree; verifier-side short-circuiting on false
    // sigma_prop is the verifier's job, not the oracle's.
    const tree = ergo_wasm._test_only_trivial_sigma_prop_tree(false);
    const input_box = ergo_wasm._test_only_ergo_box_with_tree(tree, BigInt(1_000_000));
    const tx = ergo_wasm._test_only_one_input_one_output_tx(input_box);
    const spent_boxes = new ergo_wasm.ErgoBoxes(input_box);
    const data_boxes = ergo_wasm.ErgoBoxes.empty();
    const params = ergo_wasm.Parameters.default_parameters();
    const state_ctx = ergo_wasm._test_only_synthetic_state_context(params);

    const results = ergo_wasm.compute_tx_oracle_costs(
      tx,
      spent_boxes,
      data_boxes,
      state_ctx
    );

    assert.strictEqual(results.length, 1);
    const r = results[0];
    assert.strictEqual(
      r.is_ok(),
      true,
      `trivial-false reduces cleanly: error_msg=${r.error_msg()}`
    );
    assert.strictEqual(r.cost(), BigInt(50));
  });

  it("cost-limit exceeded returns is_ok=false with partial cost", async () => {
    // max_block_cost = 1 → jit_cost_limit = 10 (raw JitCost units).
    // trivial-true wants to charge 50, which trips the limit. Per shim
    // cost_oracle.rs:342-368: cost may be 0 (rejected upfront) or > 0
    // (partial accumulation); we only assert structural failure.
    const tree = ergo_wasm._test_only_trivial_sigma_prop_tree(true);
    const input_box = ergo_wasm._test_only_ergo_box_with_tree(tree, BigInt(1_000_000));
    const tx = ergo_wasm._test_only_one_input_one_output_tx(input_box);
    const spent_boxes = new ergo_wasm.ErgoBoxes(input_box);
    const data_boxes = ergo_wasm.ErgoBoxes.empty();
    // Field order matches ergo-lib parameters::Parameters::new:
    // (block_version, storage_fee_factor, min_value_per_byte,
    //  max_block_size, max_block_cost, token_access_cost,
    //  input_cost, data_input_cost, output_cost).
    // Same as shim cost_oracle.rs:354-361.
    const tight_params = ergo_wasm._test_only_parameters_new(
      1,            // block_version
      1,            // storage_fee_factor
      360,          // min_value_per_byte
      512 * 1024,   // max_block_size
      1,            // max_block_cost = 1 → jit_cost_limit = 10
      0,            // token_access_cost
      0,            // input_cost
      0,            // data_input_cost
      0             // output_cost
    );
    const state_ctx = ergo_wasm._test_only_synthetic_state_context(tight_params);

    const results = ergo_wasm.compute_tx_oracle_costs(
      tx,
      spent_boxes,
      data_boxes,
      state_ctx
    );

    assert.strictEqual(results.length, 1);
    const r = results[0];
    assert.strictEqual(
      r.is_ok(),
      false,
      `cost-limit-exceeded should fail: cost=${r.cost()}, error_msg=${r.error_msg()}`
    );
    assert.notStrictEqual(
      r.error_msg(),
      undefined,
      "structural failure must include an error message"
    );
  });

  it("tree_version derived from spent box (V0 — mirrors shim cost_oracle.rs:370-396)", async () => {
    // V0 trivial-sigma-prop tree. Asserts is_ok==true; the production
    // code path at src/cost_oracle.rs:160-171 reads input_box.ergo_tree
    // .header().version() and overrides ctx.tree_version, replacing the
    // default V0 make_context produces. For a V0 tree this is a tautology
    // (the override sets V0 = V0), but it exercises the code path. A V1
    // case would strengthen this test; add when a V1 tree constructor lands.
    const tree = ergo_wasm._test_only_trivial_sigma_prop_tree(true);
    const input_box = ergo_wasm._test_only_ergo_box_with_tree(tree, BigInt(1_000_000));
    const tx = ergo_wasm._test_only_one_input_one_output_tx(input_box);
    const spent_boxes = new ergo_wasm.ErgoBoxes(input_box);
    const data_boxes = ergo_wasm.ErgoBoxes.empty();
    const params = ergo_wasm.Parameters.default_parameters();
    const state_ctx = ergo_wasm._test_only_synthetic_state_context(params);

    const results = ergo_wasm.compute_tx_oracle_costs(
      tx,
      spent_boxes,
      data_boxes,
      state_ctx
    );

    assert.strictEqual(results.length, 1);
    const r = results[0];
    assert.strictEqual(
      r.is_ok(),
      true,
      `tree_version override path should not throw on V0 tree: error_msg=${r.error_msg()}`
    );
    // (The tree's V0 header guarantees this passes; main signal is no throw.)
  });
});
