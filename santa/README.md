# Blitzen

sigma-rust's runner for [SANTA](https://github.com/mwaddip/santa), the Ergo consensus conformance
suite. SANTA holds test vectors whose expected answers were recorded ("blessed") from the
reference implementation (sigma-state and the JVM node); Blitzen runs them through sigma-rust and
writes down what sigma-rust answers, and SANTA compares the two. Where sigma-rust answers
differently, that is a divergence to look at, not something this runner smooths over.

It covers four tiers:

| tier | what an entry is | what Blitzen answers |
|---|---|---|
| eval | an ErgoTree, an input, the `(activated, ergoTree)` versions | `{value, cost, error}` |
| wire | bytes of a `Constant`, `Box`, `Transaction`, `SigmaBoolean`, `ErgoTree` or block-transactions section | the bytes sigma-rust writes back |
| transaction | a transaction, its input boxes, the headers, pre-header and parameters it is validated under | `{valid, cost, error}` |
| nipopow | a header chain, and an interlinks or proof request over it | the interlinks, or the proof's bytes |

The formats are SANTA's runner contract
([`docs/contract/runner-contract.md`](https://github.com/mwaddip/santa/blob/main/docs/contract/runner-contract.md)
and its `-wire`, `-transaction` and `-nipopow` companions). The runner declares `cost: true`:
SANTA grades the JIT cost of an evaluation and the block cost of an accepted transaction too.

## Run it

```bash
cargo test                                            # the runner's own tests
cargo run --release -- <vector.json | dir>            # run entries, compare with `expected`, print nice/coal
cargo run --release -- emit <vectors-dir> <out-dir>   # write one actuals file per vector (no comparison)
```

`santa-run <impl-path> <vectors-dir> <out-dir>` is the entry SANTA calls: it builds the runner and
runs `emit` (its last line through `$SANTA_BOUND`, SANTA's resource bound). `runner.json` declares
what this runner covers (`version`, `tiers`, `cost`), read by SANTA from this checkout.

## How it works

- **`eval.rs`**: evaluates an entry's tree in the canonical evaluation context the reference
  builds for a bare evaluation (SELF holding the entry's tree, `INPUTS = [SELF]`, no outputs, no
  headers, height 0, a fixed pre-header, the reference's cost limit), with the entry's input bound
  where its envelope puts it. Trees are read with `ErgoTree::sigma_parse_bytes_lenient`, since the
  vectors' roots are of any type, and evaluated with `eval::test_util::try_eval_with_deserialize`,
  the public entry (behind the `arbitrary` feature) that evaluates a root of any type. The cost
  is the context's JIT cost counter after the evaluation. A full-context entry
  (`santa-eval/v6-fullctx`) is evaluated in the real context its envelope carries: the boxes,
  headers and pre-header of a captured spend (`preheader.rs` decodes the envelope's pre-header).
  An entry whose SELF box cannot be built is not-implemented.
- **`wire.rs`**: parses a kind's bytes with sigma-rust's serializer and writes them back. The
  `ErgoTree` kind goes through the lenient parse, so the tree is re-encoded from its structure.
- **`transaction.rs`**: decodes the transaction and its boxes from their bytes, builds the
  `ErgoStateContext` from the entry's own headers, pre-header and parameters, and runs ergo-lib's
  stateful validation (`TransactionContext::validate`), whose result is an accepted
  transaction's cost. A validation error is a clean `valid: false` with sigma-rust's reason;
  bytes sigma-rust cannot read are `errored`.
- **`nipopow.rs`**: rebuilds the chain as `PoPowHeader`s and answers interlinks or
  `NipopowAlgos::prove`. The proof's bytes end with its `continuous` byte (0), which the node's
  wire encoding adds and sigma-rust's proof serializer leaves out.
- **`sval.rs`** and **`binding.rs`**: the bridge between SANTA's JSON values and sigma-rust's.
  `Long` and `BigInt` are decimal strings, bytes lower-case hex, a `SigmaProp` its bare
  `SigmaBoolean` bytes; an input given as `Value` bytes is read with sigma-rust's own reader for
  that position.
- **`main.rs`**: the CLI, and the dispatch by vector schema. Every entry gets exactly one outcome:
  a value or verdict, `errored` (sigma-rust's own refusal, a divergence where the reference
  accepted), `not-implemented`, or `panicked` (a panic, caught, or a failure of the runner itself,
  with a note).
