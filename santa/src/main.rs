//! Blitzen: sigma-rust's runner for SANTA, the Ergo consensus conformance suite
//! (<https://github.com/mwaddip/santa>). It runs SANTA's test vectors through sigma-rust and
//! writes down what sigma-rust answers; SANTA compares that with the reference implementation's
//! answers. The vector and actuals formats are SANTA's runner contract
//! (`docs/contract/runner-contract.md` for the eval tier, and its `-wire`, `-transaction` and
//! `-nipopow` companions); "runner contract §N" below refers to it. An entry's `expected` is the
//! reference's answer, recorded ("blessed") when the vector was made.
//!
//! Two modes, both producing the actuals blind (never reading `expected`):
//!
//! ```text
//! blitzen <vector.json | dir>           self-compare: run each entry, compare with its
//!                                        `expected`, print nice/coal.
//! blitzen emit <vectors-dir> <out-dir>  write <out-dir>/<same-filename>: an object mapping
//!                                        each entry's name to its actual. No comparison:
//!                                        SANTA's orchestrator compares every runner alike.
//! ```
//!
//! Self-compare uses serde_json's structural equality, which is the contract's match (§5):
//! objects key-order-insensitive, arrays order-sensitive, numbers numeric, strings exact.

mod binding;
mod eval;
mod nipopow;
mod preheader;
mod sval;
mod transaction;
mod wire;

use serde_json::Value as J;
use std::path::{Path, PathBuf};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("emit") => {
            if args.len() < 4 {
                eprintln!("usage: blitzen emit <vectors-dir> <out-dir>");
                std::process::exit(2);
            }
            emit(Path::new(&args[2]), Path::new(&args[3]));
        }
        Some(target) => self_compare(Path::new(target)),
        None => {
            eprintln!(
                "usage:\n  \
                 blitzen <vector.json | dir>           self-compare vs blessed expected\n  \
                 blitzen emit <vectors-dir> <out-dir>  emit actuals for the SANTA orchestrator"
            );
            std::process::exit(2);
        }
    }
}

/// Extract a printable message from a caught panic payload.
fn panic_note(p: Box<dyn std::any::Any + Send>) -> String {
    p.downcast_ref::<&str>()
        .map(|s| s.to_string())
        .or_else(|| p.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "non-string panic payload".to_string())
}

/// Run one entry under a panic net (runner contract §3: never panic): a panic becomes the tier's
/// `panicked` actual, built by `panicked` from the panic's message, and the run goes on. The
/// closures only read the vector, so asserting unwind-safety is sound.
fn caught(f: impl FnOnce() -> J, panicked: fn(String) -> J) -> J {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(j) => j,
        Err(p) => panicked(format!("panic: {}", panic_note(p))),
    }
}

/// The eval-shaped `panicked` actual (also the wire and NiPoPoW tiers').
fn eval_panicked(note: String) -> J {
    eval::Outcome::Panicked { note }.to_json()
}

/// The transaction-shaped `panicked` actual: it carries `valid` where eval's carries `value`.
fn tx_panicked(note: String) -> J {
    transaction::TxOutcome::Panicked { note }.to_json()
}

/// The tier a vector file's `schema` names: only the schemas this runner was written for, by exact name and
/// version.
enum Tier<'v> {
    /// `santa-eval/v6-fullctx`: each entry carries the real context of a captured spend.
    FullContextEval,
    /// `santa-nipopow/*`: the file's chain, which each entry asks interlinks or a proof over.
    NiPoPoW(&'v [J]),
    Wire,
    Transaction,
    Eval,
    /// Any other schema (a newer version of one above, or a tier this runner does not mount): every entry is
    /// not-implemented, never read as a known schema (runner-integration §3).
    Unknown,
}

fn tier_of(vector: &J) -> Tier<'_> {
    match vector["schema"].as_str().unwrap_or("") {
        "santa-eval/v1" | "santa-eval/v2" | "santa-eval/v3" | "santa-eval/v4" | "santa-eval/v5" => {
            Tier::Eval
        }
        "santa-eval/v6-fullctx" => Tier::FullContextEval,
        "santa-nipopow/v1" => Tier::NiPoPoW(
            vector["chain"]
                .as_array()
                .expect("nipopow vector missing chain"),
        ),
        "santa-wire/v1" => Tier::Wire,
        "santa-transaction/v1" => Tier::Transaction,
        _ => Tier::Unknown,
    }
}

/// Evaluate every entry of one vector file (blind), pairing each entry's `name`
/// with its actual JSON and the vector's blessed `expected` JSON. Returns empty for
/// a file carrying no `entries`. A malformed vector aborts the run via `expect`: a loud
/// error, never a silent skip (runner contract §3: every entry gets an outcome).
fn run_vector_file(path: &Path) -> Vec<(String, J, J)> {
    let text = std::fs::read_to_string(path).expect("read vector file");
    let vector: J = serde_json::from_str(&text).expect("parse vector JSON");
    let entries = match vector["entries"].as_array() {
        Some(e) => e,
        None => return Vec::new(),
    };
    let tier = tier_of(&vector);
    entries
        .iter()
        .map(|entry| {
            let name = entry["name"].as_str().unwrap_or("<unnamed>").to_string();
            let (actual, expected) = match tier {
                // the tier-neutral form: the runner cannot know the actual shape of a schema it was not written for
                Tier::Unknown => (
                    serde_json::json!({"error": "not-implemented"}),
                    entry["expected"].clone(),
                ),
                Tier::FullContextEval => (fullctx_entry(entry), entry["expected"].clone()),
                Tier::NiPoPoW(chain) => (
                    caught(|| nipopow_entry(entry, chain), eval_panicked),
                    entry["expected"].clone(),
                ),
                Tier::Wire => wire_entry(entry),
                Tier::Transaction => transaction_entry(entry),
                Tier::Eval => (eval_entry(entry), entry["expected"].clone()),
            };
            (name, actual, expected)
        })
        .collect()
}

fn nipopow_entry(entry: &J, chain: &[J]) -> J {
    match entry["kind"].as_str().unwrap_or("") {
        "nipopow_interlinks" => nipopow::run_interlinks(chain),
        "nipopow_prove" => {
            let m = entry["payload"]["m"].as_u64().expect("missing m") as u32;
            let k = entry["payload"]["k"].as_u64().expect("missing k") as u32;
            nipopow::run_prove(chain, m, k, entry["payload"]["headerId"].as_str())
        }
        _ => serde_json::json!({"error": "not-implemented"}),
    }
}

/// A wire entry's actual, and its expected: its own bytes back (a round-trip to itself).
fn wire_entry(entry: &J) -> (J, J) {
    let kind = entry["kind"].as_str().expect("wire entry missing kind");
    let bytes_hex = entry["bytes_hex"]
        .as_str()
        .expect("wire entry missing bytes_hex");
    let tree_version = entry["version"]["ergoTree"].as_u64().unwrap_or(3) as u8;
    let activated_version = entry["version"]["activated"].as_u64().map(|v| v as u8);
    let actual = caught(
        || wire::run_entry(kind, bytes_hex, tree_version, activated_version).to_json(),
        eval_panicked,
    );
    (
        actual,
        serde_json::json!({"bytes_hex": bytes_hex, "error": J::Null}),
    )
}

/// A transaction entry's actual, and its expected in the actuals vocabulary. `reason` is
/// diagnostic only, never graded, so it is left out of the expected.
fn transaction_entry(entry: &J) -> (J, J) {
    let actual = caught(|| transaction::run_entry(entry).to_json(), tx_panicked);
    let expected = serde_json::json!({
        "valid": entry["expected"]["valid"],
        "cost": entry["expected"]["cost"],
        "error": J::Null,
    });
    (actual, expected)
}

fn fullctx_entry(entry: &J) -> J {
    let tree_hex = entry["tree_bytes_hex"]
        .as_str()
        .expect("v6-fullctx entry missing tree_bytes_hex");
    let tree_bytes = hex_to_bytes(tree_hex).expect("bad tree_bytes_hex");
    let tree_v = entry["version"]["ergoTree"].as_u64().unwrap_or(0) as u8;
    let act_v = entry["version"]["activated"].as_u64().map(|v| v as u8);
    caught(
        || eval::run_entry_fullctx(&tree_bytes, &entry["context"], tree_v, act_v).to_json(),
        eval_panicked,
    )
}

fn eval_entry(entry: &J) -> J {
    let tree_hex = entry["tree_bytes_hex"]
        .as_str()
        .expect("entry missing tree_bytes_hex");
    let tree_bytes = hex_to_bytes(tree_hex).expect("bad tree_bytes_hex");
    let input = entry.get("input").filter(|v| !v.is_null());
    let inputs = entry.get("inputs").and_then(|v| v.as_array());
    let self_registers = entry.get("selfRegisters").and_then(|v| v.as_object()); // v4
    let extension = entry.get("extension").and_then(|v| v.as_object()); // v5
    let tree_v = entry["version"]["ergoTree"].as_u64().unwrap_or(0) as u8;
    let act_v = entry["version"]["activated"].as_u64().unwrap_or(0) as u8;
    caught(
        || {
            eval::run_entry(
                &tree_bytes,
                input,
                inputs,
                self_registers,
                extension,
                tree_v,
                act_v,
            )
            .to_json()
        },
        eval_panicked,
    )
}

/// Self-compare mode: evaluate the corpus and tally nice/coal vs the blessed `expected`.
fn self_compare(path: &Path) {
    let files = collect_vector_files(path);
    if files.is_empty() {
        eprintln!("no .json vector files found at {}", path.display());
        std::process::exit(2);
    }

    let (mut nice, mut coal, mut total) = (0u64, 0u64, 0u64);
    let mut coal_list: Vec<(String, String, J, J)> = Vec::new();

    for f in &files {
        let short = short_name(f);
        for (name, actual, expected) in run_vector_file(f) {
            total += 1;
            if actual == expected {
                nice += 1;
            } else {
                coal += 1;
                coal_list.push((short.clone(), name, actual, expected));
            }
        }
    }

    println!(
        "blitzen: {} nice / {} coal / {} entries across {} files",
        nice,
        coal,
        total,
        files.len()
    );
    for (file, name, actual, expected) in coal_list.iter().take(60) {
        println!("\nCOAL  {}  ::  {}", file, name);
        println!("  actual:   {}", actual);
        println!("  expected: {}", expected);
    }
    if coal_list.len() > 60 {
        println!("\n... and {} more coal entries", coal_list.len() - 60);
    }
}

/// Emit mode: write one actuals file per vector (`<out-dir>/<same-filename>`), each an
/// object mapping entry `name` → its `{ value, cost, error }` (santa-eval.actuals schema).
/// Blind — never reads `expected`; SANTA's orchestrator compares. Exit 0 once
/// every actuals file is written; a malformed vector aborts loudly (non-zero) via `expect`.
fn emit(vectors_dir: &Path, out_dir: &Path) {
    let files = collect_vector_files(vectors_dir);
    if files.is_empty() {
        eprintln!("no .json vector files found at {}", vectors_dir.display());
        std::process::exit(2);
    }
    std::fs::create_dir_all(out_dir).expect("create out dir");

    let mut written = 0u64;
    for f in &files {
        let entries = run_vector_file(f);
        if entries.is_empty() {
            continue; // not an eval vector → no actuals file
        }
        let mut actuals = serde_json::Map::new();
        for (name, actual, _expected) in entries {
            actuals.insert(name, actual);
        }
        let filename = f.file_name().expect("vector filename");
        let out_path = out_dir.join(filename);
        let text = serde_json::to_string_pretty(&J::Object(actuals)).expect("serialize actuals");
        std::fs::write(&out_path, text).expect("write actuals file");
        written += 1;
    }
    eprintln!(
        "blitzen emit: wrote {} actuals files to {}",
        written,
        out_dir.display()
    );
}

fn short_name(path: &Path) -> String {
    path.file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("?")
        .to_string()
}

fn collect_vector_files(path: &Path) -> Vec<PathBuf> {
    if path.is_dir() {
        let mut out: Vec<PathBuf> = std::fs::read_dir(path)
            .expect("read dir")
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("json"))
            .collect();
        out.sort();
        out
    } else {
        vec![path.to_path_buf()]
    }
}

pub(crate) fn hex_to_bytes(s: &str) -> Result<Vec<u8>, String> {
    if !s.len().is_multiple_of(2) {
        return Err(format!("odd-length hex: {}", s));
    }
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).map_err(|e| e.to_string()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{caught, eval_panicked, hex_to_bytes, tx_panicked};
    use crate::eval::Outcome;
    use serde_json::{json, Value as J};

    #[test]
    fn caught_turns_a_panic_into_panicked() {
        let j = caught(|| -> J { panic!("kaboom") }, eval_panicked);
        assert_eq!(j["error"], "panicked");
        assert_eq!(j["value"], J::Null);
        assert_eq!(j["cost"], J::Null);
        assert!(j["note"].as_str().unwrap().contains("kaboom"));
    }

    #[test]
    fn caught_with_the_tx_shape_carries_valid() {
        // The tx panic shape carries `valid` (santa-transaction.actuals), not eval's
        // `value`: a value-shaped panic would fail the tx actuals schema.
        let j = caught(|| -> J { panic!("tx kaboom") }, tx_panicked);
        assert_eq!(j["error"], "panicked");
        assert_eq!(j["valid"], J::Null);
        assert_eq!(j["cost"], J::Null);
        assert!(j["note"].as_str().unwrap().contains("tx kaboom"));
    }

    #[test]
    fn caught_passes_through_a_normal_outcome() {
        let j = caught(|| Outcome::Errored.to_json(), eval_panicked);
        assert_eq!(j["error"], "errored");
    }

    /// santa-eval/v4 keystone: `accept-r4-long#0` from Box.getReg_dynamic_index.json.
    /// Script: `{ SELF.getReg[Long](getVar[Int](1).get) }`
    /// SELF R4 = Long 7, var 1 (input) = Int 4.
    ///
    /// The v4 arm is exercised (selfRegisters + var 1 decode, SELF box replacement).
    /// sigma-rust currently cannot parse the dynamic getReg MethodCall tree (MethodId
    /// 19 on SBox / type id 99) — the actual outcome is `errored` today — but that is
    /// a conformance finding, not a harness property. When sigma-rust implements 99:19
    /// the actual will flip to Some(Long 7) (`error: null`) — still a valid verdict.
    ///
    /// This test pins the ENVELOPE only:
    ///   - outcome is well-formed: `error` is null or "errored" (a real verdict, never "panicked")
    ///   - if errored: no `note` field (panicked carries a note; errored must not)
    // do not pin the impl's verdict here — vectors do that; this test pins the envelope.
    #[test]
    fn v4_accept_r4_long_keystone() {
        // tree_bytes_hex from the committed vector entry.
        let tree_bytes = hex_to_bytes("1b0b00dc6313a701e4e3010405").expect("bad hex");
        // selfRegisters: R4 = Long(7)
        let self_registers_json = json!({"4": {"kind": "Long", "value": "7"}});
        let self_registers = self_registers_json.as_object().unwrap();
        // input (var 1): Int(4) — the dynamic register index
        let input_json = json!({"kind": "Int", "value": 4});
        // ergoTree v3, activated v3
        let actual = crate::eval::run_entry(
            &tree_bytes,
            Some(&input_json),
            None,
            Some(self_registers),
            None,
            3,
            3,
        )
        .to_json();
        // Harness-shape: a real verdict — either a value (error null) or errored — never panicked.
        let error = &actual["error"];
        assert!(
            error.is_null() || error == "errored",
            "v4 envelope must yield a verdict (null or errored), got: {}",
            actual
        );
        // If errored, no `note` (panicked carries a note; errored must not).
        if error == "errored" {
            assert!(
                actual.get("note").is_none(),
                "errored must not carry a note; got: {}",
                actual
            );
        }
    }

    /// A schema the runner was not written for (a newer version of one it knows, or a tier it does not mount) is
    /// answered `not-implemented` for every entry: read as a known schema, a new envelope would show as false
    /// divergences (runner-integration §3).
    #[test]
    fn an_unknown_schema_is_not_implemented() {
        use crate::run_vector_file;
        // a tree_bytes_hex and kind/bytes_hex a known arm would happily read
        for schema in ["santa-eval/v99", "santa-wire/v2", "santa-block/v1"] {
            let vector = json!({"schema": schema, "entries": [{
                "name": "e#0", "tree_bytes_hex": "000402", "kind": "Constant", "bytes_hex": "0101",
                "version": {"ergoTree": 0, "activated": 2}, "expected": {"error": null}
            }]});
            let path = std::env::temp_dir().join(format!(
                "blitzen-unknown-schema-{}.json",
                std::process::id()
            ));
            std::fs::write(&path, vector.to_string()).expect("write test vector");
            let results = run_vector_file(&path);
            assert_eq!(results.len(), 1, "{schema}");
            // the tier-neutral form, admitted by every tier's actuals schema: the runner cannot know the actual
            // shape of a schema it was not written for
            assert_eq!(
                results[0].1,
                json!({"error": "not-implemented"}),
                "{schema}"
            );
        }
    }

    /// santa-nipopow/v1 (`tests/fixtures/nipopow-chain-32.json`: SANTA's 32-header chain (from the reference node), its interlinks entry and
    /// one prove entry). This pins the arm's ENVELOPE, not develop's verdicts (the vectors do that): the interlinks
    /// entry answers one interlink vector per chain header, the prove entry a hex `proofHex` (or an honest
    /// errored / panicked), and neither is not-implemented.
    #[test]
    fn nipopow_entries_answer_interlinks_and_a_proof() {
        use crate::run_vector_file;
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/nipopow-chain-32.json");
        let results = run_vector_file(&path);
        assert_eq!(results.len(), 2);
        let (_, interlinks, _) = &results[0];
        assert_eq!(interlinks["error"], J::Null, "{interlinks}");
        assert_eq!(
            interlinks["interlinks"].as_array().map(Vec::len),
            Some(32),
            "{interlinks}"
        );
        let (_, prove, _) = &results[1];
        assert_ne!(prove["error"], "not-implemented", "{prove}");
        if prove["error"].is_null() {
            let hex = prove["proofHex"].as_str().expect("proofHex");
            assert!(
                !hex.is_empty() && hex.chars().all(|c| c.is_ascii_hexdigit()),
                "{prove}"
            );
        }
    }

    /// A wire entry without a `version` is still read (its defaults are pinned by the two tests
    /// below, with entries whose answer depends on them).
    #[test]
    fn a_wire_entry_without_versions_round_trips() {
        use crate::run_vector_file;
        // sbox_minimal of SANTA's `vectors/wire/v5/authored/Box.json`
        let hex = "c0843d09020101000000000000000000000000000000000000000000000000000000000000000000000000";
        let vector = json!({"schema": "santa-wire/v1", "entries": [
            {"name": "e#0", "kind": "Box", "bytes_hex": hex}
        ]});
        let path = std::env::temp_dir().join(format!(
            "blitzen-wire-no-version-{}.json",
            std::process::id()
        ));
        std::fs::write(&path, vector.to_string()).expect("write test vector");
        let results = run_vector_file(&path);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].1["error"], J::Null, "{}", results[0].1);
        assert_eq!(results[0].1["bytes_hex"], hex);
    }

    /// The `ergo_avltree_rust` this runner builds with is the one the repository's own manifest
    /// patches in. A patch applies only in the workspace being built and the runner is a workspace
    /// of its own, so the pin is repeated in its manifest: in both, the `[patch.crates-io]` table
    /// must name the same git source at the same rev.
    #[test]
    fn the_avltree_patch_is_the_repositorys() {
        /// The `git` and `rev` of `ergo_avltree_rust` in the manifest's `[patch.crates-io]` table.
        fn patch_pin(manifest: &std::path::Path) -> (String, String) {
            let at = manifest.display();
            let text = std::fs::read_to_string(manifest).unwrap_or_else(|e| panic!("{at}: {e}"));
            let table: Vec<&str> = text
                .lines()
                .skip_while(|l| l.trim() != "[patch.crates-io]")
                .skip(1)
                .take_while(|l| !l.trim_start().starts_with('['))
                .collect();
            let line = table
                .iter()
                .find(|l| l.trim_start().starts_with("ergo_avltree_rust"))
                .unwrap_or_else(|| panic!("{at}: no ergo_avltree_rust in [patch.crates-io]"));
            // `key = "value"`: the key's text can also occur inside a value (`git` in the URL)
            let value = |key: &str| {
                line.match_indices(key)
                    .find_map(|(i, _)| {
                        let rest = line[i + key.len()..].trim_start().strip_prefix('=')?;
                        let rest = rest.trim_start().strip_prefix('"')?;
                        rest.split('"').next().map(str::to_string)
                    })
                    .unwrap_or_else(|| panic!("{at}: the ergo_avltree_rust patch has no {key}"))
            };
            (value("git"), value("rev"))
        }
        let here = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        assert_eq!(
            patch_pin(&here.join("Cargo.toml")),
            patch_pin(&here.join("../Cargo.toml")),
            "santa/Cargo.toml and the repository's Cargo.toml patch ergo_avltree_rust differently"
        );
    }

    /// santa-eval/v6-fullctx end to end (`tests/fixtures/v6-fullctx-h111927.json`, a captured
    /// testnet spend: SigmaProp `d3` at cost 1139 on the reference): the context is rebuilt from
    /// the envelope (boxes, headers, the pre-header, the state root, the per-input extensions) and
    /// the tree evaluated in it, to the reference's value and cost.
    #[test]
    fn v6_fullctx_h111927_evaluates() {
        use crate::run_vector_file;
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/v6-fullctx-h111927.json");
        let results = run_vector_file(&path);
        assert_eq!(results.len(), 1, "expected one result");
        let (_name, actual, expected) = &results[0];
        assert_eq!(
            actual, expected,
            "h111927 actual must match blessed expected"
        );
    }

    /// Write `vector` to a session-unique temp file, then read it through `run_vector_file`.
    fn run_adhoc_vector(stem: &str, vector: J) -> Vec<(String, J, J)> {
        use crate::run_vector_file;
        let path =
            std::env::temp_dir().join(format!("blitzen-{}-{}.json", stem, std::process::id()));
        std::fs::write(&path, vector.to_string()).expect("write test vector");
        run_vector_file(&path)
    }

    /// A Transaction wire entry whose context-extension holds `Coll[SFunc]` (the v6-only
    /// function-type gate): round-trips at the runner's unversioned wire DEFAULT of
    /// `ergoTree = 3` (reader parses SFunc). A mutant that defaulted to 0 would read at
    /// tree_version 0 and refuse the SFunc type code.
    #[test]
    fn a_wire_transaction_without_versions_defaults_to_tree_version_three() {
        // `ext-c2-coll-of-func-v3-accept#22` of
        // vectors/wire/v6/authored/Transaction.extension_evaluated_values.json: a tx whose
        // input 0 carries extension `{0: Coll[SFunc((Int, Int) => Int)]}` (empty).
        let hex = "0100a19de1b5fa998df5a48630a611180690abad5270c33f23a79baba2f8840d710001008300700204040400000001c0843d0008d3010000";
        let vector = json!({"schema": "santa-wire/v1", "entries": [
            {"name": "e#0", "kind": "Transaction", "bytes_hex": hex}
        ]});
        let results = run_adhoc_vector("wire-sfunc-no-version", vector);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].1["error"], J::Null, "{}", results[0].1);
        assert_eq!(results[0].1["bytes_hex"], hex);
    }

    /// A wire Box entry whose tree is above the activated script version. The wire DEFAULT
    /// treats the entry's `version.activated` as absent — no activation context, so the box
    /// parses (no check fires). A mutant that defaulted to `Some(3)` when the field is missing
    /// would refuse a v4 tree against activated 3.
    #[test]
    fn a_wire_box_without_versions_has_no_activation_context() {
        // `box-tree-v4-reject#0` of vectors/wire/v6/authored/Box.tree_version_above_activated.json:
        // a bare box whose size-flagged tree has version 4. Without an activated version the box
        // round-trips; at activated=3 the tree is refused.
        let hex = "c0843d0c0208d30100001d823ee9ea823cc80232a19181efad41d66849c33ed5d0d6c5750b8d60f1d66400";
        let vector = json!({"schema": "santa-wire/v1", "entries": [
            {"name": "e#0", "kind": "Box", "bytes_hex": hex}
        ]});
        let results = run_adhoc_vector("wire-box-v4-no-version", vector);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].1["error"], J::Null, "{}", results[0].1);
        assert_eq!(results[0].1["bytes_hex"], hex);
    }

    /// A v6-fullctx entry whose `version.activated` is present carries that activated version
    /// to the fullctx runner. With `activated=1` and `phf.version=4`, extensions read at
    /// `read_activated=1` — an SFunc value in `input_extensions[0]` refuses (SFunc parses only
    /// at reader tree_version >= V3). A mutant that always passes `activated_version=None`
    /// would fall back to `phf.version - 1 = 3`, accept the SFunc, build parts, and the
    /// simple `SigmaProp(true)` tree would evaluate to a value.
    #[test]
    fn a_fullctx_entry_carries_activated_to_the_reader() {
        let envelope: J = serde_json::from_str(include_str!("testdata/fullctx-h210129.json"))
            .expect("fixture parses");
        let mut entry = envelope["entries"][0].clone();
        entry["version"] = json!({"activated": 1});
        // Replace the tree with `0b 02 08 d3`: SigmaProp(TrivialProp(true)) — evaluates at ANY
        // tree_version, so the only distinguisher is the extension read at `read_activated`.
        entry["tree_bytes_hex"] = json!("0b0208d3");
        entry["expected"] = json!({});
        let mut ctx = entry["context"].clone();
        // Coll[SFunc(SInt => SInt)] empty: one-arg SFunc so convert_self_extension (which calls
        // `to_script_value` on each extension value in eval) accepts the runtime form.
        ctx["input_extensions"] = json!([{
            "0": {"kind": "Value", "bytes_hex": "83007001040400"}
        }]);
        entry["context"] = ctx;
        let vector = json!({"schema": "santa-eval/v6-fullctx", "entries": [entry]});
        let results = run_adhoc_vector("fullctx-sfunc-activated-1", vector);
        assert_eq!(results.len(), 1);
        assert_eq!(
            results[0].1["error"], "errored",
            "the SFunc extension value must refuse at reader tree_version 1; got {}",
            results[0].1
        );
    }

    /// An eval entry whose `version.activated` is present carries that activated version to the
    /// eval runner. With `activated=1`, the binding reads at `act_v=1` — an SFunc value in a
    /// v5 extension refuses. A mutant that defaulted to 3 (overriding the explicit `1`) would
    /// accept the SFunc and evaluate.
    #[test]
    fn an_eval_entry_carries_activated_to_the_binding() {
        // SigmaProp(true) tree (parses at any tree_version); SFunc extension value; activated=1
        // on the entry. The binding is read at activated=1 → refused → Errored. The mutant
        // M06 would use the mapped-or-default `activated=3`, accept, and the tree evaluate.
        let hex_tree = "0b0208d3";
        let vector = json!({"schema": "santa-eval/v5", "entries": [{
            "name": "e#0",
            "tree_bytes_hex": hex_tree,
            "extension": {"0": {"kind": "Value", "bytes_hex": "83007001040400"}},
            "version": {"activated": 1, "ergoTree": 3},
            "expected": {}
        }]});
        let results = run_adhoc_vector("eval-sfunc-activated-1", vector);
        assert_eq!(results.len(), 1);
        assert_eq!(
            results[0].1["error"], "errored",
            "SFunc extension must refuse at activated=1; got {}",
            results[0].1
        );
    }

    /// An eval entry whose ONLY `version` field is `ergoTree` carries the DEFAULT activated
    /// of 0 to the eval runner. The binding reads at activated 0, so a v5 extension holding
    /// an SFunc Value refuses (SFunc type code gated to tree_version >= V3). A mutant that
    /// defaulted to 3 would accept the SFunc and evaluate the tree.
    #[test]
    fn an_eval_entry_without_activated_defaults_to_zero() {
        // Tree `0b 02 08 d3`: header v3+size, size=2, body = SigmaProp(TrivialProp(true)). Parses
        // at reader tree_version = MAX_SCRIPT_VERSION (V3) — always. The ERRORED on unmutated
        // comes from the extension decode (binding read at activated 0 refuses SFunc), not the
        // tree parse.
        let hex_tree = "0b0208d3";
        let vector = json!({"schema": "santa-eval/v5", "entries": [{
            "name": "e#0",
            "tree_bytes_hex": hex_tree,
            "extension": {"0": {"kind": "Value", "bytes_hex": "83007001040400"}},
            "version": {"ergoTree": 3},
            "expected": {}
        }]});
        let results = run_adhoc_vector("eval-sfunc-no-activated", vector);
        assert_eq!(results.len(), 1);
        assert_eq!(
            results[0].1["error"], "errored",
            "the SFunc extension must refuse at the default activated=0; got {}",
            results[0].1
        );
    }

    /// An eval entry whose ONLY `version` field is `activated` carries the DEFAULT ergoTree
    /// of 0 to the eval runner. The context's `tree_version` cell is set to V0, and a tree
    /// whose evaluation gates on `ctx.tree_version() >= V3` (a `Downcast(BigInt, Int)`) errors.
    /// A mutant that defaulted to 3 would set the cell to V3 and the Downcast would succeed.
    #[test]
    fn an_eval_entry_without_ergo_tree_defaults_to_zero() {
        // Tree bytes = `0b 05 7d 06 01 01 04`: header v3+size, size=5, body =
        // Downcast(BigInt(1), SInt). Evaluates to `Int 1` at ctx.tree_version >= V3; errors at
        // ctx.tree_version < V3.
        let hex_tree = "0b057d06010104";
        let vector = json!({"schema": "santa-eval/v2", "entries": [{
            "name": "e#0",
            "tree_bytes_hex": hex_tree,
            "version": {"activated": 3},
            "expected": {}
        }]});
        let results = run_adhoc_vector("eval-downcast-no-ergotree", vector);
        assert_eq!(results.len(), 1);
        assert_eq!(
            results[0].1["error"], "errored",
            "Downcast(BigInt, Int) at ctx.tree_version=V0 must error; got {}",
            results[0].1
        );
    }

    /// A v6-fullctx entry whose ONLY `version` field is `activated` carries the DEFAULT
    /// ergoTree of 0 to the fullctx runner. The context's `tree_version` cell is V0, and a
    /// Downcast(BigInt, Int) in the tree errors. A mutant that defaulted to 3 would succeed.
    #[test]
    fn a_fullctx_entry_without_ergo_tree_defaults_to_zero() {
        // Replace the h111927 tree with the `0b 05 7d 06 01 01 04` Downcast tree, keeping its
        // context (which has activated=3). Then REMOVE version.ergoTree.
        let envelope: J =
            serde_json::from_str(include_str!("../tests/fixtures/v6-fullctx-h111927.json"))
                .expect("fixture parses");
        let mut entry = envelope["entries"][0].clone();
        entry["tree_bytes_hex"] = json!("0b057d06010104");
        entry["version"] = json!({"activated": 3});
        // Drop the blessed expected — we are testing the runner's unmutated behaviour.
        entry["expected"] = json!({});
        let vector = json!({"schema": "santa-eval/v6-fullctx", "entries": [entry]});
        let results = run_adhoc_vector("fullctx-downcast-no-ergotree", vector);
        assert_eq!(results.len(), 1);
        assert_eq!(
            results[0].1["error"], "errored",
            "Downcast(BigInt, Int) at ctx.tree_version=V0 must error; got {}",
            results[0].1
        );
    }
}
