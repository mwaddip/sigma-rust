//! Transaction-tier validation: decode one captured entry's tx + referenced boxes from
//! canonical sigma BYTES (the consensus-unambiguous form — bytes preserve context-extension
//! wire ORDER and u64 precision that node-API JSON silently corrupts; runner-contract-transaction §2), build
//! the REAL block context the vector carries (headers + preHeader + parameters; §5 there),
//! and run ergo-lib's stateful validation — `TransactionContext::validate`, the sigma-rust
//! port of the node's `validateStateful` (per-input script verify + ERG/token preservation +
//! dust + height checks). The tx analog of [`crate::eval::run_entry`] / [`crate::wire::run_entry`].
//!
//! Outcome mapping (santa-transaction.actuals): a *verdict* is `valid: true` (with the
//! accumulated cost when the impl yields one) or `valid: false` + the impl's reason —
//! every `TxValidationError` is a clean reject, mirroring the reference node's
//! `validateStateful → Failure ⇒ invalid` mapping (TxValidate.scala). A decode/setup
//! failure (sigma-rust can't even read bytes the JVM accepted) is `errored` — no
//! verdict was reached; that too is a real divergence on an accept vector.
//!
//! An accepted transaction's cost is what `validate` returns: the accumulated block cost
//! (the transaction's initial cost, each input's reduction, the proofs' verification).

use ergo_chain_types::{BlockId, Digest32, EcPoint, Header, PreHeader, Votes};
use ergo_lib::chain::ergo_state_context::{ErgoStateContext, Headers};
use ergo_lib::chain::parameters::{Parameter, Parameters};
use ergo_lib::chain::transaction::Transaction;
use ergo_lib::wallet::tx_context::TransactionContext;
use ergotree_ir::chain::ergo_box::ErgoBox;
use ergotree_ir::ergo_tree::ErgoTreeVersion;
use ergotree_ir::serialization::sigma_byte_reader::{self, SigmaByteRead};
use ergotree_ir::serialization::SigmaSerializable; // Transaction/ErgoBox sigma_parse_bytes
use sigma_ser::ScorexSerializable; // Header/EcPoint scorex_parse_bytes

use crate::hex_to_bytes;

/// One transaction entry's outcome — the tx analog of [`crate::eval::Outcome`].
pub enum TxOutcome {
    /// Validation reached a verdict: `cost` is `Some` only on accept, `reason` only on reject (the
    /// impl's error string, diagnostic).
    Verdict {
        valid: bool,
        cost: Option<u64>,
        reason: Option<String>,
    },
    /// No verdict: the runner couldn't decode the entry or assemble the tx context.
    Errored { reason: String },
    /// A would-be panic, caught by main's never-panic net (runner contract §3).
    Panicked { note: String },
}

impl TxOutcome {
    pub fn to_json(&self) -> serde_json::Value {
        match self {
            TxOutcome::Verdict {
                valid,
                cost,
                reason,
            } => {
                let mut j = serde_json::json!({
                    "valid": valid,
                    "cost": cost,
                    "error": serde_json::Value::Null,
                });
                if let Some(r) = reason {
                    j["reason"] = serde_json::json!(r);
                }
                j
            }
            TxOutcome::Errored { reason } => serde_json::json!({
                "valid": null, "cost": null, "error": "errored", "reason": reason
            }),
            TxOutcome::Panicked { note } => serde_json::json!({
                "valid": null, "cost": null, "error": "panicked", "note": note
            }),
        }
    }
}

/// Decode a JSON array of hex-encoded `ErgoBox` bytes (`input_boxes_hex` /
/// `data_input_boxes_hex`), in tx-input order. A hex/parse refusal of bytes the reference
/// accepted is a no-verdict (`errored`) — sigma-rust couldn't read what the JVM accepted.
fn decode_boxes_hex(v: &serde_json::Value, field: &str) -> Result<Vec<ErgoBox>, String> {
    let arr = v
        .as_array()
        .ok_or_else(|| format!("{field} not an array"))?;
    let mut out = Vec::with_capacity(arr.len());
    for (i, item) in arr.iter().enumerate() {
        let h = item
            .as_str()
            .ok_or_else(|| format!("{field}[{i}] not a string"))?;
        let bytes = hex_to_bytes(h).map_err(|e| format!("{field}[{i}] bad hex: {e}"))?;
        let b = ErgoBox::sigma_parse_bytes(&bytes)
            .map_err(|e| format!("{field}[{i}] decode: {e:?}"))?;
        out.push(b);
    }
    Ok(out)
}

/// Build the `ErgoStateContext` from the vector's REAL provided context (runner-contract-transaction §5):
/// the last headers (`headers_hex`, NEWEST-first scorex bytes), the validating block's
/// `preHeader`, and the on-chain `parameters` table. This replaces the earlier synthetic
/// zeroed context — the reference and every runner now validate under the SAME carried
/// context, so there is no per-impl reconstruction to coordinate.
fn build_state_context(entry: &serde_json::Value) -> Result<ErgoStateContext, String> {
    // Headers: newest-first; 1..=10 (a node near genesis supplies fewer).
    let headers_arr = entry["headers_hex"]
        .as_array()
        .ok_or("headers_hex missing or not an array")?;
    let mut headers_vec: Vec<Header> = Vec::with_capacity(headers_arr.len());
    for (i, item) in headers_arr.iter().enumerate() {
        let h = item
            .as_str()
            .ok_or_else(|| format!("headers_hex[{i}] not a string"))?;
        let bytes = hex_to_bytes(h).map_err(|e| format!("headers_hex[{i}] bad hex: {e}"))?;
        headers_vec.push(
            Header::scorex_parse_bytes(&bytes)
                .map_err(|e| format!("headers_hex[{i}] decode: {e:?}"))?,
        );
    }
    let headers: Headers = headers_vec
        .try_into()
        .map_err(|e| format!("headers out of bounds (need 1..=10): {e:?}"))?;

    // PreHeader: the validating block's real pre-header (no synthetic `activated + 1` pin).
    let ph = &entry["preHeader"];
    let version = ph["version"].as_u64().ok_or("preHeader.version missing")? as u8;
    let parent_id_bytes: [u8; 32] = hex_to_bytes(
        ph["parentId"]
            .as_str()
            .ok_or("preHeader.parentId missing")?,
    )
    .map_err(|e| format!("preHeader.parentId bad hex: {e}"))?
    .try_into()
    .map_err(|_| "preHeader.parentId not 32 bytes".to_string())?;
    let timestamp: u64 = ph["timestamp"]
        .as_str()
        .ok_or("preHeader.timestamp missing (u64 string)")?
        .parse()
        .map_err(|_| "preHeader.timestamp not a u64".to_string())?;
    let n_bits = ph["nBits"].as_u64().ok_or("preHeader.nBits missing")? as u32;
    let height = ph["height"].as_u64().ok_or("preHeader.height missing")? as u32;
    let miner_pk_bytes = hex_to_bytes(ph["minerPk"].as_str().ok_or("preHeader.minerPk missing")?)
        .map_err(|e| format!("preHeader.minerPk bad hex: {e}"))?;
    let miner_pk = EcPoint::scorex_parse_bytes(&miner_pk_bytes)
        .map_err(|e| format!("preHeader.minerPk decode: {e:?}"))?;
    let votes_bytes: [u8; 3] = hex_to_bytes(ph["votes"].as_str().ok_or("preHeader.votes missing")?)
        .map_err(|e| format!("preHeader.votes bad hex: {e}"))?
        .try_into()
        .map_err(|_| "preHeader.votes not 3 bytes".to_string())?;
    let pre_header = PreHeader {
        version,
        parent_id: BlockId(Digest32::from(parent_id_bytes)),
        timestamp,
        n_bits,
        height,
        miner_pk: Box::new(miner_pk),
        votes: Votes(votes_bytes),
    };

    // Parameters: the carried on-chain table. Start from defaults (for `MaxBlockSize`,
    // which the vector does not carry and stateful tx validation does not consult) and
    // overwrite every provided field; set `BlockVersion` to the VOTED parameters' block version
    // (parameters.blockVersion), which gates the activated ErgoTree version AND the monotonic
    // creation-height rule. The JVM reads it from currentParameters, not the header. It defaults
    // to the pre-header's version — equal in every vector that omits the field.
    let p = &entry["parameters"];
    let pget = |k: &str| -> Result<i32, String> {
        p[k].as_i64()
            .ok_or_else(|| format!("parameters.{k} missing"))
            .map(|v| v as i32)
    };
    let mut parameters = Parameters::default();
    let t = &mut parameters.parameters_table;
    t.insert(Parameter::MaxBlockCost, pget("maxBlockCost")?);
    t.insert(Parameter::StorageFeeFactor, pget("storageFeeFactor")?);
    t.insert(Parameter::MinValuePerByte, pget("minValuePerByte")?);
    t.insert(Parameter::InputCost, pget("inputCost")?);
    t.insert(Parameter::DataInputCost, pget("dataInputCost")?);
    t.insert(Parameter::OutputCost, pget("outputCost")?);
    t.insert(Parameter::TokenAccessCost, pget("tokenAccessCost")?);
    let params_block_version = p["blockVersion"]
        .as_i64()
        .map(|v| v as i32)
        .unwrap_or(version as i32);
    t.insert(Parameter::BlockVersion, params_block_version);

    Ok(ErgoStateContext::new(pre_header, headers, parameters))
}

/// Validate one transaction entry. Produces exactly one [`TxOutcome`] (totality,
/// runner contract §3). The tx + boxes decode from canonical sigma bytes (`tx_bytes_hex` /
/// `*_boxes_hex`); the state context is built from the entry's real `headers_hex` +
/// `preHeader` + `parameters` (runner-contract-transaction §5).
pub fn run_entry(entry: &serde_json::Value) -> TxOutcome {
    let tx_bytes = match entry["tx_bytes_hex"].as_str() {
        Some(h) => match hex_to_bytes(h) {
            Ok(b) => b,
            Err(e) => {
                return TxOutcome::Errored {
                    reason: format!("tx_bytes_hex bad hex: {e}"),
                }
            }
        },
        None => {
            return TxOutcome::Errored {
                reason: "entry missing tx_bytes_hex".into(),
            }
        }
    };
    // ergo reads a v6 block's transactions at version context (blockVersion - 1,
    // blockVersion - 1), an earlier block's at the default (1, 1) (ergo v6.0.6
    // `BlockTransactions.scala:184-202`); the reader's version is sigma-rust's context.
    // From block version 4 the reader also takes the activated script version,
    // blockVersion - 1; a box (below) is read from the UTXO set, outside any context.
    // The block version is the voted parameters' (`parameters.blockVersion`, the
    // pre-header's when the entry has none), a signed byte, as in SANTA's `TxEngine`.
    let block_version = entry["parameters"]["blockVersion"]
        .as_u64()
        .or_else(|| entry["preHeader"]["version"].as_u64())
        .map(|v| v as u8 as i8);
    let mut tx_reader = sigma_byte_reader::from_bytes(&tx_bytes);
    let parsed_tx = match block_version {
        Some(v) if v >= 4 => tx_reader.with_versions(
            ErgoTreeVersion::from((v - 1) as u8),
            ErgoTreeVersion::V3,
            Transaction::sigma_parse,
        ),
        _ => tx_reader.with_tree_version(ErgoTreeVersion::V0, Transaction::sigma_parse),
    };
    let tx = match parsed_tx {
        Ok(t) => t,
        Err(e) => {
            return TxOutcome::Errored {
                reason: format!("tx decode: {e:?}"),
            }
        }
    };
    let input_boxes = match decode_boxes_hex(&entry["input_boxes_hex"], "input_boxes_hex") {
        Ok(b) => b,
        Err(e) => return TxOutcome::Errored { reason: e },
    };
    let data_input_boxes =
        match decode_boxes_hex(&entry["data_input_boxes_hex"], "data_input_boxes_hex") {
            Ok(b) => b,
            Err(e) => return TxOutcome::Errored { reason: e },
        };

    let state_context = match build_state_context(entry) {
        Ok(c) => c,
        Err(e) => return TxOutcome::Errored { reason: e },
    };

    // Assemble boxes_to_spend/data_boxes keyed by the tx's input ids. A failure here
    // (box missing, bounds) means the runner can't even pose the validation question.
    let tx_context = match TransactionContext::new(tx, input_boxes, data_input_boxes) {
        Ok(c) => c,
        Err(e) => {
            return TxOutcome::Errored {
                reason: format!("tx context: {e}"),
            }
        }
    };

    match tx_context.validate(&state_context) {
        Ok(cost) => TxOutcome::Verdict {
            valid: true,
            cost: Some(cost),
            reason: None,
        },
        Err(e) => TxOutcome::Verdict {
            valid: false,
            cost: None,
            reason: Some(e.to_string()),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::run_entry;
    use serde_json::Value as J;

    /// The keystone captured vector, committed verbatim at
    /// `tests/fixtures/bigint-downcast-2666.json` (the on-disk `santa-transaction/v1` form) so
    /// the test is hermetic in the standalone runner repo (no SANTA vector tree needed).
    /// A real testnet tx whose input script needs the v3-gated BigInt→Long Downcast —
    /// the reference answers {valid: true, cost: 14846}.
    const KEYSTONE_VECTOR: &str = include_str!("../tests/fixtures/bigint-downcast-2666.json");

    /// Shape test only: a verdict must be reached (`error: null`, `valid` a bool).
    /// Whether `valid` is true (a sigma-rust with the v6 fixes, as the reference answers) or
    /// false (upstream develop: pre-fix `reduce_to_crypto` rejects the v3-gated
    /// Downcast) is per-branch — graded by conform against `expected`, not asserted
    /// here. The verdict is logged for eyeballing.
    #[test]
    fn keystone_reaches_a_verdict() {
        let vector: J = serde_json::from_str(KEYSTONE_VECTOR).expect("fixture JSON");
        let entry = &vector["entries"][0];
        let j = run_entry(entry).to_json();
        println!("keystone bigint-downcast-2666 verdict: {j}");
        assert_eq!(j["error"], J::Null, "expected a verdict, got: {j}");
        assert!(j["valid"].is_boolean(), "valid must be a bool, got: {j}");
        if j["valid"] == J::Bool(false) {
            assert!(j["reason"].is_string(), "a reject must carry a reason: {j}");
        }
    }

    /// A malformed entry (tx bytes not decodable) is `errored` — the runner failed to
    /// reach a verdict; never a panic, never a silent skip (contract totality). The tx
    /// decode fails first, before any context assembly.
    #[test]
    fn undecodable_tx_is_errored() {
        let entry: J = serde_json::json!({
            "name": "broken",
            "tx_bytes_hex": "00"
        });
        let j = run_entry(&entry).to_json();
        assert_eq!(j["error"], "errored");
        assert_eq!(j["valid"], J::Null);
        assert_eq!(j["cost"], J::Null);
        assert!(j["reason"].is_string(), "errored must carry a reason: {j}");
    }

    fn preheader(version: u8) -> J {
        serde_json::json!({
            "version": version,
            "parentId": "0000000000000000000000000000000000000000000000000000000000000000",
            "timestamp": "0",
            "nBits": 0,
            "height": 0,
            "minerPk": "030000000000000000000000000000000000000000000000000000000000000000",
            "votes": "000000",
        })
    }
    fn parameters(block_version: u8) -> J {
        serde_json::json!({
            "blockVersion": block_version,
            "maxBlockCost": 0,
            "storageFeeFactor": 0,
            "minValuePerByte": 0,
            "inputCost": 0,
            "dataInputCost": 0,
            "outputCost": 0,
            "tokenAccessCost": 0,
        })
    }

    /// From block version 4 the transaction tier parses a tx at activated `blockVersion - 1`
    /// — a v4 block reads at activated V3. A single-output tx whose output tree has header
    /// version 4 is refused (its version is above the activated one). A mutant that passed
    /// `blockVersion` directly as activated (V4) would accept the v4 tree, fail only later on
    /// the empty headers list.
    #[test]
    fn tx_tier_uses_block_version_minus_one_as_activated() {
        // Minimal tx: 1 input (zero box_id, empty proof, empty ctx_ext), 0 data inputs,
        // 0 tokens, 1 output (value 1M, ergo_tree v4 size, body SigmaProp true, no tokens, no regs).
        let tx_hex = "0100000000000000000000000000000000000000000000000000000000000000000000000001c0843d0c0208d3000000";
        let entry = serde_json::json!({
            "name": "e#0",
            "tx_bytes_hex": tx_hex,
            "input_boxes_hex": [],
            "data_input_boxes_hex": [],
            "headers_hex": [],
            "preHeader": preheader(4),
            "parameters": parameters(4),
        });
        let j = run_entry(&entry).to_json();
        assert_eq!(j["error"], "errored");
        let reason = j["reason"].as_str().unwrap_or("");
        assert!(
            reason.contains("tx decode"),
            "expected the transaction's read to refuse a v4 tree at activated 3, got: {j}"
        );
    }

    /// Below block version 4 the transaction tier parses a tx at `tree_version = V0` (no
    /// activation context). A tx whose input context extension holds `Coll[SFunc(Int=>Int)]`
    /// — a function type gated to reader tree_version >= V3 — fails to parse. A mutant that
    /// raised the default reader tree_version to V3 for below-block-4 reads would ACCEPT the
    /// SFunc value and fail only later on the empty headers list.
    #[test]
    fn tx_tier_reads_below_block_4_at_tree_version_v0() {
        // Tx with 1 input carrying ctx-ext {0: Coll[SFunc(SInt=>SInt)]()} and 1 output.
        let tx_hex = "0100a19de1b5fa998df5a48630a611180690abad5270c33f23a79baba2f8840d71000100 83007001040400 000001c0843d0008d3010000".replace(' ', "");
        let entry = serde_json::json!({
            "name": "e#0",
            "tx_bytes_hex": tx_hex,
            "input_boxes_hex": [],
            "data_input_boxes_hex": [],
            "headers_hex": [],
            "preHeader": preheader(3),
            "parameters": parameters(3),
        });
        let j = run_entry(&entry).to_json();
        assert_eq!(j["error"], "errored");
        let reason = j["reason"].as_str().unwrap_or("");
        assert!(
            reason.contains("tx decode"),
            "expected a tx-decode failure at tree_version V0, got: {j}"
        );
    }
}
