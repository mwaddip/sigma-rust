//! The eval tier: evaluate an entry's tree in the canonical evaluation context, with the entry's
//! input bound where its envelope puts it, under the entry's `(activated, ergoTree)` versions; or,
//! for the full-context form, in the real context the entry carries.
//!
//! The canonical context is the one the reference evaluator (sigma-state) builds for a bare
//! evaluation. SELF is a box holding the entry's tree, with value 1,000,000, no tokens, no
//! additional registers, creation height 0, a zero transaction id and index 0. It holds the tree
//! as the library's strict reader keeps it (the reader of a node built on this library), and
//! otherwise as parsed ([`self_tree_of`]). INPUTS is exactly
//! `[SELF]`, OUTPUTS, the data inputs and the headers are empty, HEIGHT is 0, the pre-header and
//! the last block's state root are fixed (see [`ContextParts::context`]) and no spending input
//! carries a context extension. The entry's input goes, by envelope ([`Binding`]): to variable 1
//! of SELF's extension (v1/v2); into one extension per spending input, read by `getVarFromInput`
//! (v3); into SELF's registers R4-R9 plus variable 1 (v4); or it is SELF's whole extension (v5).
//!
//! The full-context form (`santa-eval/v6-fullctx`) carries a captured spend: its boxes, headers,
//! pre-header and per-input extensions ([`FullContextParts`]).
//!
//! Trees are read with `ErgoTree::sigma_parse_bytes_lenient`, because the vectors' roots are of any
//! type, not only SigmaProp (the reference's `deserializeErgoTree(checkType = false)`). They are
//! evaluated with `eval::test_util::try_eval_with_deserialize`, the public entry (behind the
//! `arbitrary` feature) that evaluates a root of any type: it substitutes `DeserializeContext`
//! nodes against the context, then evaluates. The root keeps its constant placeholders and the
//! context carries the tree's constants, as a node evaluates a tree; the cost is the context's JIT
//! cost counter after the evaluation, under the reference's limit ([`EVAL_JIT_COST_LIMIT`]).

use std::cell::Cell;

use ergo_chain_types::{ec_point, BlockId, Digest, EcPoint, Header, PreHeader, Votes};
use ergotree_interpreter::eval::test_util::try_eval_with_deserialize;
use ergotree_ir::chain::context::{Context, ContextExtensionProvider, ContextHeaders, TxIoVec};
use ergotree_ir::chain::context_extension::ContextExtension;
use ergotree_ir::chain::ergo_box::box_value::BoxValue;
use ergotree_ir::chain::ergo_box::{ErgoBox, NonMandatoryRegisterId, NonMandatoryRegisters};
use ergotree_ir::chain::evaluated_value::EvaluatedValue;
use ergotree_ir::chain::tx_id::TxId;
use ergotree_ir::ergo_tree::{ErgoTree, ErgoTreeVersion};
use ergotree_ir::mir::avl_tree_data::{AvlTreeData, AvlTreeFlags};
use ergotree_ir::mir::value::Value;
use ergotree_ir::serialization::SigmaSerializable;
use ergotree_ir::types::stype::SType;
use sigma_ser::ScorexSerializable;

use crate::binding;
use crate::sval;

/// One entry's outcome.
pub enum Outcome {
    /// Evaluated to a value, at this JIT cost.
    Success {
        value: serde_json::Value,
        cost: u64,
    },
    Errored,
    /// An entry whose context the runner cannot build (see [`ContextParts::new`]).
    NotImplemented,
    /// A failure that is not the library's verdict: a panic caught by `main`'s net, or a failure
    /// to decode the entry's input or encode its result. The note says which.
    Panicked {
        note: String,
    },
}

impl Outcome {
    pub fn to_json(&self) -> serde_json::Value {
        match self {
            Outcome::Success { value, cost } => {
                serde_json::json!({"value": value, "cost": cost, "error": serde_json::Value::Null})
            }
            Outcome::Errored => {
                serde_json::json!({"value": null, "cost": null, "error": "errored"})
            }
            Outcome::NotImplemented => {
                serde_json::json!({"value": null, "cost": null, "error": "not-implemented"})
            }
            Outcome::Panicked { note } => {
                serde_json::json!({"value": null, "cost": null, "error": "panicked", "note": note})
            }
        }
    }
}

/// The reference evaluator's cost limit (runner contract §2): `scriptCostLimitInEvaluator`,
/// 1,000,000 block units = 10,000,000 JIT units, from an initial cost of 0. Every context the
/// runner builds carries it: with none, a script that writes past the limit answers a value.
const EVAL_JIT_COST_LIMIT: u64 = 10_000_000;

/// What an entry binds into the canonical context, by envelope.
enum Binding {
    /// v1/v2: the entry's input, if it has one, at variable 1 of SELF's extension.
    Var1(Option<EvaluatedValue>),
    /// v3: one context extension per spending input (`getVarFromInput`); SELF's own is empty.
    PerInput(Vec<ContextExtension>),
    /// v4: SELF's registers R4-R9, and the entry's input at variable 1.
    SelfRegisters(NonMandatoryRegisters, EvaluatedValue),
    /// v5: SELF's whole extension, verbatim (keys 0..255).
    Extension(ContextExtension),
}

/// The spending inputs' context extensions, by input index.
struct InputExtensions(Vec<ContextExtension>);

impl ContextExtensionProvider for InputExtensions {
    fn context_extension(&self, input_index: usize) -> Option<&ContextExtension> {
        self.0.get(input_index)
    }
}

/// The last block's state root: a digest, every operation allowed, 32-byte keys, values of any
/// length.
fn state_root(digest: Vec<u8>) -> AvlTreeData {
    AvlTreeData {
        digest,
        tree_flags: AvlTreeFlags::new(true, true, true),
        key_length: 32,
        value_length_opt: None,
    }
}

/// What the canonical context borrows.
struct ContextParts {
    /// SELF, and INPUTS' one box.
    self_box: ErgoBox,
    extension: ContextExtension,
    input_extensions: InputExtensions,
}

impl ContextParts {
    /// `tree` is the tree SELF holds ([`self_tree_of`]). The `Err` is the entry's outcome. A SELF
    /// that cannot be built is a context this runner cannot represent: not-implemented, the cause
    /// on stderr, never a stand-in box. `ErgoBox::new` writes the box for its id, and a parsed tree
    /// can hold what has no written form (a tuple type of zero or one item). The reference keeps
    /// such a tree in a box it writes only when a script reads the box's id or bytes, and fails
    /// then.
    fn new(binding: Binding, tree: &ErgoTree) -> Result<ContextParts, Outcome> {
        let with_var1 = |v: Option<EvaluatedValue>| {
            let mut ext = ContextExtension::empty();
            if let Some(v) = v {
                ext.values.insert(1u8, v);
            }
            ext
        };
        let none = || InputExtensions(Vec::new());
        let no_registers = NonMandatoryRegisters::empty;
        let (registers, extension, input_extensions) = match binding {
            Binding::Var1(input) => (no_registers(), with_var1(input), none()),
            Binding::PerInput(exts) => (
                no_registers(),
                ContextExtension::empty(),
                InputExtensions(exts),
            ),
            Binding::SelfRegisters(regs, input) => (regs, with_var1(Some(input)), none()),
            Binding::Extension(ext) => (no_registers(), ext, none()),
        };
        let value = BoxValue::new(1_000_000).map_err(|e| Outcome::Panicked {
            note: format!("SELF value: {e:?}"),
        })?;
        // SELF is built with its registers, never given them afterwards: a box's id is the hash of
        // its bytes, registers included.
        let self_box = ErgoBox::new(
            value,
            tree.clone(),
            None, // no tokens
            registers,
            0, // creation height
            TxId::zero(),
            0, // index
        )
        .map_err(|e| {
            eprintln!("blitzen: the canonical SELF box cannot be built: {e:?}");
            Outcome::NotImplemented
        })?;
        Ok(ContextParts {
            self_box,
            extension,
            input_extensions,
        })
    }

    /// The canonical context over these parts. The pre-header is the reference's: version
    /// `activated + 1` (the block version the activated script version derives from), a zero
    /// parent id, timestamp 3, nBits 0, height 0, the group generator as miner key, zero votes.
    /// The activated script version is a context input of its own. There are no headers, and the
    /// last block's state root is the reference's dummy tree over 33 zero bytes.
    fn context(&self, tree_version: u8, activated_version: u8) -> Context<'_> {
        Context {
            height: 0,
            self_box: &self.self_box,
            outputs: &[],
            data_inputs: None,
            inputs: TxIoVec::from_vec(vec![&self.self_box])
                .expect("one input is within TxIoVec's bounds"),
            pre_header: PreHeader {
                version: activated_version + 1,
                parent_id: BlockId(Digest::zero()),
                timestamp: 3,
                n_bits: 0,
                height: 0,
                miner_pk: Box::new(ec_point::generator()),
                votes: Votes([0u8; 3]),
            },
            activated_script_version_byte: activated_version as i8,
            last_block_utxo_root: state_root(vec![0u8; 33]),
            headers: ContextHeaders::from_vec(Vec::new()).expect("no headers is within bounds"),
            extension: &self.extension,
            tree_version: Cell::new(ErgoTreeVersion::from(tree_version)),
            extension_provider: &self.input_extensions,
            jit_cost: Cell::new(0),
            jit_cost_limit: Some(EVAL_JIT_COST_LIMIT),
            constants: None,
        }
    }
}

/// An input-decode failure's outcome: the library refusing the bytes (its verdict on material
/// the reference accepted) is `errored`, a divergence; any other failure (malformed vector JSON,
/// an unsupported kind) is the runner's own, `panicked` with the cause.
fn decode_failure_outcome(e: sval::BridgeError, site: &str) -> Outcome {
    match e {
        sval::BridgeError::Refused(_) => Outcome::Errored,
        other => Outcome::Panicked {
            note: format!("{}: {:?}", site, other),
        },
    }
}

/// One extension map (`{"<var id>": <binding>}`), each value read by sigma-rust's own reader. A
/// key that is not a byte is `errored`.
fn extension_from(
    map: &serde_json::Map<String, serde_json::Value>,
    activated_version: u8,
    site: &str,
) -> Result<ContextExtension, Outcome> {
    let mut ext = ContextExtension::empty();
    for (k, v) in map {
        let id: u8 = k.parse().map_err(|_| Outcome::Errored)?;
        let c = binding::extension_binding(v, activated_version)
            .map_err(|e| decode_failure_outcome(e, site))?;
        ext.values.insert(id, c);
    }
    Ok(ext)
}

/// SELF's registers (`{"4": <binding>, …, "9": …}`), read as one register map, in id order.
fn registers_from(
    map: &serde_json::Map<String, serde_json::Value>,
    activated_version: u8,
) -> Result<NonMandatoryRegisters, Outcome> {
    let mut pairs: Vec<(NonMandatoryRegisterId, &serde_json::Value)> =
        Vec::with_capacity(map.len());
    for (k, v) in map {
        let id: u8 = k.parse().map_err(|_| Outcome::Panicked {
            note: format!("v4 selfRegisters: bad key {:?}", k),
        })?;
        let reg_id = match id {
            4 => NonMandatoryRegisterId::R4,
            5 => NonMandatoryRegisterId::R5,
            6 => NonMandatoryRegisterId::R6,
            7 => NonMandatoryRegisterId::R7,
            8 => NonMandatoryRegisterId::R8,
            9 => NonMandatoryRegisterId::R9,
            _ => {
                return Err(Outcome::Panicked {
                    note: format!("v4 selfRegisters: key {} out of R4-R9 range", id),
                })
            }
        };
        pairs.push((reg_id, v));
    }
    pairs.sort_by_key(|(rid, _)| *rid as u8);
    binding::register_map(&pairs, activated_version)
        .map_err(|e| decode_failure_outcome(e, "v4 selfRegisters decode"))
}

/// The entry's binding, by envelope: an `extension` makes it v5, else `selfRegisters` v4, else
/// `inputs` v3, else v1/v2 (its `input`, if any). A v3 entry's `input` is still decoded, first.
fn binding_of(
    input: Option<&serde_json::Value>,
    inputs: Option<&Vec<serde_json::Value>>,
    self_registers: Option<&serde_json::Map<String, serde_json::Value>>,
    extension: Option<&serde_json::Map<String, serde_json::Value>>,
    activated_version: u8,
) -> Result<Binding, Outcome> {
    // A key >= 0x80 stays in: sigma-rust's own guard decides (the reference refuses it), so an
    // implementation that wrongly accepts it shows.
    if let Some(map) = extension {
        return extension_from(map, activated_version, "v5 extension decode")
            .map(Binding::Extension);
    }
    if let Some(map) = self_registers {
        let input = input.ok_or_else(|| Outcome::Panicked {
            note: "v4 entry missing input".into(),
        })?;
        let var1 = binding::extension_binding(input, activated_version)
            .map_err(|e| decode_failure_outcome(e, "v4 input decode"))?;
        let regs = registers_from(map, activated_version)?;
        return Ok(Binding::SelfRegisters(regs, var1));
    }
    let var1 = match input {
        Some(j) => Some(
            binding::extension_binding(j, activated_version)
                .map_err(|e| decode_failure_outcome(e, "input decode"))?,
        ),
        None => None,
    };
    match inputs {
        Some(arr) => {
            let mut exts = Vec::with_capacity(arr.len());
            for inp in arr {
                exts.push(match inp.get("extension").and_then(|e| e.as_object()) {
                    Some(map) => extension_from(map, activated_version, "input decode (v3)")?,
                    None => ContextExtension::empty(),
                });
            }
            Ok(Binding::PerInput(exts))
        }
        None => Ok(Binding::Var1(var1)),
    }
}

/// Evaluate one entry. Produces exactly one [`Outcome`].
pub fn run_entry(
    tree_bytes: &[u8],
    input: Option<&serde_json::Value>,
    inputs: Option<&Vec<serde_json::Value>>,
    self_registers: Option<&serde_json::Map<String, serde_json::Value>>,
    extension: Option<&serde_json::Map<String, serde_json::Value>>,
    tree_version: u8,
    activated_version: u8,
) -> Outcome {
    let binding = match binding_of(input, inputs, self_registers, extension, activated_version) {
        Ok(binding) => binding,
        Err(outcome) => return outcome,
    };
    let tree = match ErgoTree::sigma_parse_bytes_lenient(tree_bytes) {
        Ok(t) => t,
        Err(_) => return Outcome::Errored,
    };
    let parts = match ContextParts::new(binding, &self_tree_of(tree_bytes, &tree)) {
        Ok(p) => p,
        Err(outcome) => return outcome,
    };
    evaluate(&tree, parts.context(tree_version, activated_version))
}

/// The tree SELF holds. The runner contract (§2, the SELF-tree note): as a node holds it where a
/// node can read it, otherwise the parse. The node's reader is here the library's strict one,
/// which a box is read with. It keeps a size-flagged tree whose root is not a SigmaProp, which is
/// not a script, as its bytes: the bytes its size field spans. A tree that reader refuses (for
/// one, a size field that claims more bytes than there are), an unsized tree with such a root,
/// whose box a node refuses, and a SigmaProp root are held as parsed. The tree evaluated is the
/// parse in every case.
fn self_tree_of(tree_bytes: &[u8], parsed: &ErgoTree) -> ErgoTree {
    let size_flagged = tree_bytes.first().is_some_and(|h| h & 0x08 != 0);
    let root_is_no_sigma_prop = parsed
        .root_expr()
        .is_ok_and(|root| root.tpe() != SType::SSigmaProp);
    if size_flagged && root_is_no_sigma_prop {
        // for a sized tree `sigma_parse_bytes` is the strict reader, the one a box is read with
        ErgoTree::sigma_parse_bytes(tree_bytes).unwrap_or_else(|_| parsed.clone())
    } else {
        parsed.clone()
    }
}

/// Evaluate the tree's root in the context and encode the value with its cost. The root keeps
/// its constant placeholders and the context is given the tree's constants. A tree that did not
/// parse to a root, or an evaluation error, is `errored`.
fn evaluate<'a>(tree: &'a ErgoTree, mut ctx: Context<'a>) -> Outcome {
    let root = match tree.root_expr() {
        Ok(r) => r,
        Err(_) => return Outcome::Errored,
    };
    ctx.constants = match tree.constants() {
        Ok(c) => Some(c),
        Err(_) => return Outcome::Errored,
    };
    match try_eval_with_deserialize::<Value<'static>>(root, &ctx) {
        Ok(v) => match sval::encode_value(&v) {
            Ok(value) => Outcome::Success {
                value,
                cost: ctx.jit_cost_value(),
            },
            Err(e) => Outcome::Panicked {
                note: format!("result encode: {:?}", e),
            },
        },
        Err(_) => Outcome::Errored,
    }
}

/// Why a full-context envelope gave no context: `Refused` is the library refusing material the
/// reference accepted (a box, header, pre-header or miner key it cannot read, a collection out of
/// its bounds), which is `errored`, a divergence; `Malformed` is a defect of the envelope or of
/// this runner's reading of it (a missing or mistyped field, bad hex), which is `panicked` with
/// the cause.
enum FullCtxError {
    Refused(String),
    Malformed(String),
}

/// What a full-context entry's context borrows: the captured spend, read from the envelope.
struct FullContextParts {
    inputs: Vec<ErgoBox>,
    data_inputs: Vec<ErgoBox>,
    outputs: Vec<ErgoBox>,
    self_index: usize,
    /// Newest first; up to ten, fewer near genesis.
    headers: Vec<Header>,
    pre_header: PreHeader,
    height: u32,
    last_block_utxo_root: AvlTreeData,
    activated_script_version_byte: i8,
    /// SELF's own extension: its input's.
    extension: ContextExtension,
    input_extensions: InputExtensions,
}

impl FullContextParts {
    /// Read a `santa-eval/v6-fullctx` `context` envelope (runner contract §2): the real boxes,
    /// headers and pre-header of a spend. The last block's state root is the envelope's
    /// `last_block_utxo_root_hex`, or else the newest header's. The activated script version is the
    /// entry's `version.activated`, as the reference takes it; an entry without one takes the
    /// envelope's block version minus 1.
    fn new(
        context: &serde_json::Value,
        activated_version: Option<u8>,
    ) -> Result<FullContextParts, FullCtxError> {
        let obj = context
            .as_object()
            .ok_or_else(|| FullCtxError::Malformed("context not an object".into()))?;

        // A hex-array box field (`inputs` / `data_inputs` / `outputs`).
        let parse_boxes = |key: &str| -> Result<Vec<ErgoBox>, FullCtxError> {
            let arr = obj.get(key).and_then(|v| v.as_array()).ok_or_else(|| {
                FullCtxError::Malformed(format!("context.{key} missing or not an array"))
            })?;
            let mut out = Vec::with_capacity(arr.len());
            for (i, item) in arr.iter().enumerate() {
                let h = item.as_str().ok_or_else(|| {
                    FullCtxError::Malformed(format!("context.{key}[{i}] not a string"))
                })?;
                let bytes = crate::hex_to_bytes(h)
                    .map_err(|_| FullCtxError::Malformed(format!("context.{key}[{i}] bad hex")))?;
                let b = ErgoBox::sigma_parse_bytes(&bytes)
                    .map_err(|e| FullCtxError::Refused(format!("context.{key}[{i}]: {e:?}")))?;
                out.push(b);
            }
            Ok(out)
        };
        let inputs = parse_boxes("inputs")?;
        let data_inputs = parse_boxes("data_inputs")?;
        let outputs = parse_boxes("outputs")?;

        let headers_arr = obj
            .get("headers")
            .and_then(|v| v.as_array())
            .ok_or_else(|| {
                FullCtxError::Malformed("context.headers missing or not an array".into())
            })?;
        let mut headers: Vec<Header> = Vec::with_capacity(headers_arr.len());
        for (i, item) in headers_arr.iter().enumerate() {
            let h = item.as_str().ok_or_else(|| {
                FullCtxError::Malformed(format!("context.headers[{i}] not a string"))
            })?;
            let bytes = crate::hex_to_bytes(h)
                .map_err(|_| FullCtxError::Malformed(format!("context.headers[{i}] bad hex")))?;
            headers.push(
                Header::scorex_parse_bytes(&bytes)
                    .map_err(|e| FullCtxError::Refused(format!("context.headers[{i}]: {e:?}")))?,
            );
        }

        // The pre-header has no consensus serializer: the envelope carries SANTA's own encoding
        // (`crate::preheader`), with the real block version.
        let ph_hex = obj
            .get("pre_header_hex")
            .and_then(|v| v.as_str())
            .ok_or_else(|| FullCtxError::Malformed("context.pre_header_hex missing".into()))?;
        let ph_bytes = crate::hex_to_bytes(ph_hex)
            .map_err(|_| FullCtxError::Malformed("context.pre_header_hex bad hex".into()))?;
        let phf = crate::preheader::decode(&ph_bytes)
            .map_err(|e| FullCtxError::Refused(format!("pre_header decode: {e}")))?;
        let miner_pk = EcPoint::scorex_parse_bytes(&phf.miner_pk)
            .map_err(|e| FullCtxError::Refused(format!("minerPk parse: {e:?}")))?;
        let pre_header = PreHeader {
            version: phf.version,
            parent_id: BlockId(Digest::<32>::from(phf.parent_id)),
            timestamp: phf.timestamp,
            n_bits: phf.n_bits,
            height: phf.height,
            miner_pk: Box::new(miner_pk),
            votes: Votes(phf.votes),
        };

        let height = obj.get("height").and_then(|v| v.as_u64()).ok_or_else(|| {
            FullCtxError::Malformed("context.height missing or not a number".into())
        })? as u32;
        let self_index = obj
            .get("self_index")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| {
                FullCtxError::Malformed("context.self_index missing or not a number".into())
            })? as usize;

        let digest: Vec<u8> =
            if let Some(hex) = obj.get("last_block_utxo_root_hex").and_then(|v| v.as_str()) {
                crate::hex_to_bytes(hex).map_err(|_| {
                    FullCtxError::Malformed("context.last_block_utxo_root_hex bad hex".into())
                })?
            } else {
                headers
                    .first()
                    .map(|h| h.state_root.0.to_vec())
                    .ok_or_else(|| {
                        FullCtxError::Malformed("no headers and no last_block_utxo_root_hex".into())
                    })?
            };

        // One extension per input (`getVarFromInput` by index), each value read at the activated
        // version this context takes.
        let read_activated = activated_version.unwrap_or_else(|| phf.version.wrapping_sub(1));
        let mut input_extensions: Vec<ContextExtension> = Vec::new();
        if let Some(arr) = obj.get("input_extensions").and_then(|v| v.as_array()) {
            for (i, item) in arr.iter().enumerate() {
                let mut ext = ContextExtension::empty();
                if let Some(map) = item.as_object() {
                    for (k, v) in map {
                        let id: u8 = k.parse().map_err(|_| {
                            FullCtxError::Malformed(format!("input_extensions[{i}] bad key {k:?}"))
                        })?;
                        let ev =
                            binding::extension_binding(v, read_activated).map_err(|e| match e {
                                sval::BridgeError::Refused(m) => FullCtxError::Refused(format!(
                                    "input_extensions[{i}][{k}]: {m}"
                                )),
                                other => FullCtxError::Malformed(format!(
                                    "input_extensions[{i}][{k}]: {other:?}"
                                )),
                            })?;
                        ext.values.insert(id, ev);
                    }
                }
                input_extensions.push(ext);
            }
        }

        if self_index >= inputs.len() {
            return Err(FullCtxError::Malformed(format!(
                "self_index {self_index} out of range ({} inputs)",
                inputs.len()
            )));
        }
        Ok(FullContextParts {
            inputs,
            data_inputs,
            outputs,
            self_index,
            headers,
            activated_script_version_byte: match activated_version {
                Some(v) => v as i8,
                None => (pre_header.version as i8).wrapping_sub(1),
            },
            pre_header,
            height,
            last_block_utxo_root: state_root(digest),
            extension: input_extensions
                .get(self_index)
                .cloned()
                .unwrap_or_else(ContextExtension::empty),
            input_extensions: InputExtensions(input_extensions),
        })
    }

    /// The context over these parts. A collection out of the library's bounds is `Refused`.
    fn context(&self, tree_version: u8) -> Result<Context<'_>, FullCtxError> {
        let inputs = TxIoVec::from_vec(self.inputs.iter().collect())
            .map_err(|e| FullCtxError::Refused(format!("inputs bounds: {e:?}")))?;
        let data_inputs = if self.data_inputs.is_empty() {
            None
        } else {
            Some(
                TxIoVec::from_vec(self.data_inputs.iter().collect())
                    .map_err(|e| FullCtxError::Refused(format!("data_inputs bounds: {e:?}")))?,
            )
        };
        let headers = ContextHeaders::from_vec(self.headers.clone())
            .map_err(|e| FullCtxError::Refused(format!("headers bounds: {e:?}")))?;
        Ok(Context {
            height: self.height,
            self_box: &self.inputs[self.self_index],
            outputs: &self.outputs,
            data_inputs,
            inputs,
            pre_header: self.pre_header.clone(),
            activated_script_version_byte: self.activated_script_version_byte,
            last_block_utxo_root: self.last_block_utxo_root.clone(),
            headers,
            extension: &self.extension,
            tree_version: Cell::new(ErgoTreeVersion::from(tree_version)),
            extension_provider: &self.input_extensions,
            jit_cost: Cell::new(0),
            jit_cost_limit: Some(EVAL_JIT_COST_LIMIT),
            constants: None,
        })
    }
}

/// Evaluate one `santa-eval/v6-fullctx` entry: the tree in the real context its envelope carries.
/// Produces exactly one [`Outcome`].
pub fn run_entry_fullctx(
    tree_bytes: &[u8],
    context: &serde_json::Value,
    tree_version: u8,
    activated_version: Option<u8>,
) -> Outcome {
    let refused = |e: FullCtxError| match e {
        // `errored` carries no note in the actuals; the reason goes to stderr, for diagnosis.
        FullCtxError::Refused(reason) => {
            eprintln!("blitzen: v6-fullctx reconstruction refused: {reason}");
            Outcome::Errored
        }
        FullCtxError::Malformed(note) => Outcome::Panicked { note },
    };
    let parts = match FullContextParts::new(context, activated_version) {
        Ok(p) => p,
        Err(e) => return refused(e),
    };
    let ctx = match parts.context(tree_version) {
        Ok(c) => c,
        Err(e) => return refused(e),
    };
    let tree = match ErgoTree::sigma_parse_bytes_lenient(tree_bytes) {
        Ok(t) => t,
        Err(_) => return Outcome::Errored,
    };
    evaluate(&tree, ctx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ergotree_ir::mir::constant::Constant;

    /// A library refusal is `errored`; any other bridge failure is `panicked`, the site in the note.
    #[test]
    fn decode_failure_outcome_split() {
        assert!(matches!(
            decode_failure_outcome(sval::BridgeError::Refused("x".into()), "input decode"),
            Outcome::Errored
        ));
        match decode_failure_outcome(sval::BridgeError::Decode("x".into()), "input decode") {
            Outcome::Panicked { note } => {
                assert!(note.starts_with("input decode:"), "note: {}", note)
            }
            other => panic!("expected Panicked, got {:?}", outcome_name(&other)),
        }
    }

    fn fixture_tree() -> ErgoTree {
        // built from an Expr, not parsed, so these tests do not depend on the lenient entry
        ErgoTree::new(
            ergotree_ir::ergo_tree::ErgoTreeHeader::v1(false),
            &ergotree_ir::mir::expr::Expr::Const(1i32.into()),
        )
        .expect("the fixture tree is constructible")
    }

    fn var1() -> EvaluatedValue {
        Constant::from(7i32).into()
    }

    /// The canonical context's surfaces. Each is asserted: a vector that reads one would
    /// otherwise grade against whatever the runner happened to put there.
    #[test]
    fn canonical_context_pins_the_box_surfaces() {
        let tree = fixture_tree();
        let parts = ContextParts::new(Binding::Var1(None), &tree)
            .ok()
            .expect("the parts build");
        let ctx = parts.context(3, 2);
        assert_eq!(ctx.self_box.value.as_u64(), &1_000_000u64, "SELF.value");
        assert_eq!(
            ctx.self_box.ergo_tree, tree,
            "SELF.ergoTree is the entry's tree"
        );
        assert!(ctx.self_box.tokens.is_none(), "SELF carries no tokens");
        assert_eq!(ctx.self_box.creation_height, 0, "SELF.creationHeight");
        assert_eq!(
            ctx.self_box.transaction_id,
            TxId::zero(),
            "SELF.transactionId"
        );
        assert_eq!(ctx.self_box.index, 0, "SELF.index");
        assert!(
            matches!(
                ctx.self_box
                    .additional_registers
                    .get_constant(NonMandatoryRegisterId::R4),
                Ok(None)
            ),
            "SELF.R4 is absent"
        );
        assert_eq!(ctx.inputs.as_vec().len(), 1, "INPUTS is exactly [SELF]");
        assert_eq!(
            **ctx.inputs.first(),
            *ctx.self_box,
            "INPUTS[0] is SELF by value (selfBoxIndex is a value search)"
        );
        assert!(ctx.outputs.is_empty(), "OUTPUTS is empty");
        assert!(ctx.data_inputs.is_none(), "the data inputs are empty");
        assert!(ctx.headers.is_empty(), "there are no headers");
        assert_eq!(ctx.height, 0, "HEIGHT");
        assert_eq!(
            ctx.pre_header.version, 3,
            "preHeader.version is activated + 1"
        );
        assert_eq!(
            ctx.activated_script_version_byte, 2,
            "the activated script version"
        );
        assert_eq!(
            *ctx.pre_header.miner_pk,
            ec_point::generator(),
            "preHeader.minerPk"
        );
        assert_eq!(
            ctx.last_block_utxo_root.digest,
            vec![0u8; 33],
            "the last block's state root"
        );
        assert_eq!(
            ctx.tree_version.get(),
            ErgoTreeVersion::from(3u8),
            "the tree version"
        );
    }

    /// Every context the runner builds carries the reference's cost limit and starts at cost 0;
    /// the full-context one is read off a captured envelope (`testdata/`, SANTA's
    /// `vectors/eval/v6/captured/h210129-6c51f301c418.json`).
    #[test]
    fn every_context_carries_the_cost_limit() {
        assert_eq!(EVAL_JIT_COST_LIMIT, 10_000_000);
        let want = Some(EVAL_JIT_COST_LIMIT);
        let tree = fixture_tree();
        let regs = NonMandatoryRegisters::new(vec![(NonMandatoryRegisterId::R4, 1i32.into())])
            .expect("one register");
        for (what, binding) in [
            ("v2", Binding::Var1(Some(var1()))),
            ("v3", Binding::PerInput(Vec::new())),
            ("v4", Binding::SelfRegisters(regs, var1())),
            ("v5", Binding::Extension(ContextExtension::empty())),
        ] {
            let parts = ContextParts::new(binding, &tree)
                .ok()
                .expect("the parts build");
            let ctx = parts.context(3, 3);
            assert_eq!(ctx.jit_cost_limit, want, "{what}");
            assert_eq!(ctx.jit_cost_value(), 0, "{what}");
        }
        let envelope: serde_json::Value =
            serde_json::from_str(include_str!("testdata/fullctx-h210129.json"))
                .expect("the fixture parses");
        let parts = FullContextParts::new(&envelope["entries"][0]["context"], Some(3))
            .ok()
            .expect("the captured context's parts build");
        let full = parts.context(3).ok().expect("the captured context builds");
        assert_eq!(full.jit_cost_limit, want, "v6-fullctx");
        assert_eq!(full.jit_cost_value(), 0, "v6-fullctx");
    }

    /// No spending input carries a context extension, except in v3, which carries one per entry
    /// extension; v1/v2, v4 and v5 bind into SELF's own.
    #[test]
    fn only_v3_binds_per_input_extensions() {
        let tree = fixture_tree();
        let no_input_extension = |parts: &ContextParts, what: &str| {
            for i in 0..3usize {
                assert!(
                    parts
                        .context(3, 2)
                        .extension_provider
                        .context_extension(i)
                        .is_none(),
                    "{what}: input {i} carries no extension"
                );
            }
        };

        let v2 = ContextParts::new(Binding::Var1(Some(var1())), &tree)
            .ok()
            .expect("the parts build");
        assert!(
            v2.context(3, 2).extension.values.contains_key(&1u8),
            "v2: SELF's var 1"
        );
        no_input_extension(&v2, "v2");

        let regs = NonMandatoryRegisters::new(vec![(NonMandatoryRegisterId::R4, 1i32.into())])
            .expect("one register");
        let v4 = ContextParts::new(Binding::SelfRegisters(regs, var1()), &tree)
            .ok()
            .expect("the parts build");
        let ctx = v4.context(3, 2);
        assert!(ctx.extension.values.contains_key(&1u8), "v4: SELF's var 1");
        assert!(
            matches!(
                ctx.self_box
                    .additional_registers
                    .get_constant(NonMandatoryRegisterId::R4),
                Ok(Some(_))
            ),
            "v4: SELF carries the entry's R4"
        );
        no_input_extension(&v4, "v4");

        let mut ext = ContextExtension::empty();
        ext.values.insert(0u8, var1());
        let v5 = ContextParts::new(Binding::Extension(ext), &tree)
            .ok()
            .expect("the parts build");
        assert!(
            v5.context(3, 2).extension.values.contains_key(&0u8),
            "v5: verbatim"
        );
        no_input_extension(&v5, "v5");

        let mut inp = ContextExtension::empty();
        inp.values.insert(1u8, var1());
        let v3 = ContextParts::new(Binding::PerInput(vec![inp]), &tree)
            .ok()
            .expect("the parts build");
        let ctx = v3.context(3, 2);
        assert!(
            ctx.extension.values.is_empty(),
            "v3: SELF's own extension is empty"
        );
        assert!(
            ctx.extension_provider.context_extension(0).is_some(),
            "v3: input 0's"
        );
        assert!(
            ctx.extension_provider.context_extension(1).is_none(),
            "v3: only input 0"
        );
    }

    /// A SELF that cannot be built is a context this runner cannot represent: the entry is
    /// not-implemented, never evaluated on a stand-in box. Here the root is a SigmaProp, so SELF
    /// holds the parse, and the parse holds a tuple type of one item, which has no written form
    /// (as one of no item has none): `ErgoBox::new` writes the box for its id.
    #[test]
    fn a_self_that_cannot_be_built_is_not_implemented() {
        // sigmaProp(c._1 == 1), c a constant of type (Int)
        let tree = crate::hex_to_bytes("0b0ad1938c60010402010402").expect("hex");
        let input = serde_json::json!({"kind": "Int", "value": 0});
        let out = run_entry(&tree, Some(&input), None, None, None, 3, 3);
        assert!(
            matches!(out, Outcome::NotImplemented),
            "got {}",
            outcome_name(&out)
        );
    }

    /// The tree SELF holds for an entry's bytes, and whether it is kept as bytes.
    fn self_tree(hex: &str) -> (ErgoTree, bool) {
        let bytes = crate::hex_to_bytes(hex).expect("hex");
        let parsed = ErgoTree::sigma_parse_bytes_lenient(&bytes).expect("the lenient parse");
        assert!(matches!(parsed, ErgoTree::Parsed(_)), "{hex}: the parse");
        let held = self_tree_of(&bytes, &parsed);
        let kept_as_bytes = matches!(held, ErgoTree::Unparsed { .. });
        (held, kept_as_bytes)
    }

    /// SELF holds its tree as the strict reader keeps it, otherwise the parse: the rows of the
    /// contract's SELF-tree note. `9a04020402` is `1 + 1`, an Int root that can be written;
    /// `b160010402` is `SizeOf((1))`, an Int root whose parse cannot. The root's type is the
    /// root's own: a function whose body is a SigmaProp is not a SigmaProp root.
    #[test]
    fn self_holds_its_tree_as_a_node_holds_it() {
        // size-flagged, the root not a SigmaProp: a node keeps it as its bytes
        assert!(self_tree("0b059a04020402").1, "sized, an Int root");
        assert!(
            self_tree("0b05b160010402").1,
            "sized, an Int root, unwritable"
        );
        assert!(self_tree("08059a04020402").1, "sized at version 0");
        assert!(self_tree("0b0ad9010104d19304020402").1, "a function root");
        // the size field claims more bytes than there are: a node cannot read it
        assert!(!self_tree("0b069a04020402").1, "the size lies long");
        assert!(!self_tree("0b2a9a04020402").1, "the size lies long by 37");
        // unsized, the root not a SigmaProp: a node refuses the box
        assert!(!self_tree("009a04020402").1, "unsized, an Int root");
        // a SigmaProp root: a node parses it
        assert!(!self_tree("0b03d10101").1, "sized, a SigmaProp root");
        assert!(!self_tree("00d10101").1, "unsized, a SigmaProp root");
    }

    /// A tree kept as bytes is the bytes its size field spans: a byte after the tree is not SELF's,
    /// and a size field that cuts the body short cuts SELF's tree there.
    #[test]
    fn a_tree_kept_as_bytes_is_the_span_of_its_size_field() {
        let span = |hex: &str| self_tree(hex).0.bytes().expect("the kept bytes");
        let whole = crate::hex_to_bytes("0b059a04020402").expect("hex");
        assert_eq!(span("0b059a04020402"), whole);
        assert_eq!(span("0b059a0402040205"), whole, "a trailing byte");
        assert_eq!(
            span("0b049a04020402"),
            crate::hex_to_bytes("0b049a040204").expect("hex"),
            "a size one short"
        );
    }

    /// For a tree the strict reader keeps as bytes, SELF IS the box the library reads from the
    /// box's bytes (what a node built on it holds): the same id and the same bytes, also where the
    /// parse has no written form. The box as written: value 1,000,000, the tree, creation height
    /// 0, no tokens, no registers, a zero transaction id, index 0.
    #[test]
    fn self_is_the_box_the_library_reads_from_its_bytes() {
        for tree_hex in ["0b059a04020402", "0b05b160010402", "0b03b16000"] {
            let (held, kept_as_bytes) = self_tree(tree_hex);
            assert!(kept_as_bytes, "{tree_hex}");
            let parts = ContextParts::new(Binding::Var1(None), &held)
                .ok()
                .expect("the parts build");
            let mut box_bytes = vec![0xc0, 0x84, 0x3d];
            box_bytes.extend(crate::hex_to_bytes(tree_hex).expect("hex"));
            box_bytes.extend([0u8; 3 + 32 + 1]);
            let read = ErgoBox::sigma_parse_bytes(&box_bytes).expect("the box reads");
            assert_eq!(parts.self_box.box_id(), read.box_id(), "{tree_hex}: the id");
            assert_eq!(
                parts.self_box.bytes().expect("SELF's bytes"),
                box_bytes,
                "{tree_hex}: the bytes"
            );
        }
    }

    /// Entries whose tree holds a tuple type of zero or one item under a root that is not a
    /// SigmaProp are answered: SELF holds the bytes, the parse is evaluated. The answers are the
    /// reference's (SANTA `Tuple.non_pair_value_ops` entries 45 and 46, `SelectField.non_pair`
    /// entry 0, `Self.node_form_tree_span` entry 11).
    #[test]
    fn an_entry_whose_parse_cannot_be_written_is_answered() {
        let input = serde_json::json!({"kind": "Int", "value": 0});
        let run = |tree_hex: &str, tree_version: u8, activated: u8| {
            let tree = crate::hex_to_bytes(tree_hex).expect("hex");
            run_entry(
                &tree,
                Some(&input),
                None,
                None,
                None,
                tree_version,
                activated,
            )
            .to_json()
        };
        let int = |value: i64, cost: u64| {
            let value = serde_json::json!({"kind": "Int", "value": value});
            serde_json::json!({"value": value, "cost": cost, "error": null})
        };
        assert_eq!(run("0b03b16000", 3, 3), int(0, 19), "SizeOf(())");
        assert_eq!(run("0b05b160010402", 3, 3), int(1, 19), "SizeOf((1))");
        assert_eq!(
            run("08068c6001040a01", 0, 2)["error"],
            "errored",
            "SelectField((5), 1): the evaluator's refusal"
        );
        // a node cannot read this one (its size field lies long): SELF holds the parse, and the
        // entry is evaluated, not refused
        assert_eq!(
            run("0b069a04020402", 3, 3),
            int(2, 25),
            "1 + 1, the size lies long"
        );
    }

    /// The first entry of the captured full-context vector: its tree and its context envelope.
    fn captured_entry() -> (Vec<u8>, serde_json::Value) {
        let vector: serde_json::Value =
            serde_json::from_str(include_str!("testdata/fullctx-h210129.json"))
                .expect("the fixture parses");
        let entry = &vector["entries"][0];
        let tree =
            crate::hex_to_bytes(entry["tree_bytes_hex"].as_str().expect("a tree")).expect("hex");
        (tree, entry["context"].clone())
    }

    /// A `self_index` that names no input (past the last one, or there is none) is the envelope's
    /// defect: `panicked` with the cause, never a Rust panic.
    #[test]
    fn a_full_context_self_index_out_of_range_is_panicked() {
        let (tree, context) = captured_entry();
        let mut past_the_end = context.clone();
        past_the_end["self_index"] = context["inputs"].as_array().expect("inputs").len().into();
        let mut no_inputs = context;
        no_inputs["inputs"] = serde_json::json!([]);
        for (what, ctx) in [("past the end", past_the_end), ("no inputs", no_inputs)] {
            match run_entry_fullctx(&tree, &ctx, 3, Some(3)) {
                Outcome::Panicked { note } => {
                    assert!(note.contains("self_index"), "{what}: {note}")
                }
                other => panic!("{what}: expected Panicked, got {}", outcome_name(&other)),
            }
        }
    }

    /// The activated script version is the entry's; an entry without one takes the envelope's
    /// block version minus 1.
    #[test]
    fn a_full_context_without_an_activated_version_takes_the_block_versions() {
        let (_, context) = captured_entry();
        let parts = FullContextParts::new(&context, None)
            .ok()
            .expect("the captured context's parts build");
        assert_eq!(
            parts.activated_script_version_byte,
            parts.pre_header.version as i8 - 1
        );
        let parts = FullContextParts::new(&context, Some(1))
            .ok()
            .expect("the captured context's parts build");
        assert_eq!(parts.activated_script_version_byte, 1);
    }

    /// A v4 entry's SELF is ONE box: built with the entry's registers, and INPUTS' only box (the
    /// reference's `boxesToSpend = [selfBox]`). The five answers are the reference's (sigma-state
    /// 6.0.7) for a SELF holding R4 = Int 1: its id is the id of the box WITH the register.
    #[test]
    fn a_v4_self_is_the_one_input_box_with_its_registers() {
        let registers = serde_json::json!({"4": {"kind": "Int", "value": 1}});
        let input = serde_json::json!({"kind": "Int", "value": 0});
        let eval = |tree_hex: &str| {
            let tree = crate::hex_to_bytes(tree_hex).expect("hex");
            match run_entry(&tree, Some(&input), None, registers.as_object(), None, 3, 3) {
                Outcome::Success { value, .. } => value,
                other => panic!("{tree_hex}: expected a value, got {}", outcome_name(&other)),
            }
        };
        let bytes = |hex: &str| {
            let items: Vec<serde_json::Value> = crate::hex_to_bytes(hex)
                .expect("hex")
                .into_iter()
                .map(|b| serde_json::json!({"kind": "Byte", "value": b as i8}))
                .collect();
            serde_json::json!({"kind": "Coll", "elem": {"tag": "SByte"}, "items": items})
        };
        let boolean = |b: bool| serde_json::json!({"kind": "Boolean", "value": b});
        assert_eq!(
            eval("0b0a959304020402c5a7c5a7"),
            bytes("12e4ec7a6bdd88b9932716ccae39a7607feb57b435346aeaf658b13d525bb19d"),
            "SELF.id"
        );
        assert_eq!(
            eval("0b12959304020402c5b2a4040000c5b2a4040000"),
            bytes("0feec40512b0c36c2da6a89645985b2a88a36d3f1b16e94b5ff1a301a303a8a1"),
            "INPUTS(0).id"
        );
        assert_eq!(
            eval("0b1495930402040293b2a4040000a793b2a4040000a7"),
            boolean(true),
            "INPUTS(0) == SELF"
        );
        assert_eq!(
            eval("0b0e959304020402db6508fedb6508fe"),
            serde_json::json!({"kind": "Int", "value": 0}),
            "CONTEXT.selfBoxIndex"
        );
        assert_eq!(
            eval("0b18959304020402e6c6b2a40400000404e6c6b2a40400000404"),
            boolean(true),
            "INPUTS(0).R4[Int].isDefined"
        );
    }

    fn outcome_name(o: &Outcome) -> &'static str {
        match o {
            Outcome::Success { .. } => "Success",
            Outcome::Errored => "Errored",
            Outcome::NotImplemented => "NotImplemented",
            Outcome::Panicked { .. } => "Panicked",
        }
    }

    /// A v5 (`extension` envelope) entry whose extension key is not a byte: the entry is
    /// `errored`, not `not-implemented`.
    #[test]
    fn a_v5_extension_with_a_bad_key_is_errored() {
        let tree = crate::hex_to_bytes("000402").expect("hex"); // Int 1
        let ext = serde_json::json!({"abc": {"kind": "Int", "value": 0}});
        let out = run_entry(&tree, None, None, None, ext.as_object(), 3, 3);
        assert!(
            matches!(out, Outcome::Errored),
            "got {}",
            outcome_name(&out)
        );
    }

    /// A v4 (`selfRegisters` envelope) entry whose register key is not a byte: the entry is
    /// `panicked` with the cause (the v4 arm's bad key is the runner's own defect, NOT a
    /// library refusal), the key in the note. Distinguishes from `errored` on the same key.
    #[test]
    fn a_v4_self_registers_with_a_bad_key_is_panicked() {
        let tree = crate::hex_to_bytes("000402").expect("hex"); // Int 1
        let regs = serde_json::json!({"abc": {"kind": "Int", "value": 0}});
        let input = serde_json::json!({"kind": "Int", "value": 0});
        let out = run_entry(&tree, Some(&input), None, regs.as_object(), None, 3, 3);
        match out {
            Outcome::Panicked { note } => {
                assert!(note.contains("v4 selfRegisters: bad key"), "note: {note}",)
            }
            other => panic!("expected Panicked, got {}", outcome_name(&other)),
        }
    }

    /// A v3 entry (both `inputs` AND a top-level `input`) with a top-level `input` whose kind
    /// the runner cannot decode is `panicked`: v3 still decodes `input` first so an envelope
    /// defect does not hide as `success`. Unmutated: input decode fails → `panicked`. A mutant
    /// that only decodes `input` when `inputs` is None would evaluate the tree and succeed.
    #[test]
    fn a_v3_entry_with_a_bogus_top_level_input_is_panicked() {
        let tree = crate::hex_to_bytes("000402").expect("hex"); // Int 1
        let input = serde_json::json!({"kind": "NoSuchKind", "value": 0});
        let inputs = vec![serde_json::json!({})];
        let out = run_entry(&tree, Some(&input), Some(&inputs), None, None, 3, 3);
        match out {
            Outcome::Panicked { note } => assert!(note.contains("input decode"), "note: {note}",),
            other => panic!("expected Panicked, got {}", outcome_name(&other)),
        }
    }

    /// `eval::run_entry` decodes the binding BEFORE parsing the tree: a bad tree plus a
    /// bogus input gives `panicked` (the input decode's error), not `errored` (the tree
    /// parse's). An implementation that tree-parses first would return `errored` here.
    #[test]
    fn eval_binding_decode_comes_before_tree_parse() {
        // tree bytes: header 00 then opcode ff with no node behind it — the lenient parse fails.
        let tree = crate::hex_to_bytes("00ff").expect("hex");
        let input = serde_json::json!({"kind": "NoSuchKind", "value": 0});
        let out = run_entry(&tree, Some(&input), None, None, None, 3, 3);
        match out {
            Outcome::Panicked { note } => {
                assert!(note.contains("input decode"), "note: {note}")
            }
            other => panic!("expected Panicked, got {}", outcome_name(&other)),
        }
    }

    /// `eval::run_entry_fullctx` builds the context parts BEFORE parsing the tree: a bad tree
    /// plus a malformed envelope gives `panicked` (the envelope defect), not `errored` (the
    /// tree parse's). An implementation that tree-parses first would return `errored` here.
    #[test]
    fn fullctx_parts_build_comes_before_tree_parse() {
        let (_, mut context) = captured_entry();
        // remove `height`: the envelope becomes malformed; `FullContextParts::new` returns
        // `FullCtxError::Malformed`, which the runner maps to `panicked`.
        context.as_object_mut().expect("object").remove("height");
        let bad_tree = crate::hex_to_bytes("00ff").expect("hex");
        let out = run_entry_fullctx(&bad_tree, &context, 3, Some(3));
        match out {
            Outcome::Panicked { note } => {
                assert!(note.contains("height"), "note: {note}")
            }
            other => panic!("expected Panicked, got {}", outcome_name(&other)),
        }
    }

    /// The full-context reconstruction reads the per-input `extension` values at the activated
    /// version the caller passes, not the block version derived from the pre-header. When the
    /// entry carries `activated=Some(1)` and an input extension value whose type (SFunc) only
    /// parses at reader tree_version >= V3, the reader REFUSES — the parts build fails. A mutant
    /// that reads extensions at `pre_header.version - 1` regardless of the explicit activated
    /// (block version is 4, so read_activated would be 3) would ACCEPT the same bytes.
    #[test]
    fn full_context_reads_extensions_at_the_activated_version() {
        let (_, context) = captured_entry();
        // An empty `Coll[SFunc(SInt => SInt)]`: `83 00 70 01 04 04 00`. The function type's code
        // is read from tree version 3 only.
        let value_hex = "83007001040400".to_string();
        let mut ctx = context;
        // the envelope's input_extensions[0] carries one variable; apply to self_index=0.
        ctx["input_extensions"] = serde_json::json!([{
            "0": {"kind": "Value", "bytes_hex": value_hex}
        }]);
        // Pre-header's block version is 4 (h210129 testnet). activated=Some(1) differs from
        // `phf.version - 1` = 3. Unmutated reads at 1 (refused); the mutant that always uses
        // `phf.version - 1` would read at 3 (accepted) — two different outcomes for the parts
        // build.
        let parts = FullContextParts::new(&ctx, Some(1));
        assert!(
            parts.is_err(),
            "at activated=1 the SFunc value must be refused (reader tree_version 1 < V3)"
        );
    }

    /// When no activated version is given, extensions read at `pre_header.version - 1`.
    /// A mutant that falls back to `pre_header.version` itself would read them one version
    /// higher. With pre_header.version=3 the gap crosses V3: unmutated reads at tree_v=2
    /// (no SFunc), the mutant at tree_v=3 (SFunc parses). The SFunc Value refusal is the
    /// witness; the mutant would accept the same bytes.
    #[test]
    fn full_context_without_activated_takes_pre_header_version_minus_one() {
        let (_, mut context) = captured_entry();
        // Patch the pre_header's first byte (version) from 04 to 03.
        let ph_hex = context["pre_header_hex"].as_str().expect("hex").to_string();
        assert!(
            ph_hex.starts_with("04"),
            "captured envelope's block version is 4"
        );
        let ph_v3 = format!("03{}", &ph_hex[2..]);
        context["pre_header_hex"] = serde_json::json!(ph_v3);
        // Coll[SFunc(SInt => SInt)] empty: parses at tree_v>=V3, not below.
        let value_hex = "83007001040400".to_string();
        context["input_extensions"] = serde_json::json!([{
            "0": {"kind": "Value", "bytes_hex": value_hex}
        }]);
        let parts = FullContextParts::new(&context, None);
        assert!(
            parts.is_err(),
            "with pre_header.version=3 and no activated, read_activated = 2 and the SFunc value \
             must be refused"
        );
    }
}
