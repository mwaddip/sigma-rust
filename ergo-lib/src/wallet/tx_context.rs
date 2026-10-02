//! Transaction context

use alloc::vec::Vec;
use hashbrown::hash_map::Entry;
use hashbrown::HashMap;

use crate::chain::ergo_state_context::ErgoStateContext;
use crate::chain::parameters::Parameters;
use crate::chain::transaction::ergo_transaction::{ErgoTransaction, TxValidationError};
use crate::chain::transaction::storage_rent::{
    storage_rent_verdict, StorageRentVerdict, STORAGE_CONTRACT_COST,
};
use crate::chain::transaction::{Transaction, TransactionError};
use crate::ergotree_ir::chain::ergo_box::BoxId;
use ergotree_interpreter::eval::env::Env;
use ergotree_interpreter::eval::reduce_to_crypto;
use ergotree_interpreter::eval::ReductionDiagnosticInfo;
use ergotree_interpreter::sigma_protocol::crypto_cost::estimate_crypto_cost;
use ergotree_interpreter::sigma_protocol::verifier::{
    check_soft_fork_condition, verify_signature, VerificationResult, VerifierError,
};
use ergotree_ir::chain::context::{CostLimitExceeded, TxIoVec};
use ergotree_ir::chain::ergo_box::{BoxTokens, ErgoBox};
use ergotree_ir::chain::token::{TokenAmount, TokenId};
use thiserror::Error;

use super::signing::{make_context, update_context};

/// Transaction and an additional info required for signing or verification
#[derive(PartialEq, Eq, Debug, Clone)]
pub struct TransactionContext<T: ErgoTransaction> {
    /// Unsigned transaction to sign
    pub spending_tx: T,
    /// Boxes corresponding to [`crate::chain::transaction::unsigned::UnsignedTransaction::inputs`]
    boxes_to_spend: TxIoVec<ErgoBox>,
    /// Boxes corresponding to [`crate::chain::transaction::unsigned::UnsignedTransaction::data_inputs`]
    pub(crate) data_boxes: Option<TxIoVec<ErgoBox>>,
    /// Stores the location of each BoxId in [`Self::boxes_to_spend`]
    box_index: HashMap<BoxId, u16>,
}

impl<T: ErgoTransaction> TransactionContext<T> {
    /// New TransactionContext
    pub fn new(
        spending_tx: T,
        boxes_to_spend: Vec<ErgoBox>,
        data_boxes: Vec<ErgoBox>,
    ) -> Result<Self, TransactionContextError> {
        let boxes_to_spend = TxIoVec::from_vec(boxes_to_spend).map_err(|e| match e {
            bounded_vec::BoundedVecOutOfBounds::LowerBoundError { .. } => {
                TransactionContextError::NoInputBoxes
            }
            bounded_vec::BoundedVecOutOfBounds::UpperBoundError { got, .. } => {
                TransactionContextError::TooManyInputBoxes(got)
            }
        })?;
        let data_boxes_len = data_boxes.len();
        let data_boxes = if !data_boxes.is_empty() {
            Some(
                TxIoVec::from_vec(data_boxes)
                    .map_err(|_| TransactionContextError::TooManyDataInputBoxes(data_boxes_len))?,
            )
        } else {
            None
        };

        let box_index: HashMap<BoxId, u16> = boxes_to_spend
            .iter()
            .enumerate()
            .map(|(i, b)| (b.box_id(), i as u16))
            .collect();
        for (i, unsigned_input) in spending_tx.inputs_ids().enumerate() {
            if !box_index.contains_key(&unsigned_input) {
                return Err(TransactionContextError::InputBoxNotFound(i));
            }
        }

        if let Some(data_inputs) = spending_tx.data_inputs().as_ref() {
            if let Some(data_boxes) = data_boxes.as_ref() {
                let data_box_index: HashMap<BoxId, u16> = data_boxes
                    .iter()
                    .enumerate()
                    .map(|(i, b)| (b.box_id(), i as u16))
                    .collect();
                for (i, data_input) in data_inputs.iter().enumerate() {
                    if !data_box_index.contains_key(&data_input.box_id) {
                        return Err(TransactionContextError::DataInputBoxNotFound(i));
                    }
                }
            } else {
                return Err(TransactionContextError::DataInputBoxNotFound(0));
            }
        }
        Ok(TransactionContext {
            spending_tx,
            boxes_to_spend,
            data_boxes,
            box_index,
        })
    }

    /// Returns box with given id, if it exists.
    pub fn get_input_box(&self, box_id: &BoxId) -> Option<&ErgoBox> {
        self.box_index
            .get(box_id)
            .and_then(|&idx| self.boxes_to_spend.get(idx as usize))
    }
}

/// Fixed JIT cost (in block-cost units) charged once per transaction for interpreter
/// initialization, matching Scala's `interpreterInitCost`.
pub(crate) const INTERPRETER_INIT_COST: u64 = 10_000;

/// Count (total_entries, distinct_token_count) across the given boxes' token sets.
fn count_tokens(boxes: &[ErgoBox]) -> (u64, u64) {
    let mut total_entries = 0u64;
    let mut distinct: hashbrown::HashSet<TokenId> = hashbrown::HashSet::new();
    for b in boxes {
        for t in b.tokens.iter().flatten() {
            total_entries += 1;
            distinct.insert(t.token_id);
        }
    }
    (total_entries, distinct.len() as u64)
}

/// Per-tx init cost in block-cost units. Port of PR 846's `compute_tx_init_cost`,
/// which was validated against 19,549 mainnet txs and matches Scala's
/// `ErgoTransaction.computeInitiationCost`.
fn compute_tx_init_cost(
    tx: &Transaction,
    boxes_to_spend: &[ErgoBox],
    parameters: &Parameters,
) -> u64 {
    let n_data_inputs = tx.data_inputs.as_ref().map_or(0, |d| d.len()) as u64;
    let structural = INTERPRETER_INIT_COST
        + tx.inputs.len() as u64 * parameters.input_cost() as u64
        + n_data_inputs * parameters.data_input_cost() as u64
        + tx.outputs.len() as u64 * parameters.output_cost() as u64;

    let (in_entries, in_distinct) = count_tokens(boxes_to_spend);
    let (out_entries, out_distinct) = count_tokens(tx.outputs.as_slice());
    let token_cost = (in_entries + out_entries + in_distinct + out_distinct)
        * parameters.token_access_cost() as u64;

    structural + token_cost
}

impl TransactionContext<Transaction> {
    /// Verify transaction using blockchain parameters.
    /// Returns the total accumulated script evaluation cost (in block cost units).
    ///
    /// # Panics
    /// If the state context's parameters table lacks an entry this reads, the block version
    /// among them: `Parameters`' accessors index the table.
    // This is based on validateStateful() in Ergo: https://github.com/ergoplatform/ergo/blob/48239ef98ced06617dc21a0eee5670235e362933/ergo-core/src/main/scala/org/ergoplatform/modifiers/mempool/ErgoTransaction.scala#L357
    pub fn validate(&self, state_context: &ErgoStateContext) -> Result<u64, TxValidationError> {
        // Check that input sum does not overflow. The reference implementation
        // sums with Math.addExact over longs (validateStateful), so every
        // addition is checked — a plain sum would panic in debug builds and
        // wrap in release builds before any bound check could fire.
        let input_sum = self
            .boxes_to_spend
            .iter()
            .try_fold(0i64, |a, b| a.checked_add(b.value.as_i64()))
            .ok_or(TxValidationError::InputSumOverflow)?;
        // Check that output sum does not overflow and is equal to ERG amount in inputs
        let output_sum = self
            .spending_tx
            .outputs
            .iter()
            .try_fold(0i64, |a, b| a.checked_add(b.value.as_i64()))
            .ok_or(TxValidationError::OutputSumOverflow)?;
        if input_sum != output_sum {
            return Err(TxValidationError::ErgPreservationError(
                input_sum as u64,
                output_sum as u64,
            ));
        }

        // Monotonic Box creation happens after v3: ergo's `blockVersion <= HardeningVersion`
        // (`ErgoTransaction.scala:379-384`), on the voted parameters' block version, a signed
        // byte
        let max_creation_height = if state_context.block_version() <= 2 {
            0
        } else {
            #[allow(clippy::unwrap_used)] // Unwrap is valid here since inputs can not be empty
            self.boxes_to_spend
                .iter()
                .map(|b| b.creation_height)
                .max()
                .unwrap()
        };
        // Check that outputs are not dust and aren't created in future
        for output in &self.spending_tx.outputs {
            verify_output(state_context, output, max_creation_height)?;
        }

        let in_assets = extract_assets(self.boxes_to_spend.iter().map(|b| &b.tokens))?;
        let out_assets = extract_assets(self.spending_tx.outputs.iter().map(|b| &b.tokens))?;
        verify_assets(self.spending_tx.inputs_ids(), in_assets, out_assets)?;
        // Verify input proofs with cost tracking.
        // This is usually the most expensive check so it's done last.
        let bytes_to_sign = self.spending_tx.bytes_to_sign()?;
        let max_block_cost = state_context.parameters.max_block_cost() as u64;
        let mut context = make_context(state_context, self, 0)?;

        // Per-tx init cost (gaps S1 + S2): interpreter baseline + per-input,
        // per-data-input, per-output, per-token structural costs — all front-loaded,
        // mirroring the JVM's `initialCost` in `ErgoTransaction.validateStateful`
        // (interpreterInitCost + inputs·inputCost + dataInputs·dataInputCost +
        // outputs·outputCost, plus the token-access cost). Reject upfront if it
        // already exceeds MaxBlockCost (the JVM's `maxCost >= startCost` gate) — we
        // can't honestly blame any specific input.
        let init_cost_block = compute_tx_init_cost(
            &self.spending_tx,
            self.boxes_to_spend.as_slice(),
            &state_context.parameters,
        );
        if init_cost_block > max_block_cost {
            return Err(TxValidationError::InitCostExceeded(
                init_cost_block.saturating_mul(10),
            ));
        }

        // Running tx cost in BLOCK units, mirroring the JVM's `currentTxCost`. The
        // cost limit is enforced in block units — NOT against the raw JIT accumulator.
        // The JVM floors each input's JIT cost to block cost (`JitCost.toBlockCost`,
        // i.e. `/ 10`) per input and per component (script eval and crypto verify are
        // floored separately) before accumulating, discarding each input's
        // sub-block-unit (mod-10) remainder. A tx whose floored block total equals
        // MaxBlockCost is therefore accepted even though the un-floored JIT sum exceeds
        // MaxBlockCost * 10 — the exact-fit boundary we must admit.
        let mut total_cost: u64 = init_cost_block;
        for input_idx in 0..self.spending_tx.inputs.len() {
            update_context(&mut context, self, input_idx)?;
            let input = self
                .spending_tx
                .inputs
                .get(input_idx)
                .ok_or(TransactionContextError::InputBoxNotFound(input_idx))?;
            let input_box = self
                .get_input_box(&input.box_id)
                .ok_or(TransactionContextError::InputBoxNotFound(input_idx))?;

            // Storage-rent branch of `ErgoInterpreter.verify`, costed as in
            // `ErgoTransaction.verifyInput` (`ErgoTransaction.scala:110-160`): the
            // verdict first (`txScriptValidation`), then the running cost against the
            // block limit (`bsBlockTransactionsCost`).
            match storage_rent_verdict(&input.spending_proof.proof, state_context, &context) {
                StorageRentVerdict::Verdict(false) => {
                    return Err(TxValidationError::ReducedToFalse(
                        input_idx,
                        VerificationResult {
                            result: false,
                            cost: STORAGE_CONTRACT_COST,
                            diag: ReductionDiagnosticInfo {
                                env: Env::empty(),
                                pretty_printed_expr: None,
                            },
                        },
                    ));
                }
                StorageRentVerdict::Verdict(true) => {
                    let remaining_block = max_block_cost.saturating_sub(total_cost);
                    if STORAGE_CONTRACT_COST > remaining_block {
                        return Err(TxValidationError::VerifierError(
                            input_idx,
                            VerifierError::EvalError(
                                CostLimitExceeded(remaining_block.saturating_mul(10)).into(),
                            ),
                        ));
                    }
                    total_cost += STORAGE_CONTRACT_COST;
                    continue;
                }
                StorageRentVerdict::NotApplicable => {}
            }

            // `Interpreter.verify` runs `checkSoftForkCondition` before the reduction
            // (sigmastate v6.0.6 `Interpreter.scala:362`), after the rent path. A tree above
            // the activated version fails the input. Under an activated version above the
            // supported one, a tree this interpreter cannot read is accepted unverified at
            // the input's init cost (`:317`), which is 0 here.
            if check_soft_fork_condition(&input_box.ergo_tree, &context)
                .map_err(|e| TxValidationError::VerifierError(input_idx, e.into()))?
            {
                continue;
            }

            // The JVM verifies each input with a FRESH interpreter whose
            // `costLimit = maxCost - currentTxCost` (the remaining BLOCK budget,
            // `initCost = 0`) — `ErgoTransaction.verifyInput`. Mirror it: reset the
            // accumulator and cap it at the remaining budget × 10 (the evaluator's
            // `CostAccumulator` works in the JitCost scale). The remaining budget
            // shrinks as inputs accrue, so splitting expensive work across many inputs
            // still cannot exceed MaxBlockCost in aggregate (the S4 concern) — while
            // per-input flooring keeps the exact-fit boundary byte-identical to the JVM.
            let remaining_block = max_block_cost.saturating_sub(total_cost);
            context.reset_jit_cost();
            context.jit_cost_limit = Some(remaining_block.saturating_mul(10));

            // Reduce the ErgoTree to a SigmaBoolean. The per-input JIT cap above
            // mirrors the JVM evaluator's `CostAccumulator` (throws once accumulated
            // JIT exceeds `remaining_block * 10`); `reduction.cost` is the floored
            // block cost (`(jit_after - 0) / 10`).
            let reduction = reduce_to_crypto(&input_box.ergo_tree, &context)
                .map_err(|e| TxValidationError::VerifierError(input_idx, e.into()))?;

            // Sigma-protocol verification cost, floored to block units separately
            // (JVM `addCryptoCost`: `estimateCryptoVerifyCost(...).toBlockCost`), then
            // the combined per-input block cost is checked against the remaining block
            // budget (JVM `addCostChecked` / `bsBlockTransactionsCost`, block units).
            let crypto_cost_block = estimate_crypto_cost(&reduction.sigma_prop) / 10;
            let input_cost_block = reduction.cost + crypto_cost_block;
            if input_cost_block > remaining_block {
                return Err(TxValidationError::VerifierError(
                    input_idx,
                    VerifierError::EvalError(
                        CostLimitExceeded(remaining_block.saturating_mul(10)).into(),
                    ),
                ));
            }

            let verified = verify_signature(
                reduction.sigma_prop.clone(),
                &bytes_to_sign,
                input.spending_proof.proof.as_ref(),
            )
            .map_err(|e| TxValidationError::VerifierError(input_idx, e))?;
            if !verified {
                return Err(TxValidationError::ReducedToFalse(
                    input_idx,
                    VerificationResult {
                        result: false,
                        cost: reduction.cost,
                        diag: reduction.diag,
                    },
                ));
            }

            total_cost += input_cost_block;
        }
        Ok(total_cost)
    }
}

fn verify_output(
    state_context: &ErgoStateContext,
    output: &ErgoBox,
    max_creation_height: u32,
) -> Result<(), TxValidationError> {
    // `verifyOutput` measures `ErgoBox.bytes`, which ergo writes under its default version
    // context (`ErgoTransaction.scala:171-175`, `BoxUtils.scala:41`)
    let box_size = output.bytes()?.len() as u64;
    let script_size = output.script_bytes()?.len();
    // the voted parameters' (`ErgoTransaction.scala:168`)
    let block_version = state_context.block_version();
    // Check that output is not dust
    let minimum_value = box_size * state_context.parameters.min_value_per_byte() as u64;
    if *output.value.as_u64() < minimum_value {
        return Err(TxValidationError::DustOutput(
            output.box_id(),
            output.value,
            minimum_value,
        ));
    }
    // Check that height does not exceed maximum height. Note that heights can be potentially negative in V1
    if output.creation_height as i32 > state_context.pre_header.height as i32 {
        return Err(TxValidationError::InvalidHeightError(
            output.creation_height,
        ));
    }
    if output.creation_height < max_creation_height {
        return Err(TxValidationError::MonotonicHeightError(
            output.creation_height,
            max_creation_height,
        ));
    }
    // Negative output heights were allowed in V1. sigma-rust always stores heights as unsigned integers
    if block_version != 1 && output.creation_height & (1 << 31) != 0 {
        return Err(TxValidationError::NegativeHeight);
    }
    if box_size as usize > ErgoBox::MAX_BOX_SIZE {
        return Err(TxValidationError::BoxSizeExceeded(box_size as usize));
    }
    if script_size > ErgoBox::MAX_SCRIPT_SIZE {
        return Err(TxValidationError::ScriptSizeExceeded(script_size));
    }
    Ok(())
}

// Extract all of the assets in a collection of boxes for transaction validation
fn extract_assets<'a, I: Iterator<Item = &'a Option<BoxTokens>>>(
    mut boxes: I,
) -> Result<HashMap<TokenId, TokenAmount>, TxValidationError> {
    boxes.try_fold(
        HashMap::new(),
        |mut asset_map: HashMap<TokenId, TokenAmount>, tokens| {
            tokens
                .as_ref()
                .into_iter()
                .flatten()
                .try_for_each(|token| {
                    match asset_map.entry(token.token_id) {
                        Entry::Occupied(mut occ) => {
                            *occ.get_mut() = occ.get().checked_add(&token.amount)?;
                        }
                        Entry::Vacant(vac) => {
                            vac.insert(token.amount);
                        }
                    }
                    Ok::<(), TxValidationError>(())
                })?;
            Ok(asset_map)
        },
    )
}

fn verify_assets(
    mut inputs: impl Iterator<Item = BoxId>,
    in_assets: HashMap<TokenId, TokenAmount>,
    out_assets: HashMap<TokenId, TokenAmount>,
) -> Result<(), TxValidationError> {
    // If this transaction mints a new token, it's token ID must be the ID of the first box being spent
    #[allow(clippy::unwrap_used)]
    // Inputs size is already validated so it must be of atleast size 1
    let new_token_id: TokenId = inputs.next().unwrap().into();
    for (&out_token_id, &out_amount) in &out_assets {
        if let Some(&in_amount) = in_assets.get(&out_token_id) {
            // Check that Transaction is not creating tokens out of thin air
            if in_amount < out_amount {
                return Err(TxValidationError::TokenPreservationError {
                    token_id: out_token_id,
                    in_amount: in_amount.into(),
                    out_amount: out_amount.into(),
                    new_token_id,
                });
            }
        } else if out_token_id != new_token_id {
            //minting a new token. Token amount checks are handled by the TokenAmount newtype and not needed here
            return Err(TxValidationError::TokenPreservationError {
                token_id: out_token_id,
                in_amount: 0,
                out_amount: out_amount.into(),
                new_token_id,
            });
        }
    }
    Ok(())
}

/// Transaction context errors
#[derive(Error, Debug)]
pub enum TransactionContextError {
    /// Transaction error
    #[error("Transaction error: {0}")]
    TransactionError(#[from] TransactionError),
    /// No input boxes (boxes_to_spend is empty)
    #[error("No input boxes")]
    NoInputBoxes,
    /// Too many input boxes
    #[error("Too many input boxes: {0}")]
    TooManyInputBoxes(usize),
    /// Input box not found
    #[error("Input box not found: {0}")]
    InputBoxNotFound(usize),
    /// Too many data input boxes
    #[error("Too many data input boxes: {0}")]
    TooManyDataInputBoxes(usize),
    /// Data input box not found
    #[error("Data input box not found: {0}")]
    DataInputBoxNotFound(usize),
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod test {
    use std::collections::HashMap;

    use alloc::vec::Vec;
    use ergotree_interpreter::sigma_protocol::prover::ProofBytes;
    use ergotree_ir::chain::context::TxIoVec;
    use ergotree_ir::chain::context_extension::ContextExtension;
    use ergotree_ir::chain::ergo_box::arbitrary::ArbBoxParameters;
    use ergotree_ir::chain::ergo_box::box_value::BoxValue;
    use ergotree_ir::chain::ergo_box::{
        BoxTokens, ErgoBox, ErgoBoxCandidate, NonMandatoryRegisters,
    };
    use ergotree_ir::chain::token::arbitrary::ArbTokenIdParam;
    use ergotree_ir::chain::token::{Token, TokenAmount, TokenId};
    use ergotree_ir::ergo_tree::{ErgoTree, ErgoTreeHeader};
    use ergotree_ir::mir::constant::{Constant, Literal};
    use ergotree_ir::mir::expr::Expr;
    use proptest::prelude::*;
    use proptest::strategy::Strategy;
    use proptest::test_runner::TestRng;
    use sigma_test_util::{force_any_val, force_any_val_with};

    use crate::chain::ergo_state_context::ErgoStateContext;
    use crate::chain::parameters::Parameters;
    use crate::chain::transaction::ergo_transaction::{ErgoTransaction, TxValidationError};
    use crate::chain::transaction::prover_result::ProverResult;
    use crate::chain::transaction::unsigned::UnsignedTransaction;
    use crate::chain::transaction::{Input, Transaction, TxId, UnsignedInput};
    use crate::wallet::Wallet;

    use super::TransactionContext;

    // Disperse token_count tokens across inputs
    fn disperse_tokens(inputs: u16, token_count: u8) -> Vec<Option<BoxTokens>> {
        let mut token_distribution = vec![vec![]; inputs as usize];
        for i in 0..token_count {
            let token = force_any_val_with::<Token>(ArbTokenIdParam::Arbitrary);
            token_distribution[(i as usize) % inputs as usize].push(token);
        }
        token_distribution
            .into_iter()
            .map(BoxTokens::from_vec)
            .map(Result::ok)
            .collect()
    }
    fn gen_boxes(
        min_tokens: u8,
        max_tokens: u8,
        min_inputs: u16,
        max_inputs: u16,
        ergotree_gen: impl Strategy<Value = ErgoTree>,
        height_gen: Option<BoxedStrategy<u32>>,
    ) -> impl Strategy<Value = Vec<ErgoBox>> {
        (
            min_inputs..=max_inputs,
            min_tokens..=max_tokens,
            ergotree_gen,
            height_gen.clone().unwrap_or_else(|| Just(0).boxed()),
        )
            .prop_flat_map(
                |(input_count, assets_count, proposition, creation_height)| {
                    let tokens = disperse_tokens(input_count, assets_count);
                    tokens
                        .into_iter()
                        .map(move |tokens| {
                            let box_params = ArbBoxParameters {
                                value_range: (10000000..100000000).into(),
                                ergo_tree: Just(proposition.clone()).boxed(),
                                creation_height: Just(creation_height).boxed(),
                                tokens: Just(tokens).boxed(),
                                ..Default::default()
                            };
                            ErgoBox::arbitrary_with(box_params)
                        })
                        .collect::<Vec<_>>()
                },
            )
    }
    fn valid_unsigned_transaction_from_boxes(
        mut rng: TestRng,
        boxes: &[ErgoBox],
        issue_new_token: bool,
        output_prop: ErgoTree,
        _data_boxes: &[ErgoBox],
    ) -> UnsignedTransaction {
        let input_sum = boxes.iter().map(|b| *b.value.as_u64()).sum::<u64>();
        assert!(input_sum > *BoxValue::SAFE_USER_MIN.as_u64());

        let mut assets_map: HashMap<TokenId, TokenAmount> = boxes
            .iter()
            .flat_map(|b| b.tokens.clone().into_iter().flatten())
            .map(|token| (token.token_id, token.amount))
            .collect();
        if issue_new_token {
            assets_map.insert(
                boxes[0].box_id().into(),
                rng.gen_range(1..=i64::MAX as u64).try_into().unwrap(),
            );
        }

        let parameters = Parameters::default();
        let sufficient_amount =
            ErgoBox::MAX_BOX_SIZE as u64 * parameters.min_value_per_byte() as u64;
        let max_outputs = core::cmp::min(i16::MAX as u16, (input_sum / sufficient_amount) as u16);
        let outputs = core::cmp::min(
            max_outputs,
            core::cmp::max(boxes.len() + 1, rng.gen_range(0..boxes.len() * 2)) as u16,
        );
        assert!(outputs > 0);
        assert!(sufficient_amount * (outputs as u64) <= input_sum);
        let mut output_preamounts = vec![sufficient_amount; outputs as usize];
        let mut remainder = input_sum - sufficient_amount * outputs as u64;
        while remainder > 0 {
            let idx = rng.gen_range(0..output_preamounts.len());
            if remainder < input_sum / boxes.len() as u64 {
                output_preamounts[idx] = output_preamounts[idx].checked_add(remainder).unwrap();
                remainder = 0;
            } else {
                let val = rng.gen_range(0..=remainder);
                output_preamounts[idx] = output_preamounts[idx].checked_add(val).unwrap();
                remainder -= val;
            }
        }

        let mut token_amounts: Vec<HashMap<TokenId, u64>> = vec![HashMap::new(); outputs as usize];
        let mut available_token_slots = (outputs * 255) as usize;
        while !assets_map.is_empty() && available_token_slots > 0 {
            let cur = assets_map
                .iter()
                .map(|(&token_id, &token_amount)| (token_id, token_amount))
                .next()
                .unwrap();
            let out_idx = loop {
                let idx = rng.gen_range(0..token_amounts.len());
                if token_amounts[idx].len() < 255 {
                    break idx;
                }
            };
            let contains = token_amounts[out_idx].contains_key(&cur.0);

            let amount = if *cur.1.as_u64() == 1
                || (available_token_slots < assets_map.len() * 2 && !contains)
                || rng.gen()
            {
                *cur.1.as_u64()
            } else {
                rng.gen_range(1..=*cur.1.as_u64())
            };
            if amount == *cur.1.as_u64() {
                assets_map.remove(&cur.0);
            } else {
                assets_map.entry(cur.0).and_modify(|amt| {
                    *amt = amt
                        .checked_sub(&TokenAmount::try_from(amount).unwrap())
                        .unwrap()
                });
            }
            token_amounts[out_idx]
                .entry(cur.0)
                .and_modify(|token_amount| *token_amount += amount)
                .or_insert_with(|| {
                    available_token_slots -= 1;
                    amount
                });
        }
        let output_boxes = output_preamounts
            .into_iter()
            .zip(token_amounts)
            .map(|(amount, tokens)| -> (u64, Option<BoxTokens>) {
                (
                    amount,
                    tokens
                        .into_iter()
                        .map(|(token_id, token_amount)| {
                            Token::from((token_id, TokenAmount::try_from(token_amount).unwrap()))
                        })
                        .collect::<Vec<_>>()
                        .try_into()
                        .ok(),
                )
            })
            .map(|(amount, tokens)| ErgoBoxCandidate {
                value: BoxValue::new(amount).unwrap(),
                ergo_tree: output_prop.clone(),
                tokens,
                additional_registers: NonMandatoryRegisters::empty(),
                creation_height: 0,
            })
            .collect();
        UnsignedTransaction::new_from_vec(
            boxes
                .iter()
                .map(|b| UnsignedInput::new(b.box_id(), ContextExtension::empty()))
                .collect(),
            vec![],
            output_boxes,
        )
        .unwrap()
    }
    fn valid_transaction_from_boxes(
        rng: TestRng,
        boxes: Vec<ErgoBox>,
        issue_new_token: bool,
        output_prop: ErgoTree,
        data_boxes: Vec<ErgoBox>,
    ) -> Transaction {
        let unsigned_tx = valid_unsigned_transaction_from_boxes(
            rng,
            &boxes,
            issue_new_token,
            output_prop,
            &data_boxes,
        );
        let tx_context =
            TransactionContext::new(unsigned_tx.clone(), boxes.clone(), data_boxes).unwrap();
        let wallet = Wallet::from_secrets(vec![]);
        let state_context = force_any_val();
        // Attempt to sign a transaction. If signing fails because script reduces to false or prover doesn't know some secret then return an invalid transaction
        wallet
            .sign_transaction(tx_context, &state_context, None)
            .or_else(|_| {
                Transaction::new(
                    TxIoVec::from_vec(
                        boxes
                            .iter()
                            .map(|b| Input {
                                box_id: b.box_id(),
                                spending_proof: ProverResult {
                                    proof: ProofBytes::Empty,
                                    extension: ContextExtension::empty(),
                                },
                            })
                            .collect(),
                    )
                    .unwrap(),
                    unsigned_tx.data_inputs,
                    unsigned_tx.output_candidates,
                )
            })
            .unwrap()
    }
    fn valid_transaction_gen_with_tree(
        tree: ErgoTree,
    ) -> impl Strategy<Value = (Vec<ErgoBox>, Transaction)> {
        let box_generator = gen_boxes(1, 100, 1, 100, Just(tree.clone()), None);
        (box_generator, bool::arbitrary()).prop_perturb(move |(boxes, issue_new_token), rng| {
            (
                boxes.clone(),
                valid_transaction_from_boxes(rng, boxes, issue_new_token, tree.clone(), vec![]),
            )
        })
    }

    fn valid_transaction_generator() -> impl Strategy<Value = (Vec<ErgoBox>, Transaction)> {
        let true_tree = ErgoTree::new(
            ErgoTreeHeader::v0(true),
            &Expr::Const(Constant {
                tpe: ergotree_ir::types::stype::SType::SBoolean,
                v: Literal::Boolean(true),
            }),
        )
        .unwrap();
        valid_transaction_gen_with_tree(true_tree)
    }

    fn update_asset<F: FnOnce(TokenAmount) -> TokenAmount>(
        transaction: &mut Transaction,
        boxes: &[ErgoBox],
        f: F,
    ) {
        for output in transaction.outputs.iter_mut() {
            if let Some(token) = output
                .tokens
                .iter_mut()
                .flatten()
                .find(|t| t.token_id != boxes[0].box_id().into())
            {
                token.amount = f(token.amount);
                break;
            }
        }
    }

    fn huge_value_box(index: u16) -> ErgoBox {
        ErgoBox::new(
            BoxValue::try_from(i64::MAX as u64).unwrap(),
            force_any_val::<ErgoTree>(),
            None,
            NonMandatoryRegisters::empty(),
            0,
            force_any_val::<TxId>(),
            index,
        )
        .unwrap()
    }

    fn input_for(b: &ErgoBox) -> Input {
        Input {
            box_id: b.box_id(),
            spending_proof: ProverResult {
                proof: ProofBytes::Empty,
                extension: ContextExtension::empty(),
            },
        }
    }

    // Aggregate ERG overflow must surface as a validation error (issue #881):
    // the reference implementation sums with Math.addExact, so an overflowing
    // aggregate is trapped at the addition — not a panic (debug) or a wrapped
    // total flowing into the preservation check (release). Three i64::MAX
    // boxes are each individually valid but their sum exceeds the i64 domain.
    #[test]
    fn validate_input_sum_overflow() {
        let boxes: Vec<ErgoBox> = (0..3).map(huge_value_box).collect();
        let inputs: Vec<Input> = boxes.iter().map(input_for).collect();
        let output = ErgoBoxCandidate {
            value: BoxValue::SAFE_USER_MIN,
            ergo_tree: force_any_val::<ErgoTree>(),
            tokens: None,
            additional_registers: NonMandatoryRegisters::empty(),
            creation_height: 0,
        };
        let tx = Transaction::new_from_vec(inputs, vec![], vec![output]).unwrap();
        let tx_context = TransactionContext::new(tx, boxes, vec![]).unwrap();
        let state_context: ErgoStateContext = force_any_val();
        assert!(matches!(
            tx_context.validate(&state_context),
            Err(TxValidationError::InputSumOverflow)
        ));
    }

    #[test]
    fn validate_output_sum_overflow() {
        let input_box = ErgoBox::new(
            BoxValue::SAFE_USER_MIN,
            force_any_val::<ErgoTree>(),
            None,
            NonMandatoryRegisters::empty(),
            0,
            force_any_val::<TxId>(),
            0,
        )
        .unwrap();
        let inputs = vec![input_for(&input_box)];
        let outputs: Vec<ErgoBoxCandidate> = (0..3)
            .map(|_| ErgoBoxCandidate {
                value: BoxValue::try_from(i64::MAX as u64).unwrap(),
                ergo_tree: force_any_val::<ErgoTree>(),
                tokens: None,
                additional_registers: NonMandatoryRegisters::empty(),
                creation_height: 0,
            })
            .collect();
        let tx = Transaction::new_from_vec(inputs, vec![], outputs).unwrap();
        let tx_context = TransactionContext::new(tx, vec![input_box], vec![]).unwrap();
        let state_context: ErgoStateContext = force_any_val();
        assert!(matches!(
            tx_context.validate(&state_context),
            Err(TxValidationError::OutputSumOverflow)
        ));
    }

    proptest! {

    #![proptest_config(ProptestConfig::with_cases(32))]
    #[test]
    // Test that a valid transaction is valid
    fn test_valid_transaction((boxes, tx) in valid_transaction_generator()) {
        let state_context = force_any_val();
        let tx_context = TransactionContext::new(tx, boxes, vec![]).unwrap();
        tx_context.validate(&state_context).unwrap();
    }
    #[test]
    fn test_ergo_preservation((mut boxes, mut tx) in valid_transaction_generator(), positive_delta: bool, change_output: bool) {
        let state_context = force_any_val();

        let box_value = if change_output {
            let slice: &mut [ErgoBox] = tx.outputs.as_mut();
            &mut slice[0].value
        }
        else {
            &mut boxes[0].value
        };
        if positive_delta {
            *box_value = box_value.checked_add(&BoxValue::SAFE_USER_MIN).unwrap();
        }
        else {
            *box_value = BoxValue::try_from(box_value.as_u64() - 1).unwrap();
        }

        assert!(tx.validate_stateless().is_ok());

        let tx_context = TransactionContext::new(tx, boxes, vec![]).unwrap();
        match tx_context.validate(&state_context) {
            Err(TxValidationError::ErgPreservationError(_, _)) => {},
            e => panic!("Expected validation to fail got {e:?}")
        }
    }
    #[test]
    fn test_zero_asset_creation((boxes, mut tx) in valid_transaction_generator()) {
        let state_context = force_any_val();
        update_asset(&mut tx, &boxes, |amount| amount.checked_add(&TokenAmount::MIN).unwrap());
        assert!(tx.validate_stateless().is_ok());

        let tx_context = TransactionContext::new(tx, boxes, vec![]).unwrap();
        match tx_context.validate(&state_context) {
            Err(TxValidationError::TokenPreservationError { .. } ) => {},
            other => panic!("Expected validation to fail, got {other:?}")
        }
    }
    #[test]
    fn test_asset_preservation((boxes, mut tx) in valid_transaction_generator()) {
        let state_context = force_any_val();
        update_asset(&mut tx, &boxes, |amount| amount.checked_add(&TokenAmount::MIN).unwrap());
        assert!(tx.validate_stateless().is_ok());

        let tx_context = TransactionContext::new(tx, boxes, vec![]).unwrap();
        match tx_context.validate(&state_context) {
            Err(TxValidationError::TokenPreservationError { .. } ) => {},
            other => panic!("Expected validation to fail, got {other:?}")
        }
    }
    }
    // Test that unspendable boxes can't be included in a transaction
    // TODO: When sigma-rust lands support for storage rent transactions, there should be a test that successfully passes validation when box is old enough
    #[test]
    fn test_false_proposition() {
        let state_context = force_any_val();
        let false_tree = ErgoTree::new(
            ErgoTreeHeader::v0(true),
            &Expr::Const(Constant {
                tpe: ergotree_ir::types::stype::SType::SBoolean,
                v: Literal::Boolean(false),
            }),
        )
        .unwrap();
        proptest!(|((boxes, tx) in valid_transaction_gen_with_tree(false_tree))| {
            assert!(tx.validate_stateless().is_ok());

            let tx_context = TransactionContext::new(tx, boxes, vec![]).unwrap();
            match tx_context.validate(&state_context) {
                Err(TxValidationError::ReducedToFalse(_, _)) => {},
                other => panic!("Expected validation to fail, got {other:?}")
            }
        });
    }
    // Regression for S4 (JIT_COSTING_FIX_PLAN.md): validate() must enforce the cost
    // limit cumulatively across the whole tx — inputs that individually fit must not
    // collectively exceed MaxBlockCost. We match the JVM (`ErgoTransaction.verifyInput`):
    // each input gets a FRESH budget of `maxCost - currentTxCost` (the *remaining*
    // block budget), which shrinks as inputs accrue, so the aggregate is still capped.
    // Use Const(true) inputs (50 JitCost = 5 block each, TrivialProp → 0 crypto cost)
    // and zero-out structural Parameters so init reduces to the fixed
    // INTERPRETER_INIT_COST baseline; then size max_block_cost so 2 inputs fit
    // (init + 2 × 5) and the 3rd overflows its remaining budget.
    #[test]
    fn test_validate_enforces_cumulative_jit_cost_across_inputs() {
        use ergotree_interpreter::eval::EvalError;
        use ergotree_interpreter::sigma_protocol::verifier::VerifierError;

        // Non-segregated SigmaProp(TrivialProp(true)) tree: the proposition
        // reduces via `trivial_reduce`, which charges exactly
        // `EVAL_SIGMA_PROP_CONSTANT = 50` JitCost per input. 50 is a clean
        // multiple of 10, so the JIT→block-cost round-trip in `Parameters`
        // doesn't lose precision — critical for sizing the limit at the
        // exact overflow boundary.
        let true_tree = ErgoTree::new(
            ErgoTreeHeader::v0(false),
            &Expr::Const(Constant {
                tpe: ergotree_ir::types::stype::SType::SSigmaProp,
                v: Literal::SigmaProp(alloc::boxed::Box::new(
                    ergotree_ir::sigma_protocol::sigma_boolean::SigmaProp::new(
                        ergotree_ir::sigma_protocol::sigma_boolean::SigmaBoolean::TrivialProp(true),
                    ),
                )),
            }),
        )
        .unwrap();
        proptest!(|((boxes, tx) in valid_transaction_gen_with_tree(true_tree))| {
            prop_assume!(tx.inputs.len() >= 3);

            let mut state_context: ErgoStateContext = force_any_val();
            let tx_context = TransactionContext::new(tx.clone(), boxes.clone(), vec![]).unwrap();
            // Baseline: tx must validate cleanly with default params, otherwise the
            // sample failed for non-cost-related reasons — skip.
            prop_assume!(tx_context.validate(&state_context).is_ok());

            // Zero out structural Parameters so compute_tx_init_cost reduces to the
            // fixed INTERPRETER_INIT_COST regardless of tx shape, then size the
            // budget to exactly the 3rd-input overflow boundary.
            const PER_INPUT_JIT: u64 = 50; // trivial_reduce charges EVAL_SIGMA_PROP_CONSTANT
            let init_jit = super::INTERPRETER_INIT_COST * 10;
            let limit_jit = init_jit + 2 * PER_INPUT_JIT; // 2 inputs fit, 3 overflow
            let mbc = i32::try_from(limit_jit / 10).unwrap();
            state_context.parameters = crate::chain::parameters::Parameters::new(
                1, 1_250_000, 360, 512 * 1024, mbc, 0, 0, 0, 0,
            );
            let tx_context = TransactionContext::new(tx, boxes, vec![]).unwrap();
            match tx_context.validate(&state_context) {
                Err(TxValidationError::VerifierError(_, verr)) => {
                    let is_cost = match &verr {
                        VerifierError::EvalError(EvalError::CostError(_)) => true,
                        VerifierError::EvalError(EvalError::Spanned(e)) => {
                            matches!(*e.error, EvalError::CostError(_))
                        }
                        VerifierError::ErgoTreeError(_)
                        | VerifierError::EvalError(_)
                        | VerifierError::SigParsingError(_)
                        | VerifierError::FiatShamirTreeSerializationError(_) => false,
                    };
                    prop_assert!(is_cost, "expected CostError, got {verr:?}");
                }
                other => panic!("expected cost-limit rejection, got {other:?}"),
            }
        });
    }
    #[test]
    fn an_output_is_measured_as_ergo_s_default_context_writes_it() {
        // SANTA X15, `Tuple(1, Upcast(1, Long))`, as R4. ergo measures an output by
        // `ErgoBox.bytes` (`ErgoTransaction.scala:171-175`, `BoxUtils.scala:41`), written under
        // its default version context, where the `Upcast` is written as its constant: two
        // bytes fewer than at version 3
        use super::verify_output;
        use crate::chain::parameters::Parameter;
        use ergotree_ir::serialization::SigmaSerializable;
        let candidate = ErgoBoxCandidate::sigma_parse_bytes(
            &base16::decode("c0843d0008d3010001860204027e040205").unwrap(),
        )
        .unwrap();
        let output = ErgoBox::from_box_candidate(&candidate, TxId::zero(), 0).unwrap();
        let written = output.bytes().unwrap().len() as u64;
        assert_eq!(
            output.sigma_serialize_bytes().unwrap().len() as u64,
            written + 2
        );
        let mut state_context: ErgoStateContext = force_any_val();
        state_context.pre_header.height = 1;
        let per_byte = *output.value.as_u64() / written;
        for (per_byte, dust) in [(per_byte, false), (per_byte + 1, true)] {
            state_context
                .parameters
                .parameters_table
                .insert(Parameter::MinValuePerByte, per_byte as i32);
            assert_eq!(
                matches!(
                    verify_output(&state_context, &output, 0),
                    Err(TxValidationError::DustOutput(..))
                ),
                dust,
                "{per_byte}"
            );
        }
    }

    #[test]
    fn an_output_the_default_context_cannot_write_fails_the_output_checks() {
        // SANTA `evaluated-values-spend` entry 46: C2's twin, an empty `Coll[Int => Int]`, as
        // R4. ergo's output checks write the output under its default version context, where
        // a function type has no encoding (`TypeSerializer.scala:111`): the transaction is
        // invalid (`ErgoTransaction.scala:171-175`)
        use super::verify_output;
        use ergotree_ir::serialization::SigmaSerializable;
        let candidate = ErgoBoxCandidate::sigma_parse_bytes(
            &base16::decode("c0843d0008d301000183007001040400").unwrap(),
        )
        .unwrap();
        let output = ErgoBox::from_box_candidate(&candidate, TxId::zero(), 0).unwrap();
        let mut state_context: ErgoStateContext = force_any_val();
        state_context.pre_header.height = 1;
        assert!(matches!(
            verify_output(&state_context, &output, 0),
            Err(TxValidationError::SigmaSerializationError(_))
        ));
    }

    /// The state SANTA's spends are validated in (transaction tier): its parameters and
    /// height, at the voted parameters' `block_version` and under a header of
    /// `header_version`. The headers are arbitrary: no script here reads one.
    fn santa_state_context(block_version: u8, header_version: u8) -> ErgoStateContext {
        use crate::chain::parameters::Parameter;
        let mut state_context: ErgoStateContext = force_any_val();
        state_context.pre_header.version = header_version;
        state_context.pre_header.height = 1051200;
        for (parameter, value) in [
            (Parameter::MaxBlockCost, 1000000),
            (Parameter::StorageFeeFactor, 1250000),
            (Parameter::MinValuePerByte, 360),
            (Parameter::InputCost, 2000),
            (Parameter::DataInputCost, 100),
            (Parameter::OutputCost, 100),
            (Parameter::TokenAccessCost, 100),
            (Parameter::BlockVersion, block_version as i32),
        ] {
            state_context
                .parameters
                .parameters_table
                .insert(parameter, value);
        }
        state_context
    }

    /// One of SANTA's spends: its transaction, the spent box and a data input
    type SantaSpend = (&'static str, &'static str, Option<&'static str>);

    /// One of SANTA's spends at `block_version`, the voted parameters' and the header's alike
    fn santa_spend(block_version: u8, spend: SantaSpend) -> Result<u64, TxValidationError> {
        santa_spend_under(block_version, block_version, spend)
    }

    /// One of SANTA's spends under the voted parameters' `block_version` and a header of
    /// `header_version`, read as a node reads it: a block's transactions under
    /// (blockVersion - 1) from block version 4 and with no context before (ergo v6.0.6
    /// `BlockTransactions.scala:184-202`, where the block version is a signed byte), a box
    /// from the UTXO set with none. SANTA's oracle reads the transaction at the parameters'
    /// block version, and no entry's bytes tell that from the header's.
    fn santa_spend_under(
        block_version: u8,
        header_version: u8,
        (tx, input, data_input): SantaSpend,
    ) -> Result<u64, TxValidationError> {
        use ergotree_ir::ergo_tree::ErgoTreeVersion;
        use ergotree_ir::serialization::sigma_byte_reader::{from_bytes, SigmaByteRead};
        use ergotree_ir::serialization::SigmaSerializable;
        let tx_bytes = base16::decode(tx).unwrap();
        let tx = if block_version as i8 >= 4 {
            let activated = ErgoTreeVersion::from(block_version - 1);
            from_bytes(&tx_bytes)
                .with_versions(activated, activated, Transaction::sigma_parse)
                .unwrap()
        } else {
            Transaction::sigma_parse_bytes(&tx_bytes).unwrap()
        };
        let parse_box =
            |hex: &str| ErgoBox::sigma_parse_bytes(&base16::decode(hex).unwrap()).unwrap();
        let data_boxes = data_input.map(parse_box).into_iter().collect();
        TransactionContext::new(tx, vec![parse_box(input)], data_boxes)
            .unwrap()
            .validate(&santa_state_context(block_version, header_version))
    }

    /// SANTA `tree-version-above-activated-eval`, transaction tier, at block version 4: spends
    /// whose script deserializes a Box while it runs. The Box's tree is v4 in each entry that
    /// is refused, and v3 in its twin.
    /// - #0, #1: `DeserializeContext(1, SigmaProp)`, whose script holds the Box as a constant.
    /// - #2, #3: the same script through `DeserializeRegister(R4, SigmaProp)`.
    /// - #4, #5: a `DeserializeContext` in a branch that is never taken, decoded all the same.
    /// - #6, #7: `SubstConstants` over a template whose constant 0 is the Box.
    /// - #8: a template whose own header is v4, with no Box. Its version is not compared.
    /// - #9, #10: `Global.deserializeTo[Box]`, in a v3 tree.
    const SANTA_EVAL: [SantaSpend; 11] = [
        ("01dd4eb3e616b9d69842d694abe46d57ff5798504fa4a5a749eec09a3c34c1aee40001010e33d191c1638094ebdc030c0208d3010000355c6041d6c39f3fdcf9c7b76d16aa0ac2588677ead70aa29692362f66eed2680005000000018094ebdc030008d3010000", "8094ebdc0300d40801010000bd687baa88d495611420610f616d2540816da1102f3c0cf9c01502049dcf796900", None),
        ("01f9f3b4b7b291a01ccce09312cdd25f40154a350a350d18655bd0ed0cec4731bb0001010e33d191c1638094ebdc030b0208d3010000355c6041d6c39f3fdcf9c7b76d16aa0ac2588677ead70aa29692362f66eed2680005000000018094ebdc030008d3010000", "8094ebdc0300d4080101000074e88b04fd35f5b1f5d290aff828d6ea83dcfbed4e67e8ae81a9cda6f557b5ec00", None),
        ("01b29e99974e34a1abeb19b65d7e8999c780d0e9c3c42940af8e4c22523ae1d98f00000000018094ebdc030008d3010000", "8094ebdc0300d50408000100010e33d191c1638094ebdc030c0208d3010000355c6041d6c39f3fdcf9c7b76d16aa0ac2588677ead70aa29692362f66eed26800050039c4e831ad242c5dd92703d7a0c27646e1bcd476ca8abe39313572c036e01b1900", None),
        ("01bfef4aba3978f2199992caf8ce60fa7580ef3cf27b098fa4a2b3448aebadd0df00000000018094ebdc030008d3010000", "8094ebdc0300d50408000100010e33d191c1638094ebdc030b0208d3010000355c6041d6c39f3fdcf9c7b76d16aa0ac2588677ead70aa29692362f66eed268000500ebfd10d5cd99b813fadc622466046d36c89a9e9d469725e188404aee931ca40100", None),
        ("011bde0832c5b9a3469bab9adafb4d3f84c8b99547e73a8db69daf91f886abcd1f0001010e3291c1638094ebdc030c0208d3010000355c6041d6c39f3fdcf9c7b76d16aa0ac2588677ead70aa29692362f66eed2680005000000018094ebdc030008d3010000", "8094ebdc0300d1950100d40101010101000042c56bcbeb3f93cb65fc790aa446771f8f1065edced8737888de5165020ff0c900", None),
        ("01e5c7aa85cf01feca523538a7e31499a5ddb4ec5fa1972e9f693cc7b5ecf7dbff0001010e3291c1638094ebdc030b0208d3010000355c6041d6c39f3fdcf9c7b76d16aa0ac2588677ead70aa29692362f66eed2680005000000018094ebdc030008d3010000", "8094ebdc0300d1950100d40101010101000066d0bc484e3c6815f8601cfc29272a10a8ac37aea5d7b7f05d4c23df41a90eb600", None),
        ("01e903fbd58487fe2d394e9aef7e3f3dec09bcbcdfd78ac740c3af9f0f04d234c20001010e391002638094ebdc030c0208d3010000355c6041d6c39f3fdcf9c7b76d16aa0ac2588677ead70aa29692362f66eed268000500d191c1730073010000018094ebdc030008d3010000", "8094ebdc0310010400d191b174e4e3010e830004830004730001000064818d45390b3a202031a3dbb61c1c699d9083968061b151fe62cea897cfa3a400", None),
        ("0125cb9efe258240d231a3db50bc86a34963857de772062f43f67245ca71cd37f80001010e391002638094ebdc030b0208d3010000355c6041d6c39f3fdcf9c7b76d16aa0ac2588677ead70aa29692362f66eed268000500d191c1730073010000018094ebdc030008d3010000", "8094ebdc0310010400d191b174e4e3010e8300048300047300010000e2e73c11a787ff44d6da11ed089dae4eb5900d072cf91c3fb9e0c8f3685241ac00", None),
        ("0126464b0c9eae536e779a2aa440d6ba4dc5d57504648eb73d677db1ed354d14960001010e071c050108d373000000018094ebdc030008d3010000", "8094ebdc0310010400d191b174e4e3010e8300048300047300010000802028f0b897b557086d4e5baefe3a24dcdbcf4a1713b9d69cc17429f32f2e4c00", None),
        ("01dd18174bab5d9d291c55e599690cab3c51e63d7336db4233974525ab798d55dd0001010e2d8094ebdc030c0208d3010000355c6041d6c39f3fdcf9c7b76d16aa0ac2588677ead70aa29692362f66eed268000000018094ebdc030008d3010000", "8094ebdc031b12010500d191c1dc6a04dd01e4e3010e637300010000b1edcd88cc0bb55c8fbdd87b922a26bd07645cafab8aefbf482f2d08d666d76f00", None),
        ("014d99d96b6f2194ccdf7a6f9ab744b888d4dffd623b402a2acc77efcdab5327340001010e2d8094ebdc030b0208d3010000355c6041d6c39f3fdcf9c7b76d16aa0ac2588677ead70aa29692362f66eed268000000018094ebdc030008d3010000", "8094ebdc031b12010500d191c1dc6a04dd01e4e3010e637300010000290728210a800aa2c9aa362f061e7bed6466839c36509a980859f812b7bd6a5800", None),
    ];
    #[test]
    fn a_tree_deserialized_during_a_spend_is_read_under_the_activated_version() {
        // sigmastate reduces inside `withVersions(activated, ergoTree.version)`
        // (`Interpreter.scala:366`), so `deserializeErgoTree` compares a tree that is
        // deserialized while the script runs with the activated version
        // (`ErgoTreeSerializer.scala:150-154`), and the spend is invalid. The spends the JVM
        // accepts are at its costs.
        for (i, cost) in [
            None,
            Some(12215),
            None,
            Some(12217),
            None,
            Some(12223),
            None,
            Some(12141),
            Some(12131),
            None,
            Some(12124),
        ]
        .into_iter()
        .enumerate()
        {
            let res = santa_spend(4, SANTA_EVAL[i]);
            match cost {
                Some(cost) => assert_eq!(res.as_ref().ok(), Some(&cost), "#{i}: {res:?}"),
                None => {
                    // the parser's error, as the variant or, where the evaluator reports it
                    // with the script's source, as its message
                    let refused = format!("{res:?}");
                    assert!(
                        res.is_err()
                            && (refused.contains("TreeVersionAboveActivated(4, 3)")
                                || refused.contains(
                                    "Tree version (4) is above activated script version (3)"
                                )),
                        "#{i}: {refused}"
                    );
                }
            }
        }
    }

    /// SANTA `tree-version-above-activated`, transaction tier, at block version 4: the
    /// transaction, the spent box and the data input of each entry. Every tree is a
    /// `SigmaProp(true)` constant.
    /// - #0 to #3: the spent box's tree is v4, v5, v6, v7. Invalid.
    /// - #4: a v3 tree. Valid, 12105.
    /// - #5: a data input with a v7 tree, whose script never runs. Valid, 12205.
    /// - #6: a rent collection of a v4-tree box. Valid, 12150: the rent path decides before
    ///   the script's (`ErgoInterpreter.scala:72-84`).
    /// - #7: the same without the rent variable, so on the script path. Invalid.
    const SANTA_V6: [SantaSpend; 8] = [
        ("01784afe6f780d54138e463fc99b46e6621ec688ea37e5a3d1b20f4d1b31dc18d600000000018094ebdc030008d3010000", "8094ebdc030c0208d30100003e8998e6e86bec4c436c777124ce66c2b58acf8737fed2aee42bac9952d395b400", None),
        ("01dc235e8e8143949d677f8749d9fa02743e868546d29dc54a6532e2323e52026d00000000018094ebdc030008d3010000", "8094ebdc030d0208d301000055bb18f53293651f4629c0de7168e40047a10cb1dfac2ffd3c86423ecc6285bd00", None),
        ("014a3b556dedbec7b92714a19c9f7bd499fe58943855c1f983ef2d589894db1de400000000018094ebdc030008d3010000", "8094ebdc030e0208d3010000e5e394c16a2b6c3a7940b5e3b89c611cdd348847cdbe12d541db1272833f4dc500", None),
        ("015e3d9cc254a848b9d83a06edf81db9219ace2c717dba56cb53d9e828c9c6dd8900000000018094ebdc030008d3010000", "8094ebdc030f0208d30100002aa1611cf3d12efd48c79b33877e4cff7dae5c3f6bbe3c425fea87e74fdbd5b600", None),
        ("01a9e4530eae5492e695036db56dd341a636c9a8244931702121fc6918032524a600000000018094ebdc030008d3010000", "8094ebdc030b0208d3010000c6d756e123a76296267b5585443f6bf4bb0b718b223b63f6346de52d6104e6fe00", None),
        ("017777a23fa9c0f38f98d0911bb95cdcf483b7cf50f12e9f2a9a2e5bb17101c4f0000001c8fbf459d5cf7d1b58155f4452b6c5d4a5fd627d6a5a871b9928353445dc5b6e00018094ebdc030008d3010000", "8094ebdc030008d30100004acfca69b32cd322acc6624793c8f37846b80b635c463046d457add775f2f59d00", Some("8094ebdc030f0208d30100001ab90bf71dd55341c606992b1a3e1d5b3ffb240cd04cd3746e6b312f6c9d52a100")),
        ("019953b1f1da5bc9d6ca7f6b5df1b9ef3e8136617964ea09b7d8e339250917bd8f00017f0300000001c0f79c1a0008d3c094400000", "c0f79c1a0c0208d30000003cc4e4379fd74e048d4174d7e65266dd7d3168d527101ea6d6a09f8a022ed39b00", None),
        ("019953b1f1da5bc9d6ca7f6b5df1b9ef3e8136617964ea09b7d8e339250917bd8f0000000001c0f79c1a0008d3c094400000", "c0f79c1a0c0208d30000003cc4e4379fd74e048d4174d7e65266dd7d3168d527101ea6d6a09f8a022ed39b00", None),
    ];

    /// The same at block version 3.
    /// - #0: the spent box's tree is v3. Invalid.
    /// - #1: a v2 tree. Valid, 12105.
    /// - #2, #3: an output with a v3 tree, then a v4 one, in a transaction read with no
    ///   context. Valid, 12105.
    const SANTA_V5: [SantaSpend; 4] = [
        ("0132590c975512fc1213e0c598c0ca1e1b05c46d9c2eddb68b97446471cb7e667b00000000018094ebdc030008d3010000", "8094ebdc030b0208d3010000d42d221684077e351b4d9cf1e0596d1f39182290affb16d810f018be897aebde00", None),
        ("0139e31dd67812dcc0f951ced53e9e87ba77727354dad4f4b25bde2c4dd4edf0ba00000000018094ebdc030008d3010000", "8094ebdc030a0208d30100008a93b72b483e955d0dc80f99fb9608f73e80caaa59c849b5601a87e22857427f00", None),
        ("01ff8e9a38a98d717b6e3f448ff0fbb64f7934b72016f9af0da0d31260e36d165900000000018094ebdc030b0208d3010000", "8094ebdc030008d301000029cf8401e98f0023f3010fb9d453455324fe2e43289139c8529d13a4aae2d3be00", None),
        ("018b133ddd496c74a4e746156596cd7b76fb2a9c87af7b6eb973fcbddec5a945a200000000018094ebdc030c0208d3010000", "8094ebdc030008d3010000108491b1a8b4c63a8faa89282c9d43de54da62baa2541a644047754b7c22757a00", None),
    ];

    /// SANTA `tree-version-block-version-edges`, transaction tier: the block version, then
    /// the spend. Every tree is a `SigmaProp` constant, `true` unless said.
    /// - #0, #1: block version 2. A v2 tree is invalid, as the spend check has no floor at
    ///   activated 2. A v1 tree is valid.
    /// - #2 to #6: block version 5, above what this interpreter supports. A v4 and a v5 tree
    ///   are accepted unverified, at 12100, and a v3 tree is verified, at 12105. A v4
    ///   `SigmaProp(false)` is accepted all the same, as it is not reduced. A v3 one reduces
    ///   to false.
    /// - #7: block version 0. The activated version is -1, and a v0 tree is invalid.
    /// - #8, #9: block version 128, so activated 127. A v0 tree is verified, a v4 tree
    ///   accepted unverified.
    /// - #10: block version 200, so activated -57. A v0 tree is invalid.
    const SANTA_EDGES: [(u8, SantaSpend); 11] = [
        (2, ("0114ae3c8859c951b7382273a6b0608b4bed5bcbb9808c62b2216a8ac3b0a65b9400000000018094ebdc030008d3010000", "8094ebdc030a0208d30100001867d290daf93e26943cd142e8909f934da1ae5a3d13702b18e7a098963c560d00", None)),
        (2, ("01935d1f5768727d08a52a592b21cc516f7867b4ed29f9f9bb6ab9c53fc3b7e27400000000018094ebdc030008d3010000", "8094ebdc03090208d3010000b1912f3813cffa1c46b8d0dc01bbc2e77f081f8c6a95466f53a297f8d41d301900", None)),
        (5, ("01716ba868ff0c1b3b2789af540271f36b44307b38f9b2f91856cf118995ac92c100000000018094ebdc030008d3010000", "8094ebdc030c0208d30100000e1a1f383e5ea73af4eeeb1e91afecc8f82d885ef515348a859c3f4ac7ed11d700", None)),
        (5, ("01ddcb0cc788fbfa7c957546cae341d509a9f2c82f1c2793003f635d5b96001b1200000000018094ebdc030008d3010000", "8094ebdc030d0208d30100003a9bb8b913706593e307088cfd2af43e52c14ac5a90d123e1ee3b610d762f51700", None)),
        (5, ("012a9cdc8ddedde678c7e9e8c3cf99da18b47cd7e588936c7ed70930c2ee91a0ee00000000018094ebdc030008d3010000", "8094ebdc030b0208d3010000cf0662f0cbebd14c1e623ef4dfd34a5c8f9093751f2796f85a3b4b0eee51a96300", None)),
        (5, ("01be0a41624e7557944d04741e817a7c53f4596f8940a982d7cf8198e3b361f90c00000000018094ebdc030008d3010000", "8094ebdc030c0208d2010000cefc0b1e08278658b4bf41642b483e6593809dd9859d655a79ab443474b28def00", None)),
        (5, ("01054d4d6b54021426e829eb8eba73c6474fd1a7c1189910e2311ae6b738a2793000000000018094ebdc030008d3010000", "8094ebdc030b0208d20100000436bd11fdcfd21948d2321e0b47b4743b07379ccafb8c4fe19cc11b68392b2000", None)),
        (0, ("015d6ecc694d898a5984479adb3e5794e5d973945ba599a6d34cb0158845f21db600000000018094ebdc030008d3010000", "8094ebdc030008d3010000ff892b7d68e99b9a61ee01ecb8ab4ed5ee34ffd7d669f6dbd499b7f8d8b55a2e00", None)),
        (128, ("017484138af2a1747e0e10d316b65bcbf68e4fae3a000abc38f94e3f82e40331be00000000018094ebdc030008d3010000", "8094ebdc030008d30100004af7b6af4808f1bbbb728689c843b0e701b8137e1657ac01a14db44a728d446e00", None)),
        (128, ("016fc86bed1e3cbcc1ed10d0c6ced31d6bf3d5802a4791bfba79176e6a3e4f14f900000000018094ebdc030008d3010000", "8094ebdc030c0208d30100002bf691793ced6f5f33af8ada7e362cd1acdf9434ec0eed632593249a59c64efc00", None)),
        (200, ("018c9ac6aa44428cf555a2d6c46663bbc4b07d1b0d9d0f8143c0db72785607254c00000000018094ebdc030008d3010000", "8094ebdc030008d3010000e2d4be7ffad7105eba3e5cbd7e2cc3ef69183c8538ab3bbdbd6fab85b66bb5b100", None)),
    ];

    fn is_version_error(res: &Result<u64, TxValidationError>) -> bool {
        use ergotree_interpreter::eval::EvalError;
        use ergotree_interpreter::sigma_protocol::verifier::VerifierError;
        matches!(
            res,
            Err(TxValidationError::VerifierError(
                0,
                VerifierError::EvalError(EvalError::TreeVersionAboveActivated { .. })
            ))
        )
    }

    #[test]
    fn a_spend_of_a_tree_above_the_activated_version_is_invalid() {
        // sigmastate's `checkSoftForkCondition` throws (`Interpreter.scala:325-327`)
        for (block_version, spend) in [
            (4, SANTA_V6[0]),
            (4, SANTA_V6[1]),
            (4, SANTA_V6[2]),
            (4, SANTA_V6[3]),
            (4, SANTA_V6[7]),
            (3, SANTA_V5[0]),
        ] {
            let res = santa_spend(block_version, spend);
            assert!(is_version_error(&res), "{spend:?}: {res:?}");
        }
    }

    #[test]
    fn the_version_check_reaches_no_spend_the_jvm_accepts() {
        // with the JVM's costs
        for (block_version, spend, cost) in [
            (4, SANTA_V6[4], 12105),
            (4, SANTA_V6[5], 12205),
            (4, SANTA_V6[6], 12150),
            (3, SANTA_V5[1], 12105),
            (3, SANTA_V5[2], 12105),
            (3, SANTA_V5[3], 12105),
        ] {
            let res = santa_spend(block_version, spend);
            assert_eq!(res.ok(), Some(cost), "{spend:?}");
        }
    }

    #[test]
    fn the_spend_check_at_the_edges_of_the_block_version() {
        // sigmastate's `checkSoftForkCondition` (`Interpreter.scala:298-331`), whose versions
        // are signed bytes, with the JVM's verdicts and costs
        enum Jvm {
            Valid(u64),
            /// "ErgoTree version N is higher than activated M"
            Version,
            /// the proposition reduces to false
            False,
        }
        for (i, jvm) in [
            Jvm::Version,
            Jvm::Valid(12105),
            Jvm::Valid(12100),
            Jvm::Valid(12100),
            Jvm::Valid(12105),
            Jvm::Valid(12100),
            Jvm::False,
            Jvm::Version,
            Jvm::Valid(12105),
            Jvm::Valid(12100),
            Jvm::Version,
        ]
        .into_iter()
        .enumerate()
        {
            let (block_version, spend) = SANTA_EDGES[i];
            let res = santa_spend(block_version, spend);
            match jvm {
                Jvm::Valid(cost) => assert_eq!(res.as_ref().ok(), Some(&cost), "#{i}: {res:?}"),
                Jvm::Version => assert!(is_version_error(&res), "#{i}: {res:?}"),
                Jvm::False => assert!(
                    matches!(res, Err(TxValidationError::ReducedToFalse(0, _))),
                    "#{i}: {res:?}"
                ),
            }
        }
    }

    #[test]
    fn a_tree_above_the_activated_version_is_not_reduced_for_signing() {
        // `ReducingInterpreter.reduce` goes to `fullReduction` without `checkSoftForkCondition`
        // (sigmastate v6.0.6 `ReducingInterpreter.scala:33-45`), and the reduction's
        // `VersionContext` refuses the tree (`Interpreter.scala:207`,
        // `VersionContext.scala:17-21`). SANTA measured `fullReduction` on the JVM; this path
        // follows from it and has no vector. SANTA v6 #0's spend, a v4 tree at block version
        // 4, then #4's, a v3 tree.
        use crate::chain::transaction::reduced::reduce_tx;
        use crate::wallet::signing::TxSigningError;
        use ergotree_interpreter::eval::EvalError;
        use ergotree_interpreter::sigma_protocol::prover::ProverError;
        use ergotree_ir::serialization::SigmaSerializable;
        let reduce = |(tx, input, _): SantaSpend| {
            let tx = Transaction::sigma_parse_bytes(&base16::decode(tx).unwrap()).unwrap();
            let unsigned = UnsignedTransaction::new(
                tx.inputs.mapped(|input| {
                    UnsignedInput::new(input.box_id, input.spending_proof.extension)
                }),
                tx.data_inputs,
                tx.output_candidates,
            )
            .unwrap();
            let input = ErgoBox::sigma_parse_bytes(&base16::decode(input).unwrap()).unwrap();
            reduce_tx(
                TransactionContext::new(unsigned, vec![input], vec![]).unwrap(),
                &santa_state_context(4, 4),
            )
        };
        let refused = reduce(SANTA_V6[0]);
        assert!(
            matches!(
                refused,
                Err(TxSigningError::ProverError(
                    ProverError::EvalError(EvalError::TreeVersionAboveActivated { .. }),
                    0
                ))
            ),
            "{refused:?}"
        );
        assert!(reduce(SANTA_V6[4]).is_ok());
    }

    /// SANTA `block-version-source`, transaction tier: the voted parameters' block version,
    /// the header's, then the spend. ergo judges a spend by the first (ergo v6.0.6
    /// `ErgoContext.scala:28`, `ErgoStateContext.scala:114`), and a header need not carry it
    /// between epoch starts (`exBlockVersion`, `:241`, is checked when an epoch starts, `:265`).
    /// - #0: parameters 4, header 3, a v3 tree. Valid, 12105: activated 3.
    /// - #1: parameters 4, header 5, a v4 `SigmaProp(false)`. Invalid, "ErgoTree version 4 is
    ///   higher than activated 3": a header's 5 would accept it unverified.
    /// - #2, #3: parameters 4, header 0 and 200, a v0 tree. Valid, 12105.
    /// - #4: parameters 5, header 4, a v4 `SigmaProp(false)`. Valid, 12100, unverified.
    /// - #5: parameters 4, header 2, an output created at height 1 from an input created at
    ///   5. Invalid: the height rule applies from the parameters' block version 3
    ///   (`ErgoTransaction.scala:379-384`).
    /// - #6: the same under parameters 2 and header 4. Valid, 12105: no rule yet.
    /// - #7: parameters 4, header 3, the script `CONTEXT.preHeader.version == 3` in a v3
    ///   tree. Valid, 12105: a script reads the header's version.
    const SANTA_SOURCE: [(u8, u8, SantaSpend); 8] = [
        (4, 3, ("015e57107169ac1479ad83df2334022c0b921045c5ab0b3ec3e1e3a2d27eaceedb00000000018094ebdc030008d3010000", "8094ebdc030b0208d30100003f1f83114b9dce1440c32c8344e7689dc3b836d44b4e629c05f1f4e5333d2a7c00", None)),
        (4, 5, ("011f22af797ddf093be3049bcef4cc1b2d0f0cbec9d9a90f1b65aebb333e0c3b8200000000018094ebdc030008d3010000", "8094ebdc030c0208d2010000e2e387d431a1ed10ff41ea69480b18c7c61b81db5b6ed574120a15601e2f414a00", None)),
        (4, 0, ("01547dc7f867a6fe075e2f12ce3c2a6e6e7262d158c47c43fd496101e8570ceaae00000000018094ebdc030008d3010000", "8094ebdc030008d30100003218942a814426ba0c410f8c03ca876a7d0a6e5d6c1fbf51232fad829868aab700", None)),
        (4, 200, ("01fe3015ea2b143da7a0fd9d4e965d4d7deddb5d420bdfe123f91d1cbaa25e581f00000000018094ebdc030008d3010000", "8094ebdc030008d3010000e189786b7c73a00010af9ad86a9707f7d808cd6d2af83fad6b3a01c51ff57aaa00", None)),
        (5, 4, ("01f51ae3abf35910cfb0d7502abf1d20020071af7421577b06278830223959dc3400000000018094ebdc030008d3010000", "8094ebdc030c0208d2010000f544fff9a8c2552f2f76a73387eeadedf8636e38364242143c695ad910993b5f00", None)),
        (4, 2, ("015fd4f96ddd63670779630e302668cf0ac8e3a71821f429434a2545d782006eac00000000018094ebdc030008d3010000", "8094ebdc030008d305000000b7e8d6d1274d3da9c8aca87e2e750f430841ece33a238add6a92dfe5e3d2f200", None)),
        (2, 4, ("01b5227455ea515458f7eb2aad3299d5968d7ec99585b720c9fb9032e53a50facf00000000018094ebdc030008d3010000", "8094ebdc030008d3050000764a6cae679f77fb0028a93c86715376299e72284bb6f139f866c8cc2df7710300", None)),
        (4, 3, ("012526597f1ed7c35904bbe920a2d73bc0ce65913dfc37e4b817ea98abb1a1699b00000000018094ebdc030008d3010000", "8094ebdc031b0e010203d193db6901db6503fe730001000045932ee9d7d84fc823bed895a239b45a73ab6e8f09b90d3de1a63e691af4194700", None)),
    ];

    #[test]
    fn a_spend_is_judged_by_the_voted_parameters_block_version() {
        enum Jvm {
            Valid(u64),
            /// "ErgoTree version 4 is higher than activated 3"
            Version,
            /// "Creation height of any output should be not less than ..."
            Height,
        }
        for (i, jvm) in [
            Jvm::Valid(12105),
            Jvm::Version,
            Jvm::Valid(12105),
            Jvm::Valid(12105),
            Jvm::Valid(12100),
            Jvm::Height,
            Jvm::Valid(12105),
            Jvm::Valid(12105),
        ]
        .into_iter()
        .enumerate()
        {
            let (block_version, header_version, spend) = SANTA_SOURCE[i];
            let res = santa_spend_under(block_version, header_version, spend);
            match jvm {
                Jvm::Valid(cost) => assert_eq!(res.as_ref().ok(), Some(&cost), "#{i}: {res:?}"),
                Jvm::Version => assert!(
                    is_version_error(&res) && format!("{res:?}").contains("activated_version: 3"),
                    "#{i}: {res:?}"
                ),
                Jvm::Height => assert!(
                    matches!(res, Err(TxValidationError::MonotonicHeightError(1, 5))),
                    "#{i}: {res:?}"
                ),
            }
        }
    }

    #[test]
    fn the_height_rule_compares_the_block_version_as_a_signed_byte() {
        // ergo's `blockVersion <= Header.HardeningVersion` is on `Byte`s
        // (`ErgoTransaction.scala:384`, `Header.scala:136`), so from block version 128 the
        // rule does not apply. By source, no vector: SANTA #5's transaction, whose output is
        // created below its input. Under the parameters' block version 128, so activated 127,
        // it is valid, at the cost of SANTA's spend at that block version (`SANTA_EDGES` #8).
        // Under 200 it fails on the tree's version, as every script spend does there, and not
        // on its height.
        let (_, _, spend) = SANTA_SOURCE[5];
        let res = santa_spend_under(128, 4, spend);
        assert_eq!(res.as_ref().ok(), Some(&12105), "{res:?}");
        let res = santa_spend_under(200, 4, spend);
        assert!(is_version_error(&res), "{res:?}");
    }

    #[test]
    fn a_negative_creation_height_is_refused_unless_the_parameters_block_version_is_1() {
        // ergo's `(blockVersion == 1) || out.creationHeight >= 0` is on
        // `stateContext.blockVersion`, the voted parameters', a `Byte`
        // (`ErgoTransaction.scala:168-173`): the rule holds at every block version but 1, at
        // 0 and from 128 too. By source, no vector: an output created above `Int.MaxValue`
        // does not parse (sigmastate `ErgoBoxCandidate.scala:195`), so it exists in memory
        // only.
        use super::verify_output;
        let output = ErgoBox::new(
            BoxValue::SAFE_USER_MIN,
            ErgoTree::new(ErgoTreeHeader::v0(false), &Expr::Const(true.into())).unwrap(),
            None,
            NonMandatoryRegisters::empty(),
            1 << 31,
            force_any_val::<TxId>(),
            0,
        )
        .unwrap();
        let refused = |block_version: u8, header_version: u8| {
            matches!(
                verify_output(
                    &santa_state_context(block_version, header_version),
                    &output,
                    0
                ),
                Err(TxValidationError::NegativeHeight)
            )
        };
        for block_version in [0u8, 2, 4, 128, 200, 255] {
            assert!(refused(block_version, 1), "{block_version}");
        }
        // under the parameters' block version 1 this rule exempts it, whatever the header's
        assert!(!refused(1, 2));
    }

    #[test]
    fn signing_is_activated_by_the_parameters_block_version_too() {
        // ergo's wallet and the SDK's provers take the activated version from the parameters
        // as well (ergo `ErgoProvingInterpreter.scala:76`; sigmastate
        // `AppkitProvingInterpreter.scala:198`, `ReducingInterpreter.scala:146`). By source, no
        // vector: SANTA #0's spend, a v3 tree, is signed under the parameters' block version
        // 4, and refused under 1, which `Parameters::default()` has, whatever the header's.
        use crate::wallet::signing::{sign_transaction, TxSigningError};
        use ergotree_interpreter::eval::EvalError;
        use ergotree_interpreter::sigma_protocol::prover::{ProverError, TestProver};
        use ergotree_ir::serialization::SigmaSerializable;
        let sign = |block_version: u8, header_version: u8| {
            let (_, _, (tx, input, _)) = SANTA_SOURCE[0];
            let tx = Transaction::sigma_parse_bytes(&base16::decode(tx).unwrap()).unwrap();
            let unsigned = UnsignedTransaction::new(
                tx.inputs.mapped(|input| {
                    UnsignedInput::new(input.box_id, input.spending_proof.extension)
                }),
                tx.data_inputs,
                tx.output_candidates,
            )
            .unwrap();
            let input = ErgoBox::sigma_parse_bytes(&base16::decode(input).unwrap()).unwrap();
            sign_transaction(
                &TestProver { secrets: vec![] },
                TransactionContext::new(unsigned, vec![input], vec![]).unwrap(),
                &santa_state_context(block_version, header_version),
                None,
            )
        };
        assert!(sign(4, 3).is_ok());
        let refused = sign(1, 4);
        assert!(
            matches!(
                refused,
                Err(TxSigningError::ProverError(
                    ProverError::EvalError(EvalError::TreeVersionAboveActivated { .. }),
                    0
                ))
            ),
            "{refused:?}"
        );
    }

    #[test]
    fn test_monotonic_box_creation() {
        let true_tree = ErgoTree::new(
            ErgoTreeHeader::v0(true),
            &Expr::Const(Constant {
                tpe: ergotree_ir::types::stype::SType::SBoolean,
                v: Literal::Boolean(true),
            }),
        )
        .unwrap();

        let state_context_tx_gen = |tx: &Transaction, version| {
            let height = tx
                .output_candidates
                .iter()
                .map(|b| b.creation_height)
                .max()
                .unwrap();
            let mut state_context: ErgoStateContext = force_any_val();
            state_context.pre_header.height = height;
            state_context
                .parameters
                .parameters_table
                .insert(crate::chain::parameters::Parameter::BlockVersion, version);
            state_context
        };
        let box_gen = gen_boxes(
            5,
            10,
            5,
            10,
            Just(true_tree.clone()),
            Some((0..i32::MAX as u32).boxed()),
        );
        // Generate a list of boxes. If monotonic_valid is true then monotonic height validation will pass, otherwise it will fail in tests
        let tx_gen =
            (box_gen, bool::arbitrary()).prop_perturb(|(boxes, monotonic_valid), mut rng| {
                let max_height = boxes.iter().map(|b| b.creation_height).max().unwrap();
                let mut unsigned_tx = valid_unsigned_transaction_from_boxes(
                    rng.clone(),
                    &boxes,
                    true,
                    true_tree.clone(),
                    &[],
                );
                if monotonic_valid {
                    unsigned_tx
                        .output_candidates
                        .iter_mut()
                        .for_each(|b| b.creation_height = max_height + rng.gen_range(1..1000));
                } else {
                    unsigned_tx.output_candidates.iter_mut().for_each(|b| {
                        b.creation_height = max_height.saturating_sub(rng.gen_range(1..1000))
                    });
                }
                let wallet = Wallet::from_secrets(vec![]);
                let state_context = force_any_val();
                let tx_context =
                    TransactionContext::new(unsigned_tx, boxes.clone(), vec![]).unwrap();
                let signed_tx = wallet
                    .sign_transaction(tx_context, &state_context, None)
                    .unwrap();
                (boxes, signed_tx, monotonic_valid)
            });
        proptest!(|((boxes, tx, monotonic_valid) in tx_gen)| {
            assert!(tx.validate_stateless().is_ok());

            // For blocks V1 and V2 monotonic height rule is not respected.
            let context1 = state_context_tx_gen(&tx, 1);
            let context2 = state_context_tx_gen(&tx, 2);
            // V3 enforces monotonic height rule, thus validation should fail if !monotonic_valid
            let context3 = state_context_tx_gen(&tx, 3);
            let tx_context = TransactionContext::new(tx, boxes, vec![]).unwrap();
            match tx_context.validate(&context1) {
                Ok(_) => {},
                other => panic!("Expected validation to succeed, got {other:?}")
            }
            match tx_context.validate(&context2) {
                Ok(_) => {},
                other => panic!("Expected validation to succeed, got {other:?}")
            }
            match (monotonic_valid, tx_context.validate(&context3)) {
                (true, Ok(_)) => {},
                (false, Err(TxValidationError::MonotonicHeightError(_, _))) => {},
                other => panic!("Expected validation to fail, got {other:?}")
            }
        });
    }

    // ---- storage-rent cost in the JIT validate loop ----
    mod storage_rent_cost {
        use super::super::compute_tx_init_cost;
        use crate::chain::parameters::Parameter;
        use crate::chain::transaction::ergo_transaction::TxValidationError;
        use crate::chain::transaction::storage_rent::test_support::{
            expired_box, recreated, tx_spending,
        };
        use crate::chain::transaction::storage_rent::{STORAGE_CONTRACT_COST, STORAGE_PERIOD};

        #[test]
        fn rent_input_is_charged_storage_contract_cost() {
            let b = expired_box(5_000_000_000, 0, 0);
            let p = STORAGE_PERIOD;
            let (tx_ctx, sc) = tx_spending(
                std::slice::from_ref(&b),
                &[Some(0)],
                &[recreated(&b, 5_000_000_000, p)],
                p,
            );
            let init = compute_tx_init_cost(
                &tx_ctx.spending_tx,
                tx_ctx.boxes_to_spend.as_slice(),
                &sc.parameters,
            );
            assert_eq!(tx_ctx.validate(&sc).unwrap(), init + STORAGE_CONTRACT_COST);
        }

        #[test]
        fn two_rent_inputs_cost_100() {
            let (b0, b1) = (
                expired_box(5_000_000_000, 0, 0),
                expired_box(3_000_000_000, 0, 1),
            );
            let p = STORAGE_PERIOD;
            let outs = [
                recreated(&b0, 5_000_000_000, p),
                recreated(&b1, 3_000_000_000, p),
            ];
            let (tx_ctx, sc) = tx_spending(&[b0, b1], &[Some(0), Some(1)], &outs, p);
            let init = compute_tx_init_cost(
                &tx_ctx.spending_tx,
                tx_ctx.boxes_to_spend.as_slice(),
                &sc.parameters,
            );
            assert_eq!(
                tx_ctx.validate(&sc).unwrap(),
                init + 2 * STORAGE_CONTRACT_COST
            );
        }

        #[test]
        fn rent_cost_is_checked_against_the_block_limit() {
            let b = expired_box(5_000_000_000, 0, 0);
            let p = STORAGE_PERIOD;
            let (tx_ctx, mut sc) = tx_spending(
                std::slice::from_ref(&b),
                &[Some(0)],
                &[recreated(&b, 5_000_000_000, p)],
                p,
            );
            let init = compute_tx_init_cost(
                &tx_ctx.spending_tx,
                tx_ctx.boxes_to_spend.as_slice(),
                &sc.parameters,
            );
            let limit = |sc: &mut crate::chain::ergo_state_context::ErgoStateContext, v: u64| {
                sc.parameters
                    .parameters_table
                    .insert(Parameter::MaxBlockCost, v as i32);
            };
            limit(&mut sc, init + STORAGE_CONTRACT_COST);
            assert_eq!(tx_ctx.validate(&sc).unwrap(), init + STORAGE_CONTRACT_COST);
            limit(&mut sc, init + STORAGE_CONTRACT_COST - 1);
            assert!(matches!(
                tx_ctx.validate(&sc),
                Err(TxValidationError::VerifierError(0, _))
            ));
        }
    }
}
