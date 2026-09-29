//! Ergo transaction

mod data_input;
pub mod ergo_transaction;
pub mod input;
pub mod reduced;
pub(crate) mod storage_rent;
pub mod unsigned;

use alloc::string::String;
use alloc::vec::Vec;
use bounded_vec::BoundedVec;
use ergo_chain_types::blake2b256_hash;
use ergotree_interpreter::eval::env::Env;
use ergotree_interpreter::eval::extract_sigma_boolean;
use ergotree_interpreter::eval::EvalError;
use ergotree_interpreter::eval::ReductionDiagnosticInfo;
use ergotree_interpreter::sigma_protocol::verifier::verify_signature;
use ergotree_interpreter::sigma_protocol::verifier::TestVerifier;
use ergotree_interpreter::sigma_protocol::verifier::VerificationResult;
use ergotree_interpreter::sigma_protocol::verifier::Verifier;
use ergotree_interpreter::sigma_protocol::verifier::VerifierError;
use ergotree_ir::chain::context::Context;
pub use ergotree_ir::chain::context::TxIoVec;
use ergotree_ir::chain::ergo_box::BoxId;
use ergotree_ir::chain::ergo_box::ErgoBox;
use ergotree_ir::chain::ergo_box::ErgoBoxCandidate;
use ergotree_ir::chain::token::TokenId;
pub use ergotree_ir::chain::tx_id::TxId;
use ergotree_ir::chain::IndexSet;
use ergotree_ir::ergo_tree::ErgoTreeError;
use ergotree_ir::ergo_tree::ErgoTreeVersion;
use thiserror::Error;

pub use data_input::*;
use ergotree_interpreter::sigma_protocol::prover::ProofBytes;
use ergotree_ir::serialization::sigma_byte_reader::SigmaByteRead;
use ergotree_ir::serialization::sigma_byte_reader::MAX_ARRAY_LENGTH;
use ergotree_ir::serialization::sigma_byte_writer::SigmaByteWrite;
use ergotree_ir::serialization::sigma_byte_writer::SigmaByteWriter;
use ergotree_ir::serialization::SigmaParsingError;
use ergotree_ir::serialization::SigmaSerializable;
use ergotree_ir::serialization::SigmaSerializationError;
use ergotree_ir::serialization::SigmaSerializeResult;
pub use input::*;

use crate::wallet::signing::update_context;
use crate::wallet::signing::TransactionContext;
use crate::wallet::tx_context::TransactionContextError;

use self::ergo_transaction::TxValidationError;
use self::storage_rent::{storage_rent_verdict, StorageRentVerdict, STORAGE_CONTRACT_COST};
use self::unsigned::UnsignedTransaction;

use core::convert::TryFrom;
use core::convert::TryInto;
use core::iter::FromIterator;

use super::ergo_state_context::ErgoStateContext;

/**
 * ErgoTransaction is an atomic state transition operation. It destroys Boxes from the state
 * and creates new ones. If transaction is spending boxes protected by some non-trivial scripts,
 * its inputs should also contain proof of spending correctness - context extension (user-defined
 * key-value map) and data inputs (links to existing boxes in the state) that may be used during
 * script reduction to crypto, signatures that satisfies the remaining cryptographic protection
 * of the script.
 * Transactions are not encrypted, so it is possible to browse and view every transaction ever
 * collected into a block.
 */
#[cfg_attr(feature = "json", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "json",
    serde(
        try_from = "super::json::transaction::TransactionJson",
        into = "super::json::transaction::TransactionJson"
    )
)]
#[derive(Eq, Debug, Clone)]
pub struct Transaction {
    /// transaction id
    pub(crate) tx_id: TxId,
    /// The ErgoTree version the id and the message to sign are written at: the version the
    /// transaction was read at, since ergo computes the id as it reads one (v6.0.6
    /// `ErgoTransaction.scala:68`), or [`Transaction::BUILT_ID_VERSION`]
    id_version: ErgoTreeVersion,
    /// inputs, that will be spent by this transaction.
    pub inputs: TxIoVec<Input>,
    /// inputs, that are not going to be spent by transaction, but will be reachable from inputs
    /// scripts. `dataInputs` scripts will not be executed, thus their scripts costs are not
    /// included in transaction cost and they do not contain spending proofs.
    pub data_inputs: Option<TxIoVec<DataInput>>,

    /// box candidates to be created by this transaction. Differ from [`Self::outputs`] in that
    /// they do not include transaction id and index
    pub output_candidates: TxIoVec<ErgoBoxCandidate>,

    /// Boxes to be created by this transaction. Differ from [`Self::output_candidates`] in that
    /// they include transaction id and index
    pub outputs: TxIoVec<ErgoBox>,
}

// The id version shows in the id: two transactions read at different versions are equal
// unless it changed what their ids hash
impl PartialEq for Transaction {
    fn eq(&self, other: &Self) -> bool {
        self.tx_id == other.tx_id
            && self.inputs == other.inputs
            && self.data_inputs == other.data_inputs
            && self.output_candidates == other.output_candidates
            && self.outputs == other.outputs
    }
}

impl Transaction {
    /// Maximum number of outputs
    pub const MAX_OUTPUTS_COUNT: usize = u16::MAX as usize;

    /// The version a transaction built here writes its id and message to sign at: 3, which an
    /// ergo node reads a transaction from a peer at since 6.0 (v6.0.6
    /// `ErgoNodeViewSynchronizer.scala:793`)
    const BUILT_ID_VERSION: ErgoTreeVersion = ErgoTreeVersion::V3;

    /// Creates new transaction from vectors
    pub fn new_from_vec(
        inputs: Vec<Input>,
        data_inputs: Vec<DataInput>,
        output_candidates: Vec<ErgoBoxCandidate>,
    ) -> Result<Transaction, TransactionError> {
        Transaction::new_from_vec_at(
            inputs,
            data_inputs,
            output_candidates,
            Transaction::BUILT_ID_VERSION,
        )
    }

    fn new_from_vec_at(
        inputs: Vec<Input>,
        data_inputs: Vec<DataInput>,
        output_candidates: Vec<ErgoBoxCandidate>,
        id_version: ErgoTreeVersion,
    ) -> Result<Transaction, TransactionError> {
        Ok(Transaction::new_at(
            inputs
                .try_into()
                .map_err(TransactionError::InvalidInputsCount)?,
            BoundedVec::opt_empty_vec(data_inputs)
                .map_err(TransactionError::InvalidDataInputsCount)?,
            output_candidates
                .try_into()
                .map_err(TransactionError::InvalidOutputCandidatesCount)?,
            id_version,
        )?)
    }

    /// Creates new transaction
    pub fn new(
        inputs: TxIoVec<Input>,
        data_inputs: Option<TxIoVec<DataInput>>,
        output_candidates: TxIoVec<ErgoBoxCandidate>,
    ) -> Result<Transaction, SigmaSerializationError> {
        Transaction::new_at(
            inputs,
            data_inputs,
            output_candidates,
            Transaction::BUILT_ID_VERSION,
        )
    }

    fn new_at(
        inputs: TxIoVec<Input>,
        data_inputs: Option<TxIoVec<DataInput>>,
        output_candidates: TxIoVec<ErgoBoxCandidate>,
        id_version: ErgoTreeVersion,
    ) -> Result<Transaction, SigmaSerializationError> {
        let outputs_with_zero_tx_id =
            output_candidates
                .clone()
                .enumerated()
                .try_mapped_ref(|(idx, bc)| {
                    ErgoBox::from_box_candidate(bc, TxId::zero(), *idx as u16)
                })?;
        let tx_to_sign = Transaction {
            tx_id: TxId::zero(),
            id_version,
            inputs,
            data_inputs,
            output_candidates: output_candidates.clone(),
            outputs: outputs_with_zero_tx_id,
        };
        let tx_id = tx_to_sign.calc_tx_id()?;
        let outputs = output_candidates
            .enumerated()
            .try_mapped_ref(|(idx, bc)| ErgoBox::from_box_candidate(bc, tx_id, *idx as u16))?;
        Ok(Transaction {
            tx_id,
            outputs,
            ..tx_to_sign
        })
    }

    /// Create Transaction from UnsignedTransaction and an array of proofs in the same order as
    /// UnsignedTransaction.inputs
    pub fn from_unsigned_tx(
        unsigned_tx: UnsignedTransaction,
        proofs: Vec<ProofBytes>,
    ) -> Result<Self, TransactionError> {
        let inputs = unsigned_tx
            .inputs
            .enumerated()
            .try_mapped(|(index, unsigned_input)| {
                proofs
                    .get(index)
                    .map(|proof| Input::from_unsigned_input(unsigned_input, proof.clone()))
                    .ok_or_else(|| {
                        TransactionError::InvalidArgument(format!(
                            "no proof for input index: {}",
                            index
                        ))
                    })
            })?;
        Ok(Transaction::new(
            inputs,
            unsigned_tx.data_inputs,
            unsigned_tx.output_candidates,
        )?)
    }

    fn calc_tx_id(&self) -> Result<TxId, SigmaSerializationError> {
        let bytes = self.bytes_to_sign()?;
        Ok(TxId(blake2b256_hash(&bytes)))
    }

    /// Serialized tx with empty proofs, written at the version the id is
    pub fn bytes_to_sign(&self) -> Result<Vec<u8>, SigmaSerializationError> {
        let empty_proof_inputs = self.inputs.mapped_ref(|i| i.input_to_sign());
        let tx_to_sign = Transaction {
            inputs: empty_proof_inputs,
            ..(*self).clone()
        };
        let mut data = Vec::new();
        let mut w = SigmaByteWriter::new(&mut data, None);
        w.with_tree_version(self.id_version, |w| tx_to_sign.sigma_serialize(w))?;
        Ok(data)
    }

    /// Get transaction id
    pub fn id(&self) -> TxId {
        self.tx_id
    }

    /// Check the signature of the transaction's input corresponding
    /// to the given input box, guarded by P2PK script
    pub fn verify_p2pk_input(
        &self,
        input_box: ErgoBox,
    ) -> Result<bool, TransactionSignatureVerificationError> {
        #[allow(clippy::unwrap_used)]
        // since we have a tx with tx_id at this point, serialization is safe to unwrap
        let message = self.bytes_to_sign().unwrap();
        let input = self
            .inputs
            .iter()
            .find(|input| input.box_id == input_box.box_id())
            .ok_or_else(|| {
                TransactionSignatureVerificationError::InputNotFound(input_box.box_id())
            })?;
        let sb = extract_sigma_boolean(&input_box.ergo_tree.proposition()?)?;
        Ok(verify_signature(
            sb,
            message.as_slice(),
            input.spending_proof.proof.as_ref(),
        )?)
    }
}

#[allow(missing_docs)]
#[derive(Error, Debug)]
pub enum TransactionSignatureVerificationError {
    #[error("Input with id {0:?} not found")]
    InputNotFound(BoxId),
    #[error("input signature verification failed: {0:?}")]
    VerifierError(#[from] VerifierError),
    #[error("ErgoTreeError: {0}")]
    ErgoTreeError(#[from] ErgoTreeError),
    #[error("EvalError: {0}")]
    EvalError(#[from] EvalError),
}

/// Returns distinct token ids from all given ErgoBoxCandidate's
pub fn distinct_token_ids<'a, I>(output_candidates: I) -> IndexSet<TokenId>
where
    I: IntoIterator<Item = &'a ErgoBoxCandidate>,
{
    let token_ids = output_candidates
        .into_iter()
        .flat_map(|b| b.tokens.iter().flatten().map(|t| t.token_id));

    IndexSet::<_>::from_iter(token_ids)
}

impl SigmaSerializable for Transaction {
    #[allow(clippy::unwrap_used)]
    fn sigma_serialize<W: SigmaByteWrite>(&self, w: &mut W) -> SigmaSerializeResult {
        // At the writer's version: ergo writes a transaction under the version context around
        // it, a block's at the block version from v3 blocks (v6.0.6
        // `BlockTransactions.scala:150-160`)
        // reference implementation - https://github.com/ScorexFoundation/sigmastate-interpreter/blob/9b20cb110effd1987ff76699d637174a4b2fb441/sigmastate/src/main/scala/org/ergoplatform/ErgoLikeTransaction.scala#L112-L112
        w.put_usize_as_u16_unwrapped(self.inputs.len())?;
        self.inputs.iter().try_for_each(|i| i.sigma_serialize(w))?;
        if let Some(data_inputs) = &self.data_inputs {
            w.put_usize_as_u16_unwrapped(data_inputs.len())?;
            data_inputs.iter().try_for_each(|i| i.sigma_serialize(w))?;
        } else {
            w.put_u16(0)?;
        }

        // Serialize distinct ids of tokens in transaction outputs.
        let distinct_token_ids = distinct_token_ids(&self.output_candidates);

        // Note that `self.output_candidates` is of type `TxIoVec` which has a max length of
        // `u16::MAX`. Therefore the following unwrap is safe.
        w.put_u32(u32::try_from(distinct_token_ids.len()).unwrap())?;
        distinct_token_ids
            .iter()
            .try_for_each(|t_id| t_id.sigma_serialize(w))?;

        // serialize outputs
        w.put_usize_as_u16_unwrapped(self.output_candidates.len())?;
        self.output_candidates.iter().try_for_each(|o| {
            ErgoBoxCandidate::serialize_body_with_indexed_digests(o, Some(&distinct_token_ids), w)
        })?;
        Ok(())
    }

    fn sigma_parse<R: SigmaByteRead>(r: &mut R) -> Result<Self, SigmaParsingError> {
        // ergo's `ErgoTransactionSerializer.parse` reads each transaction on a new
        // `SigmaByteReader` (ergo v6.0.6 `ErgoTransaction.scala:497-503`), a block's
        // transactions included (`BlockTransactions.scala:187-200`). So a transaction starts
        // at level 0 with empty stores, whatever an earlier one left on the same stream.
        r.with_fresh_parse_state(|r| {
            // At the reader's version: ergo reads a transaction under the version context
            // around it, (3, 3) in a v6 block, the activated version from a peer
            // (`BlockTransactions.scala:184-202`, `ErgoNodeViewSynchronizer.scala:793`)
            // reference implementation - https://github.com/ScorexFoundation/sigmastate-interpreter/blob/9b20cb110effd1987ff76699d637174a4b2fb441/sigmastate/src/main/scala/org/ergoplatform/ErgoLikeTransaction.scala#L146-L146

            // parse transaction inputs
            let inputs_count = r.get_u16()?;
            let mut inputs = Vec::new();
            for _ in 0..inputs_count {
                inputs.push(Input::sigma_parse(r)?);
            }

            // parse transaction data inputs
            let data_inputs_count = r.get_u16()?;
            let mut data_inputs = Vec::new();
            for _ in 0..data_inputs_count {
                data_inputs.push(DataInput::sigma_parse(r)?);
            }

            // parse distinct ids of tokens in transaction outputs: sigmastate reads the count
            // with `getUIntExact` and allocates with `safeNewArray`, which refuses more than
            // `MaxArrayLength` (v6.0.6 `ErgoLikeTransaction.scala:162-166`)
            let tokens_count = r.get_u32()?;
            if tokens_count as usize > MAX_ARRAY_LENGTH {
                return Err(SigmaParsingError::ArrayLengthExceeded(
                    tokens_count as usize,
                ));
            }
            let mut token_ids = IndexSet::with_hasher(Default::default());
            for _ in 0..tokens_count {
                token_ids.insert(TokenId::sigma_parse(r)?);
            }

            // parse outputs
            let outputs_count = r.get_u16()?;
            let mut outputs = Vec::new();
            for _ in 0..outputs_count {
                outputs.push(ErgoBoxCandidate::parse_body_with_indexed_digests(
                    Some(&token_ids),
                    r,
                )?)
            }

            Transaction::new_from_vec_at(inputs, data_inputs, outputs, r.tree_version())
                .map_err(|e| SigmaParsingError::Misc(format!("{}", e)))
        })
    }
}

/// Error when working with Transaction
#[allow(missing_docs)]
#[derive(Error, Eq, PartialEq, Debug, Clone)]
pub enum TransactionError {
    #[error("Tx serialization error: {0}")]
    SigmaSerializationError(#[from] SigmaSerializationError),
    #[error("Tx innvalid argument: {0}")]
    InvalidArgument(String),
    #[error("Invalid Tx inputs: {0:?}")]
    InvalidInputsCount(bounded_vec::BoundedVecOutOfBounds),
    #[error("Invalid Tx output_candidates: {0:?}")]
    InvalidOutputCandidatesCount(bounded_vec::BoundedVecOutOfBounds),
    #[error("Invalid Tx data inputs: {0:?}")]
    InvalidDataInputsCount(bounded_vec::BoundedVecOutOfBounds),
    #[error("input with index {0} not found")]
    InputNofFound(usize),
}

/// Verify transaction input's proof
pub fn verify_tx_input_proof<'ctx>(
    tx_context: &'ctx TransactionContext<Transaction>,
    ctx: &mut Context<'ctx>,
    state_context: &ErgoStateContext,
    input_idx: usize,
    bytes_to_sign: &[u8],
) -> Result<VerificationResult, TxValidationError> {
    update_context(ctx, tx_context, input_idx)?;
    let input = tx_context
        .spending_tx
        .inputs
        .get(input_idx)
        .ok_or(TransactionContextError::InputBoxNotFound(input_idx))?;
    let input_box = tx_context
        .get_input_box(&input.box_id)
        .ok_or(TransactionContextError::InputBoxNotFound(input_idx))?;
    let verifier = TestVerifier;
    // Storage-rent branch of `ErgoInterpreter.verify` (`ErgoInterpreter.scala:66-87`):
    // a final verdict costing `StorageContractCost`, else ordinary script verification.
    match storage_rent_verdict(&input.spending_proof.proof, state_context, ctx) {
        StorageRentVerdict::Verdict(result) => Ok(VerificationResult {
            result,
            cost: STORAGE_CONTRACT_COST,
            diag: ReductionDiagnosticInfo {
                env: Env::empty(),
                pretty_printed_expr: None,
            },
        }),
        StorageRentVerdict::NotApplicable => verifier
            .verify(
                &input_box.ergo_tree,
                ctx,
                input.spending_proof.proof.clone(),
                bytes_to_sign,
            )
            .map_err(|e| TxValidationError::VerifierError(input_idx, e)),
    }
}

/// Arbitrary impl
#[cfg(feature = "arbitrary")]
#[allow(clippy::unwrap_used)]
pub mod arbitrary {

    use super::*;
    use proptest::prelude::*;
    use proptest::{arbitrary::Arbitrary, collection::vec};

    impl Arbitrary for Transaction {
        type Parameters = ();

        fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
            (
                vec(any::<Input>(), 1..10),
                vec(any::<DataInput>(), 0..10),
                vec(any::<ErgoBoxCandidate>(), 1..10),
            )
                .prop_map(|(inputs, data_inputs, outputs)| {
                    Self::new_from_vec(inputs, data_inputs, outputs).unwrap()
                })
                .boxed()
        }
        type Strategy = BoxedStrategy<Self>;
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {

    use core::str::FromStr;

    use crate::chain::transaction::prover_result::ProverResult;

    use super::*;

    use ergotree_ir::{
        chain::{
            context_extension::ContextExtension,
            ergo_box::{box_value::BoxValue, NonMandatoryRegisterId, NonMandatoryRegisters},
        },
        ergo_tree::ErgoTree,
        mir::{constant::Constant, val_def::ValId},
        serialization::{
            constant_store::ConstantStore, sigma_byte_reader::from_bytes, sigma_serialize_roundtrip,
        },
        types::stype::SType,
        unsignedbigint256::UnsignedBigInt,
    };
    use indexmap::IndexMap;
    use proptest::prelude::*;
    use sigma_test_util::force_any_val;

    proptest! {

        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn tx_ser_roundtrip(v in any::<Transaction>()) {
            prop_assert_eq![sigma_serialize_roundtrip(&v), v];
        }


        #[test]
        fn tx_id_ser_roundtrip(v in any::<TxId>()) {
            prop_assert_eq![sigma_serialize_roundtrip(&v), v];
        }

    }

    #[test]
    fn test_v6_types() {
        // An output is written below tree version 3, as ergo's default version context writes
        // it (`ErgoTransaction.scala:171-175`), where an `UnsignedBigInt` register has no
        // encoding. A context extension value is written at the transaction's version, 3 for
        // one built here; rule 1019 rejects it when the transaction is read
        // (`ContextExtension.scala:61-62`).
        let mut ergo_box = ErgoBoxCandidate {
            value: BoxValue::SAFE_USER_MIN,
            ergo_tree: force_any_val::<ErgoTree>(),
            tokens: None,
            additional_registers: NonMandatoryRegisters::new([(
                NonMandatoryRegisterId::R4,
                Constant::from(UnsignedBigInt::from_str("0").unwrap()),
            )])
            .unwrap(),
            creation_height: 0,
        };
        assert!(matches!(
            Transaction::new_from_vec(
                vec![Input::new(
                    BoxId::zero(),
                    ProverResult {
                        proof: ProofBytes::Empty,
                        extension: ContextExtension::empty(),
                    },
                )],
                vec![],
                vec![ergo_box.clone()],
            ),
            Err(TransactionError::SigmaSerializationError(_))
        ));
        ergo_box.additional_registers = NonMandatoryRegisters::empty();
        let tx = Transaction::new_from_vec(
            vec![Input::new(
                BoxId::zero(),
                ProverResult {
                    proof: ProofBytes::Empty,
                    extension: ContextExtension {
                        values: IndexMap::from_iter([(
                            0,
                            UnsignedBigInt::from_str("0").unwrap().into(),
                        )]),
                    },
                },
            )],
            vec![],
            vec![ergo_box.clone()],
        )
        .unwrap();
        assert!(matches!(
            Transaction::sigma_parse_bytes(&tx.sigma_serialize_bytes().unwrap()),
            Err(SigmaParsingError::V6TypeError)
        ));
    }

    #[test]
    #[cfg(feature = "json")]
    fn test_tx_id_calc() {
        let json = r#"
        {
      "id": "9148408c04c2e38a6402a7950d6157730fa7d49e9ab3b9cadec481d7769918e9",
      "inputs": [
        {
          "boxId": "9126af0675056b80d1fda7af9bf658464dbfa0b128afca7bf7dae18c27fe8456",
          "spendingProof": {
            "proofBytes": "",
            "extension": {}
          }
        }
      ],
      "dataInputs": [],
      "outputs": [
        {
          "boxId": "b979c439dc698ce5e823b21c722a6e23721af010e4df8c72de0bfd0c3d9ccf6b",
          "value": 74187765000000000,
          "ergoTree": "101004020e36100204a00b08cd0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798ea02d192a39a8cc7a7017300730110010204020404040004c0fd4f05808c82f5f6030580b8c9e5ae040580f882ad16040204c0944004c0f407040004000580f882ad16d19683030191a38cc7a7019683020193c2b2a57300007473017302830108cdeeac93a38cc7b2a573030001978302019683040193b1a5730493c2a7c2b2a573050093958fa3730673079973089c73097e9a730a9d99a3730b730c0599c1a7c1b2a5730d00938cc7b2a5730e0001a390c1a7730f",
          "assets": [],
          "creationHeight": 284761,
          "additionalRegisters": {},
          "transactionId": "9148408c04c2e38a6402a7950d6157730fa7d49e9ab3b9cadec481d7769918e9",
          "index": 0
        },
        {
          "boxId": "e56847ed19b3dc6b72828fcfb992fdf7310828cf291221269b7ffc72fd66706e",
          "value": 67500000000,
          "ergoTree": "100204a00b08cd021dde34603426402615658f1d970cfa7c7bd92ac81a8b16eeebff264d59ce4604ea02d192a39a8cc7a70173007301",
          "assets": [],
          "creationHeight": 284761,
          "additionalRegisters": {},
          "transactionId": "9148408c04c2e38a6402a7950d6157730fa7d49e9ab3b9cadec481d7769918e9",
          "index": 1
        }
      ]
    }"#;
        let res = serde_json::from_str(json);
        let t: Transaction = res.unwrap();
        let tx_id_str: String = t.id().into();
        assert_eq!(
            "9148408c04c2e38a6402a7950d6157730fa7d49e9ab3b9cadec481d7769918e9",
            tx_id_str
        )
    }

    /// Transaction::sigma_parse with a huge declared inputs count but no data
    /// must return Err without a multi-gigabyte pre-allocation.
    #[test]
    fn transaction_parse_huge_inputs_count_returns_err() {
        use ergotree_ir::serialization::SigmaSerializable;
        use sigma_ser::vlq_encode::WriteSigmaVlqExt;
        let mut data = Vec::new();
        let mut w =
            ergotree_ir::serialization::sigma_byte_writer::SigmaByteWriter::new(&mut data, None);
        w.put_u16(u16::MAX).unwrap();
        let result = Transaction::sigma_parse_bytes(&data);
        assert!(result.is_err());
    }

    /// A transaction spending one input (empty proof, context extension `extension`), with
    /// no data inputs and one 1 ERG output guarded by `tree` (creation height 1, no tokens,
    /// no registers)
    fn tx_bytes(extension: &[u8], tree: &[u8]) -> Vec<u8> {
        let mut bytes = vec![1];
        bytes.extend(
            base16::decode("a05d90c50251aea28100ccaa1da38004fb5a04154bfa300a600f3c10d130816a")
                .unwrap(),
        );
        bytes.push(0); // proof
        bytes.extend_from_slice(extension);
        bytes.extend_from_slice(&[0, 0, 1]); // data inputs, token ids, outputs
        bytes.extend(base16::decode("8094ebdc03").unwrap());
        bytes.extend_from_slice(tree);
        bytes.extend_from_slice(&[1, 0, 0]); // creation height, tokens, registers
        bytes
    }

    /// A size-flagged v0 tree `BoolToSigmaProp(LogicalNot^107(<0x75>))`: the unknown opcode
    /// 0x75 degrades it to `Unparsed`, leaving 109 levels behind on the reader
    fn tx_degrading_at_level_109() -> Vec<u8> {
        let mut tree = vec![0x08, 0x6d, 0xd1];
        tree.extend(core::iter::repeat_n(0xef, 107));
        tree.push(0x75);
        tx_bytes(&[0], &tree)
    }

    /// `BlockValue { ValDef(1, sigmaProp(true)) } ValUse(1)`
    fn tx_defining_val_1() -> Vec<u8> {
        tx_bytes(
            &[0],
            &[0x00, 0xd8, 0x01, 0xd6, 0x01, 0x08, 0xd3, 0x72, 0x01],
        )
    }

    /// Run `f` on a thread with a 32 MiB stack: a debug build takes ~40 KB of stack per
    /// expression level, and `tx_degrading_at_level_109` nests 109 of them.
    fn on_deep_stack(f: impl FnOnce() + Send + 'static) {
        std::thread::Builder::new()
            .stack_size(32 * 1024 * 1024)
            .spawn(f)
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn transactions_on_one_reader_each_start_at_level_zero() {
        // The second transaction's `sigmaProp(true)` output takes two levels, which a reader
        // still at the first transaction's 109 would refuse at 111.
        on_deep_stack(|| {
            let second = tx_bytes(&[0], &[0x00, 0x08, 0xd3]);
            let mut r = from_bytes([tx_degrading_at_level_109(), second].concat());
            Transaction::sigma_parse(&mut r).unwrap();
            Transaction::sigma_parse(&mut r).unwrap();
        });
    }

    #[test]
    fn transactions_on_one_reader_do_not_share_val_def_types() {
        // `ValUse(1)` without a `ValDef` is rejected (sigmastate's `ValDefTypeStore` lookup
        // throws), also after a transaction that defined 1 on the same stream.
        let undefined_val_use = tx_bytes(&[0], &[0x00, 0x72, 0x01]);
        let mut r = from_bytes([tx_defining_val_1(), undefined_val_use].concat());
        Transaction::sigma_parse(&mut r).unwrap();
        assert!(matches!(
            Transaction::sigma_parse(&mut r),
            Err(SigmaParsingError::ValDefIdNotFound(ValId(1)))
        ));
    }

    #[test]
    fn transaction_parse_leaves_the_reader_state_as_it_found_it() {
        on_deep_stack(|| {
            for tx in [tx_degrading_at_level_109(), tx_defining_val_1()] {
                let mut r = from_bytes(&tx);
                r.set_level(5).unwrap();
                r.set_constant_store(ConstantStore::new(vec![1i32.into()]));
                r.val_def_type_store().insert(ValId(7), SType::SInt);
                r.set_deserialize(true);
                Transaction::sigma_parse(&mut r).unwrap();
                assert_eq!(r.level(), 5);
                assert_eq!(r.constant_store().get(0).unwrap().tpe, SType::SInt);
                assert_eq!(r.val_def_type_store().get(&ValId(7)), Some(&SType::SInt));
                assert!(r.val_def_type_store().get(&ValId(1)).is_none());
                assert!(r.was_deserialize());
            }
        });
    }

    #[test]
    fn function_type_code_in_a_transaction_is_an_error_not_a_panic() {
        // A constant of type `(Int) => Int`, in a context extension value and in a register.
        // Below ErgoTree version 3 sigmastate's `CheckTypeCode` rejects type code 112; from
        // version 3 it is the function type, whose data has no encoding
        // (`CoreDataSerializer.scala:144-146`).
        let sfunc = [0x70, 0x01, 0x04, 0x04, 0x00];
        let in_extension = tx_bytes(&[&[0x01, 0x01][..], &sfunc].concat(), &[0x00, 0x08, 0xd3]);
        for tx in [in_extension, tx_with_r4(&sfunc)] {
            assert!(matches!(
                parse_at(&tx, ErgoTreeVersion::V0),
                Err(SigmaParsingError::InvalidTypeCode(112))
            ));
            assert!(parse_at(&tx, ErgoTreeVersion::V3).is_err());
        }
    }

    /// `tx` read at `version`
    fn parse_at(tx: &[u8], version: ErgoTreeVersion) -> Result<Transaction, SigmaParsingError> {
        let mut r = from_bytes(tx);
        r.with_tree_version(version, Transaction::sigma_parse)
    }

    /// [`tx_bytes`] with an empty context extension, a `sigmaProp(true)` output tree and
    /// `value` as the output's R4
    fn tx_with_r4(value: &[u8]) -> Vec<u8> {
        let mut tx = tx_bytes(&[0], &[0x00, 0x08, 0xd3]);
        tx.pop(); // registers count
        tx.push(1);
        tx.extend_from_slice(value);
        tx
    }

    #[test]
    fn a_transaction_is_read_and_its_id_written_at_its_reader_s_version() {
        // ergo reads a v6 block's transactions at version context (3, 3), and one from a peer
        // at the activated version (v6.0.6 `BlockTransactions.scala:184-202`,
        // `ErgoNodeViewSynchronizer.scala:793`), and computes the id there
        // (`ErgoTransaction.scala:68`). From tree version 3, type code 112 is the function type
        // (`TypeSerializer.scala:211`): SANTA C2, an empty `Coll[(Int, Int) => Int]` as a
        // context extension value, and its twin `Coll[Int => Int]`. The input's proof is empty,
        // so the signed bytes are the transaction's own.
        for value in [
            &[0x83, 0x00, 0x70, 0x02, 0x04, 0x04, 0x04, 0x00][..],
            &[0x83, 0x00, 0x70, 0x01, 0x04, 0x04, 0x00],
        ] {
            let tx = tx_bytes(&[&[0x01, 0x00][..], value].concat(), &[0x00, 0x08, 0xd3]);
            let read = parse_at(&tx, ErgoTreeVersion::V3).unwrap();
            assert_eq!(read.id(), TxId(blake2b256_hash(&tx)), "{value:02x?}");
            assert_eq!(read.bytes_to_sign().unwrap(), tx, "{value:02x?}");
            assert!(
                matches!(
                    parse_at(&tx, ErgoTreeVersion::V0),
                    Err(SigmaParsingError::InvalidTypeCode(112))
                ),
                "{value:02x?}"
            );
        }
    }

    /// `bytes` with each `Upcast(1, Long)` written as the constant 1
    fn without_upcast(bytes: &[u8]) -> Vec<u8> {
        let upcast = [0x7e, 0x04, 0x02, 0x05];
        let mut out = Vec::new();
        let mut i = 0;
        while i < bytes.len() {
            if bytes[i..].starts_with(&upcast) {
                out.extend_from_slice(&[0x04, 0x02]);
                i += upcast.len();
            } else {
                out.push(bytes[i]);
                i += 1;
            }
        }
        out
    }

    #[test]
    fn an_upcast_of_a_constant_stays_in_the_id_from_version_3_and_leaves_an_output_s_id() {
        // SANTA X15, `Tuple(1, Upcast(1, Long))`, as a context extension value and as the
        // output's R4. ergo computes the id as it reads the transaction, and writes the output
        // under its default version context (1, 1) (`ErgoTransaction.scala:68`, `:171-175`);
        // below tree version 3 an `Upcast` of a constant is written as the constant
        // (`ValueSerializer.scala:157-169`). The input's proof is empty, so the signed bytes
        // are the transaction's own.
        let x15 = [0x86, 0x02, 0x04, 0x02, 0x7e, 0x04, 0x02, 0x05];
        let mut tx = tx_bytes(&[&[0x01, 0x00][..], &x15].concat(), &[0x00, 0x08, 0xd3]);
        tx.pop(); // registers count
        tx.push(1);
        tx.extend_from_slice(&x15);
        let stripped = without_upcast(&tx);
        assert_eq!(stripped.len(), tx.len() - 4);
        for (version, signed) in [(ErgoTreeVersion::V3, &tx), (ErgoTreeVersion::V0, &stripped)] {
            let read = parse_at(&tx, version).unwrap();
            assert_eq!(read.bytes_to_sign().unwrap(), *signed, "{version:?}");
            assert_eq!(read.id(), TxId(blake2b256_hash(signed)), "{version:?}");
            let output = read.outputs.first();
            let at_v3 = output.sigma_serialize_bytes().unwrap();
            let written = without_upcast(&at_v3);
            assert_eq!(written.len(), at_v3.len() - 2, "{version:?}");
            assert_eq!(output.bytes().unwrap(), written, "{version:?}");
            assert_eq!(
                output.box_id(),
                BoxId::from(blake2b256_hash(&written)),
                "{version:?}"
            );
        }
    }

    #[test]
    fn an_output_the_default_context_cannot_write_fails_the_transaction() {
        // ergo writes an output under its default version context (1, 1), where a function
        // type has no encoding (`TypeSerializer.scala:111`), and the output checks throw
        // (`ErgoTransaction.scala:171-175`): an output holding SANTA C2's twin as R4 fails the
        // transaction read at version 3 as well
        let tx = tx_with_r4(&[0x83, 0x00, 0x70, 0x01, 0x04, 0x04, 0x00]);
        assert!(parse_at(&tx, ErgoTreeVersion::V3).is_err());
        assert!(matches!(
            parse_at(&tx, ErgoTreeVersion::V0),
            Err(SigmaParsingError::InvalidTypeCode(112))
        ));
    }

    #[test]
    fn output_tree_header_bits_reach_the_tx_id() {
        // SANTA `Transaction.tree_header_bits`: sigmastate writes an output tree's header
        // byte back whole, bits 5-7 included, so the id hashes the output as received. The
        // input's proof is empty, so the signed bytes are the transaction's own.
        for tree in [
            &[0x28, 0x02, 0x08, 0xd3][..],
            &[0x48, 0x02, 0x08, 0xd3],
            &[0x88, 0x02, 0x08, 0xd3],
            &[0xe8, 0x02, 0x08, 0xd3],
            &[0xe0, 0x08, 0xd3],
        ] {
            let bytes = tx_bytes(&[0], tree);
            let tx = Transaction::sigma_parse_bytes(&bytes).unwrap();
            assert_eq!(tx.id(), TxId(blake2b256_hash(&bytes)), "{tree:02x?}");
        }
    }

    #[test]
    fn an_inputs_count_written_above_u32_is_its_low_32_bits() {
        // sigmastate reads the inputs count with `getUShort` (`ErgoLikeTransaction.scala:148`),
        // which narrows it to an `Int` before its range check: written as 2^32 + 1, it is one
        // input, and the id is over the count re-encoded
        let canonical = tx_bytes(&[0], &[0x00, 0x08, 0xd3]);
        let mut wrapped = vec![0x81, 0x80, 0x80, 0x80, 0x10];
        wrapped.extend_from_slice(&canonical[1..]);
        let tx = Transaction::sigma_parse_bytes(&wrapped).unwrap();
        assert_eq!(tx.sigma_serialize_bytes().unwrap(), canonical);
        assert_eq!(tx.id(), TxId(blake2b256_hash(&canonical)));
    }

    #[test]
    fn a_tokens_count_above_max_array_length_rejects() {
        // sigmastate reads the count of distinct token ids with `getUIntExact` and allocates
        // with `safeNewArray` (`ErgoLikeTransaction.scala:162-166`): 100001 is refused before
        // an id is read
        let canonical = tx_bytes(&[0], &[0x00, 0x08, 0xd3]);
        // the input count, the box id, an empty proof and extension, no data inputs
        let tokens_count_at = 1 + 32 + 2 + 1;
        assert_eq!(canonical[tokens_count_at], 0);
        let bytes = [
            &canonical[..tokens_count_at],
            &[0xa1, 0x8d, 0x06],
            &canonical[tokens_count_at + 1..],
        ]
        .concat();
        assert!(matches!(
            Transaction::sigma_parse_bytes(&bytes),
            Err(SigmaParsingError::ArrayLengthExceeded(n)) if n == MAX_ARRAY_LENGTH + 1
        ));
    }

    #[test]
    fn deeply_nested_extension_value_types_end_in_an_error_on_a_2_mib_stack() {
        // `Coll^n[Byte]` as a context extension value: type `0c`×(n−2) `1a`, then data
        // nested all the way down (`01`×(n−1) `00`) or an empty outer collection (`00`). On
        // a 2 MiB stack, a tokio worker's, each ends at the temporary type bound instead of
        // running the recursive type parse out of stack.
        for (n, nested_data) in [(2000, true), (2500, true), (2000, false), (2500, false)] {
            let mut value = vec![0x0c; n - 2];
            value.push(0x1a);
            if nested_data {
                value.extend(core::iter::repeat_n(0x01, n - 1));
            }
            value.push(0x00);
            let tx = tx_bytes(&[&[0x01, 0x01][..], &value].concat(), &[0x00, 0x08, 0xd3]);
            let res = std::thread::Builder::new()
                .stack_size(2 << 20)
                .spawn(move || Transaction::sigma_parse_bytes(&tx).err())
                .unwrap()
                .join()
                .unwrap();
            assert!(
                matches!(res, Some(SigmaParsingError::TypeDepthExceeded(111))),
                "n = {n}, nested data = {nested_data}: {res:?}"
            );
        }
    }
}
