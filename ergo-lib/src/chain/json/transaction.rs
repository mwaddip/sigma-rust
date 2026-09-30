use alloc::string::String;
use alloc::vec::Vec;
use core::convert::TryFrom;
use thiserror::Error;

use crate::chain::transaction::unsigned::UnsignedTransaction;
use crate::chain::transaction::{DataInput, Input, Transaction, TransactionError, UnsignedInput};
use ergotree_ir::chain::ergo_box::ErgoBox;
use ergotree_ir::chain::ergo_box::ErgoBoxCandidate;
use ergotree_ir::chain::tx_id::TxId;
use serde::{Deserialize, Serialize};

/// A transaction's JSON, with its inputs as `I` and its outputs as `O`
#[derive(Serialize, Deserialize, PartialEq, Eq, Debug, Clone)]
pub struct TransactionJson<I = Input, O = ErgoBox> {
    #[cfg_attr(feature = "json", serde(rename = "id"))]
    pub tx_id: TxId,
    /// inputs, that will be spent by this transaction.
    #[cfg_attr(feature = "json", serde(rename = "inputs"))]
    pub inputs: Vec<I>,
    /// inputs, that are not going to be spent by transaction, but will be reachable from inputs
    /// scripts. `dataInputs` scripts will not be executed, thus their scripts costs are not
    /// included in transaction cost and they do not contain spending proofs.
    #[cfg_attr(feature = "json", serde(rename = "dataInputs"))]
    pub data_inputs: Vec<DataInput>,
    #[cfg_attr(feature = "json", serde(rename = "outputs"))]
    pub outputs: Vec<O>,
}

/// An unsigned transaction's JSON, with its inputs as `I` and its outputs as `O`
#[derive(Serialize, Deserialize, PartialEq, Eq, Debug, Clone)]
pub struct UnsignedTransactionJson<I = UnsignedInput, O = ErgoBoxCandidate> {
    /// unsigned inputs, that will be spent by this transaction.
    #[cfg_attr(feature = "json", serde(rename = "inputs"))]
    pub inputs: Vec<I>,
    /// inputs, that are not going to be spent by transaction, but will be reachable from inputs
    /// scripts. `dataInputs` scripts will not be executed, thus their scripts costs are not
    /// included in transaction cost and they do not contain spending proofs.
    #[cfg_attr(feature = "json", serde(rename = "dataInputs"))]
    pub data_inputs: Vec<DataInput>,
    /// box candidates to be created by this transaction
    #[cfg_attr(feature = "json", serde(rename = "outputs"))]
    pub outputs: Vec<O>,
}

/// An unsigned transaction's id and message to sign are written at version 3, as a built
/// transaction's are, and so are its JSON's context extension and output register values: the
/// JSON reads back to the same transaction
impl Serialize for UnsignedTransaction {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let version = Transaction::BUILT_ID_VERSION;
        UnsignedTransactionJson {
            inputs: self.inputs.iter().map(|i| i.json_at(version)).collect(),
            data_inputs: self
                .data_inputs
                .as_ref()
                .map(|di| di.as_vec().clone())
                .unwrap_or_default(),
            outputs: self
                .output_candidates
                .iter()
                .map(|o| o.json_at(version))
                .collect(),
        }
        .serialize(serializer)
    }
}

impl TryFrom<UnsignedTransactionJson> for UnsignedTransaction {
    // We never return this type but () fails to compile (can't format) and ! is experimental
    type Error = String;
    fn try_from(tx_json: UnsignedTransactionJson) -> Result<Self, Self::Error> {
        UnsignedTransaction::new_from_vec(tx_json.inputs, tx_json.data_inputs, tx_json.outputs)
            .map_err(|e| format!("TryFrom<UnsignedTransactionJson> error: {0:?}", e))
    }
}

/// A transaction's JSON writes its context extension and output register values at the version
/// its id and message to sign are written at, so the JSON reads back to that id. ergo's JSON
/// writes them under the caller's version context, which its API routes leave at the default
/// (1, 1), below ErgoTree version 3 (sigma-state 6.0.6 `JsonCodecs.scala:184-185`, `:239-242`):
/// there a transaction read at 3, from a v6 block, has JSON whose values its id does not hash.
impl Serialize for Transaction {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let version = self.id_version;
        TransactionJson {
            tx_id: self.tx_id,
            inputs: self.inputs.iter().map(|i| i.json_at(version)).collect(),
            data_inputs: self
                .data_inputs
                .as_ref()
                .map(|di| di.as_vec().clone())
                .unwrap_or_default(),
            outputs: self.outputs.iter().map(|o| o.json_at(version)).collect(),
        }
        .serialize(serializer)
    }
}

/// Errors on parsing Transaction from JSON
#[derive(Error, PartialEq, Eq, Debug, Clone)]
#[allow(missing_docs)]
pub enum TransactionFromJsonError {
    #[error(
        "Tx id parsed from JSON {expected} differs from calculated from serialized bytes {actual}"
    )]
    InvalidTxId { expected: TxId, actual: TxId },
    #[error("Tx error: {0}")]
    TransactionError(#[from] TransactionError),
}

impl TryFrom<TransactionJson> for Transaction {
    type Error = TransactionFromJsonError;
    fn try_from(tx_json: TransactionJson) -> Result<Self, Self::Error> {
        let output_candidates: Vec<ErgoBoxCandidate> =
            tx_json.outputs.iter().map(|o| o.clone().into()).collect();
        let tx = Transaction::new_from_vec(tx_json.inputs, tx_json.data_inputs, output_candidates)?;
        if tx.tx_id == tx_json.tx_id {
            Ok(tx)
        } else {
            //dbg!(&tx);
            Err(TransactionFromJsonError::InvalidTxId {
                expected: tx_json.tx_id,
                actual: tx.tx_id,
            })
        }
    }
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used)]
mod tests {
    use crate::chain::transaction::unsigned::UnsignedTransaction;
    use crate::chain::transaction::Transaction;
    use ergo_chain_types::Digest32;
    use proptest::prelude::*;

    proptest! {

        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn tx_roundtrip(t in any::<Transaction>()) {
            let j = serde_json::to_string(&t)?;
            eprintln!("{}", j);
            let t_parsed: Transaction = serde_json::from_str(&j)?;
            prop_assert_eq![t, t_parsed];
        }

        #[test]
        fn unsigned_tx_roundtrip(t in any::<UnsignedTransaction>()) {
            let j = serde_json::to_string(&t)?;
            eprintln!("{}", j);
            let t_parsed: UnsignedTransaction = serde_json::from_str(&j)?;
            prop_assert_eq![t, t_parsed];
        }

    }

    #[test]
    fn item_order_preservation_685() {
        let tx_json = r#"
{
        "id" : "c8520befd345ff40fcf244b44ffe8cea29c8b116b174cfaf4f2a521604d531a4",
        "inputs" : [
          {
            "boxId" : "59f2856068c56264d290520043044ace138a3a80d414748d0e4dcd0806188546",
            "spendingProof" : {
              "proofBytes" : "",
              "extension" : {
                "0" : "04c60f",
                "5" : "0514",
                "10" : "0eee03101808cd0279aed8dea2b2a25316d5d49d13bf51c0b2c1dc696974bb4b0c07b5894e998e56040005e0e0a447040404060402040004000e201d5afc59838920bb5ef2a8f9d63825a55b1d48e269d7cecee335d637c3ff5f3f0e20003bd19d0187117f130b62e1bcab0939929ff5c7709f843c5c4dd158949285d005e201058c85a2010514040404c60f06010104d00f05e0e0a44704c60f0e691005040004000e36100204a00b08cd0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798ea02d192a39a8cc7a701730073011001020402d19683030193a38cc7b2a57300000193c2b2a57301007473027303830108cdeeac93b1a57304050005000580ade2040100d803d6017300d602b2a4730100d6037302eb027201d195ed93b1a4730393b1db630872027304d804d604db63087202d605b2a5730500d606b2db63087205730600d6077e8c72060206edededededed938cb2720473070001730893c27205d07201938c72060173099272077e730a06927ec172050699997ec1a7069d9c72077e730b067e730c067e720306909c9c7e8cb27204730d0002067e7203067e730e069c9a7207730f9a9c7ec17202067e7310067e9c73117e7312050690b0ada5d90108639593c272087313c1720873147315d90108599a8c7208018c72080273167317",
                "1" : "0e20003bd19d0187117f130b62e1bcab0939929ff5c7709f843c5c4dd158949285d0",
                "6" : "0580ade204",
                "9" : "0580b48913",
                "2" : "05e201",
                "7" : "0e201d5afc59838920bb5ef2a8f9d63825a55b1d48e269d7cecee335d637c3ff5f3f",
                "3" : "05e0e0a447",
                "8" : "0580ade204",
                "4" : "058c85a201"
              }
            }
          }
        ],
        "dataInputs" : [
        ],
        "outputs" : [
          {
            "boxId" : "0586c90d0cf6a82dab48c6a79500364ddbd6f81705f5032b03aa287de43dc638",
            "value" : 94750000,
            "ergoTree" : "101808cd0279aed8dea2b2a25316d5d49d13bf51c0b2c1dc696974bb4b0c07b5894e998e56040005e0e0a447040404060402040004000e201d5afc59838920bb5ef2a8f9d63825a55b1d48e269d7cecee335d637c3ff5f3f0e20003bd19d0187117f130b62e1bcab0939929ff5c7709f843c5c4dd158949285d005e201058c85a2010514040404c60f06010104d00f05e0e0a44704c60f0e691005040004000e36100204a00b08cd0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798ea02d192a39a8cc7a701730073011001020402d19683030193a38cc7b2a57300000193c2b2a57301007473027303830108cdeeac93b1a57304050005000580ade2040100d803d6017300d602b2a4730100d6037302eb027201d195ed93b1a4730393b1db630872027304d804d604db63087202d605b2a5730500d606b2db63087205730600d6077e8c72060206edededededed938cb2720473070001730893c27205d07201938c72060173099272077e730a06927ec172050699997ec1a7069d9c72077e730b067e730c067e720306909c9c7e8cb27204730d0002067e7203067e730e069c9a7207730f9a9c7ec17202067e7310067e9c73117e7312050690b0ada5d90108639593c272087313c1720873147315d90108599a8c7208018c72080273167317",
            "assets" : [
            ],
            "creationHeight" : 693475,
            "additionalRegisters" : {

            },
            "transactionId" : "c8520befd345ff40fcf244b44ffe8cea29c8b116b174cfaf4f2a521604d531a4",
            "index" : 0
          },
          {
            "boxId" : "4b99c2ef8496a491d176ecaf789d9e1d9aad0c2bf3e70b32e8bad73f48c722b9",
            "value" : 250000,
            "ergoTree" : "100e04000500059a0505d00f04020404040608cd03c6543ac8e8059748b1c6209ee419dd49a19ffaf5712a2f34a9412016a3a1d96708cd035b736bebf0c5393f78329f6894af84d1864c7496cc65ddc250ef60cdd75df52008cd021b63e19ab452c84cdc6687242e8494957b1f11e3750c8c184a8425f8a8171d9b05060580ade2040580a8d6b907040ad806d601b2a5730000d602b0a47301d9010241639a8c720201c18c720202d6039d9c730272027303d604b2a5730400d605b2a5730500d606b2a5730600d1968306019683020193c17201720393c27201d073079683020193c17204720393c27204d073089683020193c17205720393c27205d073099683020192c17206999972029c730a7203730b93c2a7c27206927202730c93b1a5730d",
            "assets" : [
            ],
            "creationHeight" : 693475,
            "additionalRegisters" : {

            },
            "transactionId" : "c8520befd345ff40fcf244b44ffe8cea29c8b116b174cfaf4f2a521604d531a4",
            "index" : 1
          },
          {
            "boxId" : "00ddbeb981c0b08536f72ea41e07a25adbf7bf104ee59b865619a21676e64715",
            "value" : 5000000,
            "ergoTree" : "1005040004000e36100204a00b08cd0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798ea02d192a39a8cc7a701730073011001020402d19683030193a38cc7b2a57300000193c2b2a57301007473027303830108cdeeac93b1a57304",
            "assets" : [
            ],
            "creationHeight" : 693475,
            "additionalRegisters" : {

            },
            "transactionId" : "c8520befd345ff40fcf244b44ffe8cea29c8b116b174cfaf4f2a521604d531a4",
            "index" : 2
          }
        ],
        "size" : 1562
      }
    "#;
        let tx: Transaction = serde_json::from_str(tx_json).unwrap();
        assert_eq!(
            tx.id(),
            Digest32::try_from(
                "c8520befd345ff40fcf244b44ffe8cea29c8b116b174cfaf4f2a521604d531a4".to_string()
            )
            .unwrap()
            .into()
        );
    }
}
