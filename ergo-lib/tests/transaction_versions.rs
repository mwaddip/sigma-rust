//! A transaction's id is written at a version: the one it was read at, or the one it is built at

use ergo_lib::chain::transaction::input::prover_result::ProverResult;
use ergo_lib::chain::transaction::{Input, Transaction};
use ergo_lib::ergotree_interpreter::sigma_protocol::prover::ProofBytes;
use ergo_lib::ergotree_ir::chain::context_extension::ContextExtension;
use ergo_lib::ergotree_ir::chain::ergo_box::box_value::BoxValue;
#[cfg(feature = "json")]
use ergo_lib::ergotree_ir::chain::ergo_box::ErgoBox;
use ergo_lib::ergotree_ir::chain::ergo_box::{
    BoxId, ErgoBoxCandidate, NonMandatoryRegisters, RegisterValue,
};
use ergo_lib::ergotree_ir::ergo_tree::{ErgoTree, ErgoTreeVersion};
use ergo_lib::ergotree_ir::serialization::sigma_byte_reader::{from_bytes, SigmaByteRead};
use ergo_lib::ergotree_ir::serialization::SigmaSerializable;

#[test]
#[allow(clippy::unwrap_used)]
fn a_transaction_built_at_a_version_has_the_id_a_read_at_that_version_computes() {
    // SANTA X15, `Tuple(1, Upcast(1, Long))`, as the output's R4. Below tree version 3 the
    // `Upcast` is written as the constant, so the id depends on the version it is written at.
    let x15 = RegisterValue::sigma_parse_bytes(&[0x86, 0x02, 0x04, 0x02, 0x7e, 0x04, 0x02, 0x05]);
    let output = ErgoBoxCandidate {
        value: BoxValue::try_from(1_000_000u64).unwrap(),
        ergo_tree: ErgoTree::sigma_parse_bytes(&[0x00, 0x08, 0xd3]).unwrap(),
        tokens: None,
        additional_registers: NonMandatoryRegisters::try_from(vec![x15]).unwrap(),
        creation_height: 1,
    };
    let input = Input::new(
        BoxId::zero(),
        ProverResult {
            proof: ProofBytes::Empty,
            extension: ContextExtension::empty(),
        },
    );
    let built_at = |version| {
        Transaction::new_from_vec_at(vec![input.clone()], vec![], vec![output.clone()], version)
            .unwrap()
    };
    let (v0, v3) = (built_at(ErgoTreeVersion::V0), built_at(ErgoTreeVersion::V3));
    assert_ne!(v0.id(), v3.id());
    // the transaction as written at version 3, read back at each version
    let bytes = v3.sigma_serialize_bytes().unwrap();
    for (version, built) in [(ErgoTreeVersion::V0, &v0), (ErgoTreeVersion::V3, &v3)] {
        let read = from_bytes(&bytes)
            .with_tree_version(version, Transaction::sigma_parse)
            .unwrap();
        assert_eq!(read.id(), built.id(), "{version:?}");
    }
}

#[cfg(feature = "json")]
#[test]
#[allow(clippy::unwrap_used)]
fn a_transaction_s_json_reads_back_to_its_id_whatever_version_it_was_read_or_built_at() {
    // SANTA X15 as the input's context extension variable 1 and as the output's R4. The
    // transaction's JSON writes both at the version its id is written at, so it reads back to
    // that id. An output's own JSON writes its registers at version 0, where its id is written.
    let x15 = [0x86, 0x02, 0x04, 0x02, 0x7e, 0x04, 0x02, 0x05];
    let input = Input::new(
        BoxId::zero(),
        ProverResult {
            proof: ProofBytes::Empty,
            extension: ContextExtension::sigma_parse_bytes(&[&[0x01, 0x01][..], &x15].concat())
                .unwrap(),
        },
    );
    let output = ErgoBoxCandidate {
        value: BoxValue::try_from(1_000_000u64).unwrap(),
        ergo_tree: ErgoTree::sigma_parse_bytes(&[0x00, 0x08, 0xd3]).unwrap(),
        tokens: None,
        additional_registers: NonMandatoryRegisters::try_from(vec![
            RegisterValue::sigma_parse_bytes(&x15),
        ])
        .unwrap(),
        creation_height: 1,
    };
    let built_at = |version| {
        Transaction::new_from_vec_at(vec![input.clone()], vec![], vec![output.clone()], version)
            .unwrap()
    };
    let bytes = built_at(ErgoTreeVersion::V3)
        .sigma_serialize_bytes()
        .unwrap();
    for (version, x15_written) in [
        (ErgoTreeVersion::V0, "860204020402"),
        (ErgoTreeVersion::V3, "860204027e040205"),
    ] {
        let read = from_bytes(&bytes)
            .with_tree_version(version, Transaction::sigma_parse)
            .unwrap();
        for tx in [built_at(version), read] {
            let json = serde_json::to_value(&tx).unwrap();
            let extension = &json["inputs"][0]["spendingProof"]["extension"];
            assert_eq!(extension["1"], x15_written, "{version:?}");
            assert_eq!(
                json["outputs"][0]["additionalRegisters"]["R4"], x15_written,
                "{version:?}"
            );
            let back: Transaction = serde_json::from_value(json).unwrap();
            assert_eq!(back.id(), tx.id(), "{version:?}");
            let output = tx.outputs.first();
            assert_eq!(
                back.outputs.first().box_id(),
                output.box_id(),
                "{version:?}"
            );
            let output_json = serde_json::to_value(output).unwrap();
            assert_eq!(
                output_json["additionalRegisters"]["R4"], "860204020402",
                "{version:?}"
            );
            let output_back: ErgoBox = serde_json::from_value(output_json).unwrap();
            assert_eq!(output_back.box_id(), output.box_id(), "{version:?}");
        }
    }
}
