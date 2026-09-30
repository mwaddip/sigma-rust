//! A transaction's id is written at a version: the one it was read at, or the one it is built at

use ergo_lib::chain::transaction::input::prover_result::ProverResult;
use ergo_lib::chain::transaction::{Input, Transaction};
use ergo_lib::ergotree_interpreter::sigma_protocol::prover::ProofBytes;
use ergo_lib::ergotree_ir::chain::context_extension::ContextExtension;
use ergo_lib::ergotree_ir::chain::ergo_box::box_value::BoxValue;
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
