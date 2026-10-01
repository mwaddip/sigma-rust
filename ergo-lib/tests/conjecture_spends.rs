//! Spends of sigma conjectures without children, from SANTA's `sized-tree-spend` vectors
//! (blessed on sigma-state 6.0.6). Each transaction gives the message that was signed and the
//! proof the JVM's prover made for it.

use ergo_lib::chain::transaction::Transaction;
use ergo_lib::ergotree_interpreter::sigma_protocol::prover::hint::HintsBag;
use ergo_lib::ergotree_interpreter::sigma_protocol::prover::{ProofBytes, Prover, TestProver};
use ergo_lib::ergotree_interpreter::sigma_protocol::verifier::verify_signature;
use ergo_lib::ergotree_ir::serialization::SigmaSerializable;
use ergo_lib::ergotree_ir::sigma_protocol::sigma_boolean::SigmaBoolean;

/// `sized-tree-spend` #3: the spent box's tree is the constant `CAND()`
const CAND_SPEND: &str = "0168099d3146ea9ca2c3032de8d8eac41f5e698868bd7154f4d9cb21eda596fb2c18ed8ac413b671f7eb2e295e951860a69d4fff40c4468aaf1a000000018094ebdc030008d3010000";

/// The message a transaction's inputs sign, and the proof its first input carries
#[allow(clippy::unwrap_used)]
fn message_and_proof(tx_hex: &str) -> (Vec<u8>, Vec<u8>) {
    let tx = Transaction::sigma_parse_bytes(&base16::decode(tx_hex).unwrap()).unwrap();
    let proof = Vec::<u8>::from(tx.inputs.first().spending_proof.proof.clone());
    (tx.bytes_to_sign().unwrap(), proof)
}

#[allow(clippy::unwrap_used)]
fn the_jvm_s_proof_verifies_and_is_what_the_prover_makes(sigma_bytes: &[u8], tx_hex: &str) {
    let proposition = SigmaBoolean::sigma_parse_bytes(sigma_bytes).unwrap();
    let (message, proof) = message_and_proof(tx_hex);
    assert_eq!(proof.len(), 24);
    assert!(verify_signature(proposition.clone(), &message, &proof).unwrap());
    // without a proof, and with another challenge, the spend is invalid
    assert!(!verify_signature(proposition.clone(), &message, &[]).unwrap());
    let mut wrong = proof.clone();
    wrong[23] ^= 1;
    assert!(!verify_signature(proposition.clone(), &message, &wrong).unwrap());
    let prover = TestProver { secrets: vec![] };
    let made = prover
        .generate_proof(proposition, &message, &HintsBag::empty())
        .unwrap();
    assert_eq!(made, ProofBytes::Some(proof));
}

#[test]
fn cand_without_children() {
    the_jvm_s_proof_verifies_and_is_what_the_prover_makes(&[0x96, 0x00], CAND_SPEND);
}
