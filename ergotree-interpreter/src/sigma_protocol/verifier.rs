//! Verifier

use super::dht_protocol;
use super::dht_protocol::FirstDhTupleProverMessage;
use super::fiat_shamir::FiatShamirTreeSerializationError;
use super::prover::ProofBytes;
use super::sig_serializer::SigParsingError;
use super::unchecked_tree::UncheckedDhTuple;
use super::{
    dlog_protocol,
    fiat_shamir::{fiat_shamir_hash_fn, fiat_shamir_tree_to_bytes},
    sig_serializer::parse_sig_compute_challenges,
    unchecked_tree::{UncheckedLeaf, UncheckedSchnorr},
    SigmaBoolean, UncheckedTree,
};
use crate::eval::EvalError;
use crate::eval::{reduce_to_crypto, ReductionDiagnosticInfo};
use dlog_protocol::FirstDlogProverMessage;
use ergotree_ir::chain::context::Context;
use ergotree_ir::ergo_tree::ErgoTree;
use ergotree_ir::ergo_tree::ErgoTreeError;

use derive_more::From;
use thiserror::Error;

/// Errors on proof verification
#[derive(Error, Debug, From)]
pub enum VerifierError {
    /// Failed to parse ErgoTree from bytes
    #[error("ErgoTreeError: {0}")]
    ErgoTreeError(ErgoTreeError),
    /// Failed to evaluate ErgoTree
    #[error("EvalError: {0}")]
    EvalError(EvalError),
    /// Signature parsing error
    #[error("SigParsingError: {0}")]
    SigParsingError(SigParsingError),
    /// Error while tree serialization for Fiat-Shamir hash
    #[error("Fiat-Shamir tree serialization error: {0}")]
    FiatShamirTreeSerializationError(FiatShamirTreeSerializationError),
}

/// Result of Box.ergoTree verification procedure (see `verify` method).
#[derive(Debug, Clone)]
pub struct VerificationResult {
    /// result of SigmaProp condition verification via sigma protocol
    pub result: bool,
    /// estimated cost of contract execution
    pub cost: u64,
    /// Diagnostic information about the reduction (pretty printed expr and/or env)
    pub diag: ReductionDiagnosticInfo,
}

/// Verifier for the proofs generater by [`super::prover::Prover`]
pub trait Verifier {
    /// Executes the script in a given context.
    /// Step 1: Deserialize context variables
    /// Step 2: Evaluate expression and produce SigmaProp value, which is zero-knowledge statement (see also `SigmaBoolean`).
    /// Step 3: Verify that the proof is presented to satisfy SigmaProp conditions.
    fn verify<'ctx>(
        &self,
        tree: &ErgoTree,
        ctx: &Context<'ctx>,
        proof: ProofBytes,
        message: &[u8],
    ) -> Result<VerificationResult, VerifierError> {
        let reduction_result = reduce_to_crypto(tree, ctx)?;
        let res: bool = match reduction_result.sigma_prop {
            SigmaBoolean::TrivialProp(b) => b,
            sb => {
                match proof {
                    ProofBytes::Empty => false,
                    ProofBytes::Some(proof_bytes) => {
                        // Perform Verifier Steps 1-3
                        let unchecked_tree = parse_sig_compute_challenges(&sb, proof_bytes)?;
                        // Perform Verifier Steps 4-6
                        check_commitments(unchecked_tree, message)?
                    }
                }
            }
        };
        Ok(VerificationResult {
            result: res,
            cost: reduction_result.cost,
            diag: reduction_result.diag,
        })
    }
}

/// Verify that the signature is presented to satisfy SigmaProp conditions.
pub fn verify_signature(
    sigma_tree: SigmaBoolean,
    message: &[u8],
    signature: &[u8],
) -> Result<bool, VerifierError> {
    let res: bool = match sigma_tree {
        SigmaBoolean::TrivialProp(b) => b,
        sb => {
            match signature {
                [] => false,
                _ => {
                    // Perform Verifier Steps 1-3
                    let unchecked_tree = parse_sig_compute_challenges(&sb, signature.to_vec())?;
                    // Perform Verifier Steps 4-6
                    check_commitments(unchecked_tree, message)?
                }
            }
        }
    };
    Ok(res)
}

/// Perform Verifier Steps 4-6
fn check_commitments(sp: UncheckedTree, message: &[u8]) -> Result<bool, VerifierError> {
    // Perform Verifier Step 4
    let new_root = compute_commitments(sp);
    let root_challenge = new_root.challenge();
    let mut s = fiat_shamir_tree_to_bytes(&new_root.into())?;
    s.extend_from_slice(message);
    // Verifier Steps 5-6: Convert the tree to a string `s` for input to the Fiat-Shamir hash function,
    // using the same conversion as the prover in 7
    // Accept the proof if the challenge at the root of the tree is equal to the Fiat-Shamir hash of `s`
    // (and, if applicable,  the associated data). Reject otherwise.
    let expected_challenge = fiat_shamir_hash_fn(s.as_slice());
    Ok(root_challenge == expected_challenge.into())
}

/// Verifier Step 4: For every leaf node, compute the commitment a from the challenge e and response $z$,
/// per the verifier algorithm of the leaf's Sigma-protocol.
/// If the verifier algorithm of the Sigma-protocol for any of the leaves rejects, then reject the entire proof.
pub fn compute_commitments(sp: UncheckedTree) -> UncheckedTree {
    match sp {
        UncheckedTree::UncheckedLeaf(leaf) => match leaf {
            UncheckedLeaf::UncheckedSchnorr(sn) => {
                let a = dlog_protocol::interactive_prover::compute_commitment(
                    &sn.proposition,
                    &sn.challenge,
                    &sn.second_message,
                );
                UncheckedSchnorr {
                    commitment_opt: Some(FirstDlogProverMessage { a: a.into() }),
                    ..sn
                }
                .into()
            }
            UncheckedLeaf::UncheckedDhTuple(dh) => {
                let (a, b) = dht_protocol::interactive_prover::compute_commitment(
                    &dh.proposition,
                    &dh.challenge,
                    &dh.second_message,
                );
                UncheckedDhTuple {
                    commitment_opt: Some(FirstDhTupleProverMessage::new(a, b)),
                    ..dh
                }
                .into()
            }
        },
        // sigmastate leaves an internal node as it is (`Interpreter.scala:407-409`): its children
        // are mapped where they are, not copied
        UncheckedTree::UncheckedConjecture(conj) => conj.map_children(compute_commitments).into(),
    }
}

/// Test Verifier implementation
pub struct TestVerifier;

impl Verifier for TestVerifier {}

#[allow(clippy::unwrap_used)]
#[allow(clippy::panic)]
#[cfg(test)]
#[cfg(feature = "arbitrary")]
mod tests {
    use core::convert::TryFrom;

    use crate::sigma_protocol::private_input::{DhTupleProverInput, DlogProverInput, PrivateInput};
    use crate::sigma_protocol::prover::hint::HintsBag;
    use crate::sigma_protocol::prover::{Prover, TestProver};

    use super::*;
    use ergotree_ir::mir::atleast::Atleast;
    use ergotree_ir::mir::constant::{Constant, Literal};
    use ergotree_ir::mir::expr::Expr;
    use ergotree_ir::mir::sigma_and::SigmaAnd;
    use ergotree_ir::mir::sigma_or::SigmaOr;
    use ergotree_ir::mir::value::CollKind;
    use ergotree_ir::sigma_protocol::sigma_boolean::SigmaProp;
    use ergotree_ir::types::stype::SType;
    use proptest::collection::vec;
    use proptest::prelude::*;
    use sigma_test_util::force_any_val;

    fn proof_append_some_byte(proof: &ProofBytes) -> ProofBytes {
        match proof {
            ProofBytes::Empty => panic!(),
            ProofBytes::Some(bytes) => {
                let mut new_bytes = bytes.clone();
                new_bytes.push(1u8);
                ProofBytes::Some(new_bytes)
            }
        }
    }
    proptest! {

        #![proptest_config(ProptestConfig::with_cases(16))]

        #[test]
        fn test_prover_verifier_p2pk(secret in any::<DlogProverInput>(), message in vec(any::<u8>(), 100..200)) {
            let pk = secret.public_image();
            let tree = ErgoTree::try_from(Expr::Const(pk.into())).unwrap();

            let prover = TestProver {
                secrets: vec![PrivateInput::DlogProverInput(secret)],
            };
            let res = prover.prove(&tree,
                &force_any_val::<Context>(),
                message.as_slice(),
                &HintsBag::empty());
            let proof = res.unwrap().proof;
            let verifier = TestVerifier;
            prop_assert_eq!(verifier.verify(&tree,
                                            &force_any_val::<Context>(),
                                            proof.clone(),
                                            message.as_slice())
                            .unwrap().result,
                            true);

            // possible to append bytes
            prop_assert_eq!(verifier.verify(&tree,
                                            &force_any_val::<Context>(),
                                            proof_append_some_byte(&proof),
                                            message.as_slice())
                            .unwrap().result,
                            true);

            // wrong message
            prop_assert_eq!(verifier.verify(&tree,
                                            &force_any_val::<Context>(),
                                            proof,
                                            vec![1u8; 100].as_slice())
                            .unwrap().result,
                            false);
        }

        #[test]
        fn test_prover_verifier_dht(secret in any::<DhTupleProverInput>(), message in vec(any::<u8>(), 100..200)) {
            let pk = secret.public_image().clone();
            let tree = ErgoTree::try_from(Expr::Const(pk.into())).unwrap();

            let prover = TestProver {
                secrets: vec![PrivateInput::DhTupleProverInput(secret)],
            };
            let res = prover.prove(&tree,
                &force_any_val::<Context>(),
                message.as_slice(),
                &HintsBag::empty());
            let proof = res.unwrap().proof;
            let verifier = TestVerifier;
            prop_assert_eq!(verifier.verify(&tree,
                                            &force_any_val::<Context>(),
                                            proof.clone(),
                                            message.as_slice())
                            .unwrap().result,
                            true);

            // possible to append bytes
            prop_assert_eq!(verifier.verify(&tree,
                                            &force_any_val::<Context>(),
                                            proof_append_some_byte(&proof),
                                            message.as_slice())
                            .unwrap().result,
                            true);

            // wrong message
            prop_assert_eq!(verifier.verify(&tree,
                                            &force_any_val::<Context>(),
                                            proof,
                                            vec![1u8; 100].as_slice())
                            .unwrap().result,
                            false);
        }

        #[test]
        fn test_prover_verifier_conj_and(secret1 in any::<PrivateInput>(),
                                         secret2 in any::<PrivateInput>(),
                                         message in vec(any::<u8>(), 100..200)) {
            let pk1 = secret1.public_image();
            let pk2 = secret2.public_image();
            let expr: Expr = SigmaAnd::new(vec![Expr::Const(pk1.into()), Expr::Const(pk2.into())])
                .unwrap()
                .into();
            let tree = ErgoTree::try_from(expr).unwrap();
            let prover = TestProver {
                secrets: vec![secret1, secret2],
            };
            let res = prover.prove(&tree,
                &force_any_val::<Context>(),
                message.as_slice(),
                &HintsBag::empty());
            let proof = res.unwrap().proof;
            let verifier = TestVerifier;
            let ver_res = verifier.verify(&tree,
                                          &force_any_val::<Context>(),
                                          proof,
                                          message.as_slice());
            prop_assert_eq!(ver_res.unwrap().result, true);
        }

        #[test]
        fn test_prover_verifier_conj_and_and(secret1 in any::<PrivateInput>(),
                                             secret2 in any::<PrivateInput>(),
                                             secret3 in any::<PrivateInput>(),
                                             message in vec(any::<u8>(), 100..200)) {
            let pk1 = secret1.public_image();
            let pk2 = secret2.public_image();
            let pk3 = secret3.public_image();
            let expr: Expr = SigmaAnd::new(vec![
                Expr::Const(pk1.into()),
                SigmaAnd::new(vec![Expr::Const(pk2.into()), Expr::Const(pk3.into())])
                    .unwrap()
                    .into(),
            ]).unwrap().into();
            let tree = ErgoTree::try_from(expr).unwrap();
            let prover = TestProver { secrets: vec![secret1, secret2, secret3] };
            let res = prover.prove(&tree,
                &force_any_val::<Context>(),
                message.as_slice(),
                &HintsBag::empty());
            let proof = res.unwrap().proof;
            let verifier = TestVerifier;
            let ver_res = verifier.verify(&tree,
                                          &force_any_val::<Context>(),
                                          proof,
                                          message.as_slice());
            prop_assert_eq!(ver_res.unwrap().result, true);
        }

        #[test]
        fn test_prover_verifier_conj_or(secret1 in any::<PrivateInput>(),
                                         secret2 in any::<PrivateInput>(),
                                         message in vec(any::<u8>(), 100..200)) {
            let pk1 = secret1.public_image();
            let pk2 = secret2.public_image();
            let expr: Expr = SigmaOr::new(vec![Expr::Const(pk1.into()), Expr::Const(pk2.into())])
                .unwrap()
                .into();
            let tree = ErgoTree::try_from(expr).unwrap();
            let secrets = vec![secret1, secret2];
            // any secret (out of 2) known to prover should be enough
            for secret in secrets {
                let prover = TestProver {
                    secrets: vec![secret.clone()],
                };
                let res = prover.prove(&tree,
                    &force_any_val::<Context>(),
                    message.as_slice(),
                    &HintsBag::empty());
                let proof = res.unwrap_or_else(|_| panic!("proof failed for secret: {:?}", secret)).proof;
                let verifier = TestVerifier;
                let ver_res = verifier.verify(&tree,
                                              &force_any_val::<Context>(),
                                              proof,
                                              message.as_slice());
                prop_assert_eq!(ver_res.unwrap().result, true, "verify failed on secret: {:?}", &secret);
            }
        }

        #[test]
        fn test_prover_verifier_conj_or_or(secret1 in any::<PrivateInput>(),
                                             secret2 in any::<PrivateInput>(),
                                             secret3 in any::<PrivateInput>(),
                                             message in vec(any::<u8>(), 100..200)) {
            let pk1 = secret1.public_image();
            let pk2 = secret2.public_image();
            let pk3 = secret3.public_image();
            let expr: Expr = SigmaOr::new(vec![
                Expr::Const(pk1.into()),
                SigmaOr::new(vec![Expr::Const(pk2.into()), Expr::Const(pk3.into())])
                    .unwrap()
                    .into(),
            ]).unwrap().into();
            let tree = ErgoTree::try_from(expr).unwrap();
            let secrets = vec![secret1, secret2, secret3];
            // any secret (out of 3) known to prover should be enough
            for secret in secrets {
                let prover = TestProver {
                    secrets: vec![secret.clone()],
                };
                let res = prover.prove(&tree,
                    &force_any_val::<Context>(),
                    message.as_slice(),
                    &HintsBag::empty());
                let proof = res.unwrap_or_else(|_| panic!("proof failed for secret: {:?}", secret)).proof;
                let verifier = TestVerifier;
                let ver_res = verifier.verify(&tree,
                                              &force_any_val::<Context>(),
                                              proof,
                                              message.as_slice());
                prop_assert_eq!(ver_res.unwrap().result, true, "verify failed on secret: {:?}", &secret);
            }
        }

        #[test]
        fn test_prover_verifier_atleast(secret1 in any::<DlogProverInput>(),
                                            secret2 in any::<DlogProverInput>(),
                                             secret3 in any::<DlogProverInput>(),
                                             message in vec(any::<u8>(), 100..200)) {
            let bound = Expr::Const(2i32.into());
            let inputs = Literal::Coll(
                CollKind::from_collection(
                    SType::SSigmaProp,
                    [
                        SigmaProp::from(secret1.public_image()).into(),
                        SigmaProp::from(secret2.public_image()).into(),
                        SigmaProp::from(secret3.public_image()).into(),
                    ],
                )
                .unwrap(),
            );
            let input = Constant {
                tpe: SType::SColl(SType::SSigmaProp.into()),
                v: inputs,
            }
            .into();
            let expr: Expr = Atleast::new(bound, input).unwrap().into();
            let tree = ErgoTree::try_from(expr).unwrap();
            let prover = TestProver {
                secrets: vec![secret1.into(), secret3.into()],
            };

            let res = prover.prove(&tree,
                &force_any_val::<Context>(),
                message.as_slice(),
                &HintsBag::empty());
            let proof = res.unwrap().proof;
            let verifier = TestVerifier;
            let ver_res = verifier.verify(&tree,
                &force_any_val::<Context>(),
                proof,
                message.as_slice());
            prop_assert_eq!(ver_res.unwrap().result, true)
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod empty_conjecture_tests {
    //! JVM parity: a conjecture constant is verified as it is. `CAND()`'s proof is its root
    //! challenge alone, since every child takes that challenge (`SigSerializer.scala:210-217`).
    //! `COR()` never verifies: its last child is `children(-1)`, which throws (`:228-234`), and
    //! `verifySignature` returns false for any exception (`Interpreter.scala:462-482`).
    use super::*;
    use crate::sigma_protocol::SOUNDNESS_BYTES;
    use alloc::vec::Vec;
    use ergotree_ir::serialization::SigmaSerializable;

    const MESSAGE: &[u8] = b"a message";

    /// The challenge that a proof of a tree without leaves carries: the hash of the tree's
    /// Fiat-Shamir bytes and the message (`Interpreter.scala:388-400`). A conjecture's bytes are
    /// `00`, its type, `k` for a threshold, and its child count as a Short
    /// (`UnprovenTree.scala:268-281`).
    fn root_challenge(tree: &[u8]) -> Vec<u8> {
        let hash: [u8; SOUNDNESS_BYTES] = fiat_shamir_hash_fn(&[tree, MESSAGE].concat()).into();
        hash.to_vec()
    }

    #[test]
    fn cand_without_children_verifies_with_its_root_challenge() {
        let cand = SigmaBoolean::sigma_parse_bytes(&[0x96, 0x00]).unwrap();
        let proof = root_challenge(&[0x00, 0x00, 0x00, 0x00]);
        assert!(verify_signature(cand.clone(), MESSAGE, &proof).unwrap());
        assert!(!verify_signature(cand.clone(), MESSAGE, &[]).unwrap());
        let mut wrong = proof;
        wrong[0] ^= 1;
        assert!(!verify_signature(cand, MESSAGE, &wrong).unwrap());
    }

    #[test]
    fn cor_without_children_never_verifies() {
        let cor = SigmaBoolean::sigma_parse_bytes(&[0x97, 0x00]).unwrap();
        assert!(!verify_signature(cor.clone(), MESSAGE, &[]).unwrap());
        // the proof that verifies `CAND()`, made for an OR node
        let forged = root_challenge(&[0x00, 0x01, 0x00, 0x00]);
        assert!(!matches!(verify_signature(cor, MESSAGE, &forged), Ok(true)));
    }

    #[test]
    fn a_child_count_above_32767_is_hashed_as_sigmastate_s_short() {
        // By source, no SANTA vector: `FiatShamirTree.toBytes` writes a conjecture's child count
        // as `children.length.toShort` (`UnprovenTree.scala:279-280`), which wraps above 32767.
        // `CAND(40000 × CAND())` is `96`, the count as a VLQ, and 40000 × `96 00`. In the
        // Fiat-Shamir bytes 40000 is `9c 40`.
        const N: usize = 40000;
        let cand_bytes = [&[0x96, 0xc0, 0xb8, 0x02][..], &[0x96, 0x00].repeat(N)].concat();
        let cand = SigmaBoolean::sigma_parse_bytes(&cand_bytes).unwrap();
        let tree = [
            &[0x00, 0x00, 0x9c, 0x40][..],
            &[0x00, 0x00, 0x00, 0x00].repeat(N),
        ]
        .concat();
        assert!(verify_signature(cand, MESSAGE, &root_challenge(&tree)).unwrap());
    }
}
