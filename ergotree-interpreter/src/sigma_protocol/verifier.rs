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
use crate::eval::env::Env;
use crate::eval::EvalError;
use crate::eval::{reduce_to_crypto, ReductionDiagnosticInfo};
use dlog_protocol::FirstDlogProverMessage;
use ergotree_ir::chain::context::Context;
use ergotree_ir::ergo_tree::ErgoTree;
use ergotree_ir::ergo_tree::ErgoTreeError;
use ergotree_ir::ergo_tree::ErgoTreeVersion;

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
        if check_soft_fork_condition(tree, ctx)? {
            // sigmastate's `true -> context.initCost` (`Interpreter.scala:317`): nothing was
            // evaluated
            return Ok(VerificationResult {
                result: true,
                cost: 0,
                diag: ReductionDiagnosticInfo {
                    env: Env::empty(),
                    pretty_printed_expr: None,
                },
            });
        }
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

/// sigmastate's `Interpreter.checkSoftForkCondition` (v6.0.6 `Interpreter.scala:298-331`),
/// which `verify` (`:362`) and `prove` (`ProverInterpreter.scala:128-135`) run before
/// anything else.
/// - `Ok(false)`: verify the spend.
/// - `Ok(true)`: the activated script version and the tree's version are both above
///   [`ErgoTreeVersion::MAX_SCRIPT_VERSION`]. This interpreter cannot read the tree, and the
///   spend is accepted without verification (`:304-318`).
/// - An error: the activated version is one this interpreter supports, and the tree's
///   version is above it (`:325-327`). Unlike the check at parse, this one has no floor at
///   activated 2.
///
/// A tree that did not parse has version 0, as sigmastate's `UnparsedErgoTree` has
/// (`ErgoTreeSerializer.scala:203`).
pub fn check_soft_fork_condition(tree: &ErgoTree, ctx: &Context) -> Result<bool, EvalError> {
    let tree_version = tree
        .header()
        .map(|header| header.version())
        .unwrap_or(ErgoTreeVersion::V0);
    // sigmastate compares signed bytes, and the activated version is one: ergo's is the block
    // version minus 1 as a byte (ergo v6.0.6 `ErgoContext.scala:28`), negative for a block
    // version of 0 or above 128, and then every tree is above it. So this reads the context's
    // field: the `activated_script_version()` method reads a negative version as 0.
    let activated_version = ctx.activated_script_version_byte;
    let max_supported = u8::from(ErgoTreeVersion::MAX_SCRIPT_VERSION) as i8;
    let version = u8::from(tree_version) as i8;
    if activated_version > max_supported {
        Ok(version > max_supported)
    } else if version > activated_version {
        Err(EvalError::TreeVersionAboveActivated {
            tree_version,
            activated_version,
        })
    } else {
        Ok(false)
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
    use core::convert::{TryFrom, TryInto};

    use crate::sigma_protocol::private_input::{DhTupleProverInput, DlogProverInput, PrivateInput};
    use crate::sigma_protocol::prover::hint::HintsBag;
    use crate::sigma_protocol::prover::{Prover, TestProver};
    use crate::sigma_protocol::SOUNDNESS_BYTES;

    use super::*;
    use ergotree_ir::mir::atleast::Atleast;
    use ergotree_ir::mir::constant::{Constant, Literal};
    use ergotree_ir::mir::expr::Expr;
    use ergotree_ir::mir::sigma_and::SigmaAnd;
    use ergotree_ir::mir::sigma_or::SigmaOr;
    use ergotree_ir::mir::value::CollKind;
    use ergotree_ir::sigma_protocol::sigma_boolean::cthreshold::Cthreshold;
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

    #[test]
    fn a_threshold_proof_cut_inside_its_coefficients_is_false() {
        // sigmastate reads what is left of the coefficients and of each response
        // (`SigSerializer.scala:156-163`), and the commitments computed from them do not hash
        // to the root challenge: the proof is false, and reading it is no error
        let secret = DlogProverInput::random();
        let threshold = SigmaBoolean::from(Cthreshold {
            k: 1,
            children: vec![
                secret.public_image().into(),
                DlogProverInput::random().public_image().into(),
            ]
            .try_into()
            .unwrap(),
        });
        let message = b"a message";
        let prover = TestProver {
            secrets: vec![secret.into()],
        };
        let proof = prover
            .generate_proof(threshold.clone(), message, &HintsBag::empty())
            .unwrap()
            .to_bytes();
        assert!(verify_signature(threshold.clone(), message, &proof).unwrap());
        // the root challenge, and 10 of the 24 bytes of the one coefficient
        let cut = &proof[..SOUNDNESS_BYTES + 10];
        assert!(!verify_signature(threshold, message, cut).unwrap());
    }
}

#[cfg(test)]
#[cfg(feature = "arbitrary")]
#[allow(clippy::unwrap_used)]
mod soft_fork_condition_tests {
    //! JVM parity: `Interpreter.checkSoftForkCondition` (sigmastate v6.0.6
    //! `Interpreter.scala:298-331`), which `verify` (`:362`) and `prove`
    //! (`ProverInterpreter.scala:128-135`) run before anything else. SANTA's
    //! `tree-version-above-activated` spends are replayed in ergo-lib's `tx_context`.
    use super::*;
    use ergotree_ir::serialization::SigmaSerializable;
    use sigma_test_util::force_any_val;

    /// A size-flagged tree of `version` and its root, as a box read from the UTXO set has it:
    /// read with no context, so any version parses
    fn tree(version: u8, root: u8) -> ErgoTree {
        ErgoTree::sigma_parse_bytes(&[0x08 | version, 0x02, 0x08, root]).unwrap()
    }

    /// `SigmaProp(true)` and `SigmaProp(false)`
    const TRUE: u8 = 0xd3;
    const FALSE: u8 = 0xd2;

    /// A context as ergo builds it at `block_version`: activated at the block version minus 1,
    /// as a byte (`ErgoContext.scala:28`)
    fn ctx_at_block_version(block_version: u8) -> Context<'static> {
        let mut ctx = force_any_val::<Context>();
        ctx.activated_script_version_byte = (block_version as i8).wrapping_sub(1);
        ctx
    }

    fn ctx_at(activated: u8) -> Context<'static> {
        ctx_at_block_version(activated + 1)
    }

    fn refused(version: u8, activated_version: i8) -> Result<bool, EvalError> {
        Err(EvalError::TreeVersionAboveActivated {
            tree_version: version.into(),
            activated_version,
        })
    }

    #[test]
    fn under_a_supported_activated_version_a_tree_above_it_is_refused() {
        // no floor at 2, unlike the check at parse: at activated 1 a v2 tree is refused
        for activated in 0..=3u8 {
            for version in 0..=7u8 {
                let res = check_soft_fork_condition(&tree(version, TRUE), &ctx_at(activated));
                if version > activated {
                    assert_eq!(
                        res,
                        refused(version, activated as i8),
                        "tree {version} at {activated}"
                    );
                } else {
                    assert_eq!(res, Ok(false), "tree {version} at {activated}");
                }
            }
        }
    }

    #[test]
    fn above_the_supported_version_only_a_tree_above_it_is_accepted_unverified() {
        // 127 is the highest activated version: block version 128
        for activated in [4, 5, 6, 7, 127u8] {
            for version in 0..=7u8 {
                assert_eq!(
                    check_soft_fork_condition(&tree(version, TRUE), &ctx_at(activated)),
                    Ok(version > 3),
                    "tree {version} at {activated}"
                );
            }
        }
    }

    #[test]
    fn the_activated_version_is_a_signed_byte() {
        // sigmastate's versions are `Byte`s, and the activated one is the block version minus
        // 1 as a byte (ergo `ErgoContext`). For a block version of 0, or above 128, it is
        // negative: no tree is at or below it, and none is accepted unverified. SANTA
        // `tree-version-block-version-edges` #7 and #10 are block versions 0 and 200.
        for (block_version, activated) in [
            (0u8, -1i8),
            (0x81, -128),
            (200, -57),
            (0xfe, -3),
            (0xff, -2),
        ] {
            for version in 0..=7u8 {
                assert_eq!(
                    check_soft_fork_condition(
                        &tree(version, TRUE),
                        &ctx_at_block_version(block_version)
                    ),
                    refused(version, activated),
                    "tree {version} at block version {block_version}"
                );
            }
        }
    }

    #[test]
    fn the_pre_header_s_version_does_not_decide_the_condition() {
        // SANTA `block-version-source` #0 to #3, at the interpreter: the voted parameters'
        // block version is 4, so ergo's context is activated at 3, under a header of version
        // 3, 5, 0 and 200
        let ctx = |header_version: u8| {
            let mut ctx = ctx_at(3);
            ctx.pre_header.version = header_version;
            ctx
        };
        assert_eq!(
            check_soft_fork_condition(&tree(3, TRUE), &ctx(3)),
            Ok(false)
        );
        assert_eq!(
            check_soft_fork_condition(&tree(4, FALSE), &ctx(5)),
            refused(4, 3)
        );
        for header_version in [0, 200] {
            assert_eq!(
                check_soft_fork_condition(&tree(0, TRUE), &ctx(header_version)),
                Ok(false),
                "{header_version}"
            );
        }
    }

    #[test]
    fn a_tree_that_did_not_parse_counts_as_version_0() {
        // `0c 03 d1 fd 00`: the header claims version 4, and opcode `fd` has no serializer,
        // so the tree degrades (as the output of mainnet block 545,684 does)
        let unparsed = ErgoTree::sigma_parse_bytes(&[0x0c, 0x03, 0xd1, 0xfd, 0x00]).unwrap();
        assert!(matches!(unparsed, ErgoTree::Unparsed { .. }));
        for activated in 0..=7u8 {
            assert_eq!(
                check_soft_fork_condition(&unparsed, &ctx_at(activated)),
                Ok(false),
                "{activated}"
            );
        }
    }

    fn verify(tree: &ErgoTree, ctx: &Context) -> Result<VerificationResult, VerifierError> {
        TestVerifier.verify(tree, ctx, ProofBytes::Empty, &[])
    }

    fn is_version_error(res: &Result<VerificationResult, VerifierError>) -> bool {
        matches!(
            res,
            Err(VerifierError::EvalError(
                EvalError::TreeVersionAboveActivated { .. }
            ))
        )
    }

    #[test]
    fn verify_refuses_a_tree_above_the_activated_version() {
        let res = verify(&tree(4, TRUE), &ctx_at(3));
        assert!(is_version_error(&res), "{res:?}");
        assert!(verify(&tree(3, TRUE), &ctx_at(3)).unwrap().result);
    }

    #[test]
    fn verify_has_no_floor_at_activated_2() {
        // A v2 tree at activated 1 parses under any context and is refused at the spend (SANTA
        // `tree-version-block-version-edges` #0). So is a v1 tree at activated 0, by source:
        // SANTA could not build a chain of block version 1.
        for (version, activated) in [(2u8, 1u8), (1, 0)] {
            let res = verify(&tree(version, TRUE), &ctx_at(activated));
            assert!(is_version_error(&res), "{version} at {activated}: {res:?}");
            let at_its_version = verify(&tree(version, TRUE), &ctx_at(version));
            assert!(at_its_version.unwrap().result, "{version}");
        }
        // and at block version 0 no tree is spent (#7)
        let res = verify(&tree(0, TRUE), &ctx_at_block_version(0));
        assert!(is_version_error(&res), "{res:?}");
    }

    #[test]
    fn verify_accepts_unverified_a_tree_it_cannot_read_under_a_newer_protocol() {
        // `Some(true -> context.initCost)` (`Interpreter.scala:317`): the tree is not reduced.
        // It is `SigmaProp(false)` here, which a reduction would refuse.
        let res = verify(&tree(4, FALSE), &ctx_at(4)).unwrap();
        assert!(res.result);
        assert_eq!(res.cost, 0);
        // a tree it can read is verified as usual
        assert!(!verify(&tree(3, FALSE), &ctx_at(4)).unwrap().result);
    }

    #[test]
    fn the_error_reads_as_sigmastate_s() {
        let message = |version: u8, block_version: u8| {
            check_soft_fork_condition(&tree(version, TRUE), &ctx_at_block_version(block_version))
                .unwrap_err()
                .to_string()
        };
        assert_eq!(
            message(4, 4),
            "ErgoTree version 4 is higher than activated 3"
        );
        assert_eq!(
            message(0, 0),
            "ErgoTree version 0 is higher than activated -1"
        );
        assert_eq!(
            message(0, 200),
            "ErgoTree version 0 is higher than activated -57"
        );
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
    use ergotree_ir::sigma_protocol::sigma_boolean::cand::Cand;

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

    #[test]
    fn a_child_count_above_65535_is_hashed_as_sigmastate_s_short() {
        // By source as well. No such CAND is read off the wire, where the count is a
        // `getUShort`, but a `SigmaAnd` node of up to 100000 items reduces to one.
        // `children.length.toShort` keeps the low 16 bits: 70000 is `11 70`.
        const N: usize = 70000;
        let empty = SigmaBoolean::sigma_parse_bytes(&[0x96, 0x00]).unwrap();
        let cand = SigmaBoolean::from(Cand {
            items: vec![empty; N],
        });
        let tree = [
            &[0x00, 0x00, 0x11, 0x70][..],
            &[0x00, 0x00, 0x00, 0x00].repeat(N),
        ]
        .concat();
        assert!(verify_signature(cand, MESSAGE, &root_challenge(&tree)).unwrap());
    }

    #[test]
    fn a_threshold_proof_may_end_before_its_coefficients() {
        // By source, no SANTA vector. `CTHRESHOLD(0, [CAND()])` asks a proof for one
        // coefficient, which sigmastate reads with `getBytesUnsafe`: the bytes that are left
        // (`SigSerializer.scala:250-252`, `CoreByteReader.scala:94-98`). With none the
        // polynomial is the root challenge alone (`GF2_192_Poly.scala:50-58`), the child takes
        // it, and the tree hashes as it does with the coefficient.
        let threshold = SigmaBoolean::sigma_parse_bytes(&[0x98, 0x00, 0x01, 0x96, 0x00]).unwrap();
        let proof = root_challenge(&[0x00, 0x02, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00]);
        assert!(verify_signature(threshold, MESSAGE, &proof).unwrap());
    }

    #[test]
    fn an_or_s_missing_challenge_is_not_read_as_zeros() {
        // sigmastate reads a child's challenge the same way, and then xors it into the 24
        // bytes of the node's own (`SigSerializer.scala:227-233`): a challenge that is not all
        // there throws in `Helpers.xorU` (`Helpers.scala:22-29`), which is false
        // (`Interpreter.scala:462-482`).
        let or = SigmaBoolean::sigma_parse_bytes(&[0x97, 0x02, 0x96, 0x00, 0x96, 0x00]).unwrap();
        let root = root_challenge(&[
            0x00, 0x01, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        ]);
        assert!(!matches!(
            verify_signature(or.clone(), MESSAGE, &root),
            Ok(true)
        ));
        // the root is right: with a challenge for the first child, whatever it is, it verifies
        let whole = [root, vec![5u8; SOUNDNESS_BYTES]].concat();
        assert!(verify_signature(or, MESSAGE, &whole).unwrap());
    }
}
