//! Genesis block construction for the Ergo blockchain.
//!
//! Constructs the three genesis boxes (emission, no-premine, foundation)
//! from chain parameters, matching the JVM reference implementation.

use ergotree_ir::chain::ergo_box::box_value::BoxValue;
use ergotree_ir::chain::ergo_box::box_value::BoxValueError;
use ergotree_ir::chain::ergo_box::ErgoBox;
use ergotree_ir::chain::ergo_box::NonMandatoryRegisterId;
use ergotree_ir::chain::ergo_box::NonMandatoryRegisters;
use ergotree_ir::chain::ergo_box::NonMandatoryRegistersError;
use ergotree_ir::chain::tx_id::TxId;
use ergotree_ir::mir::constant::Constant;
use ergotree_ir::serialization::SigmaSerializationError;

use super::emission::EmissionRules;
use super::emission::MonetarySettings;
use super::emission::COINS_IN_ONE_ERGO;
use super::ergo_tree_predef;
use super::ergo_tree_predef::ErgoTreePredefError;

/// Errors that can occur during genesis box construction.
#[derive(Debug, thiserror::Error)]
pub enum GenesisError {
    /// Error constructing a predefined ErgoTree.
    #[error("ergo tree predef error: {0}")]
    ErgoTreePredef(#[from] ErgoTreePredefError),
    /// Error creating a box value.
    #[error("box value error: {0}")]
    BoxValue(#[from] BoxValueError),
    /// Error serializing a box (for box ID computation).
    #[error("serialization error: {0}")]
    Serialization(#[from] SigmaSerializationError),
    /// Error creating non-mandatory registers.
    #[error("register error: {0}")]
    Register(#[from] NonMandatoryRegistersError),
}

/// Construct the three genesis boxes for an Ergo network.
///
/// Returns `(emission_box, no_premine_box, founders_box)`.
///
/// - `settings`: monetary parameters for the chain
/// - `founders_prop`: the SigmaProp constant stored in the founders box R4
///   register (controls who can spend from the foundation treasury)
///
/// Genesis boxes use `TxId::zero()`, `creation_height = 0`, and indices
/// 0, 1, 2 respectively.
pub fn genesis_boxes(
    settings: &MonetarySettings,
    founders_prop: Constant,
) -> Result<(ErgoBox, ErgoBox, ErgoBox), GenesisError> {
    let rules = EmissionRules::new(settings.clone());
    let tx_id = TxId::zero();

    let emission_box = ErgoBox::new(
        BoxValue::try_from(rules.miners_coins_total() as u64)?,
        ergo_tree_predef::emission_box_prop(settings)?,
        None,
        NonMandatoryRegisters::empty(),
        0,
        tx_id,
        0,
    )?;

    let no_premine_box = ErgoBox::new(
        BoxValue::try_from(COINS_IN_ONE_ERGO as u64)?,
        ergo_tree_predef::false_prop()?,
        None,
        NonMandatoryRegisters::empty(),
        0,
        tx_id,
        1,
    )?;

    let founders_registers =
        NonMandatoryRegisters::new(vec![(NonMandatoryRegisterId::R4, founders_prop)])?;
    let founders_value = (rules.founders_coins_total() - COINS_IN_ONE_ERGO) as u64;

    let founders_box = ErgoBox::new(
        BoxValue::try_from(founders_value)?,
        ergo_tree_predef::foundation_script(settings)?,
        None,
        founders_registers,
        0,
        tx_id,
        2,
    )?;

    Ok((emission_box, no_premine_box, founders_box))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
#[allow(clippy::panic)]
mod tests {
    use super::*;
    use ergotree_ir::sigma_protocol::sigma_boolean::SigmaBoolean;
    use ergotree_ir::sigma_protocol::sigma_boolean::SigmaProp;

    /// Build genesis boxes with TrueProp as founders proposition (simplest case).
    fn build_test_genesis() -> (ErgoBox, ErgoBox, ErgoBox) {
        let settings = MonetarySettings::default();
        let true_prop = Constant::from(SigmaProp::new(SigmaBoolean::TrivialProp(true)));
        genesis_boxes(&settings, true_prop).unwrap()
    }

    #[test]
    fn test_genesis_emission_box_value() {
        let (emission, _, _) = build_test_genesis();
        let rules = EmissionRules::new(MonetarySettings::default());
        assert_eq!(emission.value.as_i64(), rules.miners_coins_total());
    }

    #[test]
    fn test_genesis_no_premine_box_value() {
        let (_, no_premine, _) = build_test_genesis();
        assert_eq!(no_premine.value.as_i64(), COINS_IN_ONE_ERGO);
    }

    #[test]
    fn test_genesis_founders_box_value() {
        let (_, _, founders) = build_test_genesis();
        let rules = EmissionRules::new(MonetarySettings::default());
        let expected = rules.founders_coins_total() - COINS_IN_ONE_ERGO;
        assert_eq!(founders.value.as_i64(), expected);
    }

    #[test]
    fn test_genesis_total_value_equals_coins_total() {
        let (emission, no_premine, founders) = build_test_genesis();
        let rules = EmissionRules::new(MonetarySettings::default());
        let total = emission.value.as_i64() + no_premine.value.as_i64() + founders.value.as_i64();
        assert_eq!(total, rules.coins_total());
    }

    #[test]
    fn test_genesis_box_indices() {
        let (emission, no_premine, founders) = build_test_genesis();
        assert_eq!(emission.index, 0);
        assert_eq!(no_premine.index, 1);
        assert_eq!(founders.index, 2);
    }

    #[test]
    fn test_genesis_box_creation_height() {
        let (emission, no_premine, founders) = build_test_genesis();
        assert_eq!(emission.creation_height, 0);
        assert_eq!(no_premine.creation_height, 0);
        assert_eq!(founders.creation_height, 0);
    }
}
