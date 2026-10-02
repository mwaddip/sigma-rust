//! Blockchain parameters. This module defines adjustable blockchain parameters that can be voted on by miners

use ergo_lib::chain::parameters;
use wasm_bindgen::prelude::*;
extern crate derive_more;
use derive_more::{From, Into};

use crate::error_conversion::to_js;

/// Blockchain parameters
#[wasm_bindgen]
#[derive(PartialEq, Debug, Clone, Eq, From, Into)]
pub struct Parameters(pub(crate) parameters::Parameters);

#[wasm_bindgen]
impl Parameters {
    /// Return default blockchain parameters that were set at genesis, with block version 1.
    /// The parameters' block version activates scripts, for signing too: under the default
    /// only a version 0 tree is signed. A wallet passes the chain's parameters, see the
    /// constructor and `from_json`
    pub fn default_parameters() -> Parameters {
        parameters::Parameters::default().into()
    }
    /// Create new parameters from provided blockchain parameters
    #[wasm_bindgen(constructor)]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        block_version: i32,
        storage_fee_factor: i32,
        min_value_per_byte: i32,
        max_block_size: i32,
        max_block_cost: i32,
        token_access_cost: i32,
        input_cost: i32,
        data_input_cost: i32,
        output_cost: i32,
    ) -> Parameters {
        parameters::Parameters::new(
            block_version,
            storage_fee_factor,
            min_value_per_byte,
            max_block_size,
            max_block_cost,
            token_access_cost,
            input_cost,
            data_input_cost,
            output_cost,
        )
        .into()
    }
    /// Parse parameters from JSON. Supports Ergo Node API/Explorer API
    pub fn from_json(json: &str) -> Result<Parameters, JsValue> {
        serde_json::from_str(json).map(Self).map_err(to_js)
    }
    /// Get current block version
    pub fn block_version(&self) -> i32 {
        self.0.block_version()
    }
    /// Cost of storing 1 byte per Storage Period of block chain
    pub fn storage_fee_factor(&self) -> i32 {
        self.0.storage_fee_factor()
    }
    /// Minimum value per byte an output must have to not be considered dust
    pub fn min_value_per_byte(&self) -> i32 {
        self.0.min_value_per_byte()
    }
    /// Maximum size of transactions size in a block
    pub fn max_block_size(&self) -> i32 {
        self.0.max_block_size()
    }
    /// Maximum total computation cost in a block
    pub fn max_block_cost(&self) -> i32 {
        self.0.max_block_cost()
    }
    /// Cost of accessing a single token
    pub fn token_access_cost(&self) -> i32 {
        self.0.token_access_cost()
    }
    /// Validation cost per one transaction input
    pub fn input_cost(&self) -> i32 {
        self.0.input_cost()
    }
    /// Validation cost per data input
    pub fn data_input_cost(&self) -> i32 {
        self.0.data_input_cost()
    }
    /// Validation cost per one output
    pub fn output_cost(&self) -> i32 {
        self.0.output_cost()
    }
}
