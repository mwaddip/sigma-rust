import { expect, assert } from 'chai';

import { generate_block_headers } from './utils';

import * as ergo from "..";
let ergo_wasm;
beforeEach(async () => {
  ergo_wasm = await ergo;
});

// The `parameters` object of a node's `/info`
const node_parameters = {
  "outputCost": 194,
  "tokenAccessCost": 100,
  "maxBlockCost": 8001091,
  "height": 1259520,
  "maxBlockSize": 1271009,
  "dataInputCost": 100,
  "blockVersion": 4,
  "inputCost": 2407,
  "storageFeeFactor": 1250000,
  "minValuePerByte": 360
};

function expect_node_parameters(parameters) {
  expect(parameters.block_version()).to.equal(4);
  expect(parameters.storage_fee_factor()).to.equal(1250000);
  expect(parameters.min_value_per_byte()).to.equal(360);
  expect(parameters.max_block_size()).to.equal(1271009);
  expect(parameters.max_block_cost()).to.equal(8001091);
  expect(parameters.token_access_cost()).to.equal(100);
  expect(parameters.input_cost()).to.equal(2407);
  expect(parameters.data_input_cost()).to.equal(100);
  expect(parameters.output_cost()).to.equal(194);
}

// Sign, with a wallet that has no secret, a transaction that spends a box guarded by
// `0b 02 08 d3`: a version 3 tree whose proposition is true
function sign_a_spend_of_a_version_3_tree(parameters) {
  const input_contract = ergo_wasm.Contract.new(ergo_wasm.ErgoTree.from_base16_bytes('0b0208d3'));
  const input_box = new ergo_wasm.ErgoBox(ergo_wasm.BoxValue.from_i64(ergo_wasm.I64.from_str('1000000000')), 0, input_contract, ergo_wasm.TxId.zero(), 0, new ergo_wasm.Tokens());
  const recipient = ergo_wasm.Address.from_testnet_str('3WvsT2Gm4EpsM9Pg18PdY6XyhNNMqXDsvJTbbf6ihLvAmSb7u5RN');
  const unspent_boxes = new ergo_wasm.ErgoBoxes(input_box);
  const outbox_value = ergo_wasm.BoxValue.SAFE_USER_MIN();
  const outbox = new ergo_wasm.ErgoBoxCandidateBuilder(outbox_value, ergo_wasm.Contract.pay_to_address(recipient), 0).build();
  const fee = ergo_wasm.TxBuilder.SUGGESTED_TX_FEE();
  const target_balance = ergo_wasm.BoxValue.from_i64(outbox_value.as_i64().checked_add(fee.as_i64()));
  const box_selection = new ergo_wasm.SimpleBoxSelector().select(unspent_boxes, target_balance, new ergo_wasm.Tokens());
  const tx = ergo_wasm.TxBuilder.new(box_selection, new ergo_wasm.ErgoBoxCandidates(outbox), 0, fee, recipient).build();
  const block_headers = generate_block_headers();
  const pre_header = ergo_wasm.PreHeader.from_block_header(block_headers.get(0));
  const ctx = new ergo_wasm.ErgoStateContext(pre_header, block_headers, parameters);
  const wallet = ergo_wasm.Wallet.from_secrets(new ergo_wasm.SecretKeys());
  return wallet.sign_transaction(ctx, tx, unspent_boxes, ergo_wasm.ErgoBoxes.from_boxes_json([]));
}

it('Parameters from the given values', async () => {
  expect_node_parameters(new ergo_wasm.Parameters(4, 1250000, 360, 1271009, 8001091, 100, 2407, 100, 194));
});

it('Parameters from the JSON of a node', async () => {
  expect_node_parameters(ergo_wasm.Parameters.from_json(JSON.stringify(node_parameters)));
  expect(() => ergo_wasm.Parameters.from_json('{"blockVersion": 4}')).to.throw();
});

it("the parameters' block version activates scripts when signing", async () => {
  // the default parameters are those set at genesis, block version 1: activated 0
  expect(() => sign_a_spend_of_a_version_3_tree(ergo_wasm.Parameters.default_parameters()))
    .to.throw(/ErgoTree version 3 is higher than activated 0/);
  assert(sign_a_spend_of_a_version_3_tree(new ergo_wasm.Parameters(4, 1250000, 360, 1271009, 8001091, 100, 2407, 100, 194)) != null);
  assert(sign_a_spend_of_a_version_3_tree(ergo_wasm.Parameters.from_json(JSON.stringify(node_parameters))) != null);
});
