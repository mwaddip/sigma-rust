import XCTest
@testable import ErgoLib
@testable import ErgoLibC

final class ParametersTests: XCTestCase {
    /// The `parameters` object of a node's `/info`
    static let nodeParametersJSON = """
        {
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
        }
        """

    static func nodeParameters() -> Parameters {
        return Parameters(
            blockVersion: 4,
            storageFeeFactor: 1250000,
            minValuePerByte: 360,
            maxBlockSize: 1271009,
            maxBlockCost: 8001091,
            tokenAccessCost: 100,
            inputCost: 2407,
            dataInputCost: 100,
            outputCost: 194
        )
    }

    /// Sign, with a wallet that has no secret, a transaction that spends a box guarded by
    /// `0b 02 08 d3`: a version 3 tree whose proposition is true
    func signASpendOfAVersion3Tree(parameters: Parameters) throws -> Transaction {
        let inputContract = Contract(fromErgoTree: try ErgoTree(fromBase16EncodedString: "0b0208d3"))
        let txId = try TxId(withString: "93d344aa527e18e5a221db060ea1a868f46b61e4537e6e5f69ecc40334c15e38")
        let inputBox = try ErgoBox(boxValue: BoxValue(fromInt64: Int64(1000000000)), creationHeight: 0, contract: inputContract, txId: txId, index: 0, tokens: Tokens())
        let recipient = try Address(withTestnetAddress: "3WvsT2Gm4EpsM9Pg18PdY6XyhNNMqXDsvJTbbf6ihLvAmSb7u5RN")
        let unspentBoxes = ErgoBoxes()
        unspentBoxes.add(ergoBox: inputBox)
        let outboxValue = BoxValue.SAFE_USER_MIN()
        let outbox = try ErgoBoxCandidateBuilder(boxValue: outboxValue, contract: try Contract(payToAddress: recipient), creationHeight: UInt32(0)).build()
        let txOutputs = ErgoBoxCandidates()
        txOutputs.add(ergoBoxCandidate: outbox)
        let fee = TxBuilder.SUGGESTED_TX_FEE()
        let targetBalance = try BoxValue.sumOf(boxValue0: outboxValue, boxValue1: fee)
        let boxSelection = try SimpleBoxSelector().select(inputs: unspentBoxes, targetBalance: targetBalance, targetTokens: Tokens())
        let tx = try TxBuilder(boxSelection: boxSelection, outputCandidates: txOutputs, currentHeight: 0, feeAmount: fee, changeAddress: recipient).build()
        let blockHeaders = try HeaderTests.generateBlockHeadersFromJSON()
        let preHeader = PreHeader(withBlockHeader: blockHeaders.get(index: UInt(0))!)
        let ctx = try ErgoStateContext(preHeader: preHeader, headers: blockHeaders, parameters: parameters)
        let txDataInputs = try ErgoBoxes(fromJSON: [])
        return try Wallet(secrets: SecretKeys()).signTransaction(stateContext: ctx, unsignedTx: tx, boxesToSpend: unspentBoxes, dataBoxes: txDataInputs)
    }

    func testParametersFromJson() throws {
        XCTAssertNoThrow(try Parameters(withJson: ParametersTests.nodeParametersJSON))
        // a table that lacks an entry is refused
        XCTAssertThrowsError(try Parameters(withJson: "{\"blockVersion\": 4}"))
    }

    func testTheParametersBlockVersionActivatesScriptsWhenSigning() throws {
        // the default parameters are those set at genesis, block version 1: activated 0
        XCTAssertThrowsError(try signASpendOfAVersion3Tree(parameters: Parameters()))
        XCTAssertNoThrow(try signASpendOfAVersion3Tree(parameters: ParametersTests.nodeParameters()))
        XCTAssertNoThrow(try signASpendOfAVersion3Tree(parameters: try Parameters(withJson: ParametersTests.nodeParametersJSON)))
    }
}
