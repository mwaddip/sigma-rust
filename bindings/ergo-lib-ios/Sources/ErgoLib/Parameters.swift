
import Foundation
import ErgoLibC

/// Blockchain parameters that can be changed by voting
class Parameters {
    internal var pointer: ParametersPtr

    /// Create default parameters: those set at genesis, with block version 1. The parameters'
    /// block version activates scripts, for signing too: under the default only a version 0
    /// tree is signed. A wallet passes the chain's parameters, see the other initializers
    init() {
        var ptr: ParametersPtr?
        ergo_lib_parameters_default(&ptr)
        self.pointer = ptr!
    }

    /// Create parameters from the given blockchain parameters
    init(
        blockVersion: Int32,
        storageFeeFactor: Int32,
        minValuePerByte: Int32,
        maxBlockSize: Int32,
        maxBlockCost: Int32,
        tokenAccessCost: Int32,
        inputCost: Int32,
        dataInputCost: Int32,
        outputCost: Int32
    ) {
        var ptr: ParametersPtr?
        ergo_lib_parameters_new(
            blockVersion,
            storageFeeFactor,
            minValuePerByte,
            maxBlockSize,
            maxBlockCost,
            tokenAccessCost,
            inputCost,
            dataInputCost,
            outputCost,
            &ptr
        )
        self.pointer = ptr!
    }

    /// Parse parameters from JSON. Supports Ergo Node API/Explorer API
    init(withJson json: String) throws {
        var ptr: ParametersPtr?
        let error = json.withCString { cs in
            ergo_lib_parameters_from_json(cs, &ptr)
        }
        try checkError(error)
        self.pointer = ptr!
    }

    deinit {
        ergo_lib_parameters_delete(self.pointer)
    }
}
