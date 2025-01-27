console.log("Started Worker(wasm_worker.ts)");


const _jsToWasm = (uintArray) => {
  const byteVector = new Module.VectorUint8();
  uintArray.forEach((byte) => byteVector.push_back(byte));
  return byteVector;
};

const _wasmToJs = (byteVector) => {
  const uintArray = new Uint8Array(byteVector.size());
  for (let i = 0; i < byteVector.size(); i++) {
    uintArray[i] = byteVector.get(i);
  }
  return uintArray;
};

const _freeMemory = (wasmObjects) => {
  wasmObjects.forEach((obj) => obj.delete());
};


const _applyClientBlock = (data) => {
    let res;
    let resultWasm;
    const wasmBlock = _jsToWasm(data.block);
    const wasmContext = _jsToWasm(data.context);
    const wasmPt = _jsToWasm(data.pt);
    try {
        resultWasm = Module.apply_client_block(wasmBlock, wasmContext, wasmPt);
        res = _wasmToJs(resultWasm);
    } finally {
        _freeMemory([wasmBlock, wasmContext, wasmPt, resultWasm]);
    }
    return res;
}


const _generateKey = (data) => {
    let res;
    let resultWasm;
    const wasmHomseq = _jsToWasm(data.homseq);
    const wasmContext = _jsToWasm(data.context);
    try {
        resultWasm = Module.generate_key(wasmHomseq, wasmContext);
        res = [
            _wasmToJs(resultWasm.sk),
            _wasmToJs(resultWasm.sk_coefs),
            _wasmToJs(resultWasm.pk)
        ];
    } finally {
        _freeMemory([
            wasmHomseq, wasmContext, resultWasm.sk, resultWasm.sk_coefs, resultWasm.pk]);
    }
    return res;
};

const _dec = (data) => {
    let res;
    let resultWasm;
    const wasmContext = _jsToWasm(data.context);
    const wasmSk = _jsToWasm(data.sk);
    const wasmCt = _jsToWasm(data.ct);
    try {
        resultWasm = Module.dec(wasmContext, wasmSk, wasmCt);
        res = _wasmToJs(resultWasm);
    } finally {
        _freeMemory([wasmContext, wasmSk, wasmCt, resultWasm]);
    }
    return res;
};

const _enc = (data) => {
    let res;
    let resultWasm;
    const wasmContext = _jsToWasm(data.context);
    const wasmSk = _jsToWasm(data.sk);
    const wasmPt = _jsToWasm(data.pt);
    try {
        resultWasm = Module.enc(wasmContext, wasmSk, wasmPt, data.pack_for_transmission);
        res = _wasmToJs(resultWasm);
    } finally {
        _freeMemory([wasmContext, wasmSk, wasmPt, resultWasm]);
    }
    return res;
};


let wasmRuntimeInitializedHandler = async () => {
    self.postMessage("loaded_wasm_module" );

    self.onmessage = (event) => {
      const message = event.data;
      let result;
      switch (message.operation) {
        case "apply_client_block":
            result = _applyClientBlock(message.data);
            self.postMessage({ success: true, data: result });
            break;
        case 'generate_key':
            result = _generateKey(message.data);
            self.postMessage({ success: true, data: result });
            break;
        case 'dec':
            result = _dec(message.data);
            self.postMessage({ success: true, data: result });
            break;
        case 'enc':
            result = _enc(message.data);
            self.postMessage({ success: true, data: result });
            break;
        default:
          self.postMessage({ success: false, error: "Unknown operation" });
      }
    };
};

Module = { onRuntimeInitialized: wasmRuntimeInitializedHandler };
importScripts('cpp_sdk.js');
