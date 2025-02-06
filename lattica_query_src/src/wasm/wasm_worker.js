console.log("Started Worker(wasm_worker.ts)");

function _jsToWasm(uintArray) {
  const ptr = Module._malloc(uintArray.length);
  Module.HEAPU8.set(uintArray, ptr);
  return ptr;
}

function _wasmToJs(wasmArrayResult) {
  return new Uint8Array(
    Module.HEAPU8.buffer,
    wasmArrayResult.ptr,
    wasmArrayResult.length,
  ).slice();
}

const _applyClientBlock = (data) => {
  let res;
  let resultWasm;
  const wasmBlock_ptr = _jsToWasm(data.block);
  const wasmContext_ptr = _jsToWasm(data.context);
  const wasmPt_ptr = _jsToWasm(data.pt);
  try {
    resultWasm = Module.apply_client_block(
      wasmBlock_ptr,
      data.block.length,
      wasmContext_ptr,
      data.context.length,
      wasmPt_ptr,
      data.pt.length,
    );
    res = _wasmToJs(resultWasm);
  } catch (e) {
    console.log("Error:", e);
    throw e;
  } finally {
    Module._free(wasmBlock_ptr);
    Module._free(wasmContext_ptr);
    Module._free(wasmPt_ptr);
  }
  Module.freeMemory(resultWasm.ptr);
  return res;
};

const _generateKey = (data) => {
  let res;
  let resultWasm;
  const wasmHomseq_ptr = _jsToWasm(data.homseq);
  const wasmContext_ptr = _jsToWasm(data.context);
  try {
    resultWasm = Module.generate_key(
      wasmHomseq_ptr,
      data.homseq.length,
      wasmContext_ptr,
      data.context.length,
    );
    res = [
      _wasmToJs(resultWasm.sk),
      _wasmToJs(resultWasm.sk_coefs),
      _wasmToJs(resultWasm.pk),
    ];
  } catch (e) {
    console.log("Error:", e);
    throw e;
  } finally {
    Module._free(wasmHomseq_ptr);
    Module._free(wasmContext_ptr);
  }
  Module.freeMemory(resultWasm.sk.ptr);
  Module.freeMemory(resultWasm.sk_coefs.ptr);
  Module.freeMemory(resultWasm.pk.ptr);
  return res;
};

const _dec = (data) => {
  let res;
  let resultWasm;
  const wasmContext_ptr = _jsToWasm(data.context);
  const wasmSk_ptr = _jsToWasm(data.sk);
  const wasmCt_ptr = _jsToWasm(data.ct);
  try {
    resultWasm = Module.dec(
      wasmContext_ptr,
      data.context.length,
      wasmSk_ptr,
      data.sk.length,
      wasmCt_ptr,
      data.ct.length,
    );
    res = _wasmToJs(resultWasm);
  } catch (e) {
    console.log("Error:", e);
    throw e;
  } finally {
    Module._free(wasmContext_ptr);
    Module._free(wasmSk_ptr);
    Module._free(wasmCt_ptr);
  }
  Module.freeMemory(resultWasm.ptr);
  return res;
};

const _enc = (data) => {
  let res;
  let resultWasm;
  const wasmContext_ptr = _jsToWasm(data.context);
  const wasmSk_ptr = _jsToWasm(data.sk);
  const wasmPt_ptr = _jsToWasm(data.pt);

  try {
    resultWasm = Module.enc(
      wasmContext_ptr,
      data.context.length,
      wasmSk_ptr,
      data.sk.length,
      wasmPt_ptr,
      data.pt.length,
      data.pack_for_transmission,
    );

    res = _wasmToJs(resultWasm);
  } catch (e) {
    console.log("Error:", e);
    throw e;
  } finally {
    Module._free(wasmContext_ptr);
    Module._free(wasmSk_ptr);
    Module._free(wasmPt_ptr);
  }
  Module.freeMemory(resultWasm.ptr);
  return res;
};

let wasmRuntimeInitializedHandler = async () => {
  self.postMessage("loaded_wasm_module");

  self.onmessage = (event) => {
    const message = event.data;
    let result;
    switch (message.operation) {
      case "apply_client_block":
        result = _applyClientBlock(message.data);
        self.postMessage({ success: true, data: result });
        break;
      case "generate_key":
        result = _generateKey(message.data);
        self.postMessage({ success: true, data: result });
        break;
      case "dec":
        result = _dec(message.data);
        self.postMessage({ success: true, data: result });
        break;
      case "enc":
        result = _enc(message.data);
        self.postMessage({ success: true, data: result });
        break;
      default:
        self.postMessage({ success: false, error: "Unknown operation" });
    }
  };
};

Module = { onRuntimeInitialized: wasmRuntimeInitializedHandler };
importScripts("cpp_sdk.js");
