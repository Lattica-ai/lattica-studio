console.log('Started Worker(wasm_worker.ts)');

function _jsToWasm(uintArray) {
  const ptr = Module._malloc(uintArray.length);
  Module.HEAPU8.set(uintArray, ptr);
  return { ptr, length: uintArray.length };
}

function _wasmToJs(wasmBufferView) {
  return new Uint8Array(Module.HEAPU8.buffer, wasmBufferView.ptr, wasmBufferView.length).slice();
}

function freeMemory(pointers) {
  pointers.forEach((ptr) => {
    Module._free(ptr);
  });
}

const _applyClientBlock = (data) => {
  let res;
  let resultWasm;
  const wasmBlock = _jsToWasm(data.block);
  const wasmContext = _jsToWasm(data.context);
  const wasmPt = _jsToWasm(data.pt);
  try {
    resultWasm = Module.apply_client_block(wasmBlock, wasmContext, wasmPt);
    res = _wasmToJs(resultWasm);
  } catch (e) {
    console.log('Error:', e);
    throw e;
  } finally {
    freeMemory([wasmBlock.ptr, wasmContext.ptr, wasmPt.ptr]);
  }
  Module.freeMemory(resultWasm.ptr);
  return res;
};

const _generateKey = (data) => {
  let res;
  let resultWasm;
  const wasmHomseq = _jsToWasm(data.homseq);
  const wasmContext = _jsToWasm(data.extended_context);
  try {
    resultWasm = Module.generate_key(wasmHomseq, wasmContext);
    const extendedSecretKey = _wasmToJs(resultWasm.extended_secret_key);
    const pk = _wasmToJs(resultWasm.pk);
    res = [extendedSecretKey, pk];
  } catch (e) {
    console.log('Error:', e);
    throw e;
  } finally {
    freeMemory([wasmHomseq.ptr, wasmContext.ptr]);
  }
  Module.freeMemory(resultWasm.extended_secret_key.ptr);
  Module.freeMemory(resultWasm.pk.ptr);
  return res;
};

const _dec = (data) => {
  let res;
  let resultWasm;
  const wasmContext = _jsToWasm(data.context);
  const wasmSk = _jsToWasm(data.sk);
  const wasmCt = _jsToWasm(data.ct);
  try {
    resultWasm = Module.dec(wasmContext, wasmSk, wasmCt, data.as_complex);
    res = _wasmToJs(resultWasm);
  } catch (e) {
    console.log('Error:', e);
    throw e;
  } finally {
    freeMemory([wasmContext.ptr, wasmSk.ptr, wasmCt.ptr]);
  }
  Module.freeMemory(resultWasm.ptr);
  return res;
};

const _enc = (data) => {
  let res;
  let resultWasm;
  const wasmContext = _jsToWasm(data.context);
  const wasmSk = _jsToWasm(data.sk[0]);
  const wasmSkCoefs = _jsToWasm(data.sk[1]);
  const wasmPt = _jsToWasm(data.pt);

  try {
    resultWasm = Module.enc(
      wasmContext,
      wasmSk,
      wasmSkCoefs,
      wasmPt,
      data.pack_for_transmission,
      data.pt_axis_external,
      data.custom_state_name,
    );

    res = _wasmToJs(resultWasm);
  } catch (e) {
    console.log('Error:', e);
    throw e;
  } finally {
    freeMemory([wasmContext.ptr, wasmSk.ptr, wasmSkCoefs.ptr, wasmPt.ptr]);
  }
  Module.freeMemory(resultWasm.ptr);
  return res;
};

let wasmRuntimeInitializedHandler = async () => {
  self.postMessage('loaded_wasm_module');

  self.onmessage = (event) => {
    const message = event.data;
    let result;
    try {
      switch (message.operation) {
        case 'apply_client_block':
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
          self.postMessage({ success: false, error: 'Unknown operation' });
      }
    } catch (e) {
      console.error('Error in WASM worker operation:', e);
      const error = `${e.message || String(e)} (WASM memory limit: ${memoryLimitMb} MB)`;
      self.postMessage({ success: false, error });
    }
  };
};

// Define custom print function that send log messages to the main thread
function customPrint(...args) {
  const message = args.join(' ');
  console.log('[WASM Log]', message);

  // Send the log to the main thread
  postMessage({ log: true, logtype: 'WASM Log', logmsg: message });
}

// Define custom print function that send error messages to the main thread
function customPrintErr(...args) {
  const message = args.join(' ');
  console.error('[WASM Error]', message);

  // Send the error to the main thread
  postMessage({ log: true, logtype: 'WASM Error', logmsg: message });
}

const WASM_PAGE_BYTES = 64 * 1024;
const INITIAL_MEMORY_MB = 32; // memory the WASM starts with
const MAX_MEMORY_MB_TO_TRY = [3072, 2048, 1024, 512, 256]; // memory limits to try, biggest first

// Try the memory limits from the biggest down and return the first memory the browser accepts.
// * iPhones refuse a 3 GB limit.
let memoryLimitMb; // memory limit the browser accepted

function createWasmMemory() {
  let lastError;
  for (const maxMb of MAX_MEMORY_MB_TO_TRY) {
    try {
      const memory = new WebAssembly.Memory({
        initial: (INITIAL_MEMORY_MB * 1024 * 1024) / WASM_PAGE_BYTES,
        maximum: (maxMb * 1024 * 1024) / WASM_PAGE_BYTES,
        shared: true,
      });
      memoryLimitMb = maxMb;
      console.log(`WASM memory limit: ${maxMb} MB`);
      return memory;
    } catch (e) {
      console.warn(`WASM memory limit of ${maxMb} MB refused:`, e.message || e);
      lastError = e;
    }
  }
  throw lastError;
}

// Tells whether this is a worker thread, so that only the main thread allocates the memory.
const IS_WASM_THREAD = typeof self.name === 'string' && self.name.startsWith('em-pthread');

self.Module = {
  ...(IS_WASM_THREAD ? {} : { wasmMemory: createWasmMemory() }),
  print: customPrint,
  printErr: customPrintErr,
  onRuntimeInitialized: wasmRuntimeInitializedHandler,
};

const wasmPath = import.meta.url ? new URL('./cpp_sdk.js', import.meta.url).href : './cpp_sdk.js';
console.log('Loading WASM module from', wasmPath);
importScripts(wasmPath);
