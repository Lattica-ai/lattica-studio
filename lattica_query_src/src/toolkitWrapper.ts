import {
  Tensor,
  getTensorShape,
  getTensorStrides,
  flattenTensor,
  dataToTensor,
  reshapeWithStrides,
} from "./tensorUtils";
import {DataType, TensorHolder} from "./serialization/generic";

export const jsToWasm = (uintArray: Uint8Array): typeof Module.VectorUint8 => {
  const byteVector = new Module.VectorUint8();
  uintArray.forEach((byte) => byteVector.push_back(byte));
  return byteVector;
};

export const wasmToJs = (byteVector: typeof Module.VectorUint8): Uint8Array => {
  const uintArray = new Uint8Array(byteVector.size());
  for (let i = 0; i < byteVector.size(); i++) {
    uintArray[i] = byteVector.get(i);
  }
  return uintArray;
};

export const dumps_proto_tensor = (pt: Tensor): Uint8Array => {
  const shape = getTensorShape(pt);
  const strides = getTensorStrides(shape);
  const binaryData = new Uint8Array(new Float64Array(flattenTensor(pt)).buffer);

  const proto = TensorHolder.create({
    dtype: DataType.DOUBLE,  // TODO: always double for now
    sizes: shape,
    strides: strides,
    data: binaryData
  });

  return TensorHolder.encode(proto).finish();
};

export const load_proto_tensor = (serialized_pt: Uint8Array): Tensor => {
  const tensorProto = TensorHolder.decode(serialized_pt);
  const tensor = dataToTensor(tensorProto.data, tensorProto.dtype);
  return reshapeWithStrides(
    tensor,
    tensorProto.sizes,
    tensorProto.strides
  );
};

const freeMemory = (wasmObjects: (typeof Module.VectorUint8)[]): void => {
  wasmObjects.forEach((obj) => obj.delete());
};

export const applyClientBlock = (
  serialized_block: Uint8Array,
  serialized_context: Uint8Array,
  serialized_pt: Uint8Array,
) => {
  console.log("Applying client block...");
  const wasm_serialized_block = jsToWasm(serialized_block);
  const wasm_serialized_context = jsToWasm(serialized_context);
  const wasm_serialized_pt = jsToWasm(serialized_pt);

  let result: Uint8Array;
  let result_wasm: typeof Module.VectorUint8;

  try {
    result_wasm = Module.apply_client_block(
      wasm_serialized_block,
      wasm_serialized_context,
      wasm_serialized_pt,
    );
    result = wasmToJs(result_wasm);
  } finally {
    // Ensure memory is always freed
    freeMemory([
      wasm_serialized_block,
      wasm_serialized_context,
      wasm_serialized_pt,
      result_wasm,
    ]);
  }

  return result;
};

export const encrypt = (
  serialized_context: Uint8Array,
  serialized_sk: [Uint8Array, Uint8Array],
  serialized_pt: Uint8Array,
  pack_for_transmission: boolean = false,
) => {
  console.log("Encrypting...");
  const wasm_serialized_context = jsToWasm(serialized_context);
  const wasm_serialized_sk = jsToWasm(serialized_sk[0]);
  const wasm_serialized_pt = jsToWasm(serialized_pt);

  let result: Uint8Array;
  let result_wasm: typeof Module.VectorUint8;

  try {
    result_wasm = Module.enc(
        wasm_serialized_context,
        wasm_serialized_sk,
        wasm_serialized_pt,
        pack_for_transmission,
    );
    result = wasmToJs(result_wasm);
  } finally {
    // Clean up to prevent memory leaks
    freeMemory([
      wasm_serialized_context,
      wasm_serialized_sk,
      wasm_serialized_pt,
      result_wasm,
    ]);
  }

  return result;
};

export const decrypt = (
  serialized_context: Uint8Array,
  serialized_sk: [Uint8Array, Uint8Array],
  serialized_ct: Uint8Array,
) => {
  console.log("Decrypting...");
  const wasm_serialized_context = jsToWasm(serialized_context);
  const wasm_serialized_sk = jsToWasm(serialized_sk[1]);
  const wasm_serialized_ct = jsToWasm(serialized_ct);

  let result: Uint8Array;
  let result_wasm: typeof Module.VectorUint8;

  try {
    result_wasm = Module.dec(
        wasm_serialized_context,
        wasm_serialized_sk,
        wasm_serialized_ct,
    );
    result = wasmToJs(result_wasm);
  } finally {
    // Clean up to prevent memory leaks
    freeMemory([
      wasm_serialized_context,
      wasm_serialized_sk,
      wasm_serialized_ct,
      result_wasm,
    ]);
  }

  return result;
};

export const generateKey = (
  serialized_homseq: Uint8Array,
  serialized_context: Uint8Array,
): Uint8Array[] => {
  console.log("Generating key...");
  const wasm_serialized_homseq = jsToWasm(serialized_homseq);
  const wasm_serialized_context = jsToWasm(serialized_context);

  let result: Uint8Array[];
  let result_wasm: typeof Module.VectorUint8;

  try {
    result_wasm = Module.generate_key(
        wasm_serialized_homseq,
        wasm_serialized_context,
    );

    result = [
      wasmToJs(result_wasm.sk),
      wasmToJs(result_wasm.sk_coefs),
      wasmToJs(result_wasm.pk),
    ];
  } finally {
    // Clean up to prevent memory leaks
    freeMemory([
      wasm_serialized_homseq,
      wasm_serialized_context,
      result_wasm.sk,
      result_wasm.sk_coefs,
      result_wasm.pk,
    ]);
  }

  return result;
};
