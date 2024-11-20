import {Tensor, getTensorShape, getTensorStrides, flattenTensor, dataToTensor, reshapeWithStrides} from "./utils";
import {DataType, TensorHolder} from "./protos/generic_pb";
import fs from "fs";

export const jsToWasm = (uintArray: Uint8Array): typeof Module.VectorUint8 => {
    const byteVector = new Module.VectorUint8();
    uintArray.forEach(byte => byteVector.push_back(byte));
    return byteVector;
}

export const wasmToJs = (byteVector: typeof Module.VectorUint8): Uint8Array => {
    const uintArray = new Uint8Array(byteVector.size());
    for (let i = 0; i < byteVector.size(); i++) {
        uintArray[i] = byteVector.get(i);
    }
    return uintArray;
}

export const dumps_proto_tensor = (pt: Tensor): Uint8Array => {
    const shape = getTensorShape(pt);
    const strides = getTensorStrides(shape);
    const binaryData = new Uint8Array(new Float64Array(flattenTensor(pt)).buffer);

    const proto = new TensorHolder();
    proto.setDtype(DataType.DOUBLE); // TODO: always double for now
    proto.setSizesList(shape);
    proto.setStridesList(strides);
    proto.setData(binaryData);

    return proto.serializeBinary();
}

export const load_proto_tensor = (serialized_pt: Uint8Array): Tensor => {
    const tensorProto = TensorHolder.deserializeBinary(serialized_pt);
    const tensor = dataToTensor(tensorProto.getData(), tensorProto.getDtype());
    return reshapeWithStrides(tensor, tensorProto.getSizesList(), tensorProto.getStridesList());
}

const freeMemory = (wasmObjects: typeof Module.VectorUint8[]): void => {
    wasmObjects.forEach((obj) => obj.delete());
}

export const apply_client_block = (serialized_block: Uint8Array, serialized_context: Uint8Array, serialized_pt: Uint8Array) => {
    console.log("Applying client block...");
    const wasm_serialized_block = jsToWasm(serialized_block);
    const wasm_serialized_context = jsToWasm(serialized_context);
    const wasm_serialized_pt = jsToWasm(serialized_pt);
    const result_wasm = Module.apply_client_block(
        wasm_serialized_block,
        wasm_serialized_context,
        wasm_serialized_pt
    );
    const result = wasmToJs(result_wasm);

    // Clean up to prevent memory leaks
    freeMemory([wasm_serialized_block, wasm_serialized_context, wasm_serialized_pt, result_wasm]);

    return result;
}

export const enc = (serialized_context: Uint8Array, serialized_sk: [Uint8Array, Uint8Array], serialized_pt: Uint8Array, pack_for_transmission: boolean = false) => {
    console.log("Encrypting...");

    const wasm_serialized_context = jsToWasm(serialized_context);
    const wasm_serialized_sk = jsToWasm(serialized_sk[0]);
    const wasm_serialized_pt = jsToWasm(serialized_pt);
    const result_wasm = Module.enc(
        wasm_serialized_context,
        wasm_serialized_sk,
        wasm_serialized_pt,
        true
    );
    const result = wasmToJs(result_wasm);

    // Clean up to prevent memory leaks
    freeMemory([wasm_serialized_context, wasm_serialized_sk, wasm_serialized_pt, result_wasm]);

    return result;
}

export const dec = (serialized_context: Uint8Array, serialized_sk: [Uint8Array, Uint8Array], serialized_ct: Uint8Array) => {
    console.log("Decrypting...");
    const wasm_serialized_context = jsToWasm(serialized_context);
    const wasm_serialized_sk_1 = jsToWasm(serialized_sk[1]);
    const wasm_serialized_ct = jsToWasm(serialized_ct);

    const result_wasm = Module.dec(
        wasm_serialized_context,
        wasm_serialized_sk_1,
        wasm_serialized_ct,
    );
    const result = wasmToJs(result_wasm);

    // Clean up to prevent memory leaks
    freeMemory([wasm_serialized_context, wasm_serialized_sk_1, wasm_serialized_ct, result_wasm]);

    return result;
}

export const generate_key = (serialized_homseq: Uint8Array, serialized_context: Uint8Array) => {
    console.log("Generating key...");

    const wasm_serialized_homseq = jsToWasm(serialized_homseq);
    const wasm_serialized_context = jsToWasm(serialized_context);
    const result_wasm = Module.generate_key(
        wasm_serialized_homseq,
        wasm_serialized_context
    );

    const result = [
        wasmToJs(result_wasm.sk),
        wasmToJs(result_wasm.sk_coefs),
        wasmToJs(result_wasm.pk)
    ];

    // Clean up to prevent memory leaks
    freeMemory([wasm_serialized_homseq, wasm_serialized_context, result_wasm.sk, result_wasm.sk_coefs, result_wasm.pk]);

    return result;
}