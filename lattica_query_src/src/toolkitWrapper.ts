import {
  Tensor,
  getTensorShape,
  getTensorStrides,
  flattenTensor,
  dataToTensor,
  reshapeWithStrides,
} from "./tensorUtils";
import { DataType, TensorHolder } from "./serialization/generic";
import { workerManagerSingleton } from "./wasm/wasm_worker_manager";

export const dumps_proto_tensor = (pt: Tensor): Uint8Array => {
  const shape = getTensorShape(pt);
  const strides = getTensorStrides(shape);
  const binaryData = new Uint8Array(new Float64Array(flattenTensor(pt)).buffer);

  const proto = TensorHolder.create({
    dtype: DataType.DOUBLE, // TODO: always double for now
    sizes: shape,
    strides: strides,
    data: binaryData,
  });

  return TensorHolder.encode(proto).finish();
};

export const load_proto_tensor = (serialized_pt: Uint8Array): Tensor => {
  const tensorProto = TensorHolder.decode(serialized_pt);
  const tensor = dataToTensor(tensorProto.data, tensorProto.dtype);
  return reshapeWithStrides(tensor, tensorProto.sizes, tensorProto.strides);
};

export const applyClientBlock = async (
  serialized_block: Uint8Array,
  serialized_context: Uint8Array,
  serialized_pt: Uint8Array,
) => {
  workerManagerSingleton.logConsole("Applying client block...");
  return await workerManagerSingleton.callWorker("apply_client_block", {
    block: serialized_block,
    context: serialized_context,
    pt: serialized_pt,
  });
};

export const encrypt = async (
  serialized_context: Uint8Array,
  serialized_sk: [Uint8Array, Uint8Array],
  serialized_pt: Uint8Array,
  pack_for_transmission: boolean = false,
  pt_axis_external: number | undefined,
) => {
  workerManagerSingleton.logConsole("Encrypting...");
  return await workerManagerSingleton.callWorker("enc", {
    context: serialized_context,
    sk: serialized_sk[0],
    pt: serialized_pt,
    pack_for_transmission,
    pt_axis_external,
  });
};

export const decrypt = async (
  serialized_context: Uint8Array,
  serialized_sk: [Uint8Array, Uint8Array],
  serialized_ct: Uint8Array,
) => {
  workerManagerSingleton.logConsole("Decrypting...");
  return await workerManagerSingleton.callWorker("dec", {
    context: serialized_context,
    sk: serialized_sk[1],
    ct: serialized_ct,
  });
};

export const generateKey = async (
  serialized_homseq: Uint8Array,
  serialized_context: Uint8Array,
) => {
  workerManagerSingleton.logConsole("Generating key...");
  return await workerManagerSingleton.callWorker("generate_key", {
    homseq: serialized_homseq,
    context: serialized_context,
  });
};
