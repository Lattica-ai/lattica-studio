import {
  Tensor,
  getTensorShape,
  getTensorStrides,
  flattenTensor,
  dataToTensor,
  reshapeWithStrides,
} from "./tensorUtils";
import { DataType, TensorHolder } from "./serialization/generic";

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


export const applyClientBlock = (
  serialized_block: Uint8Array,
  serialized_context: Uint8Array,
  serialized_pt: Uint8Array,
) => {
  console.log("Applying client block...");
  return workerManager.callWorker("apply_client_block", {
    block: serialized_block,
    context: serialized_context,
    pt: serialized_pt
  });
};

export const encrypt = (
  serialized_context: Uint8Array,
  serialized_sk: [Uint8Array, Uint8Array],
  serialized_pt: Uint8Array,
  pack_for_transmission: boolean = false,
) => {
  console.log("Encrypting...");
  return workerManager.callWorker("enc", {
    context: serialized_context,
    sk: serialized_sk[0],
    pt: serialized_pt,
    pack_for_transmission
  });
};

export const decrypt = (
  serialized_context: Uint8Array,
  serialized_sk: [Uint8Array, Uint8Array],
  serialized_ct: Uint8Array,
) => {
  console.log("Decrypting...");
  return workerManager.callWorker("dec", {
    context: serialized_context,
    sk: serialized_sk[1],
    ct: serialized_ct
  });
};

export const generateKey = (
    serialized_homseq: Uint8Array,
    serialized_context: Uint8Array,
    ) => {
    console.log("Generating key...");
    return workerManager.callWorker("generate_key", {
        homseq: serialized_homseq,
        context: serialized_context
    });
};
