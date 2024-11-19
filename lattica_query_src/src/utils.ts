import * as worker_api from "./worker_api";
import * as toolkit from "./toolkit_wrapper";
import {DataType} from "./protos/generic_pb";
const {QueryClientBlock} = require("./protos/hom_op_pb");

export type Tensor = number | Tensor[];

export const getTensorShape = (tensor: Tensor): number[] => {
    if (typeof tensor === "number") {
        return [];
    }
    return [tensor.length, ...getTensorShape(tensor[0])];
}

export const getTensorStrides = (shape: number[]): number[] => {
    const strides = Array(shape.length).fill(1);
    for (let i = shape.length - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    return strides;
}

export const flattenTensor = (tensor: Tensor): number[] => {
    if (typeof tensor === "number") {
        return [tensor]; // Base case: a single number.
    }
    return tensor.reduce<number[]>((acc, value) => acc.concat(flattenTensor(value)), []);
}

export const dataToTensor = (data, dtype) => {
    const buffer = data.slice().buffer;  // Copy the data to a new buffer
    switch (dtype) {
        case DataType.DOUBLE:
            return new Float64Array(buffer);
        default:
            throw new Error(`Unsupported dtype: ${dtype}`);
    }
};

export const reshapeWithStrides = (flatArray, shape, strides) => {
  // Create a function to access the elements in the flattened array using strides
  function getElement(indices) {
    let index = 0;
    for (let i = 0; i < indices.length; i++) {
      index += indices[i] * strides[i];
    }
    return flatArray[index];
  }

  // Recursively create the multi-dimensional array based on shape and strides
  function createMultiDimensionalArray(dimIndex = 0, indices = []) {
    if (dimIndex === shape.length) {
      return getElement(indices);
    }

    let result = [];
    for (let i = 0; i < shape[dimIndex]; i++) {
      indices[dimIndex] = i;
      result.push(createMultiDimensionalArray(dimIndex + 1, indices));
    }
    return result;
  }

  return createMultiDimensionalArray();
}

export const imageToMatrix = (
  image: HTMLImageElement,
): Promise<number[][][]> => {
  return new Promise((resolve, reject) => {
    const canvas = document.createElement("canvas");
    canvas.width = image.width;
    canvas.height = image.height;
    const ctx = canvas.getContext("2d");

    if (!ctx) {
      reject("Unable to create 2D context");
      return;
    }

    ctx.drawImage(image, 0, 0, image.width, image.height);
    const { data, width, height } = ctx.getImageData(
      0,
      0,
      image.width,
      image.height,
    );

    const matrix = Array.from({ length: height }, (_, y) =>
      Array.from({ length: width }, (_, x) => {
        const index = (y * width + x) * 4;
        return [data[index], data[index + 1], data[index + 2], data[index + 3]];
      }),
    );

    resolve(matrix);
  });
};

// assume we get and return encrypted: first client block (if exists) was applied, and last client block (will be applied)- are not in the clientBlocks list
export const runInferenceDemo = async (
  jwt: string,
  context: Uint8Array,
  secretKey: [Uint8Array, Uint8Array],
  serialized_ct: Uint8Array,
  clientBlocks: typeof QueryClientBlock[],
  firstBlockToRun: number,
): Promise<Uint8Array> => {
  let serialized_ct_res = await worker_api.apply_hom_pipeline(
    jwt,
    serialized_ct,
    firstBlockToRun,
  );
  console.log(serialized_ct_res);
  if (clientBlocks.length == 0) {
    // just apply once and return
    return serialized_ct_res;
  }

  let serialized_pt = toolkit.dec(context, secretKey, serialized_ct_res);

  for (const block_proto of clientBlocks) {
    console.log("Applying client operators");
    serialized_pt = toolkit.apply_client_block(
      block_proto.serializeBinary(),
      context,
      serialized_pt,
    );

    serialized_ct = toolkit.enc(context, secretKey, serialized_pt, true);
    serialized_ct_res = await worker_api.apply_hom_pipeline(
      jwt,
      serialized_ct,
      block_proto.getBlockIndex() + 1,
    );
  }

  return serialized_ct_res;
};

// export const runInference = async (
//   jwt: string,
//   context: Buffer,
//   secretKey: [Buffer, Buffer],
//   pt: Tensor,
//   homseq: Buffer,
// ): Promise<Tensor> => {
//   let serialized_pt = toolkit.dumps_proto_tensor(pt);
//
//   const clientBlocks =
//     QueryClientSequentialHomOp.deserializeBinary(homseq).getClientBlocksList();
//   const isBeFirst =
//     clientBlocks.length === 0 || clientBlocks[0].getBlockIndex() !== 0;
//
//   let serialized_ct;
//   let serialized_ct_res;
//   if (isBeFirst) {
//     serialized_ct = toolkit.encrypt(context, secretKey, serialized_pt, true);
//     serialized_ct_res = await worker_api.apply_hom_pipeline(
//       jwt,
//       serialized_ct,
//       0,
//     );
//     serialized_pt = toolkit.decrypt(context, secretKey, serialized_ct_res);
//   }
//
//   for (const block_proto of clientBlocks) {
//     console.log("Applying client operators");
//     const serialized_block_proto = block_proto.serializeBinary();
//     serialized_pt = toolkit.apply_client_block(
//       serialized_block_proto,
//       context,
//       serialized_pt,
//     );
//     if (block_proto.getIsLast()) {
//       break;
//     }
//
//     const serialized_ct = toolkit.encrypt(
//       context,
//       secretKey,
//       serialized_pt,
//       true,
//     );
//     serialized_ct_res = await worker_api.apply_hom_pipeline(
//       jwt,
//       serialized_ct,
//       block_proto.getBlockIndex() + 1,
//     );
//     serialized_pt = toolkit.decrypt(context, secretKey, serialized_ct_res);
//   }
//   return toolkit.load_proto_tensor(serialized_pt);
// };

// export const to_serialized_lattica_tensor = (tensor: Tensor): Buffer =>{
//     return new Buffer([1, 2, 3]);
// }
