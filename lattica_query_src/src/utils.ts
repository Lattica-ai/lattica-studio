import * as worker_api from "./worker_api";
import * as toolkit from "./toolkit_mock";
// const {QueryClientBlock} = require("./protos/hom_op_pb");

export type Tensor = number | Tensor[];

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

// export const jsToWasm = (uintArray: Uint8Array) => {
//     const byteVector = new Module.VectorUint8();
//     uintArray.forEach(byte => byteVector.push_back(byte));
//     return byteVector;
// }

// function wasmToJs(byteVector: Module.VectorUint8) {
//     const uintArray = new Uint8Array(byteVector.size());
//     for (let i = 0; i < byteVector.size(); i++) {
//         uintArray[i] = byteVector.get(i);
//     }
//     return uintArray;
// }

// function processInputBytes() {
//     console.log("Processing input bytes...");
//     const inputBytes = new Uint8Array([10, 4, 2, 1, 4, 1, 18, 4, 4, 1, 1, 1, 26, 64, 65, 43, 0, 0, 0, 0, 0, 0, 209, 9, 0, 0, 0, 0, 0, 0, 102, 112, 0, 0, 0, 0, 0, 0, 177, 122, 0, 0, 0, 0, 0, 0, 65, 43, 0, 0, 0, 0, 0, 0, 48, 150, 0, 0, 0, 0, 0, 0, 155, 47, 0, 0, 0, 0, 0, 0, 80, 37, 0, 0, 0, 0, 0, 0, 32, 3]);
//     const byteVector = jsToWasm(inputBytes);
//     const outputPair = Module.process_bytes(byteVector);
//
//     const outputVector1 = wasmToJs(outputPair.first);
//     const outputVector2 = wasmToJs(outputPair.second);
//     console.log("Output Bytes Vector 1:", outputVector1);
//     console.log("Output Bytes Vector 2:", outputVector2);
//
//     // Clean up to prevent memory leaks
//     byteVector.delete();
//     outputPair.first.delete();
//     outputPair.second.delete();
// }

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

  let serialized_pt = toolkit.decrypt(context, secretKey, serialized_ct_res);

  for (const block_proto of clientBlocks) {
    console.log("Applying client operators");
    serialized_pt = toolkit.apply_client_block(
      block_proto.serializeBinary(),
      context,
      serialized_pt,
    );

    serialized_ct = await toolkit.encrypt(context, secretKey, serialized_pt, true);
    serialized_ct_res = await worker_api.apply_hom_pipeline(
      jwt,
      serialized_ct,
      block_proto.getBlockIndex() + 1,
    );
  }
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
