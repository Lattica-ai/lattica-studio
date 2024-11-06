import * as worker_api from "./worker_api";
import * as toolkit from "./toolkit_mock";
import { QueryClientSequentialHomOp } from "./protos/hom_op_pb";

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

export const runInference = async (
  jwt: string,
  context: Buffer | null,
  secretKey: [Buffer, Buffer] | null,
  encryptedInput: Buffer,
  homseq: Buffer | null,
): Promise<Uint8Array> => {
  const homsec_proto = QueryClientSequentialHomOp.deserializeBinary(homseq);

  // TODO: assume is_be_first=true, because we already encrypted the input
  const clientBlocks = homsec_proto.getClientBlocksList();
  const isBeFirst =
    clientBlocks.length === 0 || clientBlocks[0].getBlockIndex() !== 0;
  if (!isBeFirst) {
    throw new Error("BE must be first for this demo");
  }

  const serialized_ct_res = await worker_api.apply_hom_pipeline(
    jwt,
    encryptedInput,
    0,
  );

  let serialized_pt = toolkit.decrypt(context, secretKey, serialized_ct_res);

  for (const block_proto of clientBlocks) {
    console.log("Applying client operators");
    const serialized_block_proto = block_proto.serializeBinary();
    serialized_pt = toolkit.apply_client_block(
      serialized_block_proto,
      context,
      serialized_pt,
    );
    if (block_proto.getIsLast()) {
      break;
      //     TODO: assume this is not the case as we return encrypted result
    }

    const serialized_ct = toolkit.encrypt(
      context,
      secretKey,
      serialized_pt,
      true,
    );
    serialized_ct_res = await worker_api.apply_hom_pipeline(
      jwt,
      serialized_ct,
      block_proto.getBlockIndex() + 1,
    );
    serialized_pt = toolkit.decrypt(context, secretKey, serialized_ct_res);
  }
  return toolkit.load_proto_tensor(serialized_pt);

  // return fs.readFileSync('mock_data/ct_res');
};

// export const to_serialized_lattica_tensor = (tensor: Tensor): Buffer =>{
//     return new Buffer([1, 2, 3]);
// }
