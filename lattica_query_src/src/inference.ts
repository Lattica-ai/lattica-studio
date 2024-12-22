import * as worker_api from "./workerApi";
import * as toolkit from "./toolkitWrapper";
import { QueryClientBlock } from "./protos/hom_op_pb";

// assume we get and return encrypted: first client block (if exists) was applied, and last client block (will be applied)- are not in the clientBlocks list
export const runInferenceDemo = async (
  jwt: string,
  beUrl: string,
  context: Uint8Array,
  secretKey: [Uint8Array, Uint8Array],
  serialized_ct: Uint8Array,
  clientBlocks: QueryClientBlock[],
  firstBlockToRun: number,
): Promise<Uint8Array> => {
  let serialized_ct_res = await worker_api.apply_hom_pipeline(
    jwt,
    beUrl,
    serialized_ct,
    firstBlockToRun,
  );

  for (const block_proto of clientBlocks) {
    let serialized_pt = toolkit.decrypt(context, secretKey, serialized_ct_res);

    serialized_pt = toolkit.applyClientBlock(
      block_proto.serializeBinary(),
      context,
      serialized_pt,
    );

    serialized_ct = toolkit.encrypt(context, secretKey, serialized_pt, true);

    serialized_ct_res = await worker_api.apply_hom_pipeline(
      jwt,
      beUrl,
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