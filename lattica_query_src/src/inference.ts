import * as worker_api from "./workerApi";
import * as toolkit from "./toolkitWrapper";
import { QueryClientBlock } from "./serialization/hom_op";
import { Tensor, loads_proto_tensor, dumps_proto_tensor } from "./tensorUtils";

// assume we get and return encrypted: first client block (if exists) was applied, and last client block (will be applied)- are not in the clientBlocks list
export const runInferenceDemo = async (
  jwt: string,
  beUrl: string,
  context: Uint8Array,
  secretKey: [Uint8Array, Uint8Array],
  serialized_ct: Uint8Array,
  clientBlocks: QueryClientBlock[],
  firstBlockToRun: number,
  asComplex: boolean = false,
): Promise<Uint8Array> => {
  let serialized_ct_res = await worker_api.apply_hom_pipeline(
    jwt,
    beUrl,
    serialized_ct,
    firstBlockToRun,
  );

  for (const block_proto of clientBlocks) {
    let serialized_pt = await toolkit.decrypt(
      context,
      secretKey,
      serialized_ct_res,
      asComplex,
    );

    serialized_pt = await toolkit.applyClientBlock(
      QueryClientBlock.encode(block_proto).finish(),
      context,
      serialized_pt,
    );

    serialized_ct = await toolkit.encrypt(
      context,
      secretKey,
      serialized_pt,
      true,
      block_proto.ptAxisExternal,
    );

    serialized_ct_res = await worker_api.apply_hom_pipeline(
      jwt,
      beUrl,
      serialized_ct,
      block_proto.blockIndex + 1,
    );
  }

  return serialized_ct_res;
};

export const runInference = async (
  jwt: string,
  beUrl: string,
  context: Uint8Array,
  secretKey: [Uint8Array, Uint8Array],
  pt: Tensor,
  clientBlocks: QueryClientBlock[],
  asComplex: boolean = false,
): Promise<Tensor> => {
  let serialized_pt = dumps_proto_tensor(pt);

  const isBeFirst =
    clientBlocks.length === 0 || clientBlocks[0].blockIndex !== 0;

  let serialized_ct;
  let serialized_ct_res;
  if (isBeFirst) {
    serialized_ct = await toolkit.encrypt(
      context,
      secretKey,
      serialized_pt,
      true,
    );
    serialized_ct_res = await worker_api.apply_hom_pipeline(
      jwt,
      beUrl,
      serialized_ct,
      0,
    );
    serialized_pt = await toolkit.decrypt(
      context,
      secretKey,
      serialized_ct_res,
      asComplex,
    );
  }

  for (const block_proto of clientBlocks) {
    console.log("Applying client operators");
    const serializedBlock = QueryClientBlock.encode(block_proto).finish();
    serialized_pt = await toolkit.applyClientBlock(
      serializedBlock,
      context,
      serialized_pt,
    );
    if (block_proto.isLast) {
      break;
    }

    const serialized_ct = await toolkit.encrypt(
      context,
      secretKey,
      serialized_pt,
      true,
      block_proto.ptAxisExternal,
    );
    serialized_ct_res = await worker_api.apply_hom_pipeline(
      jwt,
      beUrl,
      serialized_ct,
      block_proto.blockIndex + 1,
    );
    serialized_pt = await toolkit.decrypt(
      context,
      secretKey,
      serialized_ct_res,
      asComplex,
    );
  }
  return loads_proto_tensor(serialized_pt);
};
