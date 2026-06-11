import * as workerApi from './workerApi';
import * as toolkit from './toolkitWrapper';
import { Tensor, loads_proto_tensor, dumps_proto_tensor } from './tensorUtils';

export const runInference = async (
  jwt: string,
  beUrl: string,
  context: Uint8Array,
  secretKey: [Uint8Array, Uint8Array],
  pt: Tensor,
  preBlock: Uint8Array,
  postBlock: Uint8Array,
  ptAxisExternal?: number,
  asComplex: boolean = false,
  contextRingSwitch: Uint8Array | null = null,
  secretKeyRingSwitch: [Uint8Array, Uint8Array] | null = null,
): Promise<Tensor> => {
  let serialized_pt = dumps_proto_tensor(pt);
  
  const encContext = ringSwitchContext ?? context;
  const encSecretKey = ringSwitchSecretKey ?? secretKey;

  console.log('Applying preprocessing client block');
  serialized_pt = await toolkit.applyClientBlock(preBlock, encContext, serialized_pt);

  console.log('Encrypting input tensor');
  let contextEnc: Uint8Array;
  let secretKeyEnc: [Uint8Array, Uint8Array];
  if (contextRingSwitch) {
    contextEnc = contextRingSwitch;
    secretKeyEnc = secretKeyRingSwitch!;
  }
  else {
    contextEnc = context;
    secretKeyEnc = secretKey;
  }
  const serialized_ct = await toolkit.encrypt(
    contextEnc,
    secretKeyEnc,
    serialized_pt,
    true,
    ptAxisExternal,
  );

  console.log('Sending encrypted data to server for homomorphic inference');
  const serialized_ct_res = await workerApi.apply_hom_pipeline(jwt, beUrl, serialized_ct);

  console.log('Decrypting result tensor');
  serialized_pt = await toolkit.decrypt(context, secretKey, serialized_ct_res, asComplex);

  console.log('Applying postprocessing client block');
  serialized_pt = await toolkit.applyClientBlock(postBlock, context, serialized_pt);

  return loads_proto_tensor(serialized_pt);
};
