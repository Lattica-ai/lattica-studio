import { Tensor } from "./utils";

export const generateKey = (
  serialized_homseq: Uint8Array,
  serialized_context: Uint8Array,
): { secretKey: [Uint8Array, Uint8Array]; evaluationKey: Uint8Array } => {
  return {
    secretKey: [
      Uint8Array.from([1, 2, 3]),
      Uint8Array.from([4, 5, 6]),
    ],
    evaluationKey: Uint8Array.from([7, 8, 9]),
  }
  // return {
  //   secretKey: [
  //     fs.readFileSync("mock_data/my_sk1"),
  //     fs.readFileSync("mock_data/my_sk2"),
  //   ],
  //   evaluationKey: fs.readFileSync("mock_data/my_pk.lpk"),
  // };
};

export const encrypt = async (  // TODO: remove async
  serialized_context: Uint8Array | null,
  serialized_sk: [Uint8Array, Uint8Array] | null,
  serialized_pt: Uint8Array,
  pack_for_transmission: boolean = false,
): Promise<Uint8Array> => {
  const mockCt = await fetch("mock_data/ser_ct1");
  return new Uint8Array(await mockCt.arrayBuffer());
};

export const decrypt = (
  context: Uint8Array | null,
  secretKey: [Uint8Array, Uint8Array] | null,
  encryptedResult: Uint8Array,
): Uint8Array => {
  return Uint8Array.from([15, 25, 35]);
};

export const apply_client_block = (
  serialized_block_proto: Uint8Array,
  serialized_context: Uint8Array,
  serialized_pt: Uint8Array,
): Uint8Array => {
  return Uint8Array.from([1, 2, 3]);
};

export const dumps_proto_tensor = (tensor: Tensor): Uint8Array => {
  return new Uint8Array([1, 2, 3]);
  // return fs.readFileSync("mock_data/ser_pt_initial");
};

export const load_proto_tensor = (serialized_pt: Uint8Array): Tensor => {
  return [123, 234, 345];
};
