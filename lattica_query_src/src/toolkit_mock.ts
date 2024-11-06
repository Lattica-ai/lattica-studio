import { Tensor } from "./utils";
import * as fs from "fs";

export const generateKey = (
  serialized_homseq: Buffer,
  serialized_context: Buffer,
): { secretKey: [Buffer, Buffer]; evaluationKey: Buffer } => {
  return {
    secretKey: [Buffer.from([3, 2, 1]), Buffer.from([31, 21, 11])],
    evaluationKey: fs.readFileSync("mock_data/my_pk.lpk"),
  };
};

export const encrypt = (
  serialized_context: Buffer | null,
  serialized_sk: [Buffer, Buffer] | null,
  serialized_pt: Buffer,
  pack_for_transmission: boolean = false,
): Buffer => {
  return Buffer.from([10, 20, 30]);
};

export const decrypt = (
  context: Buffer | null,
  secretKey: [Buffer, Buffer] | null,
  encryptedResult: Buffer,
): Buffer => {
  return Buffer.from([15, 25, 35]);
};

export const apply_client_block = (
  serialized_block_proto: Uint8Array,
  serialized_context: Buffer,
  serialized_pt: Buffer,
): Buffer => {
  return Buffer.from([1, 2, 3]);
};

export const dumps_proto_tensor = (tensor: Tensor): Buffer => {
  return Buffer.from([1, 2, 3]);
};

export const load_proto_tensor = (serialized_pt: Buffer): Tensor => {
  return [123, 234, 345];
};
