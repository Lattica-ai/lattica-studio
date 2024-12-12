// package: lattica_proto
// file: protos/key_aux.proto

import * as jspb from "google-protobuf";
import * as protos_ciphertexts_pb from "../protos/ciphertexts_pb";

export class CKKS_AuxKey extends jspb.Message {
  hasBaseKey(): boolean;
  clearBaseKey(): void;
  getBaseKey(): protos_ciphertexts_pb.CKKS_Ciphertext | undefined;
  setBaseKey(value?: protos_ciphertexts_pb.CKKS_Ciphertext): void;

  hasFullKey(): boolean;
  clearFullKey(): void;
  getFullKey(): protos_ciphertexts_pb.CKKS_Ciphertext | undefined;
  setFullKey(value?: protos_ciphertexts_pb.CKKS_Ciphertext): void;

  hasIdentityKey(): boolean;
  clearIdentityKey(): void;
  getIdentityKey(): protos_ciphertexts_pb.CKKS_Ciphertext | undefined;
  setIdentityKey(value?: protos_ciphertexts_pb.CKKS_Ciphertext): void;

  hasSquareKey(): boolean;
  clearSquareKey(): void;
  getSquareKey(): protos_ciphertexts_pb.CKKS_Ciphertext | undefined;
  setSquareKey(value?: protos_ciphertexts_pb.CKKS_Ciphertext): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): CKKS_AuxKey.AsObject;
  static toObject(
    includeInstance: boolean,
    msg: CKKS_AuxKey,
  ): CKKS_AuxKey.AsObject;
  static extensions: { [key: number]: jspb.ExtensionFieldInfo<jspb.Message> };
  static extensionsBinary: {
    [key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>;
  };
  static serializeBinaryToWriter(
    message: CKKS_AuxKey,
    writer: jspb.BinaryWriter,
  ): void;
  static deserializeBinary(bytes: Uint8Array): CKKS_AuxKey;
  static deserializeBinaryFromReader(
    message: CKKS_AuxKey,
    reader: jspb.BinaryReader,
  ): CKKS_AuxKey;
}

export namespace CKKS_AuxKey {
  export type AsObject = {
    baseKey?: protos_ciphertexts_pb.CKKS_Ciphertext.AsObject;
    fullKey?: protos_ciphertexts_pb.CKKS_Ciphertext.AsObject;
    identityKey?: protos_ciphertexts_pb.CKKS_Ciphertext.AsObject;
    squareKey?: protos_ciphertexts_pb.CKKS_Ciphertext.AsObject;
  };
}

export class RBGV_AuxKey extends jspb.Message {
  hasBaseKey(): boolean;
  clearBaseKey(): void;
  getBaseKey(): protos_ciphertexts_pb.RBGV_Ciphertext | undefined;
  setBaseKey(value?: protos_ciphertexts_pb.RBGV_Ciphertext): void;

  hasFullKey(): boolean;
  clearFullKey(): void;
  getFullKey(): protos_ciphertexts_pb.RBGV_Ciphertext | undefined;
  setFullKey(value?: protos_ciphertexts_pb.RBGV_Ciphertext): void;

  hasIdentityKey(): boolean;
  clearIdentityKey(): void;
  getIdentityKey(): protos_ciphertexts_pb.RBGV_Ciphertext | undefined;
  setIdentityKey(value?: protos_ciphertexts_pb.RBGV_Ciphertext): void;

  hasSquareKey(): boolean;
  clearSquareKey(): void;
  getSquareKey(): protos_ciphertexts_pb.RBGV_Ciphertext | undefined;
  setSquareKey(value?: protos_ciphertexts_pb.RBGV_Ciphertext): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): RBGV_AuxKey.AsObject;
  static toObject(
    includeInstance: boolean,
    msg: RBGV_AuxKey,
  ): RBGV_AuxKey.AsObject;
  static extensions: { [key: number]: jspb.ExtensionFieldInfo<jspb.Message> };
  static extensionsBinary: {
    [key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>;
  };
  static serializeBinaryToWriter(
    message: RBGV_AuxKey,
    writer: jspb.BinaryWriter,
  ): void;
  static deserializeBinary(bytes: Uint8Array): RBGV_AuxKey;
  static deserializeBinaryFromReader(
    message: RBGV_AuxKey,
    reader: jspb.BinaryReader,
  ): RBGV_AuxKey;
}

export namespace RBGV_AuxKey {
  export type AsObject = {
    baseKey?: protos_ciphertexts_pb.RBGV_Ciphertext.AsObject;
    fullKey?: protos_ciphertexts_pb.RBGV_Ciphertext.AsObject;
    identityKey?: protos_ciphertexts_pb.RBGV_Ciphertext.AsObject;
    squareKey?: protos_ciphertexts_pb.RBGV_Ciphertext.AsObject;
  };
}

export interface AuxKeyTypeMap {
  BASE_KEY: 0;
  FULL_KEY: 1;
  IDENTITY_KEY: 2;
  SQUARE_KEY: 3;
}

export const AuxKeyType: AuxKeyTypeMap;
