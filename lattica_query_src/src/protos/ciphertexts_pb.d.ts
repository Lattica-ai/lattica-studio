// package: lattica_proto
// file: protos/ciphertexts.proto

import * as jspb from "google-protobuf";
import * as protos_generic_pb from "../protos/generic_pb";
import * as protos_pt_shape_pb from "../protos/pt_shape_pb";
import * as protos_global_params_and_state_pb from "../protos/global_params_and_state_pb";

export class Ciphertext extends jspb.Message {
  hasA(): boolean;
  clearA(): void;
  getA(): protos_generic_pb.TensorHolder | undefined;
  setA(value?: protos_generic_pb.TensorHolder): void;

  hasB(): boolean;
  clearB(): void;
  getB(): protos_generic_pb.TensorHolder | undefined;
  setB(value?: protos_generic_pb.TensorHolder): void;

  hasPtShape(): boolean;
  clearPtShape(): void;
  getPtShape(): protos_pt_shape_pb.PtShape | undefined;
  setPtShape(value?: protos_pt_shape_pb.PtShape): void;

  hasState(): boolean;
  clearState(): void;
  getState(): protos_global_params_and_state_pb.State | undefined;
  setState(value?: protos_global_params_and_state_pb.State): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): Ciphertext.AsObject;
  static toObject(
    includeInstance: boolean,
    msg: Ciphertext,
  ): Ciphertext.AsObject;
  static extensions: { [key: number]: jspb.ExtensionFieldInfo<jspb.Message> };
  static extensionsBinary: {
    [key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>;
  };
  static serializeBinaryToWriter(
    message: Ciphertext,
    writer: jspb.BinaryWriter,
  ): void;
  static deserializeBinary(bytes: Uint8Array): Ciphertext;
  static deserializeBinaryFromReader(
    message: Ciphertext,
    reader: jspb.BinaryReader,
  ): Ciphertext;
}

export namespace Ciphertext {
  export type AsObject = {
    a?: protos_generic_pb.TensorHolder.AsObject;
    b?: protos_generic_pb.TensorHolder.AsObject;
    ptShape?: protos_pt_shape_pb.PtShape.AsObject;
    state?: protos_global_params_and_state_pb.State.AsObject;
  };
}

export class RBGV_Ciphertext extends jspb.Message {
  hasCiphertext(): boolean;
  clearCiphertext(): void;
  getCiphertext(): Ciphertext | undefined;
  setCiphertext(value?: Ciphertext): void;

  getCorrectionFactor(): number;
  setCorrectionFactor(value: number): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): RBGV_Ciphertext.AsObject;
  static toObject(
    includeInstance: boolean,
    msg: RBGV_Ciphertext,
  ): RBGV_Ciphertext.AsObject;
  static extensions: { [key: number]: jspb.ExtensionFieldInfo<jspb.Message> };
  static extensionsBinary: {
    [key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>;
  };
  static serializeBinaryToWriter(
    message: RBGV_Ciphertext,
    writer: jspb.BinaryWriter,
  ): void;
  static deserializeBinary(bytes: Uint8Array): RBGV_Ciphertext;
  static deserializeBinaryFromReader(
    message: RBGV_Ciphertext,
    reader: jspb.BinaryReader,
  ): RBGV_Ciphertext;
}

export namespace RBGV_Ciphertext {
  export type AsObject = {
    ciphertext?: Ciphertext.AsObject;
    correctionFactor: number;
  };
}

export class CKKS_Ciphertext extends jspb.Message {
  hasCiphertext(): boolean;
  clearCiphertext(): void;
  getCiphertext(): Ciphertext | undefined;
  setCiphertext(value?: Ciphertext): void;

  clearPtScaleList(): void;
  getPtScaleList(): Array<number>;
  setPtScaleList(value: Array<number>): void;
  addPtScale(value: number, index?: number): number;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): CKKS_Ciphertext.AsObject;
  static toObject(
    includeInstance: boolean,
    msg: CKKS_Ciphertext,
  ): CKKS_Ciphertext.AsObject;
  static extensions: { [key: number]: jspb.ExtensionFieldInfo<jspb.Message> };
  static extensionsBinary: {
    [key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>;
  };
  static serializeBinaryToWriter(
    message: CKKS_Ciphertext,
    writer: jspb.BinaryWriter,
  ): void;
  static deserializeBinary(bytes: Uint8Array): CKKS_Ciphertext;
  static deserializeBinaryFromReader(
    message: CKKS_Ciphertext,
    reader: jspb.BinaryReader,
  ): CKKS_Ciphertext;
}

export namespace CKKS_Ciphertext {
  export type AsObject = {
    ciphertext?: Ciphertext.AsObject;
    ptScaleList: Array<number>;
  };
}
