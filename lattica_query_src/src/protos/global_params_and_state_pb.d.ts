// package: lattica_proto
// file: protos/global_params_and_state.proto

import * as jspb from "google-protobuf";
import * as protos_generic_pb from "../protos/generic_pb";
import * as protos_slot_permutations_pb from "../protos/slot_permutations_pb";
import * as protos_mod_params_and_state_pb from "../protos/mod_params_and_state_pb";
import * as protos_crt_params_and_state_pb from "../protos/crt_params_and_state_pb";

export class State extends jspb.Message {
  hasModState(): boolean;
  clearModState(): void;
  getModState(): protos_mod_params_and_state_pb.ModState | undefined;
  setModState(value?: protos_mod_params_and_state_pb.ModState): void;

  hasCrtState(): boolean;
  clearCrtState(): void;
  getCrtState(): protos_crt_params_and_state_pb.CrtState | undefined;
  setCrtState(value?: protos_crt_params_and_state_pb.CrtState): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): State.AsObject;
  static toObject(includeInstance: boolean, msg: State): State.AsObject;
  static extensions: { [key: number]: jspb.ExtensionFieldInfo<jspb.Message> };
  static extensionsBinary: {
    [key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>;
  };
  static serializeBinaryToWriter(
    message: State,
    writer: jspb.BinaryWriter,
  ): void;
  static deserializeBinary(bytes: Uint8Array): State;
  static deserializeBinaryFromReader(
    message: State,
    reader: jspb.BinaryReader,
  ): State;
}

export namespace State {
  export type AsObject = {
    modState?: protos_mod_params_and_state_pb.ModState.AsObject;
    crtState?: protos_crt_params_and_state_pb.CrtState.AsObject;
  };
}

export class GParams extends jspb.Message {
  getGExp(): number;
  setGExp(value: number): void;

  getGBaseBits(): number;
  setGBaseBits(value: number): void;

  hasGVec(): boolean;
  clearGVec(): void;
  getGVec(): protos_generic_pb.TensorHolder | undefined;
  setGVec(value?: protos_generic_pb.TensorHolder): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): GParams.AsObject;
  static toObject(includeInstance: boolean, msg: GParams): GParams.AsObject;
  static extensions: { [key: number]: jspb.ExtensionFieldInfo<jspb.Message> };
  static extensionsBinary: {
    [key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>;
  };
  static serializeBinaryToWriter(
    message: GParams,
    writer: jspb.BinaryWriter,
  ): void;
  static deserializeBinary(bytes: Uint8Array): GParams;
  static deserializeBinaryFromReader(
    message: GParams,
    reader: jspb.BinaryReader,
  ): GParams;
}

export namespace GParams {
  export type AsObject = {
    gExp: number;
    gBaseBits: number;
    gVec?: protos_generic_pb.TensorHolder.AsObject;
  };
}

export class Params extends jspb.Message {
  getN(): number;
  setN(value: number): void;

  getLogn(): number;
  setLogn(value: number): void;

  getErrStd(): number;
  setErrStd(value: number): void;

  getSkStd(): number;
  setSkStd(value: number): void;

  getPtScale(): number;
  setPtScale(value: number): void;

  hasGParams(): boolean;
  clearGParams(): void;
  getGParams(): GParams | undefined;
  setGParams(value?: GParams): void;

  hasPtGParams(): boolean;
  clearPtGParams(): void;
  getPtGParams(): GParams | undefined;
  setPtGParams(value?: GParams): void;

  hasCrtParams(): boolean;
  clearCrtParams(): void;
  getCrtParams(): protos_crt_params_and_state_pb.CrtParams | undefined;
  setCrtParams(value?: protos_crt_params_and_state_pb.CrtParams): void;

  hasPtPackParams(): boolean;
  clearPtPackParams(): void;
  getPtPackParams(): protos_crt_params_and_state_pb.CrtParams | undefined;
  setPtPackParams(value?: protos_crt_params_and_state_pb.CrtParams): void;

  hasPtModState(): boolean;
  clearPtModState(): void;
  getPtModState(): protos_mod_params_and_state_pb.ModState | undefined;
  setPtModState(value?: protos_mod_params_and_state_pb.ModState): void;

  hasPtPackState(): boolean;
  clearPtPackState(): void;
  getPtPackState(): protos_crt_params_and_state_pb.CrtState | undefined;
  setPtPackState(value?: protos_crt_params_and_state_pb.CrtState): void;

  hasNRange(): boolean;
  clearNRange(): void;
  getNRange(): protos_generic_pb.TensorHolder | undefined;
  setNRange(value?: protos_generic_pb.TensorHolder): void;

  hasPermsBaseCrt(): boolean;
  clearPermsBaseCrt(): void;
  getPermsBaseCrt(): protos_slot_permutations_pb.CrtPermutations | undefined;
  setPermsBaseCrt(value?: protos_slot_permutations_pb.CrtPermutations): void;

  hasXi(): boolean;
  clearXi(): void;
  getXi(): protos_generic_pb.Complex | undefined;
  setXi(value?: protos_generic_pb.Complex): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): Params.AsObject;
  static toObject(includeInstance: boolean, msg: Params): Params.AsObject;
  static extensions: { [key: number]: jspb.ExtensionFieldInfo<jspb.Message> };
  static extensionsBinary: {
    [key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>;
  };
  static serializeBinaryToWriter(
    message: Params,
    writer: jspb.BinaryWriter,
  ): void;
  static deserializeBinary(bytes: Uint8Array): Params;
  static deserializeBinaryFromReader(
    message: Params,
    reader: jspb.BinaryReader,
  ): Params;
}

export namespace Params {
  export type AsObject = {
    n: number;
    logn: number;
    errStd: number;
    skStd: number;
    ptScale: number;
    gParams?: GParams.AsObject;
    ptGParams?: GParams.AsObject;
    crtParams?: protos_crt_params_and_state_pb.CrtParams.AsObject;
    ptPackParams?: protos_crt_params_and_state_pb.CrtParams.AsObject;
    ptModState?: protos_mod_params_and_state_pb.ModState.AsObject;
    ptPackState?: protos_crt_params_and_state_pb.CrtState.AsObject;
    nRange?: protos_generic_pb.TensorHolder.AsObject;
    permsBaseCrt?: protos_slot_permutations_pb.CrtPermutations.AsObject;
    xi?: protos_generic_pb.Complex.AsObject;
  };
}
