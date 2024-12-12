// package: lattica_proto
// file: protos/crt_params_and_state.proto

import * as jspb from "google-protobuf";
import * as protos_generic_pb from "../protos/generic_pb";
import * as protos_mod_params_and_state_pb from "../protos/mod_params_and_state_pb";

export class CrtState extends jspb.Message {
  hasModState(): boolean;
  clearModState(): void;
  getModState(): protos_mod_params_and_state_pb.ModState | undefined;
  setModState(value?: protos_mod_params_and_state_pb.ModState): void;

  hasActivePsiArr(): boolean;
  clearActivePsiArr(): void;
  getActivePsiArr(): protos_generic_pb.TensorHolder | undefined;
  setActivePsiArr(value?: protos_generic_pb.TensorHolder): void;

  hasActivePsiInvArr(): boolean;
  clearActivePsiInvArr(): void;
  getActivePsiInvArr(): protos_generic_pb.TensorHolder | undefined;
  setActivePsiInvArr(value?: protos_generic_pb.TensorHolder): void;

  hasActiveNInvList(): boolean;
  clearActiveNInvList(): void;
  getActiveNInvList(): protos_generic_pb.TensorHolder | undefined;
  setActiveNInvList(value?: protos_generic_pb.TensorHolder): void;

  hasActiveT(): boolean;
  clearActiveT(): void;
  getActiveT(): protos_generic_pb.TensorHolder | undefined;
  setActiveT(value?: protos_generic_pb.TensorHolder): void;

  hasActiveInvT(): boolean;
  clearActiveInvT(): void;
  getActiveInvT(): protos_generic_pb.TensorHolder | undefined;
  setActiveInvT(value?: protos_generic_pb.TensorHolder): void;

  hasActiveDftMatBase(): boolean;
  clearActiveDftMatBase(): void;
  getActiveDftMatBase(): protos_generic_pb.TensorHolder | undefined;
  setActiveDftMatBase(value?: protos_generic_pb.TensorHolder): void;

  hasActiveDftInvMatBase(): boolean;
  clearActiveDftInvMatBase(): void;
  getActiveDftInvMatBase(): protos_generic_pb.TensorHolder | undefined;
  setActiveDftInvMatBase(value?: protos_generic_pb.TensorHolder): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): CrtState.AsObject;
  static toObject(includeInstance: boolean, msg: CrtState): CrtState.AsObject;
  static extensions: { [key: number]: jspb.ExtensionFieldInfo<jspb.Message> };
  static extensionsBinary: {
    [key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>;
  };
  static serializeBinaryToWriter(
    message: CrtState,
    writer: jspb.BinaryWriter,
  ): void;
  static deserializeBinary(bytes: Uint8Array): CrtState;
  static deserializeBinaryFromReader(
    message: CrtState,
    reader: jspb.BinaryReader,
  ): CrtState;
}

export namespace CrtState {
  export type AsObject = {
    modState?: protos_mod_params_and_state_pb.ModState.AsObject;
    activePsiArr?: protos_generic_pb.TensorHolder.AsObject;
    activePsiInvArr?: protos_generic_pb.TensorHolder.AsObject;
    activeNInvList?: protos_generic_pb.TensorHolder.AsObject;
    activeT?: protos_generic_pb.TensorHolder.AsObject;
    activeInvT?: protos_generic_pb.TensorHolder.AsObject;
    activeDftMatBase?: protos_generic_pb.TensorHolder.AsObject;
    activeDftInvMatBase?: protos_generic_pb.TensorHolder.AsObject;
  };
}

export class CrtParams extends jspb.Message {
  getM(): number;
  setM(value: number): void;

  getN(): number;
  setN(value: number): void;

  getLogn(): number;
  setLogn(value: number): void;

  hasPerm(): boolean;
  clearPerm(): void;
  getPerm(): protos_generic_pb.TensorHolder | undefined;
  setPerm(value?: protos_generic_pb.TensorHolder): void;

  hasPermPairs(): boolean;
  clearPermPairs(): void;
  getPermPairs(): protos_generic_pb.TensorHolder | undefined;
  setPermPairs(value?: protos_generic_pb.TensorHolder): void;

  clearTwoHalfIndicesList(): void;
  getTwoHalfIndicesList(): Array<protos_generic_pb.TensorHolder>;
  setTwoHalfIndicesList(value: Array<protos_generic_pb.TensorHolder>): void;
  addTwoHalfIndices(
    value?: protos_generic_pb.TensorHolder,
    index?: number,
  ): protos_generic_pb.TensorHolder;

  clearAlternatingIndicesList(): void;
  getAlternatingIndicesList(): Array<protos_generic_pb.TensorHolder>;
  setAlternatingIndicesList(value: Array<protos_generic_pb.TensorHolder>): void;
  addAlternatingIndices(
    value?: protos_generic_pb.TensorHolder,
    index?: number,
  ): protos_generic_pb.TensorHolder;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): CrtParams.AsObject;
  static toObject(includeInstance: boolean, msg: CrtParams): CrtParams.AsObject;
  static extensions: { [key: number]: jspb.ExtensionFieldInfo<jspb.Message> };
  static extensionsBinary: {
    [key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>;
  };
  static serializeBinaryToWriter(
    message: CrtParams,
    writer: jspb.BinaryWriter,
  ): void;
  static deserializeBinary(bytes: Uint8Array): CrtParams;
  static deserializeBinaryFromReader(
    message: CrtParams,
    reader: jspb.BinaryReader,
  ): CrtParams;
}

export namespace CrtParams {
  export type AsObject = {
    m: number;
    n: number;
    logn: number;
    perm?: protos_generic_pb.TensorHolder.AsObject;
    permPairs?: protos_generic_pb.TensorHolder.AsObject;
    twoHalfIndicesList: Array<protos_generic_pb.TensorHolder.AsObject>;
    alternatingIndicesList: Array<protos_generic_pb.TensorHolder.AsObject>;
  };
}
