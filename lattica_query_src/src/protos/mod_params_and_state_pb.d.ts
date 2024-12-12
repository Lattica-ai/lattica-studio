// package: lattica_proto
// file: protos/mod_params_and_state.proto

import * as jspb from "google-protobuf";
import * as protos_generic_pb from "../protos/generic_pb";

export class ModState extends jspb.Message {
  hasActiveQ(): boolean;
  clearActiveQ(): void;
  getActiveQ(): protos_generic_pb.BigInt | undefined;
  setActiveQ(value?: protos_generic_pb.BigInt): void;

  getActiveLenQList(): number;
  setActiveLenQList(value: number): void;

  hasActiveQList(): boolean;
  clearActiveQList(): void;
  getActiveQList(): protos_generic_pb.TensorHolder | undefined;
  setActiveQList(value?: protos_generic_pb.TensorHolder): void;

  hasActiveRows(): boolean;
  clearActiveRows(): void;
  getActiveRows(): protos_generic_pb.TensorHolder | undefined;
  setActiveRows(value?: protos_generic_pb.TensorHolder): void;

  hasActiveCols(): boolean;
  clearActiveCols(): void;
  getActiveCols(): protos_generic_pb.TensorHolder | undefined;
  setActiveCols(value?: protos_generic_pb.TensorHolder): void;

  clearActiveReconstructionTermsList(): void;
  getActiveReconstructionTermsList(): Array<protos_generic_pb.BigInt>;
  setActiveReconstructionTermsList(
    value: Array<protos_generic_pb.BigInt>,
  ): void;
  addActiveReconstructionTerms(
    value?: protos_generic_pb.BigInt,
    index?: number,
  ): protos_generic_pb.BigInt;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): ModState.AsObject;
  static toObject(includeInstance: boolean, msg: ModState): ModState.AsObject;
  static extensions: { [key: number]: jspb.ExtensionFieldInfo<jspb.Message> };
  static extensionsBinary: {
    [key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>;
  };
  static serializeBinaryToWriter(
    message: ModState,
    writer: jspb.BinaryWriter,
  ): void;
  static deserializeBinary(bytes: Uint8Array): ModState;
  static deserializeBinaryFromReader(
    message: ModState,
    reader: jspb.BinaryReader,
  ): ModState;
}

export namespace ModState {
  export type AsObject = {
    activeQ?: protos_generic_pb.BigInt.AsObject;
    activeLenQList: number;
    activeQList?: protos_generic_pb.TensorHolder.AsObject;
    activeRows?: protos_generic_pb.TensorHolder.AsObject;
    activeCols?: protos_generic_pb.TensorHolder.AsObject;
    activeReconstructionTermsList: Array<protos_generic_pb.BigInt.AsObject>;
  };
}
