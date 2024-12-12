// package: lattica_proto
// file: protos/slot_permutations.proto

import * as jspb from "google-protobuf";
import * as protos_generic_pb from "../protos/generic_pb";

export class CrtPermutations extends jspb.Message {
  getHasNpermsDim(): boolean;
  setHasNpermsDim(value: boolean): void;

  hasNewIdxs(): boolean;
  clearNewIdxs(): void;
  getNewIdxs(): protos_generic_pb.TensorHolder | undefined;
  setNewIdxs(value?: protos_generic_pb.TensorHolder): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): CrtPermutations.AsObject;
  static toObject(
    includeInstance: boolean,
    msg: CrtPermutations,
  ): CrtPermutations.AsObject;
  static extensions: { [key: number]: jspb.ExtensionFieldInfo<jspb.Message> };
  static extensionsBinary: {
    [key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>;
  };
  static serializeBinaryToWriter(
    message: CrtPermutations,
    writer: jspb.BinaryWriter,
  ): void;
  static deserializeBinary(bytes: Uint8Array): CrtPermutations;
  static deserializeBinaryFromReader(
    message: CrtPermutations,
    reader: jspb.BinaryReader,
  ): CrtPermutations;
}

export namespace CrtPermutations {
  export type AsObject = {
    hasNpermsDim: boolean;
    newIdxs?: protos_generic_pb.TensorHolder.AsObject;
  };
}
