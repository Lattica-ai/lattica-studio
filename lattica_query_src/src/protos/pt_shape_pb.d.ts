// package: lattica_proto
// file: protos/pt_shape.proto

import * as jspb from "google-protobuf";

export class PtShape extends jspb.Message {
  getInternalN(): number;
  setInternalN(value: number): void;

  clearExternalNsList(): void;
  getExternalNsList(): Array<number>;
  setExternalNsList(value: Array<number>): void;
  addExternalNs(value: number, index?: number): number;

  clearInternalShapeList(): void;
  getInternalShapeList(): Array<number>;
  setInternalShapeList(value: Array<number>): void;
  addInternalShape(value: number, index?: number): number;

  getNumNs(): number;
  setNumNs(value: number): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): PtShape.AsObject;
  static toObject(includeInstance: boolean, msg: PtShape): PtShape.AsObject;
  static extensions: { [key: number]: jspb.ExtensionFieldInfo<jspb.Message> };
  static extensionsBinary: {
    [key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>;
  };
  static serializeBinaryToWriter(
    message: PtShape,
    writer: jspb.BinaryWriter,
  ): void;
  static deserializeBinary(bytes: Uint8Array): PtShape;
  static deserializeBinaryFromReader(
    message: PtShape,
    reader: jspb.BinaryReader,
  ): PtShape;
}

export namespace PtShape {
  export type AsObject = {
    internalN: number;
    externalNsList: Array<number>;
    internalShapeList: Array<number>;
    numNs: number;
  };
}
