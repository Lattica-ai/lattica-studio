// package: lattica_proto
// file: protos/generic.proto

import * as jspb from "google-protobuf";

export class TensorHolder extends jspb.Message {
  clearSizesList(): void;
  getSizesList(): Array<number>;
  setSizesList(value: Array<number>): void;
  addSizes(value: number, index?: number): number;

  clearStridesList(): void;
  getStridesList(): Array<number>;
  setStridesList(value: Array<number>): void;
  addStrides(value: number, index?: number): number;

  getData(): Uint8Array | string;
  getData_asU8(): Uint8Array;
  getData_asB64(): string;
  setData(value: Uint8Array | string): void;

  getDtype(): DataTypeMap[keyof DataTypeMap];
  setDtype(value: DataTypeMap[keyof DataTypeMap]): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): TensorHolder.AsObject;
  static toObject(
    includeInstance: boolean,
    msg: TensorHolder,
  ): TensorHolder.AsObject;
  static extensions: { [key: number]: jspb.ExtensionFieldInfo<jspb.Message> };
  static extensionsBinary: {
    [key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>;
  };
  static serializeBinaryToWriter(
    message: TensorHolder,
    writer: jspb.BinaryWriter,
  ): void;
  static deserializeBinary(bytes: Uint8Array): TensorHolder;
  static deserializeBinaryFromReader(
    message: TensorHolder,
    reader: jspb.BinaryReader,
  ): TensorHolder;
}

export namespace TensorHolder {
  export type AsObject = {
    sizesList: Array<number>;
    stridesList: Array<number>;
    data: Uint8Array | string;
    dtype: DataTypeMap[keyof DataTypeMap];
  };
}

export class BigInt extends jspb.Message {
  getData(): string;
  setData(value: string): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): BigInt.AsObject;
  static toObject(includeInstance: boolean, msg: BigInt): BigInt.AsObject;
  static extensions: { [key: number]: jspb.ExtensionFieldInfo<jspb.Message> };
  static extensionsBinary: {
    [key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>;
  };
  static serializeBinaryToWriter(
    message: BigInt,
    writer: jspb.BinaryWriter,
  ): void;
  static deserializeBinary(bytes: Uint8Array): BigInt;
  static deserializeBinaryFromReader(
    message: BigInt,
    reader: jspb.BinaryReader,
  ): BigInt;
}

export namespace BigInt {
  export type AsObject = {
    data: string;
  };
}

export class Complex extends jspb.Message {
  getReal(): number;
  setReal(value: number): void;

  getImag(): number;
  setImag(value: number): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): Complex.AsObject;
  static toObject(includeInstance: boolean, msg: Complex): Complex.AsObject;
  static extensions: { [key: number]: jspb.ExtensionFieldInfo<jspb.Message> };
  static extensionsBinary: {
    [key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>;
  };
  static serializeBinaryToWriter(
    message: Complex,
    writer: jspb.BinaryWriter,
  ): void;
  static deserializeBinary(bytes: Uint8Array): Complex;
  static deserializeBinaryFromReader(
    message: Complex,
    reader: jspb.BinaryReader,
  ): Complex;
}

export namespace Complex {
  export type AsObject = {
    real: number;
    imag: number;
  };
}

export interface DataTypeMap {
  FLOAT: 0;
  DOUBLE: 1;
  INT32: 2;
  INT64: 3;
  BOOL: 4;
  COMPLEX_FLOAT: 5;
  COMPLEX_DOUBLE: 6;
}

export const DataType: DataTypeMap;
