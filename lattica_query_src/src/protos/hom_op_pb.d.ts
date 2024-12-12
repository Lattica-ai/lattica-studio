// package: lattica_proto
// file: protos/hom_op.proto

import * as jspb from "google-protobuf";
import * as protos_key_aux_pb from "../protos/key_aux_pb";
import * as protos_generic_pb from "../protos/generic_pb";
import * as protos_pt_shape_pb from "../protos/pt_shape_pb";

export class UnfoldApplyClearParams extends jspb.Message {
  getKernelSize(): number;
  setKernelSize(value: number): void;

  getPadding(): number;
  setPadding(value: number): void;

  getStride(): number;
  setStride(value: number): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): UnfoldApplyClearParams.AsObject;
  static toObject(
    includeInstance: boolean,
    msg: UnfoldApplyClearParams,
  ): UnfoldApplyClearParams.AsObject;
  static extensions: { [key: number]: jspb.ExtensionFieldInfo<jspb.Message> };
  static extensionsBinary: {
    [key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>;
  };
  static serializeBinaryToWriter(
    message: UnfoldApplyClearParams,
    writer: jspb.BinaryWriter,
  ): void;
  static deserializeBinary(bytes: Uint8Array): UnfoldApplyClearParams;
  static deserializeBinaryFromReader(
    message: UnfoldApplyClearParams,
    reader: jspb.BinaryReader,
  ): UnfoldApplyClearParams;
}

export namespace UnfoldApplyClearParams {
  export type AsObject = {
    kernelSize: number;
    padding: number;
    stride: number;
  };
}

export class ReshapeApplyClearParams extends jspb.Message {
  clearDimsList(): void;
  getDimsList(): Array<number>;
  setDimsList(value: Array<number>): void;
  addDims(value: number, index?: number): number;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): ReshapeApplyClearParams.AsObject;
  static toObject(
    includeInstance: boolean,
    msg: ReshapeApplyClearParams,
  ): ReshapeApplyClearParams.AsObject;
  static extensions: { [key: number]: jspb.ExtensionFieldInfo<jspb.Message> };
  static extensionsBinary: {
    [key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>;
  };
  static serializeBinaryToWriter(
    message: ReshapeApplyClearParams,
    writer: jspb.BinaryWriter,
  ): void;
  static deserializeBinary(bytes: Uint8Array): ReshapeApplyClearParams;
  static deserializeBinaryFromReader(
    message: ReshapeApplyClearParams,
    reader: jspb.BinaryReader,
  ): ReshapeApplyClearParams;
}

export namespace ReshapeApplyClearParams {
  export type AsObject = {
    dimsList: Array<number>;
  };
}

export class ConstAddApplyClearParams extends jspb.Message {
  hasData(): boolean;
  clearData(): void;
  getData(): protos_generic_pb.TensorHolder | undefined;
  setData(value?: protos_generic_pb.TensorHolder): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): ConstAddApplyClearParams.AsObject;
  static toObject(
    includeInstance: boolean,
    msg: ConstAddApplyClearParams,
  ): ConstAddApplyClearParams.AsObject;
  static extensions: { [key: number]: jspb.ExtensionFieldInfo<jspb.Message> };
  static extensionsBinary: {
    [key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>;
  };
  static serializeBinaryToWriter(
    message: ConstAddApplyClearParams,
    writer: jspb.BinaryWriter,
  ): void;
  static deserializeBinary(bytes: Uint8Array): ConstAddApplyClearParams;
  static deserializeBinaryFromReader(
    message: ConstAddApplyClearParams,
    reader: jspb.BinaryReader,
  ): ConstAddApplyClearParams;
}

export namespace ConstAddApplyClearParams {
  export type AsObject = {
    data?: protos_generic_pb.TensorHolder.AsObject;
  };
}

export class ConstMulApplyClearParams extends jspb.Message {
  hasData(): boolean;
  clearData(): void;
  getData(): protos_generic_pb.TensorHolder | undefined;
  setData(value?: protos_generic_pb.TensorHolder): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): ConstMulApplyClearParams.AsObject;
  static toObject(
    includeInstance: boolean,
    msg: ConstMulApplyClearParams,
  ): ConstMulApplyClearParams.AsObject;
  static extensions: { [key: number]: jspb.ExtensionFieldInfo<jspb.Message> };
  static extensionsBinary: {
    [key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>;
  };
  static serializeBinaryToWriter(
    message: ConstMulApplyClearParams,
    writer: jspb.BinaryWriter,
  ): void;
  static deserializeBinary(bytes: Uint8Array): ConstMulApplyClearParams;
  static deserializeBinaryFromReader(
    message: ConstMulApplyClearParams,
    reader: jspb.BinaryReader,
  ): ConstMulApplyClearParams;
}

export namespace ConstMulApplyClearParams {
  export type AsObject = {
    data?: protos_generic_pb.TensorHolder.AsObject;
  };
}

export class ConstMatMulApplyClearParams extends jspb.Message {
  hasData(): boolean;
  clearData(): void;
  getData(): protos_generic_pb.TensorHolder | undefined;
  setData(value?: protos_generic_pb.TensorHolder): void;

  hasDataPtShape(): boolean;
  clearDataPtShape(): void;
  getDataPtShape(): protos_pt_shape_pb.PtShape | undefined;
  setDataPtShape(value?: protos_pt_shape_pb.PtShape): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): ConstMatMulApplyClearParams.AsObject;
  static toObject(
    includeInstance: boolean,
    msg: ConstMatMulApplyClearParams,
  ): ConstMatMulApplyClearParams.AsObject;
  static extensions: { [key: number]: jspb.ExtensionFieldInfo<jspb.Message> };
  static extensionsBinary: {
    [key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>;
  };
  static serializeBinaryToWriter(
    message: ConstMatMulApplyClearParams,
    writer: jspb.BinaryWriter,
  ): void;
  static deserializeBinary(bytes: Uint8Array): ConstMatMulApplyClearParams;
  static deserializeBinaryFromReader(
    message: ConstMatMulApplyClearParams,
    reader: jspb.BinaryReader,
  ): ConstMatMulApplyClearParams;
}

export namespace ConstMatMulApplyClearParams {
  export type AsObject = {
    data?: protos_generic_pb.TensorHolder.AsObject;
    dataPtShape?: protos_pt_shape_pb.PtShape.AsObject;
  };
}

export class ApplyClearParams extends jspb.Message {
  hasUnfold(): boolean;
  clearUnfold(): void;
  getUnfold(): UnfoldApplyClearParams | undefined;
  setUnfold(value?: UnfoldApplyClearParams): void;

  hasReshape(): boolean;
  clearReshape(): void;
  getReshape(): ReshapeApplyClearParams | undefined;
  setReshape(value?: ReshapeApplyClearParams): void;

  hasConstAdd(): boolean;
  clearConstAdd(): void;
  getConstAdd(): ConstAddApplyClearParams | undefined;
  setConstAdd(value?: ConstAddApplyClearParams): void;

  hasConstMul(): boolean;
  clearConstMul(): void;
  getConstMul(): ConstMulApplyClearParams | undefined;
  setConstMul(value?: ConstMulApplyClearParams): void;

  hasConstMatMul(): boolean;
  clearConstMatMul(): void;
  getConstMatMul(): ConstMatMulApplyClearParams | undefined;
  setConstMatMul(value?: ConstMatMulApplyClearParams): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): ApplyClearParams.AsObject;
  static toObject(
    includeInstance: boolean,
    msg: ApplyClearParams,
  ): ApplyClearParams.AsObject;
  static extensions: { [key: number]: jspb.ExtensionFieldInfo<jspb.Message> };
  static extensionsBinary: {
    [key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>;
  };
  static serializeBinaryToWriter(
    message: ApplyClearParams,
    writer: jspb.BinaryWriter,
  ): void;
  static deserializeBinary(bytes: Uint8Array): ApplyClearParams;
  static deserializeBinaryFromReader(
    message: ApplyClearParams,
    reader: jspb.BinaryReader,
  ): ApplyClearParams;
}

export namespace ApplyClearParams {
  export type AsObject = {
    unfold?: UnfoldApplyClearParams.AsObject;
    reshape?: ReshapeApplyClearParams.AsObject;
    constAdd?: ConstAddApplyClearParams.AsObject;
    constMul?: ConstMulApplyClearParams.AsObject;
    constMatMul?: ConstMatMulApplyClearParams.AsObject;
  };
}

export class QueryClientBlock extends jspb.Message {
  getBlockIndex(): number;
  setBlockIndex(value: number): void;

  getIsLast(): boolean;
  setIsLast(value: boolean): void;

  clearOpTypesList(): void;
  getOpTypesList(): Array<HomOpTypeMap[keyof HomOpTypeMap]>;
  setOpTypesList(value: Array<HomOpTypeMap[keyof HomOpTypeMap]>): void;
  addOpTypes(
    value: HomOpTypeMap[keyof HomOpTypeMap],
    index?: number,
  ): HomOpTypeMap[keyof HomOpTypeMap];

  clearApplyClearParamsListList(): void;
  getApplyClearParamsListList(): Array<ApplyClearParams>;
  setApplyClearParamsListList(value: Array<ApplyClearParams>): void;
  addApplyClearParamsList(
    value?: ApplyClearParams,
    index?: number,
  ): ApplyClearParams;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): QueryClientBlock.AsObject;
  static toObject(
    includeInstance: boolean,
    msg: QueryClientBlock,
  ): QueryClientBlock.AsObject;
  static extensions: { [key: number]: jspb.ExtensionFieldInfo<jspb.Message> };
  static extensionsBinary: {
    [key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>;
  };
  static serializeBinaryToWriter(
    message: QueryClientBlock,
    writer: jspb.BinaryWriter,
  ): void;
  static deserializeBinary(bytes: Uint8Array): QueryClientBlock;
  static deserializeBinaryFromReader(
    message: QueryClientBlock,
    reader: jspb.BinaryReader,
  ): QueryClientBlock;
}

export namespace QueryClientBlock {
  export type AsObject = {
    blockIndex: number;
    isLast: boolean;
    opTypesList: Array<HomOpTypeMap[keyof HomOpTypeMap]>;
    applyClearParamsListList: Array<ApplyClearParams.AsObject>;
  };
}

export class QueryClientSequentialHomOp extends jspb.Message {
  clearKeyTypesList(): void;
  getKeyTypesList(): Array<
    protos_key_aux_pb.AuxKeyTypeMap[keyof protos_key_aux_pb.AuxKeyTypeMap]
  >;
  setKeyTypesList(
    value: Array<
      protos_key_aux_pb.AuxKeyTypeMap[keyof protos_key_aux_pb.AuxKeyTypeMap]
    >,
  ): void;
  addKeyTypes(
    value: protos_key_aux_pb.AuxKeyTypeMap[keyof protos_key_aux_pb.AuxKeyTypeMap],
    index?: number,
  ): protos_key_aux_pb.AuxKeyTypeMap[keyof protos_key_aux_pb.AuxKeyTypeMap];

  clearClientBlocksList(): void;
  getClientBlocksList(): Array<QueryClientBlock>;
  setClientBlocksList(value: Array<QueryClientBlock>): void;
  addClientBlocks(value?: QueryClientBlock, index?: number): QueryClientBlock;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): QueryClientSequentialHomOp.AsObject;
  static toObject(
    includeInstance: boolean,
    msg: QueryClientSequentialHomOp,
  ): QueryClientSequentialHomOp.AsObject;
  static extensions: { [key: number]: jspb.ExtensionFieldInfo<jspb.Message> };
  static extensionsBinary: {
    [key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>;
  };
  static serializeBinaryToWriter(
    message: QueryClientSequentialHomOp,
    writer: jspb.BinaryWriter,
  ): void;
  static deserializeBinary(bytes: Uint8Array): QueryClientSequentialHomOp;
  static deserializeBinaryFromReader(
    message: QueryClientSequentialHomOp,
    reader: jspb.BinaryReader,
  ): QueryClientSequentialHomOp;
}

export namespace QueryClientSequentialHomOp {
  export type AsObject = {
    keyTypesList: Array<
      protos_key_aux_pb.AuxKeyTypeMap[keyof protos_key_aux_pb.AuxKeyTypeMap]
    >;
    clientBlocksList: Array<QueryClientBlock.AsObject>;
  };
}

export interface HomOpTypeMap {
  AUXPROPAGATEKEY: 0;
  CONSTADD: 1;
  CONSTMUL: 2;
  MUL: 3;
  SQUARE: 4;
  MODSWITCH: 5;
  SUMSLOTSBASEKEY: 6;
  SUMSLOTSFULLKEY: 7;
  MATMULBASEKEY: 8;
  MATMULFULLKEY: 9;
  UNFOLD: 10;
  RESHAPE: 11;
}

export const HomOpType: HomOpTypeMap;
