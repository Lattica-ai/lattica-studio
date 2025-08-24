import * as serverApi from "./serverApi";
import * as workerApi from "./workerApi";
import {
  Tensor,
  vecToTensor,
  tensor1DToVec,
  imageToTensor,
  tensorToImage,
  dumps_proto_tensor,
  loads_proto_tensor,
} from "./tensorUtils";

import * as toolkit from "./toolkitWrapper";

import { runInferenceDemo, runInference } from "./inference";
import {
  QueryClientSequentialHomOp,
  QueryClientBlock,
  ClientData,
} from "./serialization/hom_op";

import { triggerEvkCopy } from "./dev_utils";
import {
  initWasm,
  workerManagerSingleton,
  setLogCallback,
} from "./wasm/wasm_worker_manager";
import { getDemoToken } from "./serverApi";
export {
  initWasm,
  vecToTensor,
  tensor1DToVec,
  imageToTensor,
  tensorToImage,
  setLogCallback,
  getDemoToken,
}; // Re-export for frontend

export class LatticaQueryClient {
  public readonly jwt: string;
  public readonly beUrl: string;
  protected initialized: boolean = false;
  private serializedContext!: Uint8Array;
  private clientBlocks!: QueryClientBlock[];
  private allClientBlocks!: QueryClientBlock[];
  private secretKey!: [Uint8Array, Uint8Array];
  private firstClientBlock!: Uint8Array;
  private lastClientBlock!: Uint8Array;
  private modelMetadata: Uint8Array | undefined = undefined;
  private asComplex: boolean = false;

  constructor(jwt: string, beUrl?: string) {
    if (!jwt) throw new Error("JWT token is required");
    this.jwt = jwt;
    this.beUrl = beUrl ?? serverApi.BE_URL;
  }

  protected async getClientData(): Promise<Uint8Array> {
    const serializedClientData = await workerApi.get_user_init_data(
      this.jwt,
      this.beUrl,
    );

    const clientData = ClientData.decode(serializedClientData);

    this.serializedContext = clientData.serializedContext;

    const homseq = QueryClientSequentialHomOp.decode(
      clientData.serializedClientSequentialHomOp,
    );
    this.asComplex = homseq.asComplex;
    this.modelMetadata = homseq.preprocessingData;
    this.clientBlocks = homseq.clientBlocks;
    this.allClientBlocks = homseq.clientBlocks.slice();
    workerManagerSingleton.logConsole(
      `Number of client blocks: ${this.clientBlocks.length}`,
    );

    this.firstClientBlock = QueryClientBlock.encode(
      this.clientBlocks.shift()!,
    ).finish(); // remove first from list

    this.lastClientBlock = QueryClientBlock.encode(
      this.clientBlocks.pop()!,
    ).finish(); // remove last from list

    // TODO: we can apply serializeBinary() to all remaining blocks in clientBlocks list, to save time runInference

    return clientData.serializedClientSequentialHomOp;
  }

  protected async generateKey(serializedHomseq: Uint8Array) {
    const [sk, skCoefs, evaluationKey] = await toolkit.generateKey(
      serializedHomseq,
      this.serializedContext,
    );

    this.secretKey = [sk, skCoefs];

    console.log("PK size:", evaluationKey.length);

    return evaluationKey;
  }

  async init(): Promise<void> {
    await initWasm();

    workerManagerSingleton.logConsole("Getting context from server...");
    const serializedHomseq = await this.getClientData();
    workerManagerSingleton.logConsole("Context received.");

    const evaluationKey = await this.generateKey(serializedHomseq);

    workerManagerSingleton.logConsole("Uploading evaluation key...");
    await serverApi.uploadEk(this.jwt, this.beUrl, evaluationKey);
    workerManagerSingleton.logConsole("Evaluation key uploaded.");

    await workerApi.preprocessEk(this.jwt, this.beUrl);

    this.initialized = true;
  }

  getModelMetadata(): Uint8Array | undefined {
    if (!this.initialized) throw new Error("Client must be initialized first");
    return this.modelMetadata;
  }

  async encrypt(pt: Tensor): Promise<Uint8Array> {
    if (!this.initialized) throw new Error("Client must be initialized first");

    workerManagerSingleton.logConsole(`Plain text shape: (${pt.shape})`);

    let serialized_pt = dumps_proto_tensor(pt);
    // apply first client block
    serialized_pt = await toolkit.applyClientBlock(
      this.firstClientBlock,
      this.serializedContext,
      serialized_pt,
    );

    const pt_axis_external = QueryClientBlock.decode(
      this.firstClientBlock,
    ).ptAxisExternal;

    return await toolkit.encrypt(
      this.serializedContext,
      this.secretKey,
      serialized_pt,
      true,
      pt_axis_external,
    );
  }

  async apply_query(encryptedInput: Uint8Array): Promise<Uint8Array> {
    if (!this.initialized) throw new Error("Client must be initialized first");

    return await runInferenceDemo(
      this.jwt,
      this.beUrl,
      this.serializedContext,
      this.secretKey,
      encryptedInput,
      this.clientBlocks,
      this.asComplex,
    );
  }

  async decrypt(encryptedResult: Uint8Array): Promise<Tensor> {
    if (!this.initialized) throw new Error("Client must be initialized first");
    let decryptedOutput = await toolkit.decrypt(
      this.serializedContext,
      this.secretKey,
      encryptedResult,
      this.asComplex,
    );

    // apply last client block
    decryptedOutput = await toolkit.applyClientBlock(
      this.lastClientBlock,
      this.serializedContext,
      decryptedOutput,
    );

    return loads_proto_tensor(decryptedOutput);
  }

  async runQuery(pt: Tensor): Promise<Tensor> {
    if (!this.initialized) throw new Error("Client must be initialized first");
    return await runInference(
      this.jwt,
      this.beUrl,
      this.serializedContext,
      this.secretKey,
      pt,
      this.allClientBlocks,
      this.asComplex,
    );
  }
}

export class LatticaQueryClientLocalDev extends LatticaQueryClient {
  constructor(jwt: string) {
    super(jwt, "http://localhost:3050");
  }

  async init(): Promise<void> {
    await initWasm();

    const serializedHomseq = await this.getClientData();

    const evaluationKey = await this.generateKey(serializedHomseq);

    const { s3Key } = await serverApi.sendHttpRequest(
      this.jwt,
      this.beUrl,
      "/api/token/get_pk_upload_url",
    );
    await serverApi.alertUploadComplete(this.jwt, this.beUrl, s3Key); // fake alert to change status of EVK

    // This is the only difference from the parent class
    await triggerEvkCopy(this.jwt, this.beUrl, evaluationKey, s3Key);

    await workerApi.preprocessEk(this.jwt, this.beUrl);

    this.initialized = true;
  }
}
