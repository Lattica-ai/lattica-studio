import * as serverApi from "./serverApi";
import * as workerApi from "./workerApi";
import {
  Tensor,
  getTensorShape,
  imageToTensor,
  tensorToImage,
} from "./tensorUtils";
import { setLogCallback } from "./wasm/wasm_worker_manager";
export { imageToTensor, tensorToImage, setLogCallback }; // Re-export for frontend
import * as toolkit from "./toolkitWrapper";

import { runInferenceDemo } from "./inference";
import {
  QueryClientSequentialHomOp,
  QueryClientBlock,
} from "./serialization/hom_op";

import { triggerEvkCopy } from "./dev_utils";
import { initWasm } from "./wasm/wasm_worker_manager";
export { initWasm };

export class LatticaQueryClient {
  public readonly jwt: string;
  public readonly beUrl: string;
  protected initialized: boolean = false;
  private serializedContext: Uint8Array | null = null;
  private clientBlocks: QueryClientBlock[] = [];
  private secretKey: [Uint8Array, Uint8Array] | null = null;
  private applyBlockBeforeEncryption: boolean = false;
  private firstClientBlock: Uint8Array | undefined = undefined;
  private applyBlockAfterDecryption: boolean = false;
  private lastClientBlock: Uint8Array | undefined = undefined;

  constructor(jwt: string, beUrl?: string) {
    if (!jwt) throw new Error("JWT token is required");
    this.jwt = jwt;
    if (!beUrl) throw new Error("BE URL is required");
    this.beUrl = beUrl;
  }

  protected async getClientData(): Promise<Uint8Array> {
    const { serializedHomseq, serializedContext } =
      await workerApi.get_user_init_data(this.jwt, this.beUrl);
    this.serializedContext = serializedContext;

    this.clientBlocks =
      QueryClientSequentialHomOp.decode(serializedHomseq).clientBlocks;
    console.log("number of client blocks:", this.clientBlocks.length);

    // check if need to apply client block before encryption
    this.applyBlockBeforeEncryption =
      this.clientBlocks.length > 0 && this.clientBlocks[0].blockIndex === 0;
    if (this.applyBlockBeforeEncryption) {
      this.firstClientBlock = QueryClientBlock.encode(
        this.clientBlocks.shift()!,
      ).finish(); // remove first from list
    }
    // check if need to apply client block after decryption
    this.applyBlockAfterDecryption =
      this.clientBlocks.length > 0 &&
      this.clientBlocks[this.clientBlocks.length - 1].isLast;
    if (this.applyBlockAfterDecryption) {
      this.lastClientBlock = QueryClientBlock.encode(
        this.clientBlocks.pop()!,
      ).finish(); // remove last from list
    }

    // TODO: we can apply serializeBinary() to all remaining blocks in clientBlocks list, to save time runInference

    return serializedHomseq;
  }

  protected async generateKey(serializedHomseq: Uint8Array) {
    const [sk, skCoefs, evaluationKey] = await toolkit.generateKey(
      serializedHomseq,
      this.serializedContext!,
    );

    this.secretKey = [sk, skCoefs];

    console.log("PK size:", evaluationKey.length);

    return evaluationKey;
  }

  async init(): Promise<void> {
    await initWasm();

    const serializedHomseq = await this.getClientData();

    const evaluationKey = await this.generateKey(serializedHomseq);

    await serverApi.uploadEk(this.jwt, this.beUrl, evaluationKey);

    await workerApi.preprocessEk(this.jwt, this.beUrl);

    this.initialized = true;
  }

  async encrypt(pt: Tensor): Promise<Uint8Array> {
    if (!this.initialized) throw new Error("Client must be initialized first");

    console.log("pt shape:", getTensorShape(pt));

    let serialized_pt = toolkit.dumps_proto_tensor(pt);
    // check if need to apply first client block
    if (this.applyBlockBeforeEncryption) {
      serialized_pt = await toolkit.applyClientBlock(
        this.firstClientBlock!,
        this.serializedContext!,
        serialized_pt,
      );
    }

    return await toolkit.encrypt(
      this.serializedContext!,
      this.secretKey!,
      serialized_pt,
      true,
    );
  }

  async apply_query(encryptedInput: Uint8Array): Promise<Uint8Array> {
    if (!this.initialized) throw new Error("Client must be initialized first");

    return await runInferenceDemo(
      this.jwt,
      this.beUrl,
      this.serializedContext!,
      this.secretKey!,
      encryptedInput,
      this.clientBlocks,
      this.applyBlockBeforeEncryption ? 1 : 0,
    );
  }

  async decrypt(encryptedResult: Uint8Array): Promise<Tensor> {
    if (!this.initialized) throw new Error("Client must be initialized first");
    let decryptedOutput = await toolkit.decrypt(
      this.serializedContext!,
      this.secretKey!,
      encryptedResult,
    );

    // check if last block is on client
    if (this.applyBlockAfterDecryption) {
      decryptedOutput = await toolkit.applyClientBlock(
        this.lastClientBlock!,
        this.serializedContext!,
        decryptedOutput,
      );
    }

    return toolkit.load_proto_tensor(decryptedOutput);
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
    console.log(s3Key);
    await serverApi.alertUploadComplete(this.jwt, this.beUrl, s3Key); // fake alert to change status of EVK

    // This is the only difference from the parent class
    await triggerEvkCopy(this.jwt, this.beUrl, evaluationKey, s3Key);

    await workerApi.preprocessEk(this.jwt, this.beUrl);

    this.initialized = true;
  }
}
