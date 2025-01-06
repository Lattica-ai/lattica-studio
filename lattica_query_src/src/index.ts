import * as serverApi from "./serverApi";
import * as workerApi from "./workerApi";
import { Tensor, getTensorShape, imageToTensor } from "./tensorUtils";
export { imageToTensor }; // Re-export for frontend
import * as toolkit from "./toolkitWrapper";

import { runInferenceDemo } from "./inference";
import {
  QueryClientSequentialHomOp,
  QueryClientBlock,
} from "./serialization/hom_op";

import { triggerEvkCopy } from "./dev_utils";

declare global {
  interface Window {
    Module: any;
  }
  var Module: any;
}

let initPromise: Promise<void> | null = null;

async function initWasm() {
  if (initPromise) {
    return initPromise;
  }

  console.log("Starting WASM initialization...");

  initPromise = (async () => {
    try {
      // Import the Module factory function
      // @ts-ignore
      const moduleFactory = await import("../dist/wasm/cpp_sdk.js");

      const wasmUrl = import.meta.url
        ? new URL("../dist/wasm/cpp_sdk.wasm", import.meta.url).href
        : "../dist/wasm/cpp_sdk.wasm";

      // Create and initialize the module
      const module = await moduleFactory.default({
        locateFile: () => wasmUrl,
        onRuntimeInitialized: () => {
          console.log("WASM runtime initialized");
        },
      });

      // Store it globally if needed
      window.Module = module;

      return module;
    } catch (error) {
      console.error("Failed to initialize WASM module:", error);
      initPromise = null;
      throw error;
    }
  })();

  return initPromise;
}

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

  protected generateKey(serializedHomseq: Uint8Array): Uint8Array {
    const [sk, skCoefs, evaluationKey] = toolkit.generateKey(
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

    const evaluationKey = this.generateKey(serializedHomseq);

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
      serialized_pt = toolkit.applyClientBlock(
        this.firstClientBlock!,
        this.serializedContext!,
        serialized_pt,
      );
    }

    return toolkit.encrypt(
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
    let decryptedOutput = toolkit.decrypt(
      this.serializedContext!,
      this.secretKey!,
      encryptedResult,
    );

    // check if last block is on client
    if (this.applyBlockAfterDecryption) {
      decryptedOutput = toolkit.applyClientBlock(
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
    super(jwt, "http://localhost:8080");
  }

  async init(): Promise<void> {
    await initWasm();

    const serializedHomseq = await this.getClientData();

    const evaluationKey = this.generateKey(serializedHomseq);

    const s3Key = await serverApi.uploadEk(this.jwt, this.beUrl, evaluationKey);

    // This is the only difference from the parent class
    await triggerEvkCopy(this.jwt, this.beUrl, evaluationKey, s3Key);

    await workerApi.preprocessEk(this.jwt, this.beUrl);

    this.initialized = true;
  }
}
