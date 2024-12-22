import * as server from "./serverApi";
import * as worker_api from "./workerApi";
import { Tensor, getTensorShape } from "./tensorUtils";
import * as toolkit from "./toolkitWrapper";
import "./wasm/cpp_sdk.js";

import { runInferenceDemo } from "./inference";
import {
  QueryClientSequentialHomOp,
  QueryClientBlock,
} from "./protos/hom_op_pb";

import { triggerEvkCopy } from "./dev_utils";

export {};
declare global {
  var Module: any;
}

async function initWasm() {
  try {
    // Wait for the module to be fully initialized
    await new Promise<void>((resolve) => {
      // If module is already initialized
      if (Module && Module.ready) {
        resolve();
        return;
      }

      // If not, wait for initialization
      if (!Module) {
        window.Module = {};
      }

      Module.onRuntimeInitialized = () => {
        resolve();
      };
    });

    // Now you can use the module
    console.log("WASM Module is ready!");
  } catch (error) {
    console.error("Failed to load WASM module:", error);
  }
}

class LatticaQueryClient {
  protected readonly jwt: string;
  public readonly beUrl: string = "https://gini.api.b543.lattica.dev";
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
    if (beUrl) this.beUrl = beUrl;
  }

  protected async getClientData(): Promise<Uint8Array> {
    // const initData = await worker_api.get_user_init_data(this.jwt, this.beUrl);
    const { serializedHomseq, serializedContext } = await worker_api.get_user_init_data(this.jwt, this.beUrl);
    this.serializedContext = serializedContext;

    this.clientBlocks = QueryClientSequentialHomOp.deserializeBinary(
        serializedHomseq,
    ).getClientBlocksList();
    console.log("number of client blocks:", this.clientBlocks.length);

    // check if need to apply client block before encryption
    this.applyBlockBeforeEncryption =
        this.clientBlocks.length > 0 &&
        this.clientBlocks[0].getBlockIndex() === 0;
    if (this.applyBlockBeforeEncryption) {
      this.firstClientBlock = this.clientBlocks.shift()?.serializeBinary(); // remove first from list
    }
    // check if need to apply client block after decryption
    this.applyBlockAfterDecryption =
        this.clientBlocks.length > 0 &&
        this.clientBlocks[this.clientBlocks.length - 1].getIsLast();
    if (this.applyBlockAfterDecryption) {
      this.lastClientBlock = this.clientBlocks.pop()?.serializeBinary(); // remove last from list
    }

    // TODO: we can apply serializeBinary() to all remaining blocks in clientBlocks list, to save time runInference

    return serializedHomseq;
  }

  protected generateKey(serializedHomseq: Uint8Array): Uint8Array {
    const [sk, skCoefs, evaluationKey] = toolkit.generateKey(
      serializedHomseq,
      this.serializedContext,
    );
    // let xxx = await fetch("mock_data/my_pk");
    // const evaluationKey = new Uint8Array(await xxx.arrayBuffer());
    // xxx = await fetch("mock_data/my_sk1");
    // const sk = new Uint8Array(await xxx.arrayBuffer());
    // xxx = await fetch("mock_data/my_sk2");
    // const skCoefs = new Uint8Array(await xxx.arrayBuffer());

    this.secretKey = [sk, skCoefs];

    console.log("PK size:", evaluationKey.length);

    return evaluationKey;
  }

  async init(): Promise<void> {
    await initWasm();

    const serializedHomseq = await this.getClientData();

    const evaluationKey = this.generateKey(serializedHomseq);

    await server.uploadEk(this.jwt, this.beUrl, evaluationKey);

    await worker_api.preprocessEk(this.jwt, this.beUrl);

    this.initialized = true;
  }

  async encrypt(pt: Tensor): Promise<Uint8Array> {
    if (!this.initialized) throw new Error("Client must be initialized first");

    let serialized_pt = toolkit.dumps_proto_tensor(pt);
    // check if need to apply first client block
    if (this.applyBlockBeforeEncryption) {
      serialized_pt = toolkit.applyClientBlock(
        this.firstClientBlock,
        this.serializedContext,
        serialized_pt,
      );
    }

    return toolkit.encrypt(
      this.serializedContext,
      this.secretKey,
      serialized_pt,
      true,
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
      this.applyBlockBeforeEncryption ? 1 : 0,
    );
  }

  async decrypt(encryptedResult: Uint8Array): Promise<Tensor> {
    if (!this.initialized) throw new Error("Client must be initialized first");
    let decryptedOutput = toolkit.decrypt(
      this.serializedContext,
      this.secretKey,
      encryptedResult,
    );

    // check if last block is on client
    if (this.applyBlockAfterDecryption) {
      decryptedOutput = toolkit.applyClientBlock(
        this.lastClientBlock,
        this.serializedContext,
        decryptedOutput,
      );
    }

    return toolkit.load_proto_tensor(decryptedOutput);
  }
}

class LatticaQueryClientLocalDev extends LatticaQueryClient {
  constructor(jwt: string) {
    super(jwt, "http://localhost:8080/api");
  }

  async init(): Promise<void> {
    await initWasm();

    const serializedHomseq = await this.getClientData();

    const evaluationKey = this.generateKey(serializedHomseq);

    const s3Key = await server.uploadEk(this.jwt, this.beUrl, evaluationKey);

    await triggerEvkCopy(this.jwt, this.beUrl, evaluationKey, s3Key);

    await worker_api.preprocessEk(this.jwt, this.beUrl);

    this.initialized = true;
  }
}

async function example(latticaClient: LatticaQueryClient, pt: Tensor) {
  console.time("Offline phase");
  await latticaClient.init();
  console.log("Client initialized.");
  console.timeEnd("Offline phase");

  console.time("Online phase");
  const ct = await latticaClient.encrypt(pt);
  console.log("Encrypted.");
  const ct_res = await latticaClient.apply_query(ct);
  console.log("Query finished.");
  const pt_res = await latticaClient.decrypt(ct_res);
  console.log("Decrypted.");
  console.timeEnd("Online phase");

  const pt_expected = await worker_api.apply_clear(jwt, latticaClient.beUrl, pt);
  console.log("Apply Clear finished");

  console.log('result shape:', getTensorShape(pt_res), 'expected shape:', getTensorShape(pt_expected));
  console.log('result:', pt_res);
  console.log('expected:', pt_expected);

  return pt_res;
}

// const jwt = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6IkZTRzBSMmVuWFYzZXJSR0pMaFFidSJ9.eyJ0b2tlbklkIjoiNWUyNmEwNzAtNWQxOC00ZWE2LThiZGItODc5NjYwNmRiN2NiIiwiaXNzIjoiaHR0cHM6Ly9kZXYtZjVrZ3Iwa25yd2pvYWF3MS51cy5hdXRoMC5jb20vIiwic3ViIjoiUFFJUVVpN24zbUpqSnBNdmFKRkQzQ09kTzRwMzVCc3ZAY2xpZW50cyIsImF1ZCI6Imh0dHBzOi8vbXlhcGkvYXBpIiwiaWF0IjoxNzM0MDIwMTQxLCJleHAiOjE3MzY2MTIxNDEsInNjb3BlIjoidXNlciBkZWZhdWx0IiwiZ3R5IjoiY2xpZW50LWNyZWRlbnRpYWxzIiwiYXpwIjoiUFFJUVVpN24zbUpqSnBNdmFKRkQzQ09kTzRwMzVCc3YiLCJwZXJtaXNzaW9ucyI6WyJ1c2VyIiwiZGVmYXVsdCJdfQ.GQz-nENvI_kGESL895dircH8G4ATYeXDKoBhB_4D8FEy2lyhTKqWlSecydJJohRLKEoI2AVyWJb6hG2tXkLJnqWuyYcHun7-bSF9oqmIRthRyblWaam-9jJlfvEm7V8BmlsRVj4JceggV05y-Ty3---faxxaoKTIryx9OFuePzUFJJFy1gjfM_hfDDV_alxPYBU07fh309b8g2rgilaD11-0hL55igMtkNItMBU5poRdHCxAokrM6zdxXodMDxfImr483VS-Dt9_MwH9283IbVi54dDBDtvzBEMIoenky8lgfG79C3aK2g5pH1xFwfN_p1nKWAktqbeg4fGl2wLdUQ";
const jwt = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6IkZTRzBSMmVuWFYzZXJSR0pMaFFidSJ9.eyJ0b2tlbklkIjoiZGE3Yjc0Y2UtNDAxYi00MTU0LTljZjItYWM1NjBlNDUyMTg1IiwiaXNzIjoiaHR0cHM6Ly9kZXYtZjVrZ3Iwa25yd2pvYWF3MS51cy5hdXRoMC5jb20vIiwic3ViIjoiUFFJUVVpN24zbUpqSnBNdmFKRkQzQ09kTzRwMzVCc3ZAY2xpZW50cyIsImF1ZCI6Imh0dHBzOi8vbXlhcGkvYXBpIiwiaWF0IjoxNzM0ODU0MzYwLCJleHAiOjE3Mzc0NDYzNjAsInNjb3BlIjoidXNlciBkZWZhdWx0IiwiZ3R5IjoiY2xpZW50LWNyZWRlbnRpYWxzIiwiYXpwIjoiUFFJUVVpN24zbUpqSnBNdmFKRkQzQ09kTzRwMzVCc3YiLCJwZXJtaXNzaW9ucyI6WyJ1c2VyIiwiZGVmYXVsdCJdfQ.pKqBPbdvmONF9RWIYNq9sB7vck48pzLNn6e3A1IADN8SruiPkxD-QFanJhk7xCTQUzDiMqstkRYG69WD3EmV-HxsqoZnyTXAfeM2w8O5SW-p2YPjjuwY8aau2JpDXUdSCg6jQAnzPFOy3y2rADhJHRCLFnAommCA8kKrVvXA4HuFPXAdeWguVct4x53y5bpA0aX7ym-8cxUZILetfq9GQvqN2-DC-MMAnA0Ginq4ZH5iwRQMzahuAlUrfPKTx2amMftjnfLYmB7NUBQx_6o3fnVGnnC8XJITmAYvo8eex5ejsmh-Yy2VrTIDZ1OBdKznvIBnbEV27ubgN46cb9_P_g";

// const BE_URL = "http://localhost:8080/api";
// const BE_URL = "https://api-alb-509680522.eu-central-1.elb.amazonaws.com/api";

// const client = new LatticaQueryClient(jwt, BE_URL);
// console.log("Client created");

// const client = new LatticaQueryClientLocalDev(jwt);
// console.log("DEV client created");

const pt = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,97,222,253,253,128,9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,45,252,252,252,252,253,194,65,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,32,222,252,252,252,252,253,252,202,70,38,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,116,253,252,227,66,45,169,252,252,252,232,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,116,253,252,79,0,0,13,211,252,252,252,106,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,231,255,253,69,0,0,3,170,253,253,253,116,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,209,253,252,90,0,0,118,252,252,252,252,116,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,116,253,252,234,184,184,253,252,252,252,252,116,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,21,211,252,252,252,252,253,252,252,252,252,116,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,23,158,252,252,252,253,178,168,252,252,116,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,95,116,0,0,81,253,253,116,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,184,252,252,116,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,184,252,252,32,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,38,240,252,210,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,151,252,252,85,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,170,253,253,23,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,97,252,252,252,22,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,138,252,252,252,22,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,138,252,252,218,14,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,23,158,221,35,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0];
for (let i = 0; i < pt.length; i++) {
  pt[i] /= 255;
}

// example(client, pt).then(() => {
//   console.log("FINISHED");
// });
