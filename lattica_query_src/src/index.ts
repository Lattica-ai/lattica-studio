const toolkit = require("./toolkit_mock");
const server = require("./server");
const worker_api = require("./worker_api");
const {Tensor, runInferenceDemo} = require("./utils");
const {QueryClientSequentialHomOp, QueryClientBlock} = require("./protos/hom_op_pb");
const {MOCK_BE} = require("./config");

import * as xxx from './xxx';
// import Module from './cpp_sdk.js';
//
// Module.onRuntimeInitialized = function () {
//     console.log("WASM Module is ready!");
//     const byteVector = new Module.VectorUint8();
//     console.log(byteVector);
// };

class LatticaQueryClient {
  private readonly jwt: string;
  private image: HTMLImageElement | null = null;
  private initialized: boolean = false;
  private serializedContext: Uint8Array | null = null;
  private serializedHomseq: Uint8Array | null = null;
  private clientBlocks: typeof QueryClientBlock[] = [];
  private secretKey: [Uint8Array, Uint8Array] | null = null;
  private applyBlockBeforeEncryption: boolean = false;
  private firstClientBlock: Uint8Array | undefined = undefined;
  private applyBlockAfterDecryption: boolean = false;
  private lastClientBlock: Uint8Array | undefined = undefined;

  constructor(jwt: string) {
    if (!jwt) throw new Error("JWT token is required");
    this.jwt = jwt;
  }

  async init(): Promise<void> {
    const initData = await worker_api.get_user_init_data(this.jwt);
    this.serializedContext = initData.serializedContext;
    this.serializedHomseq = initData.serializedHomseq;
    console.log('serializedContext len:', this.serializedContext.length);
    console.log('serializedHomseq len:', this.serializedHomseq.length);

    this.clientBlocks = QueryClientSequentialHomOp.deserializeBinary(
      this.serializedHomseq,
    ).getClientBlocksList();
    console.log('clientBlocks len:', this.clientBlocks.length);

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

    const { secretKey, evaluationKey } = toolkit.generateKey(
      this.serializedHomseq,
      this.serializedContext,
    );
    this.secretKey = secretKey;

    if (!MOCK_BE) {
      await server.uploadEk(this.jwt, evaluationKey);
      await worker_api.preprocessEk(this.jwt);
    }

    this.initialized = true;
  }

  async encrypt(pt: typeof Tensor): Promise<Uint8Array> {
    if (!this.initialized) throw new Error("Client must be initialized first");

    // const pt = await utils.imageToMatrix(image)

    let serialized_pt = toolkit.dumps_proto_tensor(pt);
    let encryptedInput;
    // check if need to apply first client block
    if (this.applyBlockBeforeEncryption) {
      serialized_pt = toolkit.apply_client_block(
        this.firstClientBlock,
        this.serializedContext,
        serialized_pt,
      );
    }

    encryptedInput = toolkit.encrypt(
      this.serializedContext,
      this.secretKey,
      serialized_pt,
      true,
    );

    return encryptedInput;
  }

  async apply_query(encryptedInput: Uint8Array): Promise<Uint8Array> {
    if (!this.initialized) throw new Error("Client must be initialized first");

    const encryptedOutput = await runInferenceDemo(
      this.jwt,
      this.serializedContext,
      this.secretKey,
      encryptedInput,
      this.clientBlocks,
      this.applyBlockBeforeEncryption ? 1 : 0,
    );
    console.log("apply finished");

    return encryptedOutput;
  }

  async decrypt(encryptedResult: Uint8Array): Promise<typeof Tensor> {
    if (!this.initialized) throw new Error("Client must be initialized first");
    let decryptedOutput = toolkit.decrypt(
      this.serializedContext,
      this.secretKey,
      encryptedResult,
    );

    // check if last block is on client
    if (this.applyBlockAfterDecryption) {
      decryptedOutput = toolkit.apply_client_block(
        this.lastClientBlock,
        this.serializedContext,
        decryptedOutput,
      );
    }

    return toolkit.load_proto_tensor(decryptedOutput);
  }
}

async function processImage(pt: typeof Tensor) {
  await client.init();
  console.log("Client initialized:");
  const ct = await client.encrypt(pt);
  console.log("Encrypted:");
  const ct_res = await client.apply_query(ct);
  console.log("Query result:");
  const pt_res = await client.decrypt(ct_res);
  console.log("Decrypted:");
  return pt_res;
}

const client = new LatticaQueryClient(
  "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6IkZTRzBSMmVuWFYzZXJSR0pMaFFidSJ9.eyJ0b2tlbklkIjoiMGU2NGEwNmEtYmZkNC00MTY2LWJiZDktOTkzYmI1MWMyZGNjIiwiaXNzIjoiaHR0cHM6Ly9kZXYtZjVrZ3Iwa25yd2pvYWF3MS51cy5hdXRoMC5jb20vIiwic3ViIjoiUFFJUVVpN24zbUpqSnBNdmFKRkQzQ09kTzRwMzVCc3ZAY2xpZW50cyIsImF1ZCI6Imh0dHBzOi8vbXlhcGkvYXBpIiwiaWF0IjoxNzMwODM5NTY1LCJleHAiOjE3MzM0MzE1NjUsInNjb3BlIjoidXNlciBkZWZhdWx0IiwiZ3R5IjoiY2xpZW50LWNyZWRlbnRpYWxzIiwiYXpwIjoiUFFJUVVpN24zbUpqSnBNdmFKRkQzQ09kTzRwMzVCc3YiLCJwZXJtaXNzaW9ucyI6WyJ1c2VyIiwiZGVmYXVsdCJdfQ.6grZW1Qek8lG8ciDEQ7d0MjmStuNYbVymzWp1Kbcl8TIxL8HlQE6FJNFKx0mOQlzwZuirzm-achQoHp2Uhn9lcBJ00vWWDmC0j7PqugYFo2PegVBr58xp8cuCxueAlgbZbCOZoqmZ61PHHVEJsd_THWOu6GNTLdTxy7SbKB8W756uofVK-hyFlP3DbP2-bWuzWI6QuwijnfFbOsRGd-U_zYJ_TVcasjLetbEpPWiHaxxOoBdRTDdPuvgINbfiGyBZcZySG5cGsqKAJpvcSR_KPlya1dJwz6qBn1-P061EKn3VWWG7tEm3JJaKnOnQI6VOE1G0cYwcHqKsexmgjM7EA",
);
console.log("Client created:", client);

const pt = [
  [
    [1, 2],
    [3, 4],
  ],
  [
    [5, 6],
    [7, 8],
  ],
];

processImage(pt).then((pt_res) => {
  console.log("Result:", pt_res);
});
