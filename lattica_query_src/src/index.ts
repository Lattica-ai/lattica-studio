const toolkit = require("./toolkit_mock");
const server = require("./server");
const worker_api = require("./worker_api");
const {Tensor, runInferenceDemo} = require("./utils");
const {QueryClientSequentialHomOp, QueryClientBlock} = require("./protos/hom_op_pb");
const {MOCK_BE} = require("./config");

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

    encryptedInput = await toolkit.encrypt(
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
  "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6IkZTRzBSMmVuWFYzZXJSR0pMaFFidSJ9.eyJ0b2tlbklkIjoiZjBiYjY1YzctMTM0ZS00MmEzLTkwZTktMjM5M2NmMDM3OGM0IiwiaXNzIjoiaHR0cHM6Ly9kZXYtZjVrZ3Iwa25yd2pvYWF3MS51cy5hdXRoMC5jb20vIiwic3ViIjoiUFFJUVVpN24zbUpqSnBNdmFKRkQzQ09kTzRwMzVCc3ZAY2xpZW50cyIsImF1ZCI6Imh0dHBzOi8vbXlhcGkvYXBpIiwiaWF0IjoxNzMxNTc3NjUxLCJleHAiOjE3MzQxNjk2NTEsInNjb3BlIjoidXNlciBkZWZhdWx0IiwiZ3R5IjoiY2xpZW50LWNyZWRlbnRpYWxzIiwiYXpwIjoiUFFJUVVpN24zbUpqSnBNdmFKRkQzQ09kTzRwMzVCc3YiLCJwZXJtaXNzaW9ucyI6WyJ1c2VyIiwiZGVmYXVsdCJdfQ.njJTM9yPWygz-wt8mOJXsRF3Q9BAtHXwCmKYBKAs0Hgf-92PplBChE8OzOC1QrriuW5AJPlhEArE27b1xT_oQ2pKfykxAgfaBjQb-KQJ4gclFjHXu2FIHAkmm-CHqjtOEWmc4E8o-7ToVkJEFdpRUZkKSjXOrPM9zYfohGVgeGN9sLLOpJbSYF-6pq7-HkVO4GZjwOhWNMHxEcnemSB3rwly0R0Hqf6OzKA-FIKGwYdmumOVDxZ5Drih6IZaBHNC04Qa3xyhHFo36MR2YPr4AWLVGeui32qR6aQv3Q5X5UdhjMy5Ijg4ja67E0EO3nvS4qkvBSunBSNJ6_XZkD_q7w",
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
