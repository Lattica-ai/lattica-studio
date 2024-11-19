const server = require("./server");
const worker_api = require("./worker_api");
import * as toolkit from "./toolkit_wrapper";
import './wasm/cpp_sdk.js';

const {Tensor, runInferenceDemo} = require("./utils");
const {QueryClientSequentialHomOp, QueryClientBlock} = require("./protos/hom_op_pb");
const {MOCK_BE} = require("./config");

export {};
declare global {
    var Module: any;  // You can replace 'any' with your specific module interface
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
        console.log('WASM Module is ready!');
    } catch (error) {
        console.error('Failed to load WASM module:', error);
    }
}

class LatticaQueryClient {
  private readonly jwt: string;
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
    await initWasm();

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

    const [sk, skCoefs, evaluationKey] = toolkit.generate_key(
      this.serializedHomseq,
      this.serializedContext,
    );
    console.log('sk len:', sk.length);
    console.log('skCoefs len:', skCoefs.length);
    console.log('evaluationKey len:', evaluationKey.length);
    this.secretKey = [sk, skCoefs];

    if (!MOCK_BE) {
      await server.uploadEk(this.jwt, evaluationKey);
      await worker_api.preprocessEk(this.jwt);
    }

    this.initialized = true;
  }

  async encrypt(pt: typeof Tensor): Promise<Uint8Array> {
    if (!this.initialized) throw new Error("Client must be initialized first");

    let serialized_pt = toolkit.dumps_proto_tensor(pt);
    // check if need to apply first client block
    if (this.applyBlockBeforeEncryption) {
      serialized_pt = toolkit.apply_client_block(
        this.firstClientBlock,
        this.serializedContext,
        serialized_pt,
      );
    }

    return toolkit.enc(
        this.serializedContext,
        this.secretKey,
        serialized_pt,
        true,
    );
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
    let decryptedOutput = toolkit.dec(
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
  console.log(ct_res);
  console.log(ct_res.length);
  const pt_res = await client.decrypt(ct_res);
  console.log("Decrypted:");
  return pt_res;
}

const jwt = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6IkZTRzBSMmVuWFYzZXJSR0pMaFFidSJ9.eyJ0b2tlbklkIjoiZjBiYjY1YzctMTM0ZS00MmEzLTkwZTktMjM5M2NmMDM3OGM0IiwiaXNzIjoiaHR0cHM6Ly9kZXYtZjVrZ3Iwa25yd2pvYWF3MS51cy5hdXRoMC5jb20vIiwic3ViIjoiUFFJUVVpN24zbUpqSnBNdmFKRkQzQ09kTzRwMzVCc3ZAY2xpZW50cyIsImF1ZCI6Imh0dHBzOi8vbXlhcGkvYXBpIiwiaWF0IjoxNzMxNTc3NjUxLCJleHAiOjE3MzQxNjk2NTEsInNjb3BlIjoidXNlciBkZWZhdWx0IiwiZ3R5IjoiY2xpZW50LWNyZWRlbnRpYWxzIiwiYXpwIjoiUFFJUVVpN24zbUpqSnBNdmFKRkQzQ09kTzRwMzVCc3YiLCJwZXJtaXNzaW9ucyI6WyJ1c2VyIiwiZGVmYXVsdCJdfQ.njJTM9yPWygz-wt8mOJXsRF3Q9BAtHXwCmKYBKAs0Hgf-92PplBChE8OzOC1QrriuW5AJPlhEArE27b1xT_oQ2pKfykxAgfaBjQb-KQJ4gclFjHXu2FIHAkmm-CHqjtOEWmc4E8o-7ToVkJEFdpRUZkKSjXOrPM9zYfohGVgeGN9sLLOpJbSYF-6pq7-HkVO4GZjwOhWNMHxEcnemSB3rwly0R0Hqf6OzKA-FIKGwYdmumOVDxZ5Drih6IZaBHNC04Qa3xyhHFo36MR2YPr4AWLVGeui32qR6aQv3Q5X5UdhjMy5Ijg4ja67E0EO3nvS4qkvBSunBSNJ6_XZkD_q7w";
const client = new LatticaQueryClient(jwt);
console.log("Client created");

// const pt = [
//   [
//     [1, 2],
//     [3, 4],
//   ],
//   [
//     [5, 6],
//     [7, 8],
//   ],
// ];
const pt = new Float64Array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,97,222,253,253,128,9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,45,252,252,252,252,253,194,65,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,32,222,252,252,252,252,253,252,202,70,38,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,116,253,252,227,66,45,169,252,252,252,232,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,116,253,252,79,0,0,13,211,252,252,252,106,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,231,255,253,69,0,0,3,170,253,253,253,116,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,209,253,252,90,0,0,118,252,252,252,252,116,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,116,253,252,234,184,184,253,252,252,252,252,116,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,21,211,252,252,252,252,253,252,252,252,252,116,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,23,158,252,252,252,253,178,168,252,252,116,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,95,116,0,0,81,253,253,116,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,184,252,252,116,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,184,252,252,32,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,38,240,252,210,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,151,252,252,85,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,170,253,253,23,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,97,252,252,252,22,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,138,252,252,252,22,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,138,252,252,218,14,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,23,158,221,35,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]);
for (let i = 0; i < pt.length; i++) {
    pt[i] /= 255;
}
console.log(pt.reduce((sum, value) => sum + value, 0));

// worker_api.apply_clear(jwt, pt).then((pt_res) => {
//     console.log("APPLY CLEAR RESULT:", pt_res);
// });

processImage(pt).then((pt_res) => {
  console.log("Result:", pt_res);
});
