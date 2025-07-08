import { LatticaQueryClient } from "../src";
import * as toolkitWrapper from "../src/toolkitWrapper";
import * as serverApi from "../src/serverApi";

(global as any).URL.createObjectURL = jest.fn(() => "blob://dummy");

/* ------------------------------------------------------------------ */
/*  Module mocks                                                      */
/* ------------------------------------------------------------------ */

// WASM bootstrap
jest.mock("../src/wasm/wasm_worker_manager", () => ({
  initWasm: jest.fn(() => Promise.resolve()),
  workerManagerSingleton: { logConsole: jest.fn() },
}));

// Crypto / tensor helpers
jest.mock("../src/toolkitWrapper", () => ({
  generateKey: jest.fn(() => [
    new Uint8Array([11]), // sk
    new Uint8Array([22]), // skCoefs
    new Uint8Array([7]), // evaluation key to upload
  ]),
  encrypt: jest.fn(() => new Uint8Array([42])),
  decrypt: jest.fn(() => new Uint8Array([99])),
  applyClientBlock: jest.fn(),
  logConsole: jest.fn(),
}));

// Worker HTTP helpers
jest.mock("../src/workerApi", () => ({
  get_user_init_data: jest.fn(() => Promise.resolve(new Uint8Array([1]))),
  preprocessEk: jest.fn(() => Promise.resolve()),
}));

// Backend HTTP helpers
jest.mock("../src/serverApi", () => ({
  BE_URL: "https://be.example.com",
  uploadEk: jest.fn(() => Promise.resolve({ ok: true })),
  sendHttpRequest: jest.fn(() => Promise.resolve({ s3Key: "s3://dummy" })),
  alertUploadComplete: jest.fn(() => Promise.resolve()),
}));

// Protobuf decode stubs
jest.mock("../src/serialization/hom_op", () => ({
  ClientData: {
    decode: jest.fn(() => ({
      serializedContext: new Uint8Array([10]),
      serializedClientSequentialHomOp: new Uint8Array([20]),
    })),
  },
  QueryClientSequentialHomOp: {
    decode: jest.fn(() => ({
      preprocessingData: undefined,
      clientBlocks: [],
    })),
  },
}));

// Inference RPC
const mockTensor = { data: "tensor-out" };
jest.mock("../src/inference", () => ({
  runInference: jest.fn(() => Promise.resolve(mockTensor)),
  runInferenceDemo: jest.fn(),
}));

/* ------------------------------------------------------------------ */
/*  Tests                                                             */
/* ------------------------------------------------------------------ */

describe("LatticaQueryClient (Node-side)", () => {
  const jwt = "dummy-jwt";
  const beUrl = "https://be.example.com";
  let client: LatticaQueryClient;

  beforeEach(() => {
    jest.clearAllMocks();
    client = new LatticaQueryClient(jwt, beUrl);
  });

  it("constructor with JWT", () => {
    expect(() => new LatticaQueryClient(jwt)).not.toThrow();
  });

  it("constructor throws when JWT is empty", () => {
    expect(() => new LatticaQueryClient("")).toThrow();
  });

  it("throws if runQuery() is called before init()", async () => {
    const nakedClient = new LatticaQueryClient(jwt, beUrl); // no init()

    const dummyTensor = { data: "tensor-in" } as any;
    await expect(nakedClient.runQuery(dummyTensor)).rejects.toThrow(
      "Client must be initialized first",
    );
  });

  it("encrypt() before init() throws", async () => {
    const dummyTensor = { data: "pt" } as any;
    await expect(client.encrypt(dummyTensor)).rejects.toThrow(
      "Client must be initialized first",
    );
  });

  it("decrypt() before init() throws", async () => {
    const dummyCt = new Uint8Array([1]);
    await expect(client.decrypt(dummyCt)).rejects.toThrow(
      "Client must be initialized first",
    );
  });

  it("second init() call cause extra uploadEk", async () => {
    await client.init();
    await client.init(); // should short-circuit

    expect(serverApi.uploadEk).toHaveBeenCalledTimes(2);
  });

  it("falls back to serverApi.BE_URL when no beUrl is supplied", async () => {
    const fallbackClient = new LatticaQueryClient(jwt); // ← omit beUrl
    await fallbackClient.init();

    // uploadEk must receive the mocked constant from serverApi mock
    expect(serverApi.uploadEk).toHaveBeenCalledWith(
      jwt,
      serverApi.BE_URL, // ← proves the default kicked in
      new Uint8Array([7]),
    );
  });

  it("init() generates an evaluation key and uploads it exactly once", async () => {
    await client.init();

    expect(toolkitWrapper.generateKey).toHaveBeenCalledTimes(1);
    expect(serverApi.uploadEk).toHaveBeenCalledTimes(1);
    expect(serverApi.uploadEk).toHaveBeenCalledWith(
      jwt,
      beUrl,
      new Uint8Array([7]),
    );
  });

  it("runQuery() does a full encrypt → inference → decrypt round-trip", async () => {
    await client.init();

    const inputTensor = { data: "tensor-in" } as any;
    const result = await client.runQuery(inputTensor);

    const { runInference } = require("../src/inference");
    expect(runInference).toHaveBeenCalledTimes(1);

    //  runInference(jwt, beUrl, serializedContext, secretKey, pt, clientBlocks)
    const args = (runInference as jest.Mock).mock.calls[0];
    expect(args[0]).toBe(jwt);
    expect(args[1]).toBe(beUrl);
    expect(args[5]).toEqual([]); // clientBlocks passed through

    expect(result).toBe(mockTensor); // User sees the inference result
  });
});
