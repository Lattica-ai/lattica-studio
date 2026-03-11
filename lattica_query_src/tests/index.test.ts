import { LatticaQueryClient } from '../src';
import * as toolkitWrapper from '../src/toolkitWrapper';
import * as serverApi from '../src/serverApi';
import * as inference from '../src/inference';
import { Tensor, vecToTensor, dumps_proto_tensor } from '../src/tensorUtils';
import { ClientData, ClientModel, ClientBlock } from '../src/serialization/hom_op';

(global as any).URL.createObjectURL = jest.fn(() => 'blob://dummy');

/* ------------------------------------------------------------------ */
/*  Module mocks                                                      */
/* ------------------------------------------------------------------ */

const dummyTensor: Tensor = vecToTensor([1, 2, 3, 4, 5]);
const dummySerializedTensor = dumps_proto_tensor(dummyTensor);

const dummySerializedCt = new Uint8Array([42]);

jest.mock('../src/wasm/wasm_worker_manager', () => ({
  initWasm: jest.fn(() => Promise.resolve()),
  workerManagerSingleton: { logConsole: jest.fn() },
}));

jest.mock('../src/toolkitWrapper', () => ({
  generateKey: jest.fn(() => [
    new Uint8Array([11]), // sk
    new Uint8Array([22]), // skCoefs
    new Uint8Array([7]), // evaluation key to upload
  ]),
  encrypt: jest.fn(() => dummySerializedCt),
  decrypt: jest.fn(() => dummySerializedTensor),
  applyClientBlock: jest.fn((...args: any[]) => args[2]), // return pt
  logConsole: jest.fn(),
}));

jest.mock('../src/workerApi', () => ({
  get_user_init_data: jest.fn(async () => {
    const model = ClientModel.create({
      preprocessBlock: ClientBlock.create({}),
      postprocessBlock: ClientBlock.create({}),
    });
    const serializedModel = ClientModel.encode(model).finish();
    const clientData = ClientData.create({ serializedModel });
    return ClientData.encode(clientData).finish();
  }),
  preprocessEk: jest.fn(() => Promise.resolve()),
  apply_hom_pipeline: jest.fn((...args: any[]) => args[2]), // return ct
}));

jest.mock('../src/serverApi', () => ({
  BE_URL: 'https://be.prod.com',
  uploadEk: jest.fn(() => Promise.resolve({ ok: true })),
  sendHttpRequest: jest.fn(() => Promise.resolve({ s3Key: 's3://dummy' })),
  alertUploadComplete: jest.fn(() => Promise.resolve()),
}));

/* ------------------------------------------------------------------ */
/*  Tests                                                             */
/* ------------------------------------------------------------------ */

describe('LatticaQueryClient (Node-side)', () => {
  const jwt = 'dummy-jwt';
  const beUrl = 'https://be.local.com';
  let client: LatticaQueryClient;

  beforeEach(() => {
    jest.clearAllMocks();
    client = new LatticaQueryClient(jwt, beUrl);
  });

  it('constructor with JWT', () => {
    expect(() => new LatticaQueryClient(jwt)).not.toThrow();
  });

  it('constructor throws when JWT is empty', () => {
    expect(() => new LatticaQueryClient('')).toThrow();
  });

  it('throws if runQuery() is called before init()', async () => {
    const nakedClient = new LatticaQueryClient(jwt, beUrl); // no init()

    await expect(nakedClient.runQuery(dummyTensor)).rejects.toThrow(
      'Client must be initialized first',
    );
  });

  it('encrypt() before init() throws', async () => {
    await expect(client.encrypt(dummyTensor)).rejects.toThrow('Client must be initialized first');
  });

  it('encrypt() returns ct', async () => {
    await client.init();
    const result = await client.encrypt(dummyTensor);

    expect(result).toStrictEqual(dummySerializedCt);
  });

  it('decrypt() before init() throws', async () => {
    await expect(client.decrypt(dummySerializedCt)).rejects.toThrow(
      'Client must be initialized first',
    );
  });

  it('decrypt() returns pt', async () => {
    await client.init();
    const result = await client.decrypt(dummySerializedCt);

    expect(result).toStrictEqual(dummyTensor);
  });

  it('second init() call cause extra uploadEk', async () => {
    await client.init();
    await client.init(); // should short-circuit

    expect(serverApi.uploadEk).toHaveBeenCalledTimes(2);
  });

  it('falls back to serverApi.BE_URL when no beUrl is supplied', async () => {
    const fallbackClient = new LatticaQueryClient(jwt); // ← omit beUrl
    await fallbackClient.init();

    // uploadEk must receive the mocked constant from serverApi mock
    expect(serverApi.uploadEk).toHaveBeenCalledWith(
      jwt,
      serverApi.BE_URL, // ← proves the default kicked in
      new Uint8Array([7]),
    );
  });

  it('init() generates an evaluation key and uploads it exactly once', async () => {
    await client.init();

    expect(toolkitWrapper.generateKey).toHaveBeenCalledTimes(1);
    expect(serverApi.uploadEk).toHaveBeenCalledTimes(1);
    expect(serverApi.uploadEk).toHaveBeenCalledWith(jwt, beUrl, new Uint8Array([7]));
  });

  it('runQuery() does a full encrypt → inference → decrypt round-trip', async () => {
    const spy = jest.spyOn(inference, 'runInference');

    await client.init();

    const result = await client.runQuery(dummyTensor);

    expect(spy).toHaveBeenCalledTimes(1);

    //  runInference(jwt, beUrl, serializedContext, secretKey, pt, clientBlocks)dummyCt
    const args = spy.mock.calls[0];
    expect(args[0]).toBe(jwt);
    expect(args[1]).toBe(beUrl);

    expect(result).toStrictEqual(dummyTensor);
  });
});
