// tests/LatticaQueryClient.test.ts
import {LatticaQueryClient} from '../src';
import * as worker_api from '../src/worker_api';
import * as server from '../src/server';
import * as toolkit from '../src/toolkit_wrapper';

// Mock the worker_api and server endpoints
jest.mock('../src/worker_api');
jest.mock('../src/server');
jest.mock('../src/toolkit_wrapper');

describe('LatticaQueryClient', () => {
  const mockJwt = 'test-jwt-token';
  const mockContext = new Uint8Array([1, 2, 3]);
  const mockHomseq = new Uint8Array([4, 5, 6]);
  const mockSecretKey: [Uint8Array, Uint8Array] = [new Uint8Array([7, 8]), new Uint8Array([9, 10])];
  const mockEvaluationKey = new Uint8Array([11, 12]);

  beforeEach(() => {
    // Reset all mocks
    jest.clearAllMocks();

    // Mock worker_api responses
    (worker_api.get_user_init_data as jest.Mock).mockResolvedValue({
      serializedContext: mockContext,
      serializedHomseq: mockHomseq
    });
    (worker_api.preprocessEk as jest.Mock).mockResolvedValue(undefined);

    // Mock toolkit responses
    (toolkit.generate_key as jest.Mock).mockReturnValue([
      mockSecretKey[0],
      mockSecretKey[1],
      mockEvaluationKey
    ]);

    // Mock server responses
    (server.uploadEk as jest.Mock).mockResolvedValue(undefined);

    // Mock global Module
    global.Module = {
      onRuntimeInitialized: null,
      ready: true
    };
  });

  describe('initialization', () => {
    it('should throw error if no JWT provided', () => {
      expect(() => new LatticaQueryClient('')).toThrow('JWT token is required');
    });

    it('should initialize successfully', async () => {
      const client = new LatticaQueryClient(mockJwt);
      await expect(client.init()).resolves.not.toThrow();

      expect(worker_api.get_user_init_data).toHaveBeenCalledWith(mockJwt);
      expect(toolkit.generate_key).toHaveBeenCalledWith(mockHomseq, mockContext);
      expect(server.uploadEk).toHaveBeenCalledWith(mockJwt, mockEvaluationKey);
      expect(worker_api.preprocessEk).toHaveBeenCalledWith(mockJwt);
    });

    it('should handle WASM initialization failure', async () => {
      // Save the original Module
      const originalModule = global.Module;

      try {
        // Set Module to an object that will trigger an error via onAbort
        global.Module = {
          onRuntimeInitialized: null,
          onAbort: null,
          ready: false
        };

        const client = new LatticaQueryClient(mockJwt);
        const initPromise = client.init();

        // Trigger the abort
        global.Module.onAbort(new Error('WASM initialization failed'));

        await expect(initPromise).rejects.toThrow('Failed to initialize WASM module');
      } finally {
        // Restore the original Module
        global.Module = originalModule;
      }
    }, 10000);
  });

  describe('encryption flow', () => {
    let client: LatticaQueryClient;
    const mockInput = [1, 2, 3];
    const mockEncrypted = new Uint8Array([4, 5, 6]);

    beforeEach(async () => {
      client = new LatticaQueryClient(mockJwt);
      (toolkit.dumps_proto_tensor as jest.Mock).mockReturnValue(new Uint8Array([1, 2, 3]));
      (toolkit.enc as jest.Mock).mockReturnValue(mockEncrypted);
      await client.init();
    });

    it('should encrypt data successfully', async () => {
      const encrypted = await client.encrypt(mockInput);

      expect(toolkit.dumps_proto_tensor).toHaveBeenCalledWith(mockInput);
      expect(toolkit.enc).toHaveBeenCalledWith(
        mockContext,
        mockSecretKey,
        expect.any(Uint8Array),
        true
      );
      expect(encrypted).toEqual(mockEncrypted);
    });

    it('should throw error if not initialized', async () => {
      const uninitializedClient = new LatticaQueryClient(mockJwt);
      await expect(uninitializedClient.encrypt(mockInput))
        .rejects.toThrow('Client must be initialized first');
    });
  });

  // Add more test cases for decrypt and apply_query methods
});