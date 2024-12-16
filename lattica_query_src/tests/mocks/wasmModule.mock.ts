// tests/mocks/wasmModule.mock.ts
const MockModule = {
  ready: false,
  onRuntimeInitialized: () => {},
  // Add any other WASM functions you need to mock
  apply_client_block: jest.fn(),
  enc: jest.fn(),
  dec: jest.fn(),
  generate_key: jest.fn()
};

export default MockModule;