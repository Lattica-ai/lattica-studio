// jest.config.ts
import type { Config } from 'jest';

const config: Config = {
  preset: 'ts-jest',
  // Folders jest scans, so the Python package next to them is skipped.
  roots: ['<rootDir>/src', '<rootDir>/tests'],
  testEnvironment: 'jsdom',
  setupFiles: ['./jest.setup.ts'],
  moduleNameMapper: {
    '^(\\.{1,2}/.*)\\.js$': '$1',
    './wasm/cpp_sdk.js': '<rootDir>/tests/mocks/wasmModule.mock.ts'
  },
  transform: {
    '^.+\\.tsx?$': ['ts-jest', {
      tsconfig: {
        allowJs: true,
      },
      useESM: true,
    }],
    '^.+\\.js$': ['ts-jest', {
      useESM: true,
    }],
  },
  extensionsToTreatAsEsm: ['.ts', '.tsx'],
};

export default config;