import { defineConfig } from 'tsup';

export default defineConfig({
  entry: ['src/index.ts', 'src/wasm/wasm_worker.js'],
  format: ['esm', 'cjs'],
  dts: true,
  clean: true,
  sourcemap: true,
  noExternal: ['ndarray'],
});
