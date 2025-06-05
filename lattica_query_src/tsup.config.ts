import { defineConfig } from 'tsup';
import { copyFile, mkdir } from 'fs/promises';
import path from 'path';
import { createRequire } from 'module';
const require = createRequire(import.meta.url); // Create a require function for use in ESM

export default defineConfig({
  entry: ['src/index.ts', 'src/wasm/wasm_worker.js', 'src/examples/CIFAR10.ts'],
  format: ['esm', 'cjs'],
  dts: true,
  clean: true,
  sourcemap: true,
  onSuccess: async () => {
    // Ensure dist/wasm directory exists
    await mkdir('dist/wasm', { recursive: true });

    // Path to fhe-core in node_modules
    // require.resolve finds the package.json, then we derive the build_wasm path
    const fheCorePackageJsonPath = require.resolve('fhe-core/package.json');
    const fheCoreBasePath = path.dirname(fheCorePackageJsonPath);

    // Copy WASM files from fhe-core/build_wasm to dist/wasm/
    await copyFile(path.join(fheCoreBasePath, 'build_wasm/cpp_sdk.wasm'), 'dist/wasm/cpp_sdk.wasm');
    await copyFile(path.join(fheCoreBasePath, 'build_wasm/cpp_sdk.js'), 'dist/wasm/cpp_sdk.js');
  },
});
