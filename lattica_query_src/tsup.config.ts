import { defineConfig } from "tsup";
import { copyFile, mkdir } from "fs/promises";

export default defineConfig({
  entry: [
    "src/index.ts",
    "src/examples/mnist_fc.ts",
    "src/examples/sharpen.ts",
    "src/examples/im2tensor.ts",
  ],
  format: ["esm", "cjs"],
  dts: true,
  clean: true,
  sourcemap: true,
  onSuccess: async () => {
    // Create wasm directory in dist
    await mkdir("dist/wasm", { recursive: true });

    // Copy WASM files maintaining the directory structure
    await copyFile("src/wasm/cpp_sdk.wasm", "dist/wasm/cpp_sdk.wasm");
    await copyFile("src/wasm/cpp_sdk.js", "dist/wasm/cpp_sdk.js");
  },
});
