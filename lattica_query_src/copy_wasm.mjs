// Copy the FHE core WASM (cpp_sdk.js + cpp_sdk.wasm) into dist/wasm/.
// * With --required a missing WASM fails; otherwise it only prints a notice.
import { access, copyFile, mkdir } from 'node:fs/promises';
import path from 'node:path';

// Folder holding the WASM: LATTICA_FHE_CORE_WASM_DIR, else the sibling FHE core build.
const source =
  process.env.LATTICA_FHE_CORE_WASM_DIR ?? '../../lattice-toolkit/toolkit/fhe_core/build_wasm';
const target = 'dist/wasm';
const files = ['cpp_sdk.js', 'cpp_sdk.wasm'];
const required = process.argv.includes('--required');

const exists = (file) =>
  access(path.join(source, file)).then(
    () => true,
    () => false,
  );
if (!(await Promise.all(files.map(exists))).every(Boolean)) {
  console.error(`No FHE core WASM in ${source}: build it or set LATTICA_FHE_CORE_WASM_DIR.`);
  process.exit(required ? 1 : 0);
}
await mkdir(target, { recursive: true });
for (const file of files) {
  await copyFile(path.join(source, file), path.join(target, file));
}
