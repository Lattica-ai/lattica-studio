const fs = require("fs");
const path = require("path");

// Define type for package.json
interface PackageJson {
  version: string;
  [key: string]: unknown;
}

// Read package.json from the parent directory
const packageJsonPath = path.resolve(__dirname, "../package.json");
const pkg = require(packageJsonPath) as PackageJson;

// Create version.ts file
fs.writeFileSync(
  path.resolve(__dirname, "../src/version.ts"),
  `// This file is auto-generated. Do not edit directly.
export const VERSION = "${pkg.version}";
`,
);

console.log(`Version updated to ${pkg.version}`);
