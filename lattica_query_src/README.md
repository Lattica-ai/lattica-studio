# Lattica Query Client

A TypeScript client for performing lattica queries on images.

## Installation

1. Create a GitHub Personal Access Token with `read:packages` permission:
   - Go to GitHub Settings → Developer Settings → Personal Access Tokens → Tokens (classic)
   - Generate new token
   - Select `read:packages` scope
   - Copy the token

2. Create or edit `.npmrc` in your project root:
```plaintext
@Lattica-ai:registry=https://npm.pkg.github.com
//npm.pkg.github.com/:_authToken=YOUR_GITHUB_PAT
```
3. Install the package:
```bash
npm install @Lattica-ai/lattica-query-client
```

## Usage

```typescript
import { LatticaQueryClient } from 'lattica-query-client';

const client = new LatticaQueryClient('your-jwt-token');

async function processImage(encodedImage: string) {
    await client.init();
    const encryptedImage = await client.encrypt(encodedImage);
    const queryResult = await client.apply_query(encryptedImage);
    const result = await client.decrypt(queryResult);
    return result;
}
```

## API

### `constructor(jwt: string)`
Creates a new LatticaQueryClient instance with the provided JWT token.

### `init(): Promise<void>`
Initializes the client. Must be called before any other methods.

### `encrypt(encodedImage: string): Promise<Uint8Array>`
Encrypts the provided encoded image.

### `apply_query(encryptedInput: Uint8Array): Promise<Uint8Array>`
Applies the query to the encrypted image.

### `decrypt(encryptedResult: Uint8Array): Promise<string>`
Decrypts the processed image and returns it.
```