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

async function processImage(image: HTMLImageElement) {
    await client.encrypt(image);
    await client.apply_query();
    const result = await client.decrypt();
    return result;
}
```

## API

### `constructor(jwt: string)`
Creates a new LatticaQueryClient instance with the provided JWT token.

### `encrypt(image: HTMLImageElement): Promise<boolean>`
Encrypts the provided image.

### `apply_query(): Promise<boolean>`
Applies the query to the encrypted image.

### `decrypt(): Promise<HTMLImageElement>`
Decrypts the processed image and returns it.



