# Lattica Query Client

A TypeScript client for performing lattica queries on images.

## Installation

```bash
npm install lattica-query-client
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

