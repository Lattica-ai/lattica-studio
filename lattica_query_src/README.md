# LatticaAI Query Client

A TypeScript client for performing lattica queries on images.

## Installation
Install the package:
```bash
npm install @lattica-ai/query
```

## Usage

```typescript
import { LatticaQueryClient } from '@lattica-ai/query';

const client = new LatticaQueryClient('your-jwt-token');

async function processImage(encodedImage: string) {
    await client.init();
    const encryptedImage = await client.encrypt(encodedImage);
    const queryResult = await client.apply_query(encryptedImage);
    const result = await client.decrypt(queryResult);
    return result;
}
```

## Documentation

See documentation at: https://platformdocs.lattica.ai/

## License

Proprietary - © Lattica AI

See license details in the `LICENSE.txt` file.

---

For more information, visit [https://www.lattica.ai](https://www.lattica.ai)
