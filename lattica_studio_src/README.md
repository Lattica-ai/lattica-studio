# Lattica Studio

Lattica Studio is the deployment client for [Lattica](https://www.lattica.ai)
homomorphic workloads.

This package focuses on the full model-lifecycle flow around a homomorphic
pipeline: deploy/compile, worker management, token management, and encrypted
query execution.

## Installation

```bash
pip install lattica-studio
```

Requires Python `>=3.11`.

## End-to-end flow (what must happen)

To use a homomorphic pipeline in Lattica, the flow is:

1. **Deploy + compile**
   - Register or update a model in the Lattica platform.
   - Associate the model with the homomorphic pipeline.
   - Trigger backend compilation.

2. **Start a worker before serving encrypted queries**
   - A worker must be up to process queries.
   - Worker runtime can be slow to start and incurs cost.
   - Start workers only when needed and stop them when done.

3. **Create a query token (model owner step)**
   - The model owner creates a query token.
   - Query clients use this token to access the model.

4. **Query-client one-time setup per token/model context**
   - Generate keys.
   - Register the evaluation key (EK).
   - Keep the secret key (SK) client-side for encrypt/decrypt.

5. **Send encrypted queries**
   - Encrypt inputs with SK-side client logic.
   - Send encrypted payload to the model worker.
   - Decrypt returned encrypted outputs with SK.

## Important development guidance

If the **pipeline architecture changes**, redeploy/compile is required and you
should run the full setup flow again for that new compiled model context.

Because worker runtime costs money, we recommend staged development rather than
running the entire flow every iteration.

## Single source-of-truth script

Use this script as the canonical flow reference:

- `lattica_studio/example.py`

The script is intentionally staged, so you can run only subsets of operations
by toggling stage flags (for example, deploy/compile only, or query only on an
already-running model).

It also uses `try/finally` to ensure workers are stopped when requested.

## Suggested stage-by-stage usage

During development, use these stages incrementally:

- Stage A: deploy + compile only.
- Stage B (one-time setup per compiled pipeline context): create query token, start worker,
  generate keys, register EK, then stop worker.
- Stage C (recurring query runs): when needed, start worker and send encrypted queries,
  encrypting/decrypting with the already registered key context.

This gives fast feedback while minimizing unnecessary worker runtime.

## License

Distributed under the [LatticaAI Internal Use License](./LICENSE.md) — internal,
non-commercial, research, or evaluation use only.

Copyright © LatticaAI Inc. All rights reserved.
