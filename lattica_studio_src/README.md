# Lattica Studio

`lattica-studio` is the control-plane client for [Lattica](https://www.lattica.ai)
homomorphic workloads.

It handles model lifecycle operations:

- deploy and compile encrypted pipelines
- start and stop worker sessions
- create and manage query tokens
- coordinate first-time encrypted query setup

Encrypted inference requests are executed by `lattica-query` after this setup.

## Requirements

- Python `>=3.11`
- A valid Lattica account license key

## Install

Published package:

```bash
pip install lattica-studio
```

## Quickstart

Set your license key:

```bash
export LATTICA_LICENSE_KEY="<your-license-key>"
```

Run the end-to-end MNIST example:

```bash
python -m lattica_studio.example
```

The example in [`lattica_studio/example.py`](lattica_studio/example.py) builds a
pipeline artifact, deploys it, starts a worker, creates a token, generates keys,
and sends encrypted queries.

## Core workflow

Use this sequence in production and development:

1. Deploy and compile a model pipeline.
2. Start a worker session for that model.
3. Create a query token (owner/admin step).
4. In query clients, generate keys and upload evaluation key (EK).
5. Send encrypted queries and decrypt responses client-side.

## Practical iteration strategy

Worker runtime is billed while active, so keep workers up only when needed.

Recommended staged iteration:

- **Stage A (pipeline changes):** deploy and compile.
- **Stage B (one-time per compiled context):** create token, generate keys,
  register EK.
- **Stage C (repeated queries):** start worker only for query windows, then stop.

If the pipeline architecture changes, treat it as a new encrypted context:
redeploy, create a fresh token, and regenerate keys.

## Resource inspection helper

Use the table-printing helper to inspect account resources:

```bash
python -m lattica_studio.list_and_display models
python -m lattica_studio.list_and_display workers
python -m lattica_studio.list_and_display tokens
python -m lattica_studio.list_and_display all
```

`list_and_display` reads `LATTICA_LICENSE_KEY` by default, or accepts
`--license-key`.

## More docs

- [Monorepo overview and broader quickstart](../README.md)
- [`lattica-build` source package docs](../lattica_build_src/README.md)

## License

Distributed under the [Lattica Studio License](./LICENSE.md): internal,
non-commercial, research, or evaluation use only.

Copyright (c) LatticaAI Inc. All rights reserved.
