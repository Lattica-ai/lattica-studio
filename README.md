# Lattica Studio

Build, deploy, and serve **fully homomorphic encryption (FHE) models** on the
[Lattica](https://www.lattica.ai) platform, from a PyTorch-style pipeline
definition to an encrypted inference endpoint.

Your data is encrypted before it leaves the client and stays encrypted
throughout inference. The server computes on ciphertext and never sees the
plaintext input, the plaintext output, or your secret key.

Inference runs on cloud-hosted GPU accelerators, which can significantly reduce latency versus many CPU-only FHE setups. You also avoid managing custom CUDA kernel compilation yourself.

This repository contains two Python packages:

| Package | Role |
| --- | --- |
| [`lattica-build`](./lattica_build_src) | Define a homomorphic computation graph, bind weights, plan FHE parameters, and emit a deployable artifact. Runs entirely locally. |
| [`lattica-studio`](./lattica_studio_src) | Deploy and compile that artifact on the platform, then manage models, workers, and query tokens. |

Both require **Python 3.11+**.

## How it fits together

```
       ┌─────────────────────────┐
       │      lattica-build      │   define pipeline + FHE params
       │        (local)          │   →  artifact.zip
       └────────────┬────────────┘
                    │
                    ▼
       ┌─────────────────────────┐
       │     lattica-studio      │   deploy → compile → start GPU worker
       │   (control plane API)   │   →  model_id, query token
       └────────────┬────────────┘
                    │
                    ▼
       ┌─────────────────────────┐
       │      lattica-query      │   keygen, encrypt, query, decrypt
       │  (client, holds the SK) │   →  plaintext result
       └─────────────────────────┘
```

The secret key never leaves the client. Only the evaluation key and ciphertexts
are sent to the server.

## Installation

```bash
pip install lattica-studio
```

This pulls in `lattica-build` (pipeline construction) and `lattica-query`
(client-side encryption and querying) as dependencies.

To work from a checkout of this repository:

```bash
pip install -e ./lattica_build_src
pip install -e ./lattica_studio_src
```

## Quickstart

### 1. Build an artifact locally

No account needed for this step. Build the packaged MNIST example:

```bash
lattica-build --pipeline-module lattica_build.examples.example_mnist_fc --out mnist.zip
```

On success the command writes `mnist.zip` and prints a JSON summary listing
`hom_pipeline.json` and `hom_pipeline.safetensors`.

Add `--print_graph` to inspect the computation graph, or point at your own file:

```bash
lattica-build my_pipeline.py --out my_pipeline.zip
```

Any pipeline module or file just needs to expose two callables:

```python
def build_pipeline() -> HomomorphicPipeline: ...
def build_params() -> HomParams: ...
```

### 2. Deploy, serve, and query

The remaining steps need a Lattica account license key. Set it once:

```bash
export LATTICA_LICENSE_KEY="..."
```

```python
import os

import torch

from lattica_build import build
from lattica_build.examples import example_mnist_fc
from lattica_query import QueryClient
from lattica_studio import LatticaStudio

studio = LatticaStudio(os.environ["LATTICA_LICENSE_KEY"])
x = torch.zeros(example_mnist_fc.INPUT_SHAPE)

# Build locally, then deploy and compile on the platform.
artifact = build(
    example_mnist_fc.build_pipeline(),
    example_mnist_fc.build_params(),
    "mnist.zip",
)
model_id = studio.deploy(artifact, "my-mnist-model")

# A GPU worker must be running to serve encrypted queries.
with studio.workers.running(model_id, stop_on_exit=True):
    token = studio.tokens.create(model_id, save_as="my-mnist-model")

    client = QueryClient(token)

    # Generates FHE keys and uploads the evaluation key.
    # The secret key never leaves this machine.
    sk = client.generate_key()

    # x is a plain torch tensor shaped like the pipeline's input.
    result = client.run_query(sk, x)    # encrypt → infer on ciphertext → decrypt
    print(result.argmax(dim=-1))
```

`studio.deploy_pipeline(...)` combines the build and deploy steps if you don't
need the artifact on disk:

```python
model_id = studio.deploy_pipeline(
    example_mnist_fc.build_pipeline(),
    example_mnist_fc.build_params(),
    "my-mnist-model",
)
```

Deploying under a name that already exists redeploys into that model rather than
creating a duplicate. Active workers are stopped and the model is recompiled.

A complete, runnable version of the flow above lives in
[`lattica_studio_src/lattica_studio/example.py`](./lattica_studio_src/lattica_studio/example.py).

### Display tables quickly

If you just want to inspect resources in a printable table, use the helper
script:

```bash
cd lattica_studio_src
python -m lattica_studio.list_and_display models
python -m lattica_studio.list_and_display workers
python -m lattica_studio.list_and_display tokens
python -m lattica_studio.list_and_display all
```

It uses `LATTICA_LICENSE_KEY` by default (or pass `--license-key ...`).

## Working efficiently

Worker runtime is billed while a worker is up, so keep workers running only as
long as you are actually serving queries. The
`studio.workers.running(model_id, stop_on_exit=True)` context manager stops the
worker on exit, including when the block raises.

Two things to keep in mind while iterating:

- **Changing the pipeline architecture invalidates the key context.** Redeploy,
  then create a fresh token and regenerate keys before querying again.
- **Reuse a compiled model across sessions.** Deployment and key setup only need
  to happen when the pipeline changes. To query an existing model, look it up by
  name with `studio.models.get_id_by_name(...)` and reuse a saved token with
  `studio.tokens.load(...)`.

## API overview

`LatticaStudio` exposes deployment directly and groups everything else by
resource.

**Deployment**

```python
studio.deploy(artifact, model_name, instance_type=..., num_devices=1)
studio.deploy_pipeline(
    pipeline,
    params,
    model_name,
    instance_type=...,
    num_devices=1,
    display_graph=False,
)
```

Both register (or reuse) the model, upload the artifact, and block until
compilation finishes, raising `CompilationError` or `CompilationTimeoutError`
if it doesn't.

**`studio.models`**

```python
studio.models.list()                       # → list[Model]
studio.models.get(model_id)                # → Model
studio.models.find_by_name(name)           # → Model | None
studio.models.get_id_by_name(name)         # → model_id
studio.models.update(model_id, ...)        # name, description, visibility, instance_type, ...
studio.models.activate(model_id)
studio.models.deactivate(model_id)
studio.models.set_visibility(model_id, visibility)
studio.models.display(studio.models.list())    # printable table
```

**`studio.workers`**

```python
studio.workers.running(model_id, stop_on_exit=True)   # context manager (preferred)
studio.workers.get_or_start(model_id)                 # reuse a ready worker, else start one
studio.workers.start(model_id)                        # blocks until ready
studio.workers.active(model_id)
studio.workers.stop(model_id=..., session_id=...)
studio.workers.list_sessions(model_id=..., from_date=..., to_date=...)
```

**`studio.tokens`**

```python
studio.tokens.create(model_id, name=None, save_as=None)   # save_as caches it locally
studio.tokens.load(name)                                  # load a cached token
studio.tokens.get(token)                                  # → TokenInfo
studio.tokens.list(status=..., model_id=...)
studio.tokens.assign(token_id, model_id)
studio.tokens.unassign(token_id, model_id)
studio.tokens.update(token_id, name=..., note=..., status=...)
studio.tokens.delete(token_id)
```

**`studio.account` / `studio.finance`**

```python
studio.account.get()
studio.account.update(company_name=..., contact_name=..., email=..., phone_number=...)
studio.finance.get_credits()
studio.finance.list_transactions()
```

### Instance types

Pass an `InstanceType` to `deploy`,
`deploy_pipeline`, or `models.update`:

```python
from lattica_studio.types import InstanceType

studio.deploy(artifact, "my-model", instance_type=InstanceType.G7E_2XLARGE)
```

| Instance type | Compute class |
| --- | --- |
| `G4DN_XLARGE` | GPU |
| `G5_2XLARGE` | GPU |
| `G6E_2XLARGE` | GPU |
| `G7E_2XLARGE` | GPU (default) |
| `G7E_12XLARGE` | GPU |

`num_devices` defaults to `1` and is fixed when a model is created. Redeploying
an existing model name with a different value raises `ValueError`; use a new
model name to change the device count.

### Errors

Everything raised by the SDK derives from `LatticaStudioError`, a subclass of
`RuntimeError`, so a single `except` clause covers the whole surface:

```python
from lattica_studio.exceptions import LatticaStudioError
```

Individual subclasses such as `CompilationError` and `WorkerStartupTimeoutError`
are available in the same module when you want to handle a specific failure.

## Repository layout

```
lattica_build_src/          # lattica-build package
  lattica_build/
    base_classes/           # HomomorphicPipeline, HomOp, tracing, graph printing
    operators/              # FHE, polynomial, shape, comparison, composite ops
    params/                 # FHE parameter planning, level/scale budgeting
    examples/               # runnable pipeline definitions
    build.py                # build() API and lattica-build CLI

lattica_studio_src/         # lattica-studio package
  lattica_studio/
    studio.py               # LatticaStudio entry point
    deployment.py           # deploy / compile orchestration
    resources/              # models, workers, tokens, account, finance
    types.py                # Model, Worker, TokenInfo, InstanceType
    exceptions.py
    example.py              # end-to-end reference script
```

Deeper documentation lives alongside the code:

| Topic | Doc |
| --- | --- |
| Runnable pipeline examples | [`examples/README.md`](./lattica_build_src/lattica_build/examples/README.md) |
| Pipeline API and data binding | [`base_classes/README.md`](./lattica_build_src/lattica_build/base_classes/README.md) |
| Operator composition model | [`operators/README.md`](./lattica_build_src/lattica_build/operators/README.md) |
| FHE level and scale planning | [`params/README.md`](./lattica_build_src/lattica_build/params/README.md) |

## Troubleshooting

**`CompilationError` right after deploy.** The pipeline built locally but the
backend rejected it. The exception message includes the backend's compilation
error. Check FHE parameter budgets first, see
[`params/README.md`](./lattica_build_src/lattica_build/params/README.md).

**`WorkerStartupTimeoutError`.** A worker normally becomes ready in roughly 20
seconds. If it times out, retry, or raise the timeout with
`studio.workers.start(model_id, timeout=1200)`.

**Queries fail after a redeploy.** Changing the pipeline invalidates the key
context. Create a fresh token and regenerate keys.

**`ResourceNotFoundError` from `get_id_by_name`.** The model name doesn't exist
on this account. List what's there with
`studio.models.display(studio.models.list())`.

## License

Each package is distributed under the license in its own directory:

- [`lattica_build_src/LICENSE.md`](./lattica_build_src/LICENSE.md)
- [`lattica_studio_src/LICENSE.md`](./lattica_studio_src/LICENSE.md)

Please read the applicable license before use.

Copyright © LatticaAI Inc. All rights reserved.
