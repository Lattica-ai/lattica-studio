# Lattica Build

`lattica-build` is the pipeline-construction SDK for Lattica workloads.

Use it to define a homomorphic computation graph, attach required tensors/constants,
validate metadata compatibility before execution, and emit a deployable artifact.

This package is for pipeline construction only. Use `lattica-studio` for deployment
and runtime lifecycle operations.

Clear execution is also available locally through `HomOp.forward_clear(...)` and
`HomomorphicPipeline.forward_clear(...)`. It uses ordinary torch tensors and the
data attached to operators, and does not modify or serialize the homomorphic graph.
Pass the same `HomParams` used for the build when clear execution includes
packing-sensitive client operators such as `Repeat`.

## Installation

Requirements:

- Python `>=3.11`

Install:

```bash
pip install lattica-build
```

## What this produces

The output is a serialized pipeline artifact (zip) containing:

- `hom_pipeline.json`
- `hom_pipeline.safetensors`

You hand this artifact to `lattica-studio`.

## 60-second quickstart

Build the packaged branching example:

```bash
lattica-build --pipeline-module lattica_build.examples.advanced.branching --out /tmp/quickstart_branching.zip
```

Expected result:

- command exits successfully,
- `/tmp/quickstart_branching.zip` exists,
- command prints a JSON summary including members `hom_pipeline.json` and `hom_pipeline.safetensors`.


## First-time docs path

| Topic | Doc |
| --- | --- |
| Repository overview and end-to-end workflow | [`../README.md`](../README.md) |
| Quickstart and runnable examples | [`examples/README.md`](lattica_build/examples/README.md) |
| Pipeline API and data binding (`HomomorphicPipeline`, `set_data`, serialization) | [`base_classes/README.md`](lattica_build/base_classes/README.md) |
| Operator composition model | [`operators/README.md`](lattica_build/operators/README.md) |
| Level/scale budget planning | [`params/README.md`](lattica_build/params/README.md) |

## Python API

The same build flow is available without the CLI:

```python
from lattica_build import build
from lattica_build.examples.advanced import branching

artifact = build(
    branching.build_pipeline(),
    branching.build_params(),
    "branching.zip",
    display_graph=True,
)

print(artifact.path)
```

`build(...)` returns a `BuildArtifact`; pass that object directly to
`LatticaStudio.deploy(...)`.

## Build-to-deploy workflow

1. Build and serialize a pipeline with `lattica-build`.
2. Pass the artifact to `lattica-studio`.
3. Use `lattica-studio` for deployment/runtime lifecycle operations.
