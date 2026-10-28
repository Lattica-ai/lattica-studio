# Operators

This package is the public gateway for building encrypted computation graphs.
It defines the operator model used to transform encrypted values during inference.

Scope note: this package is about graph construction and serialization.
Deployment lifecycle (upload/deploy/run/monitor) is handled by
`lattica-studio`.

## What lives here

| Area | Purpose |
| --- | --- |
| `client_ops.py` | client-only operations, preprocessing, and helpers around encrypted execution |
| `arithmetic/`, `comparison/`, `fhe/`, `ml/`, `polynomials/`, `shape/`, `slots/` | homomorphic operators used in encrypted graph execution |
| `composite/` | operators that combine other operators into larger reusable graph blocks |

## Core concepts

### `HomValue`

`HomValue` is the value carried through a graph step.
In practice, you can think of it as the encrypted tensor/ciphertext plus metadata that tracks:

- shape
- level
- scale

Operators may change one or more of these properties, and correctness depends on keeping them consistent.

### `HomOp`

`HomOp` is the execution unit in the encrypted graph.
Each operator is expected to provide three behaviors:

- `forward(...)`: runtime encrypted transformation
- `infer_output_shape(...)`: static shape propagation
- `infer_output_level_and_scale(...)`: static level/scale propagation

This lets a graph be validated and planned before encrypted execution.

## Base ops vs composite ops

### Base ops

Base ops are the primitive homomorphic operations.
They are identifiable by `OP_TYPE = HomOpType.<X>` and correspond to concrete backend operation types.

Examples include arithmetic, comparisons, shape transforms, slot operations, and FHE lifecycle steps.

Base ops are typically where explicit level/scale and shape transition rules are implemented.

### Composite ops

Composite ops are built by wiring multiple ops into a higher-level graph behavior.
They usually orchestrate existing ops rather than introducing a new backend primitive.

Because they are composed from base ops, output shape and level/scale behavior are generally inferred from the underlying graph and often do not need custom override logic.

## Composition model

A graph executes as connected `HomOp` nodes operating on `HomValue` objects.
Composition is not limited to a single sequential chain; operators can be arranged
into reusable graph structures.

1. `forward(...)` executes encrypted computation step-by-step.
2. `infer_output_shape(...)` predicts tensor compatibility across the graph.
3. `infer_output_level_and_scale(...)` predicts ciphertext viability (level/scale budget) across the graph.

This split between runtime execution and static inference is the key design pattern in this operators package.

For a detailed explanation of level-state metadata (`active_rows`,
`active_cols`, `pt_scale`) and how ops consume levels, see
[`params/README.md`](../params/README.md).

## Handoff to deployment

After your operator graph is built and serialized, pass the generated artifact
to `lattica-studio` for backend lifecycle control.

## Package map

| Package | Purpose |
| --- | --- |
| [`arithmetic/`](arithmetic/README.md) | add/sub/mul and constant arithmetic variants |
| [`comparison/`](comparison/README.md) | comparison-style approximations, `min`, `max`, and related utilities |
| [`composite/`](composite/README.md) | composition containers such as sequential module lists |
| [`fhe/`](fhe/README.md) | bootstrap, ring switch, mod switch lifecycle operations |
| [`ml/`](ml/README.md) | model-style graph blocks (conv, matmul, linear, pooling) |
| [`polynomials/`](polynomials/README.md) | polynomial evaluation and activation-style approximations |
| [`shape/`](shape/README.md) | reshape/slice/squeeze/unsqueeze transforms |
| [`slots/`](slots/README.md) | rotate, running/sum reductions, slot expansions |




