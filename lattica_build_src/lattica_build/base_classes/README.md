# Base Classes

This folder defines the core API you use to build and serialize pipelines in `lattica-build`.

Use this layer to:
- define a pipeline graph,
- attach operator data,
- set input metadata,
- serialize a build artifact.

Use `lattica-studio` for deployment and lifecycle operations after serialization.

## Main types

- `HomomorphicPipeline`: top-level pipeline object.
- `HomOp`: base class for all operators.
- `HomValue`: encrypted-value metadata object carried through graph tracing.

## `HomomorphicPipeline` mental model

A pipeline can have up to three sections:
- `client_pre`: optional client-side preprocessing ops,
- `hom`: required homomorphic graph,
- `client_post`: optional client-side postprocessing ops.

Most workloads start with only `hom`.

Practical section behavior:
- `client_pre` and `client_post` run on clear client-side values, not encrypted values.
- `hom` is the encrypted computation section and is the required core of the pipeline.
- `client_pre` feeds the primary hom input path; if you have multiple hom inputs,
  be explicit about how those additional inputs are shaped and scaled.

## Inputs and shapes

`input_shape` can be:
- a single shape (for one hom input), or
- a dict `{input_name: shape}` for multi-input graphs.

`custom_scales` is for additional encrypted inputs (not the primary input).
Keys must match input names in `input_shape`.

`custom_n_slots` sets the sub-ring slot count of the additional encrypted inputs.
The primary input takes its own from `HomParams.n_slots`.
Keys must match input names in `input_shape`, and anything left unspecified is packed
on the full ring (`internal_n = n // 2`).
See [`params/README.md`](../params/README.md#slot-packing-internal_n-vs-n_slots).

`n_axis` controls slot-axis interpretation for homomorphic layout-sensitive ops.

When to set `n_axis`:
- set it when your homomorphic ops are layout-sensitive (for example, matrix-style ops
  where slot packing direction matters),
- leave it unset when your graph is axis-agnostic.

## Attaching tensors and constants (`set_data`)

You typically bind op tensors after constructing the graph.

At pipeline level:
- `pipeline.set_data(name_or_path, *data, section=...)`

At op level:
- `op.set_data(*data, name=...)`

### Name/path forms

`name` supports:
- `None`: current op,
- `int`: child index (for module-list style composites),
- `tuple[int, ...]`: nested child indices,
- dotted string paths (composite-specific if supported by the op tree).

Examples:
- `pipeline.set_data(0, fc1_weight)` sets data on the first child op in `hom`.
- `pipeline.set_data((2, 1), tensor)` sets data on a nested child path.
- `pipeline.set_data("encoder.proj", weight)` uses a named path when supported by your composite op.

Practical guidance:
- bind leaf-op data as close as possible to construction,
- use stable numeric paths for deeply nested composites,
- keep a small mapping in your app if the graph is large.

## What serialization produces

`pipeline.save(path, hom_params)` writes a zip artifact with:
- `hom_pipeline.json`
- `hom_pipeline.safetensors`

These files contain:
- serialized op graph,
- tracing metadata,
- tensor payloads required by the graph.

This artifact is the handoff boundary to `lattica-studio`.

## `HomValue` as metadata contract

`HomValue` represents logical encrypted values during tracing and carries:
- `tensor_shape`,
- `n_axis`,
- `active_rows`,
- `active_cols`,
- `pt_scale`.

Operator inference updates this metadata so shape and level/scale compatibility can be validated before execution.

Why this matters for users:
- most build-time failures happen because two connected ops disagree on one of these
  metadata fields,
- tracing metadata lets you catch those issues before deployment.

## Validation and common failure patterns

Common issues you will see while serializing:

- Input-name mismatch:
  - `input_shape` keys must match `hom.forward(...)` parameter names.
- Missing data for ops that require tensors:
  - call `set_data(...)` on all required leaf ops.
- Invalid custom scales:
  - `custom_scales` cannot override the primary input name.
- Invalid custom slot counts:
  - `custom_n_slots` cannot override the primary input name,
  - its keys must all appear in `input_shape`,
  - values should be powers of two, as for `HomParams.n_slots`.
- Shape incompatibility:
  - broadcast or axis constraints fail in operator inference.
- Level/scale incompatibility:
  - row/col budget mismatch or scale join mismatch (especially at add/sub joins).

Debug workflow when validation fails:
1. Build with `--print_graph` to inspect traced value flow.
2. Check the first node where `shape`, `n_axis`, or `pt_scale` diverges from expectation.
3. Verify the matching op data was bound via `set_data` on the intended op path.

## Recommended build workflow

1. Construct `hom` graph (and optional client sections).
2. Bind required data with `set_data(...)`.
3. Define `input_shape` (and optional `custom_scales` / `custom_n_slots`).
4. Choose `HomParams` (see [`params/README.md`](../params/README.md)).
5. Serialize with `pipeline.save(...)`.
6. Hand artifact to `lattica-studio`.

