# Examples

This folder contains build-only examples for `lattica-build`.

Use these modules to construct and serialize homomorphic pipelines. For deployment
and runtime lifecycle, pass the resulting artifact to `lattica-studio`.

## What the examples compute

- `lattica_build.examples.example_branching`: defines a branching graph for `compare(x**2, x - 0.2)` on input shape `(4,)`.
- `lattica_build.examples.example_linear`: defines a linear graph `y = W x + b` with explicit `set_data(weight, bias)` on input shape `(3,)`.
- `lattica_build.examples.example_mnist_fc`: defines the MNIST FC demo graph (reshape -> linear -> square -> linear -> softmax) with packaged weights.
- `lattica_build.build`: CLI entrypoint that accepts a pipeline-definition module or file and builds it.

Each example writes a zip artifact containing:
- `hom_pipeline.json`
- `hom_pipeline.safetensors`

## How to run

```bash
lattica-build --pipeline-module lattica_build.examples.example_branching --out /tmp/quickstart_branching.zip
lattica-build --pipeline-module lattica_build.examples.example_linear --out /tmp/quickstart_linear.zip
lattica-build --pipeline-module lattica_build.examples.example_mnist_fc --out /tmp/quickstart_mnist_fc.zip
```

## Module vs file input

`lattica-build` supports two ways to point at a pipeline definition:

- `--pipeline-module <import.path>`: use this when the pipeline lives in an importable module
  (packaged examples, or your own installed package).
- `pipeline_file.py` (positional path): use this when you have a standalone local script.

Both forms require the target to define:
- `build_pipeline()`
- `build_params()`

Example using your own local file:

```bash
lattica-build /path/to/my_pipeline.py --out /tmp/my_pipeline.zip
```

Expected result:
- command exits successfully,
- output zip file exists,
- JSON summary prints artifact path, graph size, tensor size, and zip members.

## Print the compiled graph

You have two ways to print the graph.

1) Print immediately after build:

```bash
lattica-build --pipeline-module lattica_build.examples.example_branching --out /tmp/quickstart_branching.zip --print_graph
lattica-build --pipeline-module lattica_build.examples.example_linear --out /tmp/quickstart_linear.zip --print_graph
lattica-build --pipeline-module lattica_build.examples.example_mnist_fc --out /tmp/quickstart_mnist_fc.zip --print_graph
```

2) Print from an existing artifact or standalone JSON:

```bash
python -m lattica_build.base_classes.print_graph_structure /tmp/quickstart_branching.zip
python -m lattica_build.base_classes.print_graph_structure /path/to/hom_pipeline.json
```

The graph printer implementation lives in [`base_classes/print_graph_structure.py`](../base_classes/print_graph_structure.py).

## How to interpret graph output

- Section headers (`client_pre`, `hom`, `client_post`) show where each operation runs.
- Tree indentation and node ids (`[0.1]`, `[0.4.2]`) show composition and execution order.
- Op names (`MatMul`, `ConstAdd`, `Compare`, `PolyEval`, ...) identify the primitive/composite steps used by your graph.
- `inputs`/`output` value ids (`v0`, `v1`, ...) show how values flow between nodes.
- `shape=(...)` is the traced ciphertext/value shape; `@` marks the slots axis (`n_axis`).
- `scale=...` is the plaintext scale metadata (`pt_scale`) at that point in the graph.

Use this output to answer practical questions such as:
- where scales change,
- where branch outputs reconnect,
- and which op introduces an unexpected shape or metadata transition.

## Debugging with breakpoints in `forward`

You can debug value propagation by stepping through `forward` code in the example ops.

- Branching example: set breakpoints in `BranchRejoinCompare.forward` in [`examples/example_branching.py`](example_branching.py).
- Linear example: set breakpoints in `build_pipeline` in [`examples/example_linear.py`](example_linear.py), and in `HomLinear` internals if needed.
- Run `python -m lattica_build.build` in debug mode with the target pipeline file argument.

When inspecting intermediate values, focus on:
- `tensor_shape`,
- `n_axis`,
- `active_rows` / `active_cols`,
- `pt_scale`.

This is often the fastest way to understand why a composed graph behaves differently than expected.

## After examples

Once an example artifact builds successfully, the next step is to hand that artifact
to `lattica-studio` for deployment/runtime lifecycle operations.

