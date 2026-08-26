# Lattica Build examples

These examples define and serialize homomorphic pipelines using only
`lattica-build`. They do not deploy workloads and do not depend on
`lattica-studio` or the Lattica backend runtime.

Every runnable module exposes the two functions accepted by the build CLI:

- `build_pipeline()` constructs a `HomomorphicPipeline` and binds constants.
- `build_params()` returns the matching `HomParams`.

## Start with a small example

Build a single-operator pipeline:

```bash
lattica-build \
  --pipeline-module lattica_build.examples.basic.power \
  --out /tmp/power.zip \
  --print_graph
```

Build an example with bound matrix data:

```bash
lattica-build \
  --pipeline-module lattica_build.examples.basic.matmul \
  --out /tmp/matmul.zip
```

Each output zip contains `hom_pipeline.json` and
`hom_pipeline.safetensors`. The command also prints a JSON summary with the
artifact path, graph size, tensor size, and zip members.

## Example catalog

### Basic operators and composition

These examples use an explicit ring degree of `n=2**13`.
Their module names begin with `lattica_build.examples.basic`.

| Module | Demonstrates |
| --- | --- |
| `power` | Squaring without an automatic modulus switch |
| `add_sub` | Two named inputs and a branch of add/subtract operations |
| `const_mul` | Multiplication by a bound plaintext tensor |
| `matmul` | Matrix multiplication with deterministic bound data |
| `axis_sum` | Reduction along one tensor axis |
| `rotate_sum` | Rotation and summation over selected offsets |
| `running_sum` | Staged running sums across packed slots |
| `sum_slots` | Staged slot summation |
| `expand` | Client-side repeat followed by slot expansion |
| `reshape` | Tensor reshape with an explicit slots axis |
| `squeeze` | Removing a size-one tensor dimension |
| `unsqueeze` | Adding a size-one tensor dimension |
| `slice` | Client and homomorphic tensor slicing |
| `mod_switch` | An explicit modulus switch after a square |
| `ring_switch` | Ring switching followed by square and constant multiply |
| `module_list` | A dynamically populated `ModuleListHomOp` |

### Guided and application examples

These modules live under `lattica_build.examples.advanced`.

| Module | Demonstrates |
| --- | --- |
| `linear` | A linear layer with explicit weight and bias data |
| `branching` | Branch/rejoin topology with a comparison |
| `sharpen` | Grouped convolution for image sharpening |
| `mnist_fc` | Reshape, two FC layers, square, and client softmax |
| `bootstrap` | Two bootstrapping operations |
| `bitonic_sort` | Composite compare/exchange stages with bootstrapping |
| `resnet20` | A full ResNet-20 pipeline using downloaded weights |

`resnet20` downloads a pretrained model when its pipeline is built. All basic
examples are self-contained.

## Use the Python API

```python
from lattica_build import build
from lattica_build.examples.advanced import mnist_fc

artifact = build(
    mnist_fc.build_pipeline(),
    mnist_fc.build_params(),
    "mnist-fc.zip",
    display_graph=True,
)
print(artifact.path)
```

## Build a local file

The CLI accepts either an importable module or a standalone Python file. A
local file must define the same `build_pipeline()` and `build_params()` hooks:

```bash
lattica-build /path/to/my_pipeline.py --out /tmp/my-pipeline.zip
```

Use `--pipeline-module` for installed packages and the positional file form
while iterating on a script.

## Inspect a compiled graph

Print during a build with `--print_graph`, or inspect an existing artifact:

```bash
python -m lattica_build.base_classes.print_graph_structure /tmp/power.zip
python -m lattica_build.base_classes.print_graph_structure /path/to/hom_pipeline.json
```

The output is grouped into `client_pre`, `hom`, and `client_post`. Node ids
show composition order, value ids show data flow, `shape` includes the traced
slots axis (`@`), and `scale` shows plaintext scale metadata.

For debugging, set a breakpoint in an example's `build_pipeline()` or an
operator's `forward()` method and inspect `tensor_shape`, `n_axis`,
`active_rows`, `active_cols`, and `pt_scale`.

## Deployment boundary

Building ends with the serialized artifact. Pass that artifact to
`lattica-studio` only in deployment code; consumers that merely construct or
test pipelines need `lattica-build` but not `lattica-studio`.
