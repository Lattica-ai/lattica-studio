# ML Operators

This folder contains model-oriented operators used to build encrypted inference graphs.

## What is here

| Operator | Description |
| --- | --- |
| `HomConv` | base convolution op (`OP_TYPE = HomOpType.Conv`) |
| `HomMatMul` | base matmul-like op (`OP_TYPE = HomOpType.MatMul`) |
| `HomAvgPool` | average pooling implemented as fixed depthwise convolution |
| `HomLinear` | composite op (`HomMatMul` + optional bias add) |
| `HomBatchNorm` | composite op (`const_mul` + `const_add`) |
| `HomConvBnFused` | pre-fuses Conv+BatchNorm tensors, then runs one conv op |
| `HomMuxConv` | convolution on the multiplexed packing (`OP_TYPE = HomOpType.MuxConv`) |
| `HomMuxConvBnFused` | `HomMuxConv` with a BatchNorm folded into its weight and bias in `set_data` |

## Base vs composite in this folder

- Base ops map directly to backend op types (`HomConv`, `HomMuxConv`, `HomMatMul`).
- Composite ops are graphs built from base ops (`HomLinear`, `HomBatchNorm`).
- `HomAvgPool` and `HomConvBnFused` both execute as a single `HomConv` backend op after parameter setup.

## Shape and scale notes

- `HomMatMul` broadcasts input shape against configured `dims`, then removes `mul_axis`.
- `HomMatMul` may update `n_axis` when `mul_axis` intersects the packed axis.
- `HomLinear` bias shape is derived from `dims` by removing `mul_axis`.
- `HomBatchNorm` stores affine constants as `(-1, 1, 1)` for channel-first broadcasting.
- `HomMuxConv` takes the slots on the last axis, packed by its `MuxLayout`. Axes before them are a batch of maps,
  one ciphertext each, all convolved with the same weights; the output is `(*batch, output slots)`.
- A strided `HomMuxConv` packs its output into the smaller map's period by default, which makes its rotations
  depend on the output row. With `keep_gap=True` (and `C_out = stride * C_in`) the output keeps the gap the stride
  opens: its channels fill the skipped columns and its rows stay `stride` times as far apart. Far fewer rotations,
  but the map keeps the input's period. The convs after it take that `row_gap` and keep the gap too. The larger
  period costs the slot-count-dependent ops after it (bootstraps, the plaintexts' size), so it pays off where
  convs dominate. The conv leaves the slots between the map's rows at zero; an op that sends zero elsewhere (a
  polynomial with a constant term) fills them, which later convs ignore but a sum over all slots does not.

## Parameterization notes

- `HomConvBnFused` computes fused tensors from Conv+BN stats before execution.
- `HomAvgPool` uses a fixed normalized kernel (`1 / (kh * kw)`).

