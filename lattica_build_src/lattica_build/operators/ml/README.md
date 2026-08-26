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

## Base vs composite in this folder

- Base ops map directly to backend op types (`HomConv`, `HomMatMul`).
- Composite ops are graphs built from base ops (`HomLinear`, `HomBatchNorm`).
- `HomAvgPool` and `HomConvBnFused` both execute as a single `HomConv` backend op after parameter setup.

## Shape and scale notes

- `HomMatMul` broadcasts input shape against configured `dims`, then removes `mul_axis`.
- `HomMatMul` may update `n_axis` when `mul_axis` intersects the packed axis.
- `HomLinear` bias shape is derived from `dims` by removing `mul_axis`.
- `HomBatchNorm` stores affine constants as `(-1, 1, 1)` for channel-first broadcasting.

## Parameterization notes

- `HomConvBnFused` computes fused tensors from Conv+BN stats before execution.
- `HomAvgPool` uses a fixed normalized kernel (`1 / (kh * kw)`).

