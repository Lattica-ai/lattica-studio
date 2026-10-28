# Shape Operators

This folder contains tensor-shape transformation operators for `HomValue` metadata.

## What is here

| Operator | Description |
| --- | --- |
| `HomReshape` | base reshape op (`OP_TYPE = HomOpType.Reshape`) |
| `HomSlice` | base single-axis slice/index op (`OP_TYPE = HomOpType.Slice`) |
| `HomSqueeze` | base size-1 axis removal op (`OP_TYPE = HomOpType.Squeeze`) |
| `HomUnsqueeze` | base singleton axis insertion op (`OP_TYPE = HomOpType.Unsqueeze`) |

## Packed-axis (`n_axis`) behavior

- Shape ops honor packed-axis constraints from shape tracing.
- `HomSlice` cannot slice along `n_axis`.
- `HomSqueeze`/axis removal paths cannot remove `n_axis`.
- `HomReshape` must preserve packed-axis consistency; reshape across `n_axis` is rejected.

## Scope note

These ops transform tensor axes/lengths. For slot-axis reductions, use slot operators (for example `HomSumSlots`) rather than shape reducers.

