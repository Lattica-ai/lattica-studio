# Arithmetic Operators

This folder contains primitive arithmetic operators over `HomValue` tensors.

## What is here

| Operator | Description |
| --- | --- |
| `HomAdd` / `HomSub` | base add/sub ops (`OP_TYPE = HomOpType.Add`) |
| `HomMul` | base multiplication op (`OP_TYPE = HomOpType.Mul`) |
| `HomConstAdd` | add plaintext constant tensor (`OP_TYPE = HomOpType.ConstAdd`) |
| `HomConstMul` | multiply by plaintext constant tensor (`OP_TYPE = HomOpType.ConstMul`) |
| `HomAxisSum` | tensor-axis reduction by summation (`OP_TYPE = HomOpType.AxisSum`) |

## Shape and scale notes

- Add/mul/const ops use tensor broadcasting rules from shape tracing.
- `HomAdd` and `HomMul` of two ciphertexts take equal slot periods (`n_slots`), or a period of 1 (a value in every
  slot, e.g. a `HomSumSlots` with `reduce_output=False`) with any other, which the output takes; other pairs raise
  `ValueError`. A product of other periods would be a layout of their lcm, a sum only where the longer period has
  no zero padding; the backend takes neither. A period-1 input added to a value with zero padding raises: it
  would fill the padding.
- `HomMul` optionally reduces one axis after multiplication (`axis_sum`).
- `HomAxisSum` is for tensor-axis reduction only.
- For slot-axis reduction, use `HomSumSlots` (see [`slots/README.md`](../slots/README.md)).

