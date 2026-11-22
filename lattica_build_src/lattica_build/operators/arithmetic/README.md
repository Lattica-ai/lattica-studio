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
  slot) with any other, which the output takes; other pairs raise `ValueError`. Adding a period-1 input to a zero-padded value also raises: it would fill the padding.
- `HomMul` optionally reduces one axis after multiplication (`axis_sum`).
- `HomAxisSum` is for tensor-axis reduction only.
- For slot-axis reduction, use `HomSumSlots` (see [`slots/README.md`](../slots/README.md)).

