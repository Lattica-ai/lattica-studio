# FHE Operators

This folder contains FHE lifecycle operators that change ciphertext state (levels/ring/packing).

## What is here

| Operator | Description |
| --- | --- |
| `Bootstrap` | base bootstrap op (`OP_TYPE = HomOpType.Bootstrap`) |
| `HomModSwitch` | base modulus-switch op (`OP_TYPE = HomOpType.ModSwitch`) |
| `HomRingSwitch` | base ring-switch op (`OP_TYPE = HomOpType.RingSwitch`) |

## Operational intent

- `Bootstrap` refreshes ciphertext level budget and can optionally force a target output scale.
- `HomModSwitch` drops selected rows/columns from the active modulus chain and updates scale accordingly.
- `HomRingSwitch` changes packing/ring context and refreshes level state.

## Shape and scale notes

- `HomRingSwitch` leaves the shape unchanged.
- `Bootstrap`/`HomRingSwitch` use refreshed level state from bootstrap tracing.
- `HomModSwitch` consumes level by dropping primes and scales down by dropped-prime product.

