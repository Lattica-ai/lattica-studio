# Comparison Operators

This folder contains comparison-style operators built from polynomial sign approximation and arithmetic composition.

## What is here

| Operator / Item | Description |
| --- | --- |
| `HomCompare` | composite greater-than indicator based on sign approximation |
| `HomMinMax` | composite min/max core built on `HomCompare` |
| `HomMin` | specialization of `HomMinMax` for minimum |
| `HomMax` | specialization of `HomMinMax` for maximum |
| `comparison_utils.py` | calibration helpers for accuracy conversion |

## Output semantics

- `HomSign` (used internally) approximates a discrete sign transition.
- `HomCompare` computes `(approx_sign(a - b) + 1) / 2`.
- Output is near 1 for `a > b`, near 0 for `a < b`, with transition around equality.

## Shape and usage notes

- Comparison-based binary ops currently run elementwise on inputs with identical shapes.
- Prefer `HomMin`/`HomMax` in application graphs; `HomMinMax` is the generic core.
- Current min/max calibration in this folder assumes values in `[0, 1]`.

## If your values are outside `[0, 1]`

- Normalize into `[0, 1]` before comparison/min/max.
- Apply compare/min/max in normalized space.
- If needed, map results back to your original value range.

This keeps approximation behavior predictable and avoids calibration mismatch.

