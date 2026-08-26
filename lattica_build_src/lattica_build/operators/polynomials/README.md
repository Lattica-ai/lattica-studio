# Polynomial Operators

This folder contains polynomial evaluation primitives and activation-style approximations.

## What is here

| Operator / Item | Description |
| --- | --- |
| `HomPolyEvalBase` / `HomPolyEval` | polynomial evaluation building blocks |
| `HomSign` | composite sign-transition approximation |
| `HomRelu`, `HomSquare`, threshold/indicator variants | activation-style polynomial approximations |
| utility modules (`polynomial_evaluation_utils.py`, `remez_utils.py`) | polynomial fitting and Remez-based approximation support |

## Role in operator graphs

- Polynomial ops are frequently used as nonlinear approximations in encrypted models.
- Some classes are base backend ops; others are composites that chain polynomial stages.

## Semantics notes

- `HomSign` approximates a discrete sign transition and is controlled by `x_accuracy` and `y_accuracy`.
- Polynomial composition often trades off precision, depth, and level consumption.

