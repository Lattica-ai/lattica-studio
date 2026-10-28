# Composite Operators

This folder contains composition containers used to build operator graphs from reusable blocks.

## What is here

| Operator | Description |
| --- | --- |
| `ModuleListHomOp` | container of `HomOp` nodes (no direct forward computation) |
| `SequentialHomOp` | applies child ops in order |

## Composition model

- Composite ops orchestrate other ops and usually do not define a new backend `OP_TYPE`.
- They are building blocks for graph structure, not just a flat operator list.
- `SequentialHomOp` is one composition pattern; larger graph patterns can be built on top.

