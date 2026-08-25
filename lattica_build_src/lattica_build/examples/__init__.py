"""See `examples/README.md` for usage details."""

from .example_branching import BranchRejoinCompare, build_params as build_branching_params, build_pipeline as build_branching_pipeline
from .example_linear import build_params as build_linear_params, build_pipeline as build_linear_pipeline
from .example_mnist_fc import BATCH, INPUT_SHAPE, build_params as build_mnist_fc_params, build_pipeline as build_mnist_fc_pipeline

__all__ = [
    "BATCH",
    "INPUT_SHAPE",
    "BranchRejoinCompare",
    "build_branching_pipeline",
    "build_branching_params",
    "build_linear_pipeline",
    "build_linear_params",
    "build_mnist_fc_pipeline",
    "build_mnist_fc_params",
]

