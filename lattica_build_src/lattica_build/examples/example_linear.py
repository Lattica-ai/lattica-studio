"""See `examples/README.md` for usage details."""

from __future__ import annotations

import torch

from lattica_build.base_classes.hom_pipeline import HomomorphicPipeline
from lattica_build.operators.ml.h_linear import HomLinear
from lattica_build.params.params import DecompositionType, HomParams


def build_pipeline() -> HomomorphicPipeline:
    """Build a simple single-op linear pipeline."""
    hom = HomLinear(dims=(4, 3), bias=True)
    weight = torch.tensor(
        [
            [0.25, -0.5, 1.0],
            [1.5, 0.0, -0.75],
            [-1.0, 0.5, 0.25],
            [0.125, -0.25, 0.75],
        ],
        dtype=torch.float32,
    )
    bias = torch.tensor([0.1, -0.2, 0.3, 0.0], dtype=torch.float32)
    hom.set_data(weight, bias)
    return HomomorphicPipeline(hom=hom, input_shape=(17, 1, 3,), n_axis=-1)


def build_params() -> HomParams:
    """Create params with row/col precision structure for a tiny quickstart build."""
    return HomParams(
        n=2**14,
        full_q_list_precision=((61, 35), (55,)),
        pt_scale=2**30,
        decomposition_type=DecompositionType.BV,
    )

