"""See `examples/README.md` for usage details."""

from __future__ import annotations

from pathlib import Path

import torch

from lattica_build.base_classes.hom_pipeline import HomomorphicPipeline
from lattica_build.operators.client_ops import Softmax
from lattica_build.operators.composite.sequential import SequentialHomOp
from lattica_build.operators.arithmetic.h_const_add import HomConstAdd
from lattica_build.operators.arithmetic.h_const_mul import HomConstMul
from lattica_build.operators.ml.h_linear import HomLinear
from lattica_build.operators.polynomials.h_square import HomSquare
from lattica_build.operators.shape.h_reshape import HomReshape
from lattica_build.params.params import DecompositionType, HomParams


BATCH = 100
INPUT_SHAPE = (BATCH, 1, 1, 28 * 28)


def _load_model_weights() -> tuple[torch.Tensor, torch.Tensor]:
    """Load pre-trained MNIST FC weights from the packaged checkpoint."""
    model_path = Path(__file__).with_name("data") / "mnist_fc_model.pt"
    model = torch.load(model_path, weights_only=True, map_location="cpu")
    return model["l1.weight"], model["l2.weight"]


def build_pipeline() -> HomomorphicPipeline:
    """Build FC -> square -> FC with client reshape and softmax postprocess."""
    fc1_weight, fc2_weight = _load_model_weights()

    pipeline = HomomorphicPipeline(
        client_pre=[
            HomConstMul(dims=(1,)).set_data(1.0 / (255.0 * 0.3081)),
            HomConstAdd(dims=(1,)).set_data(-0.1307 / 0.3081),
            HomReshape(INPUT_SHAPE),
        ],
        hom=SequentialHomOp(
            HomLinear(fc1_weight.shape, bias=False, with_modswitch=False),
            HomSquare(with_modswitch=False),
            HomLinear(fc2_weight.shape, bias=False, with_modswitch=False),
        ),
        client_post=[
            Softmax(-1),
        ],
        n_axis=0,
        input_shape=INPUT_SHAPE,
    )

    pipeline.set_data(0, fc1_weight)
    pipeline.set_data(2, fc2_weight)
    return pipeline


def build_params() -> HomParams:
    """Build params used by the public MNIST FC demo."""
    return HomParams(
        full_q_list_precision=((61,), (45,)),
        n=2**8,
        pt_scale=2**20,
        decomposition_type=DecompositionType.BV,
    )
