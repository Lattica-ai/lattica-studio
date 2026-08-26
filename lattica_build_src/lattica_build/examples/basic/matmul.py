import torch

from lattica_build.base_classes.hom_pipeline import HomomorphicPipeline
from lattica_build.operators.ml.h_mat_mul import HomMatMul
from lattica_build.params.params import HomParams


MATRIX_DIMS = (16, 128, 1)
INPUT_SHAPE = (128, 512)


def build_pipeline() -> HomomorphicPipeline:
    pipeline = HomomorphicPipeline(
        hom=HomMatMul(MATRIX_DIMS, mul_axis=1),
        input_shape=INPUT_SHAPE,
    )
    generator = torch.Generator().manual_seed(0)
    pipeline.set_data(None, torch.rand(MATRIX_DIMS, generator=generator))
    return pipeline


def build_params() -> HomParams:
    return HomParams(
        n=2**13,
        full_q_list_precision=((61,), (61,)),
        pt_scale=2**32,
        num_special_primes=1,
    )
