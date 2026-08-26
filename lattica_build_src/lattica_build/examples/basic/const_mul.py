import torch

from lattica_build.base_classes.hom_pipeline import HomomorphicPipeline
from lattica_build.operators.arithmetic.h_const_mul import HomConstMul
from lattica_build.params.params import HomParams


INPUT_SHAPE = (8, 16)
DATA_SHAPE = (4, 1, 8, 16)


def build_pipeline() -> HomomorphicPipeline:
    generator = torch.Generator().manual_seed(0)
    pipeline = HomomorphicPipeline(
        hom=HomConstMul(dims=DATA_SHAPE),
        input_shape=INPUT_SHAPE,
    )
    pipeline.set_data(None, torch.rand(DATA_SHAPE, generator=generator))
    return pipeline


def build_params() -> HomParams:
    return HomParams(
        n=2**13,
        full_q_list_precision=((61,), (61,)),
        pt_scale=2**32,
        num_special_primes=1,
    )
