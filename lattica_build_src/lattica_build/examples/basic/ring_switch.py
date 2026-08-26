import torch

from lattica_build.base_classes.hom_pipeline import HomomorphicPipeline
from lattica_build.operators.arithmetic.h_const_mul import HomConstMul
from lattica_build.operators.composite.sequential import SequentialHomOp
from lattica_build.operators.fhe.h_ring_switch import HomRingSwitch
from lattica_build.operators.polynomials.h_square import HomSquare
from lattica_build.params.params import HomParams


LOG_N = 13
LOG_N_SUBRING = 4
INPUT_SCALE = 2**30


def build_pipeline(
    log_n: int = LOG_N,
    log_n_subring: int = LOG_N_SUBRING,
) -> HomomorphicPipeline:
    input_shape = (3, 2 ** (log_n_subring - 1))
    switched_shape = (3, 2 ** (log_n - 1))
    pipeline = HomomorphicPipeline(
        hom=SequentialHomOp(
            HomRingSwitch(log_n_subring=log_n_subring),
            HomSquare(),
            HomConstMul(dims=switched_shape),
        ),
        input_shape=input_shape,
    )
    generator = torch.Generator().manual_seed(0)
    pipeline.set_data(2, torch.rand(switched_shape, generator=generator))
    return pipeline


def build_params(
    log_n: int = LOG_N,
    input_scale: int = INPUT_SCALE,
) -> HomParams:
    return HomParams(
        full_q_list_precision=((60, 30),),
        n=2**log_n,
        sk_hw=192,
        pt_scale=input_scale,
        num_special_primes=6,
    )
