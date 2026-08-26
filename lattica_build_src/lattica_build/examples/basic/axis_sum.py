from lattica_build.base_classes.hom_pipeline import HomomorphicPipeline
from lattica_build.operators.arithmetic.h_axis_sum import HomAxisSum
from lattica_build.params.params import HomParams


INPUT_SHAPE = (16, 32, 128)
DIM = 1
KEEP_DIM = True


def build_pipeline() -> HomomorphicPipeline:
    return HomomorphicPipeline(
        hom=HomAxisSum(dim=DIM, keep_dim=KEEP_DIM),
        input_shape=INPUT_SHAPE,
    )


def build_params() -> HomParams:
    return HomParams(
        n=2**13,
        full_q_list_precision=((61,), (61,)),
        pt_scale=2**32,
        num_special_primes=1,
    )
