from lattica_build.base_classes.hom_pipeline import HomomorphicPipeline
from lattica_build.operators.slots.h_sum_slots import HomSumSlots
from lattica_build.params.params import HomParams


INPUT_SHAPE = (32,)
K = 8
STAGE_SIZES = [2, 4]


def build_pipeline() -> HomomorphicPipeline:
    return HomomorphicPipeline(
        hom=HomSumSlots(k=K, stage_sizes=STAGE_SIZES),
        input_shape=INPUT_SHAPE,
    )


def build_params() -> HomParams:
    return HomParams(
        n=2**13,
        full_q_list_precision=((61,), (61, 30), (61,)),
        pt_scale=2**44,
        num_special_primes=1,
    )
