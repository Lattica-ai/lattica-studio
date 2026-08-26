from lattica_build.base_classes.hom_pipeline import HomomorphicPipeline
from lattica_build.operators.slots.h_expand import HomExpand
from lattica_build.params.params import HomParams


K = 8
K_AXIS = 1
STAGE_SIZES = [2, 4]
STAGES_PER_LEVEL = 2


def build_pipeline() -> HomomorphicPipeline:
    return HomomorphicPipeline(
        hom=HomExpand(K, K_AXIS, STAGE_SIZES, STAGES_PER_LEVEL),
        input_shape=(K,),
    )


def build_params() -> HomParams:
    return HomParams(
        n=2**13,
        full_q_list_precision=((60,), (60,)),
        pt_scale=2**30,
        num_special_primes=1,
        n_slots=K,
    )
