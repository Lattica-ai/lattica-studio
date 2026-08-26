from lattica_build.base_classes.hom_pipeline import HomomorphicPipeline
from lattica_build.operators.slots.h_running_sum import HomRunningSum
from lattica_build.params.params import HomParams


K = 8
STAGE_SIZES = [4, 2]
STAGES_PER_LEVEL = 2
NUM_BLOCKS = 3


def build_pipeline() -> HomomorphicPipeline:
    return HomomorphicPipeline(
        hom=HomRunningSum(
            k=K,
            stage_sizes=STAGE_SIZES,
            stages_per_level=STAGES_PER_LEVEL,
        ),
        input_shape=(2**5 * NUM_BLOCKS,),
    )


def build_params() -> HomParams:
    return HomParams(
        n=2**13,
        full_q_list_precision=((60,), (60,)),
        pt_scale=2**30,
        num_special_primes=1,
    )
