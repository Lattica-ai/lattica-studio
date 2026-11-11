from lattica_build.base_classes.hom_pipeline import HomomorphicPipeline
from lattica_build.base_classes.pipeline_wrapper import PipelineWrapper
from lattica_build.operators.slots.h_rotate_sum_elementwise import HomRotateSumElementwise
from lattica_build.params.params import HomParams

N = 2 ** 8
N_SLOTS = N // 2
ROTATIONS = (0, 1, -3, 1, N_SLOTS + 2)
INPUT_SHAPE = (len(ROTATIONS), 2 * N_SLOTS)


class Pipeline(PipelineWrapper):
    def build_pipeline(self) -> HomomorphicPipeline:
        return HomomorphicPipeline(
            hom=HomRotateSumElementwise(rotations=ROTATIONS),
            input_shape=INPUT_SHAPE,
        )

    def build_params(self) -> HomParams:
        return HomParams(
            n=N,
            full_q_list_precision=((61,), (61,)),
            pt_scale=2 ** 32,
            num_special_primes=1,
        )
