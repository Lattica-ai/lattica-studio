from lattica_build.base_classes.hom_pipeline import HomomorphicPipeline
from lattica_build.operators.slots.h_sum_slots import HomSumSlots
from lattica_build.params.params import HomParams
INPUT_SHAPE = (28,)
NB_COLS = 4
STAGE_SIZES = [2, 4]
from lattica_build.base_classes.pipeline_wrapper import PipelineWrapper

class Pipeline(PipelineWrapper):

    def build_pipeline(self) -> HomomorphicPipeline:
        return HomomorphicPipeline(
            hom=HomSumSlots(nb_cols=NB_COLS, stage_sizes=STAGE_SIZES),
            input_shape=INPUT_SHAPE,
        )

    def build_params(self) -> HomParams:
        return HomParams(
            n=2 ** 13,
            full_q_list_precision=((61,), (61, 30), (61,)),
            pt_scale=2 ** 44,
            num_special_primes=1,
            # 28 values in 32 slots: slots 28..31 are zero padding, so the last of
            # each sum's 32 / NB_COLS slots adds nothing
            n_slots=32,
        )
