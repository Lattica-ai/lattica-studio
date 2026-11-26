import torch
from lattica_build.base_classes.hom_pipeline import HomomorphicPipeline
from lattica_build.operators.ml.h_ct_mat_mul import HomCtMatMul
from lattica_build.params.params import HomParams
D = 8
from lattica_build.base_classes.pipeline_wrapper import PipelineWrapper

class Pipeline(PipelineWrapper):

    def build_pipeline(self) -> HomomorphicPipeline:
        return HomomorphicPipeline(
            hom=HomCtMatMul(D),
            input_shape={"a": (D * D,), "b": (D * D,)},
            custom_n_slots={"b": D * D},
        )

    def build_params(self) -> HomParams:
        return HomParams(
            n=2 ** 13,
            n_slots=D * D,
            full_q_list_precision=((60, 30), (60, 30), (60, 30)),
            pt_scale=2 ** 30,
            num_special_primes=2,
        )

    def _set_custom_data(self) -> None:
        self.custom_data = {"b": torch.rand(D * D) - 0.5}
