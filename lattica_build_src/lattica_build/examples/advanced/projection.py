"""See `examples/README.md` for usage details."""
import torch

from lattica_build.base_classes.hom_pipeline import HomomorphicPipeline
from lattica_build.base_classes.pipeline_wrapper import PipelineWrapper
from lattica_build.operators.ml.h_mat_mul import HomMatMul
from lattica_build.params.params import HomParams


class Pipeline(PipelineWrapper):
    """A language model's projection: a width x width plaintext weight times one ciphertext per token, the token's
    width values in a slot period of width. The weight is kept as its integer coefficients (`pack_at_apply`)."""

    width = 4096
    tokens = 8

    def build_pipeline(self) -> HomomorphicPipeline:
        pipeline = HomomorphicPipeline(
            hom=HomMatMul((self.width, 1, self.width), mul_axis=-1, pack_at_apply=True),
            input_shape=(self.tokens, self.width),
        )
        weight = torch.randn(self.width, 1, self.width, generator=torch.Generator().manual_seed(0))
        pipeline.set_data(None, weight / self.width ** 0.5)
        return pipeline

    def build_params(self) -> HomParams:
        return HomParams(
            n=2 ** 16,
            n_slots=self.width,
            full_q_list_precision=((60, 30), (60, 30), (60, 30)),
            pt_scale=2 ** 30,
            num_special_primes=2,
        )

    def _set_example_pt(self) -> torch.Tensor:
        return torch.rand(self.tokens, self.width, dtype=torch.float64) - 0.5
