import torch
from lattica_build.base_classes.hom_op import HomOp
from lattica_build.base_classes.hom_pipeline import HomomorphicPipeline
from lattica_build.base_classes.hom_value import HomValue
from lattica_build.operators.shape.h_slice import HomSlice
from lattica_build.params.params import HomParams
INPUT_SHAPE = (4, 3, 50)
N_AXIS = 2

class _HomSliceBlock(HomOp):

    def forward(self, x: HomValue) -> HomValue:
        x = x[1:3, ...]
        return x[:, 2, :]
from lattica_build.base_classes.pipeline_wrapper import PipelineWrapper

class Pipeline(PipelineWrapper):

    def build_pipeline(self) -> HomomorphicPipeline:
        return HomomorphicPipeline(
            client_pre=[HomSlice(dim=2, key=slice(10, 40, 2))],
            hom=_HomSliceBlock(),
            input_shape=INPUT_SHAPE,
            n_axis=N_AXIS,
        )

    def build_params(self) -> HomParams:
        return HomParams(
            n=2 ** 13,
            full_q_list_precision=((61,), (61,)),
            pt_scale=2 ** 32,
            num_special_primes=1,
        )

    def compute_expected(self, example_pt: torch.Tensor) -> torch.Tensor:
        """The client slice on dim 2 followed by the two homomorphic slices."""
        return example_pt[1:3, 2, 10:40:2]
