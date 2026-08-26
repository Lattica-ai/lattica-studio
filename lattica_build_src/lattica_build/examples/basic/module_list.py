import torch

from lattica_build.base_classes.hom_op import HomOp
from lattica_build.base_classes.hom_pipeline import HomomorphicPipeline
from lattica_build.base_classes.hom_value import HomValue
from lattica_build.operators.arithmetic.h_const_add import HomConstAdd
from lattica_build.operators.composite.module_list import ModuleListHomOp
from lattica_build.params.params import HomParams


INPUT_SHAPE = (8, 16)
NUM_OPS = 3


class OpListBlock(HomOp):
    def __init__(self) -> None:
        super().__init__()
        self.ops = ModuleListHomOp(
            [HomConstAdd(dims=INPUT_SHAPE) for _ in range(NUM_OPS)]
        )

    def forward(self, x: HomValue) -> HomValue:
        for op in self.ops:
            x = op(x)
        return x


def build_pipeline() -> HomomorphicPipeline:
    pipeline = HomomorphicPipeline(hom=OpListBlock(), input_shape=INPUT_SHAPE)
    generator = torch.Generator().manual_seed(0)
    for index in range(NUM_OPS):
        pipeline.set_data(f"ops.{index}", torch.rand(INPUT_SHAPE, generator=generator))
    return pipeline


def build_params() -> HomParams:
    return HomParams(
        n=2**13,
        full_q_list_precision=((61,), (61,)),
        pt_scale=2**32,
        num_special_primes=1,
    )
