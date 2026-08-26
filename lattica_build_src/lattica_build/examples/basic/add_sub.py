from lattica_build.base_classes.hom_op import HomOp
from lattica_build.base_classes.hom_pipeline import HomomorphicPipeline
from lattica_build.base_classes.hom_value import HomValue
from lattica_build.operators.arithmetic.h_add import HomAdd
from lattica_build.operators.arithmetic.h_sub import HomSub
from lattica_build.params.params import HomParams


INPUT_SHAPE_A = (17, 1, 128)
INPUT_SHAPE_B = (1, 42, 128)


class AddSubBlock(HomOp):
    def __init__(self) -> None:
        super().__init__()
        self.add = HomAdd()
        self.sub = HomSub()

    def forward(self, x: HomValue, y: HomValue) -> HomValue:
        return self.sub(self.add(x, x), y)


def build_pipeline() -> HomomorphicPipeline:
    return HomomorphicPipeline(
        hom=AddSubBlock(),
        input_shape={"x": INPUT_SHAPE_A, "y": INPUT_SHAPE_B},
    )


def build_params() -> HomParams:
    return HomParams(
        n=2**13,
        full_q_list_precision=((61,), (61,)),
        pt_scale=2**32,
        num_special_primes=1,
    )
