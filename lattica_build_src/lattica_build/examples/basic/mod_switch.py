from lattica_build.base_classes.hom_op import HomOp
from lattica_build.base_classes.hom_pipeline import HomomorphicPipeline
from lattica_build.base_classes.hom_value import HomValue
from lattica_build.operators.fhe.h_mod_switch import HomModSwitch
from lattica_build.operators.polynomials.h_square import HomSquare
from lattica_build.params.params import HomParams


INPUT_SHAPE = (4,)


class _ModSwitchPipeline(HomOp):
    def __init__(self) -> None:
        super().__init__()
        self.square = HomSquare(with_modswitch=False)
        self.mod_switch = HomModSwitch(variant=1, cols_to_drop=[0])

    def forward(self, x: HomValue) -> HomValue:
        return self.mod_switch(self.square(x))


def build_pipeline() -> HomomorphicPipeline:
    return HomomorphicPipeline(hom=_ModSwitchPipeline(), input_shape=INPUT_SHAPE)


def build_params() -> HomParams:
    return HomParams(
        n=2**13,
        full_q_list_precision=((60, 30),),
        pt_scale=2**30,
        num_special_primes=1,
    )
