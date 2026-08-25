from lattica_build.base_classes.hom_op import HomOp
from lattica_build.base_classes.hom_pipeline import HomomorphicPipeline
from lattica_build.base_classes.hom_value import HomValue
from lattica_build.operators.fhe.h_bootstrap import Bootstrap
from lattica_build.params.bootstrapping_params import BootstrappingVariant
from lattica_build.params.params import HomParams

LOG_N = 10
LOG_N_SUBRING = 8
INPUT_SCALE = 2 ** 45


def build_pipeline() -> HomomorphicPipeline:
    """Construct a bootstrapping homomorphic pipeline."""
    class g(HomOp):
        def __init__(self, log_n_subring):
            super().__init__()
            self.boot_1 = Bootstrap(log_n_subring=log_n_subring)
            self.boot_2 = Bootstrap(log_n_subring=log_n_subring)

        def forward(self, x: HomValue) -> HomValue:
            x = self.boot_1(x)
            x = self.boot_2(x)
            return x

    hom_pipeline = HomomorphicPipeline(
        hom=g(log_n_subring=LOG_N_SUBRING),
        input_shape=(2 ** (LOG_N - 1),)
    )

    return hom_pipeline


def build_params() -> HomParams:
    return HomParams(
        n=2 ** LOG_N,
        full_q_list_precision=(
            (60,),
            (60,),
            (60,),
            (60,),
        ),
        pt_scale=INPUT_SCALE,
        sk_hw=192,
        num_special_primes=6,
        num_init_rows=2,
        bootstrapping_variant=BootstrappingVariant.REAL,
    )
