from lattica_build.base_classes.hom_op import HomOp
from lattica_build.base_classes.hom_pipeline import HomomorphicPipeline
from lattica_build.base_classes.hom_value import HomValue
from lattica_build.operators.fhe.h_bootstrap import Bootstrap
from lattica_build.params.bootstrapping_params import BootstrappingVariant
from lattica_build.params.params import HomParams


LOG_N = 10
LOG_N_SUBRING = 8
INPUT_SCALE = 2 ** 45


class _BootstrapTwice(HomOp):
    def __init__(self, log_n_subring: int) -> None:
        super().__init__()
        self.first = Bootstrap(log_n_subring=log_n_subring)
        self.second = Bootstrap(log_n_subring=log_n_subring)

    def forward(self, x: HomValue) -> HomValue:
        return self.second(self.first(x))


def build_pipeline(
    log_n: int = LOG_N,
    log_n_subring: int = LOG_N_SUBRING,
) -> HomomorphicPipeline:
    """Construct a bootstrapping homomorphic pipeline."""
    return HomomorphicPipeline(
        hom=_BootstrapTwice(log_n_subring),
        # The logical shape is one sub-ring period; repeating it across the
        # log_n ring is enc()'s job, driven by HomParams.n_slots below.
        input_shape=(2 ** (log_n_subring - 1),),
    )


def build_params(
    log_n: int = LOG_N,
    input_scale: int = INPUT_SCALE,
    log_n_subring: int = LOG_N_SUBRING,
) -> HomParams:
    return HomParams(
        n=2 ** log_n,
        n_slots=2 ** (log_n_subring - 1),
        full_q_list_precision=(
            (60,),
            (60,),
            (60,),
            (60,),
        ),
        pt_scale=input_scale,
        sk_hw=192,
        num_special_primes=6,
        num_init_rows=2,
        bootstrapping_variant=BootstrappingVariant.REAL,
    )
