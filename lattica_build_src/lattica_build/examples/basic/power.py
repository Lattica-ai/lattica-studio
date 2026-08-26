from lattica_build.base_classes.hom_pipeline import HomomorphicPipeline
from lattica_build.operators.polynomials.h_square import HomSquare
from lattica_build.params.params import HomParams


INPUT_SHAPE = (4,)


def build_pipeline() -> HomomorphicPipeline:
    return HomomorphicPipeline(
        hom=HomSquare(with_modswitch=False),
        input_shape=INPUT_SHAPE,
    )


def build_params() -> HomParams:
    return HomParams(
        n=2**13,
        full_q_list_precision=((61,), (61,)),
        pt_scale=2**32,
        num_special_primes=1,
    )
