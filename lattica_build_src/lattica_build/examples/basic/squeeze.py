from lattica_build.base_classes.hom_pipeline import HomomorphicPipeline
from lattica_build.operators.shape.h_squeeze import HomSqueeze
from lattica_build.params.params import HomParams


INPUT_SHAPE = (1, 128, 512)


def build_pipeline() -> HomomorphicPipeline:
    return HomomorphicPipeline(
        hom=HomSqueeze(0),
        input_shape=INPUT_SHAPE,
    )


def build_params() -> HomParams:
    return HomParams(
        n=2**13,
        full_q_list_precision=((61,), (61,)),
        pt_scale=2**32,
        num_special_primes=1,
    )
