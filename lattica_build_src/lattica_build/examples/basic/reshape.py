from lattica_build.base_classes.hom_pipeline import HomomorphicPipeline
from lattica_build.operators.shape.h_reshape import HomReshape
from lattica_build.params.params import HomParams


INPUT_SHAPE = (1, 128, 512)
OUTPUT_SHAPE = (128, 4, 128)


def build_pipeline() -> HomomorphicPipeline:
    return HomomorphicPipeline(
        hom=HomReshape(OUTPUT_SHAPE),
        input_shape=INPUT_SHAPE,
        n_axis=1,
    )


def build_params() -> HomParams:
    return HomParams(
        n=2**13,
        full_q_list_precision=((61,), (61,)),
        pt_scale=2**32,
        num_special_primes=1,
    )
