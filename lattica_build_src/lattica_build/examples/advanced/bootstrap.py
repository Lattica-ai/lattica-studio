import torch

from lattica_build.base_classes.hom_op import HomOp
from lattica_build.base_classes.hom_pipeline import HomomorphicPipeline
from lattica_build.base_classes.hom_value import HomValue
from lattica_build.operators.fhe.h_bootstrap import Bootstrap
from lattica_build.params.bootstrapping_params import BootstrappingVariant
from lattica_build.params.params import HomParams
from lattica_build.base_classes.pipeline_wrapper import PipelineWrapper

LOG_N = 16
LOG_N_SUBRING = 15
INPUT_SCALE = 2 ** 45


class Pipeline(PipelineWrapper):

    def build_pipeline(self) -> HomomorphicPipeline:
        """Construct a bootstrapping homomorphic pipeline."""
        return HomomorphicPipeline(
            hom=Bootstrap(),
            # The logical shape is one sub-ring period; repeating it across the
            # log_n ring is enc()'s job, driven by HomParams.n_slots below.
            input_shape=(2 ** LOG_N_SUBRING,),
        )

    def build_params(self) -> HomParams:
        return HomParams(
            n=2 ** LOG_N,
            n_slots=2 ** LOG_N_SUBRING,
            full_q_list_precision=(
                (60,),
                (60,),
                (60,),
                (60,),
                (60,),
                (60,),
                (60,),
            ),
            pt_scale=INPUT_SCALE,
            err_std=3.19,
            sk_hw=192,
            num_special_primes=7,
            num_init_rows=0,
            bootstrapping_variant=BootstrappingVariant.SLIM,
        )

    def verify_results(self, actual: torch.Tensor, expected: torch.Tensor) -> None:
        """Verify that the actual result is close to the expected result."""
        # report mean and max absolute error, and assert that the mean is below a threshold
        mean_abs_error = torch.mean(torch.abs(actual - expected))
        max_abs_error = torch.max(torch.abs(actual - expected))
        print(f"Mean absolute error: {mean_abs_error.item()}")
        print(f"Max absolute error: {max_abs_error.item()}")
        super().verify_results(actual, expected)