import torch
from lattica_build.base_classes.hom_pipeline import HomomorphicPipeline
from lattica_build.operators.arithmetic.h_const_mul import HomConstMul
from lattica_build.operators.composite.sequential import SequentialHomOp
from lattica_build.operators.fhe.h_ring_switch import HomRingSwitch
from lattica_build.operators.polynomials.h_square import HomSquare
from lattica_build.params.params import HomParams
LOG_N = 13
# The logical input period, which can be smaller than the slots of the
# sub-ring the client encrypts in (HomRingSwitch's default, 2**11).
N_SLOTS = 2 ** 3
INPUT_SCALE = 2 ** 30
from lattica_build.base_classes.pipeline_wrapper import PipelineWrapper

class Pipeline(PipelineWrapper):

    def build_pipeline(self) -> HomomorphicPipeline:
        input_shape = (3, N_SLOTS)
        pipeline = HomomorphicPipeline(
            hom=SequentialHomOp(
                HomRingSwitch(),
                HomSquare(),
                HomConstMul(dims=input_shape),
            ),
            input_shape=input_shape,
        )
        generator = torch.Generator().manual_seed(0)
        pipeline.set_data(2, torch.rand(input_shape, generator=generator))
        return pipeline

    def build_params(self) -> HomParams:
        return HomParams(
            full_q_list_precision=((60, 30),),
            n=2**LOG_N,
            # The input repeats with period N_SLOTS across the sub-ring and the ring
            # HomRingSwitch switches up to, rather than filling n/2 slots.
            n_slots=N_SLOTS,
            sk_hw=192,
            pt_scale=INPUT_SCALE,
            num_special_primes=6,
        )
