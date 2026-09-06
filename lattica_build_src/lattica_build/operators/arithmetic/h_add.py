"""See `operators/arithmetic/README.md` for usage details."""

from lattica_build.base_classes.hom_op import HomOp
from lattica_build.base_classes.hom_value import HomValue
from lattica_build.params.level_and_scale_tracing import infer_adjust_levels_and_scale
from lattica_build.params.shape_tracing import infer_broadcast_output_shape
from lattica_build.serialization.hom_op_pb2 import HomOpType

class HomAdd(HomOp):
    """Elementwise homomorphic addition (or subtraction) with broadcasting.

    This is a base operator (`OP_TYPE = HomOpType.Add`) that aligns input
    ciphertext level/scale before combining values.

    Args:
        is_sub: When `True`, the backend executes subtraction semantics.
    """

    OP_TYPE = HomOpType.Add

    def __init__(self, is_sub: bool = False) -> None:
        super().__init__()
        self.is_sub = is_sub

    def forward_clear(self, input_1, input_2):
        return input_1 - input_2 if self.is_sub else input_1 + input_2

    def infer_output_shape(
        self,
        input_1: HomValue,
        input_2: HomValue,
        **kwargs,
    ) -> HomValue:
        """Infer elementwise broadcast shape for both inputs."""
        return infer_broadcast_output_shape(input_1, input_2)

    def infer_output_level_and_scale(
        self,
        input_1: HomValue,
        input_2: HomValue,
        **kwargs,
    ) -> HomValue:
        """Infer adjusted level/scale needed for compatible addition."""
        return infer_adjust_levels_and_scale(input_1, input_2)
