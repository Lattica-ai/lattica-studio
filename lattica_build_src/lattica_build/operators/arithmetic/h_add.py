"""See `operators/arithmetic/README.md` for usage details."""

from lattica_build.base_classes.hom_op import HomOp
from lattica_build.base_classes.hom_value import HomValue
from lattica_build.params.level_and_scale_tracing import infer_adjust_levels_and_scale
from lattica_build.params.shape_tracing import has_zero_padding, infer_broadcast_output_shape
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
        internal_n: int | None = None,
        **kwargs,
    ) -> HomValue:
        """Infer elementwise broadcast shape for both inputs.

        A period-1 input added to a zero-padded one raises: it would fill the padding, which the slot operators
        read. A product keeps the padding zero, so HomMul takes it."""
        output = infer_broadcast_output_shape(input_1, input_2)
        if input_1.n_slots != input_2.n_slots and has_zero_padding(output, internal_n):
            length = output.tensor_shape[output.n_axis]
            raise ValueError(f"Cannot {'subtract' if self.is_sub else 'add'} ciphertexts of periods (n_slots) "
                             f"{input_1.n_slots} and {input_2.n_slots} with {length} values on the slot axis: "
                             f"the period-1 one would fill the other's zero padding")
        return output

    def infer_output_level_and_scale(
        self,
        shaped_output: HomValue,
        input_2: HomValue,
        **kwargs,
    ) -> HomValue:
        """Infer adjusted level/scale needed for compatible addition: shaped_output (the broadcast shape, at the
        first input's level) at the lower input's level and scale."""
        return infer_adjust_levels_and_scale(shaped_output, input_2)
