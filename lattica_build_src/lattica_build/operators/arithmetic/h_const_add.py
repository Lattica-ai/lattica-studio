"""See `operators/arithmetic/README.md` for usage details."""

from lattica_build.base_classes.hom_op import HomOp
from lattica_build.base_classes.hom_value import HomValue, TensorShape
from lattica_build.params.shape_tracing import (
    fills_zero_padding, infer_broadcast_output_shape, infer_constant_packing_n_slots,
)
from lattica_build.serialization.hom_op_pb2 import HomOpType

class HomConstAdd(HomOp):
    """Add a plaintext constant tensor (with broadcasting) to a `HomValue`.

    This is a base operator (`OP_TYPE = HomOpType.ConstAdd`).

    Args:
        dims: Plaintext constant tensor dimensions used for broadcast inference.
    """

    OP_TYPE = HomOpType.ConstAdd

    def __init__(self, dims: TensorShape) -> None:
        super().__init__()
        self.dims = dims

    def infer_output_shape(self, input: HomValue, internal_n=None, **kwargs) -> HomValue:
        """Infer output shape from broadcasting `input` with constant dims, and its period as packed.

        Raises when the output has a larger n_slots than `input` and is zero-padded: the repeats of `input` would fill
        the padding, which the slot operators read."""
        output = infer_broadcast_output_shape(input, dims=self.dims)
        output = output.make_copy(n_slots=infer_constant_packing_n_slots(input, self.dims, internal_n))
        if fills_zero_padding(input, output, internal_n):
            length = output.tensor_shape[output.n_axis]
            raise ValueError(f"Cannot add a constant to a ciphertext of period (n_slots) {input.n_slots} with {length} "
                             f"values on the slot axis: the result has period {output.n_slots}, and the ciphertext's "
                             f"repeats would fill its zero padding")
        return output

    def forward_clear(self, input):
        data = self._require_clear_data().to(device=input.device, dtype=input.dtype)
        return input + data
