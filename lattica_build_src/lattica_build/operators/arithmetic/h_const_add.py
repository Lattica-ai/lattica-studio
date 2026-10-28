"""See `operators/arithmetic/README.md` for usage details."""

from lattica_build.base_classes.hom_op import HomOp
from lattica_build.base_classes.hom_value import HomValue, TensorShape
from lattica_build.params.shape_tracing import infer_broadcast_output_shape
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

    def infer_output_shape(self, input: HomValue, **kwargs) -> HomValue:
        """Infer output shape from broadcasting `input` with constant dims."""
        return infer_broadcast_output_shape(input, dims=self.dims)

    def forward_clear(self, input):
        data = self._require_clear_data().to(device=input.device, dtype=input.dtype)
        return input + data
