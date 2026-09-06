"""See `operators/shape/README.md` for usage details."""

from lattica_build.base_classes.hom_op import HomOp
from lattica_build.base_classes.hom_value import HomValue, to_pos_axis
from lattica_build.params.shape_tracing import infer_remove_axis_output_shape
from lattica_build.serialization.hom_op_pb2 import HomOpType

class HomSqueeze(HomOp):
    """Base squeeze operator that removes a size-1 axis.

    Args:
        dim: Axis index to remove; that axis must be size 1.
    """

    OP_TYPE = HomOpType.Squeeze

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def infer_output_shape(self, input: HomValue, **kwargs) -> HomValue:
        """Infer squeezed shape; raise when selected axis is not size 1."""
        axis = to_pos_axis(self.dim, input.tensor_shape)
        if input.tensor_shape[axis] != 1:
            raise ValueError(
                f"HomSqueeze can only squeeze a dimension of size 1, "
                f"but axis {axis} of {input.tensor_shape} has size {input.tensor_shape[axis]}."
            )
        return infer_remove_axis_output_shape(input, axis)

    def forward_clear(self, input):
        return input.squeeze(self.dim)

