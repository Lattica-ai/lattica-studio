"""See `operators/shape/README.md` for usage details."""

from lattica_build.base_classes.hom_op import HomOp
from lattica_build.base_classes.hom_value import HomValue, to_pos_axis

from lattica_build.params.shape_tracing import infer_insert_axis_output_shape
from lattica_build.serialization.hom_op_pb2 import HomOpType

class HomUnsqueeze(HomOp):
    """Base unsqueeze operator that inserts a size-1 axis.

    Args:
        dim: Axis index where the singleton dimension is inserted.
    """

    OP_TYPE = HomOpType.Unsqueeze

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def infer_output_shape(self, input: HomValue, **kwargs) -> HomValue:
        """Infer output shape after inserting a singleton axis at `dim`."""
        axis = to_pos_axis(self.dim, (*input.tensor_shape, 1))
        return infer_insert_axis_output_shape(input, axis, 1)

