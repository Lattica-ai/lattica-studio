"""See `operators/arithmetic/README.md` for usage details."""

from lattica_build.base_classes.hom_op import HomOp
from lattica_build.base_classes.hom_value import HomValue
from lattica_build.params.shape_tracing import infer_reduce_axis_output_shape
from lattica_build.serialization.hom_op_pb2 import HomOpType

class HomAxisSum(HomOp):
    """Reduce a `HomValue` by summing across one axis.

    This is a base operator (`OP_TYPE = HomOpType.AxisSum`).

    Args:
        dim: Axis index to reduce.
        keep_dim: Whether to retain the reduced axis with size 1.

    Notes:
        Reducing the ciphertext slot axis (`n_axis`) is not supported here.
        Use `HomSumSlots` (`slots/h_sum_slots.py`) for slot-axis reductions.
    """

    OP_TYPE = HomOpType.AxisSum

    def __init__(self, dim: int, keep_dim: bool = False) -> None:
        super().__init__()
        self.dim = dim
        self.keep_dim = keep_dim

    def infer_output_shape(self, input: HomValue, **kwargs) -> HomValue:
        """Infer reduced output shape for the configured axis sum."""
        return infer_reduce_axis_output_shape(input, self.dim, self.keep_dim)
