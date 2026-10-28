"""See `operators/shape/README.md` for usage details."""

from lattica_build.base_classes.hom_op import HomOp
from lattica_build.base_classes.hom_value import HomValue
from lattica_build.params.shape_tracing import to_pos_axis, infer_remove_axis_output_shape
from lattica_build.serialization.hom_op_pb2 import HomOpType

class HomSlice(HomOp):
    """Base single-axis slice/index operator.

    Supports `int` (drop axis) and `slice` (keep axis) keys on one dimension.

    Args:
        dim: Axis to slice.
        key: Integer index (drops the axis) or `slice` (keeps the axis).
    """

    OP_TYPE = HomOpType.Slice

    def __init__(self, dim: int, key: int | slice) -> None:
        super().__init__()
        self.dim = dim
        if isinstance(key, int):
            self.start = key
            self.end = None
            self.step = 1
            self.drop_dim = True
        elif isinstance(key, slice):
            self.start = 0 if key.start is None else key.start
            self.end = key.stop
            self.step = 1 if key.step is None else key.step
            self.drop_dim = False
        else:
            raise TypeError(
                f"HomSlice key must be int or slice, got {type(key).__name__}."
            )

    def infer_output_shape(self, input: HomValue, **kwargs) -> HomValue:
        """Infer output shape and enforce axis/index validity checks."""
        dim = to_pos_axis(self.dim, input.tensor_shape)
        if dim == input.n_axis:
            raise ValueError(f"Cannot slice {dim=} because n_axis == dim.")

        size = input.tensor_shape[dim]
        if self.drop_dim:
            idx = self.start if self.start >= 0 else self.start + size
            if idx < 0 or idx >= size:
                raise IndexError(
                    f"HomSlice index {self.start} is out of bounds for axis {dim} with size {size}."
                )
            return infer_remove_axis_output_shape(input, dim)

        if self.step == 0:
            raise ValueError("HomSlice step must not be 0.")

        start_n, end_n, step_n = slice(self.start, self.end, self.step).indices(size)
        new_len = len(range(start_n, end_n, step_n))
        if new_len == 0:
            raise ValueError("HomSlice produced an empty dimension.")

        new_shape = list(input.tensor_shape)
        new_shape[dim] = new_len
        return input.make_copy(tensor_shape=new_shape)

    def forward_clear(self, input):
        index = [slice(None)] * input.ndim
        if self.drop_dim:
            index[self.dim] = self.start
        else:
            index[self.dim] = slice(self.start, self.end, self.step)
        return input[tuple(index)]
