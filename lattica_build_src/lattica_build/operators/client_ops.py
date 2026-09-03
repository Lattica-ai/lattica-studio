"""See `operators/README.md` for usage details."""

from lattica_build.base_classes.hom_op import ClientOp
from lattica_build.base_classes.hom_value import HomValue
from lattica_build.params.shape_tracing import to_pos_axis
from lattica_build.serialization.hom_op_pb2 import HomOpType
import torch


class Softmax(ClientOp):
    OP_TYPE = HomOpType.Softmax

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward_clear(self, input):
        return torch.softmax(input, dim=self.dim)


class Clamp(ClientOp):
    OP_TYPE = HomOpType.Clamp

    def __init__(self, min_val: float, max_val: float) -> None:
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward_clear(self, input):
        return torch.clamp(input, self.min_val, self.max_val)


class Repeat(ClientOp):
    OP_TYPE = HomOpType.Repeat

    def __init__(self, dim: int = 0) -> None:
        super().__init__()
        self.dim = dim

    def infer_output_shape(self, input: HomValue, internal_n: int | None = None, **kwargs) -> HomValue:
        if internal_n is None:
            raise ValueError("Repeat output shape requires internal_n.")
        axis = to_pos_axis(self.dim, input.tensor_shape)
        output_shape = input.tensor_shape[:axis] + (internal_n,) + input.tensor_shape[axis + 1:]
        return input.make_copy(tensor_shape=output_shape)

    def forward_clear(self, input, internal_n=None):
        # Repeat's target dimension comes from packing parameters. Preserve
        # the clear tensor when that parameter is unavailable.
        if internal_n is None:
            from lattica_build.base_classes.hom_op import HomOp
            internal_n = HomOp._clear_option("internal_n")
        if internal_n is None:
            return input
        axis = self.dim if self.dim >= 0 else input.ndim + self.dim
        repeats = [1] * input.ndim
        if internal_n % input.shape[axis] != 0:
            raise ValueError("Repeat internal_n must be divisible by the input dimension")
        repeats[axis] = internal_n // input.shape[axis]
        return input.repeat(*repeats)
