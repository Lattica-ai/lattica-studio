"""See `operators/README.md` for usage details."""

from lattica_build.base_classes.hom_op import ClientOp
from lattica_build.base_classes.hom_value import HomValue
from lattica_build.params.shape_tracing import to_pos_axis
from lattica_build.serialization.hom_op_pb2 import HomOpType


class Softmax(ClientOp):
    OP_TYPE = HomOpType.Softmax

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim


class Clamp(ClientOp):
    OP_TYPE = HomOpType.Clamp

    def __init__(self, min_val: float, max_val: float) -> None:
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val


class Repeat(ClientOp):
    OP_TYPE = HomOpType.Repeat

    def __init__(self, dim: int = 0) -> None:
        super().__init__()
        self.dim = dim

    def infer_output_shape(self, input: HomValue, n_slots: int | None = None, **kwargs) -> HomValue:
        if n_slots is None:
            raise ValueError("Repeat output shape requires n_slots.")
        axis = to_pos_axis(self.dim, input.tensor_shape)
        output_shape = input.tensor_shape[:axis] + (n_slots,) + input.tensor_shape[axis + 1:]
        return input.make_copy(tensor_shape=output_shape)