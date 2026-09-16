"""See `operators/README.md` for usage details."""

from lattica_build.base_classes.hom_op import ClientOp
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

