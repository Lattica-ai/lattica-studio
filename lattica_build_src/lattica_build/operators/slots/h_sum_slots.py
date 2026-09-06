"""See `operators/slots/README.md` for usage details."""

from typing import Optional, Sequence

from lattica_build.base_classes.hom_op import HomOp
from lattica_build.serialization.hom_op_pb2 import HomOpType

class HomSumSlots(HomOp):
    """Base slot-axis reduction operator.

    Use this operator family for slot-axis summation semantics instead of
    tensor-axis reduction ops like `HomAxisSum`.

    In this model, `HomSumSlots` uses rotation/addition semantics and does not
    consume levels.

    Args:
        k: Optional slot span to reduce (defaults to full slot count).
        stage_sizes: Optional explicit stage schedule for the summation tree.
    """

    OP_TYPE = HomOpType.SumSlots

    def __init__(
        self,
        k: Optional[int] = None,
        stage_sizes: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        self.k = k
        self.stage_sizes = stage_sizes

    def forward_clear(self, input):
        dim = -1
        k = self.k if self.k is not None else input.shape[dim]
        return input[..., :k].sum(dim=dim, keepdim=True).expand_as(input).clone()
