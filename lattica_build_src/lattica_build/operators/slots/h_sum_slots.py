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
        k: Optional number of slots each sum adds, n / k apart (defaults to full
            slot count): slot j gets x[j] + x[j + n / k] + ... for n slots.
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
        n_slots = input.shape[-1]
        k = self.k if self.k is not None else n_slots
        groups = input.reshape(*input.shape[:-1], k, n_slots // k)
        return groups.sum(dim=-2, keepdim=True).expand_as(groups).reshape(input.shape).clone()
