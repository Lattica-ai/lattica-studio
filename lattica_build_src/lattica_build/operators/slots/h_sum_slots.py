"""See `operators/slots/README.md` for usage details."""

from typing import Optional, Sequence

from lattica_build.base_classes.hom_value import HomValue
from lattica_build.operators.slots.ring_utils import SlotOp, check_span, from_ring, input_layout, to_ring
from lattica_build.serialization.hom_op_pb2 import HomOpType

class HomSumSlots(SlotOp):
    """Base slot-axis reduction operator.

    Use this operator family for slot-axis summation semantics instead of
    tensor-axis reduction ops like `HomAxisSum`.

    In this model, `HomSumSlots` uses rotation/addition semantics and does not
    consume levels.

    Args:
        k: Optional number of slots each sum adds, n_slots / k apart (defaults to
            n_slots): slot j gets x[j] + x[j + n_slots / k] + ....
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

    def infer_output_shape(self, input: HomValue, internal_n=None) -> HomValue:
        """The sum folds the input's period n_slots (internal_n without one) to n_slots / k."""
        n_slots = input.n_slots if input.n_slots is not None else internal_n
        k = self.k if self.k is not None else n_slots
        return input.make_copy(n_slots=n_slots // k)

    def forward_clear(self, input):
        """The encrypted sum on the ciphertext's slots: in each period of n_slots slots (zero padding included),
        slot j gets the k slots n_slots / k apart. The result repeats at n_slots / k and is read at the input's
        positions."""
        value, internal_n = input_layout(self, input)
        n_slots = value.n_slots
        k = self.k if self.k is not None else n_slots
        check_span(self, k, n_slots, "the period")
        ring = to_ring(input, value, internal_n)
        periods = ring.unflatten(-1, (internal_n // n_slots, k, n_slots // k))
        sums = periods.sum(dim=-2, keepdim=True).expand(periods.shape)
        return from_ring(sums.flatten(-3), value)
