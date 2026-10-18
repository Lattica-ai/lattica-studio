"""See `operators/slots/README.md` for usage details."""

from typing import Optional, Sequence

import torch

from lattica_build.base_classes.hom_op import HomOp
from lattica_build.base_classes.hom_value import HomValue
from lattica_build.serialization.hom_op_pb2 import HomOpType

class HomSumSlots(HomOp):
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
        """Sum the slots as the ciphertext does: slot j gets the k slots n_slots / k apart,
        n_slots being the input's traced period (untraced: its length rounded up to a power of 2)."""
        # Take the period n_slots from the traced input, or guess it from the axis length when there is none.
        input_values = self._clear_option("input_values")
        n_slots = input_values[0].get("n_slots") if input_values and input_values[0] else None
        if n_slots is None:
            n_slots = 1 << (input.shape[-1] - 1).bit_length()
        k = self.k if self.k is not None else n_slots
        # Zero-pad the axis to one period, add the k slots n_slots / k apart, read back at the input's positions.
        period = input[..., :n_slots]
        period = torch.nn.functional.pad(period, (0, n_slots - period.shape[-1]))
        sums = period.reshape(*period.shape[:-1], k, n_slots // k).sum(dim=-2)
        return sums[..., torch.arange(input.shape[-1], device=input.device) % (n_slots // k)]
