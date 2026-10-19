"""See `operators/slots/README.md` for usage details."""

from typing import Optional, Sequence

import torch

from lattica_build.base_classes.hom_op import HomOp
from lattica_build.base_classes.hom_value import HomValue, to_neg_axis
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
        if k > n_slots:
            raise ValueError(f"k={k} is larger than the input's {n_slots}-slot period.")
        return input.make_copy(n_slots=n_slots // k)

    def forward_clear(self, input):
        """The encrypted sum on the ciphertext's slots, along the input's traced slot axis n_axis
        with its traced period n_slots: the axis is zero-padded to n_slots, and slot j gets the
        k slots n_slots / k apart. The result repeats at n_slots / k and is read at the input's
        positions."""
        if not self._input_values:
            raise ValueError("HomSumSlots needs its input's traced slot period: run it through "
                             "HomomorphicPipeline.forward_clear")
        value = self._input_values[0]
        n_slots = value["n_slots"]
        # The slot axis counted from the end, so leading batch axes don't shift it.
        axis = -1 if value["n_axis"] is None else to_neg_axis(value["n_axis"], value["tensor_shape"])
        x = input.movedim(axis, -1)
        k = self.k if self.k is not None else n_slots
        period = x[..., :n_slots]
        period = torch.nn.functional.pad(period, (0, n_slots - period.shape[-1]))
        sums = period.reshape(*period.shape[:-1], k, n_slots // k).sum(dim=-2)
        return sums[..., torch.arange(x.shape[-1], device=x.device) % (n_slots // k)].movedim(-1, axis)
