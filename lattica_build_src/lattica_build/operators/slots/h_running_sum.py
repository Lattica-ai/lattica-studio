"""See `operators/slots/README.md` for usage details."""

import math
import torch
from typing import Optional, Sequence

from lattica_build.base_classes.hom_value import HomValue, to_pos_axis
from lattica_build.operators.slots.ring_utils import SlotOp, from_ring, input_layout, to_ring
from lattica_build.params.level_and_scale_tracing import infer_modswitch_by_num_levels
from lattica_build.serialization.hom_op_pb2 import HomOpType

class HomRunningSum(SlotOp):
    """Base slot running-sum operator over configured stage schedule.

    Args:
        cumsum_axis: Optional logical axis hint for cumulative summation.
        blocks_axis_external: Optional external blocking-axis hint.
        k: Optional number of rows each period of `n_slots` slots is summed
            over (defaults to `n_slots`): the period is `k` rows of
            `n_slots / k` columns.
        stage_sizes: Optional explicit stage schedule. When provided, it
            defines stage count as `len(stage_sizes)` and should satisfy
            `prod(stage_sizes) == k`.
        stages_per_level: Optional number of stages grouped per consumed level.
            It should divide the selected stage count exactly.
        rows_budget: Optional allowed modulus rows for modswitch inference.

    The sums run within each period, so the output keeps the input's shape and
    n_slots. The input's slot axis must have no zero padding, which the sums
    would add: it holds n_slots values, or whole rings when n_slots is
    internal_n.
    """

    OP_TYPE = HomOpType.RunningSum

    def __init__(
        self,
        cumsum_axis: Optional[int] = None,
        blocks_axis_external: Optional[int] = None,
        k: Optional[int] = None,
        stage_sizes: Sequence[int] | None = None,
        stages_per_level: int | None = None,
        rows_budget: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        self.cumsum_axis = cumsum_axis
        self.blocks_axis_external = blocks_axis_external
        self.k = k
        self.stage_sizes = stage_sizes
        self.stages_per_level = stages_per_level
        self.rows_budget = rows_budget

    def _sums_slot_axis(self, input: HomValue) -> bool:
        """Whether the running sum is along the slot axis of `input` (the default) rather than another axis."""
        return self.cumsum_axis is None or to_pos_axis(self.cumsum_axis, input.tensor_shape) == input.n_axis

    def infer_output_shape(self, input: HomValue, internal_n=None) -> HomValue:
        """The output has the input's shape. The slot axis must hold n_slots values, or whole rings when n_slots is
        internal_n."""
        if input.n_axis is None:
            return input
        length = input.tensor_shape[input.n_axis]
        whole_rings = input.n_slots == internal_n and length % internal_n == 0
        if length != input.n_slots and not whole_rings:
            raise ValueError(f"RunningSum takes a slot axis of n_slots={input.n_slots} values, or of whole rings "
                             f"when n_slots is internal_n={internal_n}: got {length}")
        return input

    def infer_output_level_and_scale(
        self,
        input: HomValue,
        hom_params=None,
        **kwargs,
    ) -> HomValue:
        """Compute exact level consumption from stages/log2(k), then apply modswitch.

        Running-sum stages include masked multiplications, so they consume
        levels. Along another axis than the slot axis, the running sum adds whole ciphertexts
        only, which consumes no level.
        """
        if not self._sums_slot_axis(input):
            return input
        if self.stage_sizes is None:
            k = self.k if self.k is not None else input.n_slots
            num_levels = int(math.log2(k))
        else:
            num_levels = len(self.stage_sizes)
        if self.stages_per_level is not None:
            num_levels //= self.stages_per_level
        return infer_modswitch_by_num_levels(
            hom_params,
            input,
            num_levels=num_levels,
            rows_budget=self.rows_budget,
        )

    def forward_clear(self, input):
        """Along another axis than the slot axis, a cumsum. Along the slot axis, the ciphertext views each period of
        n_slots slots as k rows of n_slots / k columns (slot s at row s // cols, column s % cols) and sums down the
        rows of all blocks: row r of block 0, row r of block 1, ..., then row r + 1. The blocks are the rings of the
        slot axis, or the entries of `blocks_axis_external`. The ring repeats the period, so every repeat gets the
        same sums."""
        value, internal_n = input_layout(self, input)
        if not self._sums_slot_axis(value):
            return input.cumsum(dim=self.cumsum_axis)

        n_slots = value.n_slots
        periods = to_ring(input, value, internal_n)[..., :n_slots]  # (..., blocks, n_slots)
        blocks_axis = None
        if self.blocks_axis_external is not None:
            if periods.shape[-2] != 1:
                raise ValueError("blocks_axis_external needs the slot axis to fit in one ring")
            blocks_axis = to_pos_axis(self.blocks_axis_external, value.tensor_shape)
            if blocks_axis == value.n_axis:
                raise ValueError(f"blocks_axis_external={self.blocks_axis_external} is the slot axis; "
                                 "the blocks must be on another axis")
            # The axis among those to_ring keeps before the slot axis
            blocks_axis -= int(blocks_axis > value.n_axis)
            periods = periods.squeeze(-2).movedim(blocks_axis, -2)

        k = self.k if self.k is not None else n_slots
        grid = periods.unflatten(-1, (k, n_slots // k)).transpose(-3, -2)  # (..., k, blocks, cols)
        sums = grid.flatten(-3, -2).cumsum(dim=-2).unflatten(-2, grid.shape[-3:-1])
        sums = sums.transpose(-3, -2).flatten(-2)  # (..., blocks, n_slots)

        if blocks_axis is not None:
            sums = sums.movedim(-2, blocks_axis).unsqueeze(-2)
        return from_ring(sums, value)
