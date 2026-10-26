"""See `operators/slots/README.md` for usage details."""

from typing import Sequence

import torch

from lattica_build.base_classes.hom_value import HomValue
from lattica_build.operators.slots.ring_utils import SlotOp, from_ring, full_period, input_layout, to_ring

from lattica_build.params.shape_tracing import infer_insert_axis_output_shape
from lattica_build.serialization.hom_op_pb2 import HomOpType

class HomRotateSum(SlotOp):
    """Base slot-rotation operator with optional accumulation.

    Args:
        rotations: Rotation offsets to apply.
        perform_sum: When `True`, sums all rotated ciphertexts into one output.
        add_identity_rotation: Whether to include the unrotated input in the
            output set. Sum mode always includes the unrotated input once.
        full_period_output: Whether the output's slot axis holds the whole
            period (default): the rotations bring values into the zero padding
            of a shorter slot axis, so its length is rounded up to a multiple
            of n_slots. Otherwise it keeps the input's length.
    """

    OP_TYPE = HomOpType.RotateSum

    def __init__(
        self,
        rotations: Sequence[int] | set[int],
        perform_sum: bool = True,
        add_identity_rotation: bool = False,
        full_period_output: bool = True,
    ) -> None:
        super().__init__()
        self.rotations = tuple(rotations)
        self.perform_sum = perform_sum
        self.add_identity_rotation = add_identity_rotation
        self.full_period_output = full_period_output

    def infer_output_shape(self, input: HomValue, internal_n=None, **kwargs) -> HomValue:
        """Keep shape for sum mode, or stack per-rotation outputs on axis 0. With full_period_output, the slot axis
        is rounded up to a multiple of its period n_slots (internal_n without one)."""
        n_slots = input.n_slots if input.n_slots is not None else internal_n
        if self.full_period_output and input.n_axis is not None and n_slots is not None:
            input = full_period(input, n_slots)
        if self.perform_sum:
            return input

        num_out_rotations = len(self.rotations) + int(self.add_identity_rotation)
        return infer_insert_axis_output_shape(input, 0, num_out_rotations)

    def forward_clear(self, input):
        """Each rotation rotates every ring of the ciphertext, so the zero padding of the period comes in after the
        input's last value, not its first values. With full_period_output, the output reads the whole period."""
        value, internal_n = input_layout(self, input)
        ring = to_ring(input, value, internal_n)
        output = full_period(value, value.n_slots) if self.full_period_output else value
        values = [from_ring(torch.roll(ring, -rotation, dims=-1), output) for rotation in self.rotations]
        if self.perform_sum or self.add_identity_rotation:
            values.insert(0, from_ring(ring, output))
        if self.perform_sum:
            return sum(values[1:], values[0])
        return torch.stack(values, dim=0)
