"""See `operators/slots/README.md` for usage details."""

from typing import Sequence

import torch

from lattica_build.base_classes.hom_op import HomOp
from lattica_build.base_classes.hom_value import HomValue

from lattica_build.params.shape_tracing import infer_insert_axis_output_shape
from lattica_build.serialization.hom_op_pb2 import HomOpType

class HomRotateSum(HomOp):
    """Base slot-rotation operator with optional accumulation.

    Args:
        rotations: Rotation offsets to apply.
        perform_sum: When `True`, sums all rotated ciphertexts into one output.
        add_identity_rotation: Whether to include the unrotated input in the
            output set/sum.
    """

    OP_TYPE = HomOpType.RotateSum

    def __init__(
        self,
        rotations: Sequence[int] | set[int],
        perform_sum: bool = True,
        add_identity_rotation: bool = False,
    ) -> None:
        super().__init__()
        self.rotations = tuple(rotations)
        self.perform_sum = perform_sum
        self.add_identity_rotation = add_identity_rotation

    def infer_output_shape(self, input: HomValue, **kwargs) -> HomValue:
        """Keep shape for sum mode, or stack per-rotation outputs on axis 0."""
        if self.perform_sum:
            return input

        num_out_rotations = len(self.rotations) + int(self.add_identity_rotation)
        return infer_insert_axis_output_shape(input, 0, num_out_rotations)

    def forward_clear(self, input):
        values = [torch.roll(input, -rotation, dims=-1) for rotation in self.rotations]
        if self.add_identity_rotation:
            values.insert(0, input)
        if self.perform_sum:
            return sum(values, torch.zeros_like(input))
        return torch.stack(values, dim=0)
