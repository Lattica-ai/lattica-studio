"""See `operators/slots/README.md` for usage details."""

from collections.abc import Sequence
from operator import index

import torch

from lattica_build.base_classes.hom_value import HomValue
from lattica_build.operators.slots.ring_utils import SlotOp, from_ring, input_layout, to_ring
from lattica_build.params.shape_tracing import infer_remove_axis_output_shape
from lattica_build.serialization.hom_op_pb2 import HomOpType


class HomRotateSumElementwise(SlotOp):
    """Rotate each ciphertext on ``elements_axis`` once, then sum that axis.

    ``rotations[i]`` is the left rotation of element ``i`` along the input's
    slot axis. An identity rotation, if present, must be first. Unlike
    ``HomRotateSum``, this operator never adds an implicit identity term.
    It does not consume a modulus level or change the plaintext scale.
    """

    OP_TYPE = HomOpType.RotateSumElementwise

    def __init__(self, rotations: Sequence[int], elements_axis: int = 0) -> None:
        super().__init__()
        self.rotations = tuple(index(rotation) for rotation in rotations)
        self.elements_axis = index(elements_axis)
        if not self.rotations:
            raise ValueError("rotations must contain at least one element")
        if 0 in self.rotations[1:]:
            raise ValueError("an identity rotation must be the first element")

    def _validated_axis(self, shape, n_axis):
        rank = len(shape)
        if not -rank <= self.elements_axis < rank:
            raise ValueError(f"elements_axis {self.elements_axis} is out of range for rank {rank}")
        axis = self.elements_axis % rank
        if axis == n_axis:
            raise ValueError("elements_axis cannot be the slot axis")
        if shape[axis] != len(self.rotations):
            raise ValueError("the elements axis size must equal the number of rotations")
        return axis

    def infer_slot_output_shape(self, input: HomValue, internal_n=None, **kwargs) -> HomValue:
        axis = self._validated_axis(input.tensor_shape, input.n_axis)
        return infer_remove_axis_output_shape(input, axis)

    def forward_clear(self, input):
        value, internal_n = input_layout(self, input)
        axis = self._validated_axis(input.shape, value.n_axis)
        elements = to_ring(input, value, internal_n).unbind(axis - (axis > value.n_axis))
        result = torch.roll(elements[0], -self.rotations[0], dims=-1)
        for element, rotation in zip(elements[1:], self.rotations[1:], strict=True):
            result = result + torch.roll(element, -rotation, dims=-1)
        return from_ring(result, infer_remove_axis_output_shape(value, axis))
