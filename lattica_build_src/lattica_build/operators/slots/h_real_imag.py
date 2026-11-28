"""See `operators/slots/README.md` for usage details."""

import torch

from lattica_build.base_classes.hom_op import HomOp
from lattica_build.base_classes.hom_value import HomValue
from lattica_build.params.shape_tracing import infer_insert_axis_output_shape, infer_remove_axis_output_shape
from lattica_build.serialization.hom_op_pb2 import HomOpType


def _paired_axis(shape) -> int | None:
    """The first axis with more than one entry."""
    return next((axis for axis, size in enumerate(shape) if size > 1), None)


class HomCombineRealImag(HomOp):
    """Halves the first axis with more than one entry, which must not be the slot axis: its first ceil(m/2) entries
    take the other floor(m/2) as imaginary parts, and with an odd m the last entry of the first half stays real. The
    axis keeps ceil(m/2) entries, or goes when that is 1: [x, y] -> x + i * y, the inverse of `HomSplitRealImag` up
    to a factor of 2.

    Exact, with no key switch and no level.
    """

    OP_TYPE = HomOpType.CombineRealImag

    def infer_output_shape(self, input: HomValue, **kwargs) -> HomValue:
        axis = _paired_axis(input.tensor_shape)
        if axis is None or axis == input.n_axis:
            raise ValueError(f"HomCombineRealImag takes an axis with more than one entry ahead of the slot values; "
                             f"got {tuple(input.tensor_shape)} with slots on axis {input.n_axis}")
        n_first = (input.tensor_shape[axis] + 1) // 2
        if n_first == 1:
            return infer_remove_axis_output_shape(input, axis)
        shape = list(input.tensor_shape)
        shape[axis] = n_first
        return input.make_copy(tensor_shape=tuple(shape))

    def forward_clear(self, input):
        axis = _paired_axis(input.shape)
        if axis is None:
            raise ValueError(f"HomCombineRealImag takes an axis with more than one entry; got {tuple(input.shape)}")
        n_entries = input.shape[axis]
        n_first = (n_entries + 1) // 2
        z = input.narrow(axis, 0, n_first) + 0j
        z.narrow(axis, 0, n_entries - n_first).add_(1j * input.narrow(axis, n_first, n_entries - n_first))
        return z.squeeze(axis) if n_first == 1 else z


class HomSplitRealImag(HomOp):
    """A complex value as twice its real and imaginary parts, stacked on a new first axis:
    z -> [2 Re(z), 2 Im(z)], which are z + conj(z) and -i * (z - conj(z)).
    """

    OP_TYPE = HomOpType.SplitRealImag

    def infer_output_shape(self, input: HomValue, **kwargs) -> HomValue:
        return infer_insert_axis_output_shape(input, 0, 2)

    def forward_clear(self, input):
        z = input.to(torch.complex128)
        return 2 * torch.stack((z.real, z.imag))
