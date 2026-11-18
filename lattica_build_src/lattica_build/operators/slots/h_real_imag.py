"""See `operators/slots/README.md` for usage details."""

import torch

from lattica_build.base_classes.hom_op import HomOp
from lattica_build.base_classes.hom_value import HomValue
from lattica_build.params.shape_tracing import infer_insert_axis_output_shape, infer_remove_axis_output_shape
from lattica_build.serialization.hom_op_pb2 import HomOpType


class HomCombineRealImag(HomOp):
    """Two real values, stacked on the first axis, as the real and imaginary parts of one: [x, y] -> x + i * y.

    Exact, with no key switch and no level.
    """

    OP_TYPE = HomOpType.CombineRealImag

    def infer_output_shape(self, input: HomValue, **kwargs) -> HomValue:
        if input.tensor_shape[0] != 2 or input.n_axis == 0:
            raise ValueError(f"HomCombineRealImag expects [real, imaginary] on the first axis, before the slots; "
                             f"got {tuple(input.tensor_shape)} with slots on axis {input.n_axis}")
        return infer_remove_axis_output_shape(input, 0)

    def forward_clear(self, input):
        return input[0] + 1j * input[1]


class HomSplitRealImag(HomOp):
    """A complex value as twice its real and imaginary parts, stacked on a new first axis:
    z -> [2 Re(z), 2 Im(z)], which are z + conj(z) and -i * (z - conj(z)).

    One key switch and no level. Halving would take a level, so the factor 2 is left to a constant next to it.
    """

    OP_TYPE = HomOpType.SplitRealImag

    def infer_output_shape(self, input: HomValue, **kwargs) -> HomValue:
        return infer_insert_axis_output_shape(input, 0, 2)

    def forward_clear(self, input):
        z = input.to(torch.complex128)
        return 2 * torch.stack((z.real, z.imag))
