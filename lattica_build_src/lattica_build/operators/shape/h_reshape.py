"""See `operators/shape/README.md` for usage details."""

import math

import torch

from lattica_build.base_classes.hom_op import HomOp
from lattica_build.base_classes.hom_value import HomValue, TensorShape
from lattica_build.serialization.hom_op_pb2 import HomOpType

class HomReshape(HomOp):
    """Base reshape operator for `HomValue` tensors.

    Reshape follows packed-axis (`n_axis`) safety checks from shape tracing.

    Args:
        dims: Target tensor shape for the payload tensor.
    """

    OP_TYPE = HomOpType.Reshape

    def __init__(self, dims: TensorShape) -> None:
        super().__init__()
        self.dims = dims

    def infer_output_shape(self, input: HomValue, **kwargs) -> HomValue:
        """Infer reshaped output metadata for configured target dims."""
        try:
            output_shape = tuple(torch.empty(input.tensor_shape, device="meta").reshape(self.dims).shape)
        except RuntimeError as error:
            raise ValueError("Reshape is incompatible with input size.") from error

        if input.n_axis is None:
            return input.make_copy(tensor_shape=output_shape)

        n_dim = input.tensor_shape[input.n_axis]
        input_prefix = math.prod(input.tensor_shape[:input.n_axis])
        input_suffix = math.prod(input.tensor_shape[input.n_axis + 1:])

        for output_axis, output_dim in enumerate(output_shape):
            if output_dim != n_dim:
                continue

            output_prefix = math.prod(output_shape[:output_axis])
            output_suffix = math.prod(output_shape[output_axis + 1:])
            if output_prefix == input_prefix and output_suffix == input_suffix:
                return input.make_copy(tensor_shape=output_shape, n_axis=output_axis)

        raise ValueError("HomReshape cannot reshape across n_axis.")

    def forward_clear(self, input):
        return input.reshape(self.dims)

