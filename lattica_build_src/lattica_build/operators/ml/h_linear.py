"""See `operators/ml/README.md` for usage details."""

import torch

from lattica_build.base_classes.hom_value import TensorShape
from lattica_build.operators.arithmetic.h_const_add import HomConstAdd
from lattica_build.operators.composite.sequential import SequentialHomOp
from lattica_build.operators.ml.h_mat_mul import HomMatMul


class HomLinear(SequentialHomOp):
    """Linear layer as `mat_mul` plus optional `const_add` bias.

    This is a composite operator that builds a small graph from base ops.

    Args:
        dims: Shape of the plaintext weight tensor.
        bias: Whether to append a bias `HomConstAdd` stage.
        mul_axis: Axis reduced by the matmul stage.
        mul_scale: Optional plaintext scale-up applied in matmul inference.
        with_modswitch: Whether to apply optional post-op modswitch in the
            matmul stage.
    """

    def __init__(
        self,
        dims: TensorShape,
        bias: bool = True,
        mul_axis: int = -1,
        mul_scale: int = None,
        with_modswitch: bool = True,
    ) -> None:
        self.bias = bias
        self.mul_axis = mul_axis

        self.h_weight = HomMatMul(
            dims,
            mul_axis=mul_axis,
            pt_scale=mul_scale,
            with_modswitch=with_modswitch,
        )
        ops = [self.h_weight]

        if bias:
            if mul_axis < 0:
                mul_axis += len(dims)  # to positive
            bias_shape = [*dims[:mul_axis], *dims[mul_axis + 1 :]]
            self.h_bias = HomConstAdd(dims=bias_shape)
            ops.append(self.h_bias)

        super().__init__(*ops)

    def set_data(self, *data: torch.Tensor, **kwargs) -> None:
        """Set weight and optional bias tensors for the composed linear op.

        Args:
            *data: `weight` followed by `bias` when `self.bias` is enabled.
        """
        if self.bias and len(data) == 1:
            raise ValueError("Bias is enabled but no bias data was provided")

        self.h_weight.set_data(data[0])

        if self.bias:
            self.h_bias.set_data(data[1])
