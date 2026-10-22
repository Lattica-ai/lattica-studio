"""See `operators/polynomials/README.md` for usage details."""

import math
from typing import Sequence

import torch

from lattica_build.base_classes.hom_op import HomOp
from lattica_build.base_classes.hom_value import HomValue
from lattica_build.params.level_and_scale_tracing import infer_optional_modswitch, infer_modswitch_by_num_levels
from lattica_build.serialization.hom_op_pb2 import HomOpType

class HomPolyEvalBase(HomOp):
    """Base polynomial evaluation operator.

    This base op evaluates packed Chebyshev-like coefficient tensors over the
    input ciphertext and infers level consumption from polynomial degree.

    Args:
        cheb_coefs: Chebyshev coefficients, lowest degree first. Leading dimensions are treated as batch
            dimensions and are prepended to the output shape.
        left: Lower bound of the approximation interval.
        right: Upper bound of the approximation interval.
        tol: Numerical tolerance for interval-rescale decisions.
        rows_budget: Optional allowed modulus rows for internal modswitch.
    """

    OP_TYPE = HomOpType.PolyEval

    def __init__(
        self,
        cheb_coefs,
        left: float = -1,
        right: float = 1,
        tol: float = 1e-8,
        rows_budget: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        self.cheb_coefs = cheb_coefs
        self.left = left
        self.right = right
        self.tol = tol
        self.rows_budget = rows_budget

    def infer_output_shape(self, input: HomValue, **kwargs) -> HomValue:
        coef_batch_shape = tuple(torch.as_tensor(self.cheb_coefs).shape[:-1])
        output_shape = coef_batch_shape + input.tensor_shape
        if input.n_axis is None:
            return input.make_copy(tensor_shape=output_shape)
        return input.make_copy(
            tensor_shape=output_shape,
            n_axis=input.n_axis + len(coef_batch_shape),
        )

    def forward_clear(self, input):
        x = input * (2 / (self.right - self.left))
        x = x + (-(self.right + self.left) / (self.right - self.left))
        coefs = torch.as_tensor(self.cheb_coefs, dtype=input.dtype, device=input.device)
        b_next = torch.zeros_like(x)
        b_curr = torch.zeros_like(x)
        for coefficient in reversed(coefs.unbind(-1)):   # Clenshaw: highest degree first
            coefficient = coefficient.reshape(coefs.shape[:-1] + (1,) * input.ndim)
            b_next, b_curr = b_curr, 2 * x * b_curr - b_next + coefficient
        return b_curr - x * b_next

    def infer_output_level_and_scale(
        self,
        input: HomValue,
        hom_params=None,
        **kwargs,
    ) -> HomValue:
        needs_rescale = (not self.left == -1) or not self.right == 1
        x_rescale_mul = 2 / (self.right - self.left)
        if needs_rescale and abs(x_rescale_mul - 1) > self.tol:
            input = infer_optional_modswitch(
                hom_params,
                input,
                with_modswitch=True,
                rows_budget=self.rows_budget,
                op_scale_up=None,
            )

        degree = self.cheb_coefs.shape[-1] - 1
        # the backend's depth-optimal tree, at every degree
        num_levels = math.ceil(math.log2(degree + 1))
        input = infer_modswitch_by_num_levels(
            hom_params,
            input,
            num_levels=num_levels,
            rows_budget=self.rows_budget,
        )
        return input
