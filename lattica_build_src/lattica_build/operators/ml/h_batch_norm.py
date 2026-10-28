"""See `operators/ml/README.md` for usage details."""

from typing import Union

import torch

from lattica_build.operators.arithmetic.h_const_add import HomConstAdd
from lattica_build.operators.arithmetic.h_const_mul import HomConstMul
from lattica_build.operators.composite.sequential import SequentialHomOp


class HomBatchNorm(SequentialHomOp):
    """BatchNorm as a composite `const_mul` + `const_add` graph.

    The op converts BatchNorm parameters into affine constants:
    `scale = gamma / sqrt(var + eps)` and `bias = beta - mean * scale`.

    Args:
        This composite op has no constructor parameters; data is provided by
        `set_data`.
    """

    def __init__(self) -> None:
        self.h_scale = HomConstMul()  # TODO need to set dims
        self.h_bias = HomConstAdd()  # TODO need to set dims

        super().__init__(self.h_scale, self.h_bias)

    def set_data(
        self,
        mean: torch.Tensor,
        var: torch.Tensor,
        gamma: Union[torch.Tensor, None],
        beta: Union[torch.Tensor, None],
        eps: float = 1e-5,
    ) -> None:
        """Set BatchNorm statistics and derive affine constants.

        `gamma`/`beta` default to ones/zeros when omitted. Parameters are stored
        as `(-1, 1, 1)` for channel-first broadcast usage.

        Args:
            mean: BatchNorm running mean shaped `[C]`.
            var: BatchNorm running variance shaped `[C]`.
            gamma: BatchNorm scale parameter shaped `[C]` (or `None`).
            beta: BatchNorm shift parameter shaped `[C]` (or `None`).
            eps: Epsilon added for numerical stability.
        """
        if mean is None or var is None:
            raise ValueError("Mean and variance must be provided")

        if gamma is None:
            gamma = torch.ones(mean.shape)

        if beta is None:
            beta = torch.zeros(mean.shape)

        scale = gamma / (var + eps) ** 0.5
        bias = beta - mean * scale
        self.h_scale.set_data(scale.view(-1, 1, 1))
        self.h_bias.set_data(bias.view(-1, 1, 1))
