"""See `operators/ml/README.md` for usage details."""

from typing import Optional, Tuple

import torch

from lattica_build.operators.ml.h_conv import HomConv


class HomConvBnFused(HomConv):
    """Convolution with BatchNorm fused into weight and bias tensors.

    This keeps runtime execution as a single convolution op after parameter
    preprocessing.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Spatial kernel size `(kh, kw)` or scalar.
        stride: Spatial stride `(sh, sw)` or scalar.
        padding: Spatial zero-padding `(ph, pw)` or scalar.
        groups: Grouped convolution factor.
        dilation: Spatial dilation `(dh, dw)` or scalar.
        image_hw: Optional input image spatial size hint `(h, w)`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Tuple[int, int],
        stride: int | Tuple[int, int],
        padding: int | Tuple[int, int],
        groups: int = 1,
        dilation: Optional[int | Tuple[int, int]] = (1, 1),
        image_hw: Optional[int | Tuple[int, int]] = None,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups,
            bias=True,
            dilation=dilation,
            image_hw=image_hw,
        )

    # Fused weights formulas:
    # W_fused = γ · (W / √(σ² + ε))
    # B_fused = β + γ · ((B - μ) / √(σ² + ε))
    # where:

    # - W: Conv2d weight (conv.weight)
    # - B: Conv2d bias (conv.bias)
    # - μ: BatchNorm running mean (bn.running_mean)
    # - σ²: BatchNorm running variance (bn.running_var)
    # - γ: BatchNorm scale parameter (bn.weight)
    # - β: BatchNorm shift parameter (bn.bias)
    # - ε: BatchNorm epsilon for numerical stability
    # - delta: Additional rescaling factor
    def set_data(
        self,
        weight,
        bias,
        mean,
        var,
        gamma,
        beta,
        eps: float = 1e-5,
        delta: float = 1.0,
        **kwargs,
    ) -> None:
        """Fuse Conv+BN tensors and store equivalent convolution parameters.

        Args:
            weight: Convolution weights shaped `[C_out, C_in/groups, kh, kw]`.
            bias: Convolution bias shaped `[C_out]` (or `None`).
            mean: BatchNorm running mean (`mu`) shaped `[C_out]`.
            var: BatchNorm running variance (`sigma^2`) shaped `[C_out]`.
            gamma: BatchNorm scale (`gamma`) shaped `[C_out]` (or `None`).
            beta: BatchNorm shift (`beta`) shaped `[C_out]` (or `None`).
            eps: BatchNorm epsilon for numerical stability.
            delta: Extra multiplicative factor applied after fusion.
        """
        if bias is None:
            bias = torch.zeros(mean.shape)

        if gamma is None:
            gamma = torch.ones(mean.shape)

        if beta is None:
            beta = torch.zeros(mean.shape)

        scale = gamma / (var + eps) ** 0.5
        W_fused = delta * (weight * scale.view(-1, 1, 1, 1))

        # Compute the fused bias
        B_fused = delta * (beta + (bias - mean) * scale)

        super().set_data(W_fused, B_fused)
