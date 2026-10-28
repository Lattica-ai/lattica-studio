"""See `operators/ml/README.md` for usage details."""

from typing import Optional, Tuple

import torch

from lattica_build.operators.ml.h_conv import HomConv


class HomAvgPool(HomConv):
    """Average pooling implemented as fixed-weight depthwise convolution.

    Args:
        channels: Number of input/output channels.
        kernel_size: Spatial window size for averaging.
        stride: Step between pooling windows.
        dilation: Dilation applied to the averaging kernel.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int | Tuple[int, int],
        stride: int | Tuple[int, int],
        dilation: Optional[int | Tuple[int, int]] = (1, 1),
    ):
        super().__init__(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            groups=channels,
            dilation=dilation,
        )

        avg_kernel = torch.ones(self.kernel_shape) / (
            self.kernel_size[0] * self.kernel_size[1]
        )
        super().set_data(avg_kernel)
