"""See `operators/ml/README.md` for usage details."""

import math
from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as functional

from lattica_build.base_classes.hom_op import HomOp
from lattica_build.base_classes.hom_value import HomValue
from lattica_build.params.level_and_scale_tracing import infer_optional_modswitch
from lattica_build.serialization.hom_op_pb2 import HomOpType


def _normalize_tuple(value, n, name):
    """Normalize an int/iterable argument into an `n`-tuple of ints."""

    if isinstance(value, int):
        return (value,) * n

    value_error = ValueError(
        f"The `{name}` argument must be a tuple of {n} integers. "
        f"Received: {value}"
    )

    try:
        value_tuple = tuple(value)
    except TypeError:
        raise value_error
    if len(value_tuple) != n:
        raise value_error
    for single_value in value_tuple:
        try:
            int(single_value)
        except (ValueError, TypeError):
            raise value_error
    return value_tuple


def conv_output_hw(image_hw, kernel_size, stride, padding, dilation):
    """(H, W) of a conv's output, as torch.nn.Conv2d computes it, from (h, w) pairs."""
    return tuple((size + 2 * pad - dil * (k - 1) - 1) // s + 1
                 for size, k, s, pad, dil in zip(image_hw, kernel_size, stride, padding, dilation))


class HomConvBase(HomOp):
    """The geometry, clear execution and BatchNorm fold the convolution operators share."""

    def _set_geometry(self, kernel_size, stride, padding, dilation, image_hw=None):
        """Store kernel_size, stride, padding, dilation and image_hw as 2-tuples of ints."""
        self.kernel_size = _normalize_tuple(kernel_size, 2, "kernel_size")
        self.stride = _normalize_tuple(stride, 2, "stride")
        self.padding = (tuple((k - 1) // 2 for k in self.kernel_size) if padding is None
                        else _normalize_tuple(padding, 2, "padding"))
        self.dilation = _normalize_tuple(dilation, 2, "dilation")
        self.image_hw = None if image_hw is None else _normalize_tuple(image_hw, 2, "image_hw")
        self.output_hw = None if image_hw is None else conv_output_hw(
            self.image_hw, self.kernel_size, self.stride, self.padding, self.dilation)

    @staticmethod
    def _fold_batch_norm(weight, bias, mean, var, gamma, beta, eps=1e-5, delta=1.0):
        """Fold a BatchNorm into the preceding conv:

            W_fused = δ · γ · W / √(σ² + ε)
            B_fused = δ · (β + γ · (B - μ) / √(σ² + ε))

        with W, B the conv's weight and bias, μ, σ² the BatchNorm's running mean and
        variance, γ, β its scale and shift, ε its eps and δ an extra rescaling (delta).
        bias, gamma and beta may be None (0, 1 and 0)."""
        if bias is None:
            bias = torch.zeros(mean.shape)
        if gamma is None:
            gamma = torch.ones(mean.shape)
        if beta is None:
            beta = torch.zeros(mean.shape)
        scale = gamma / (var + eps) ** 0.5
        return delta * (weight * scale.view(-1, 1, 1, 1)), delta * (beta + (bias - mean) * scale)

    def _clear_conv2d(self, x, stride, padding, groups=1):
        data = self._require_clear_data()
        weight, *bias = (t.to(x) for t in (data if isinstance(data, tuple) else (data,)))
        return functional.conv2d(x, weight, *bias, stride=stride, padding=padding,
                                 dilation=self.dilation, groups=groups)


class HomConv(HomConvBase):
    """Base homomorphic 2D convolution operator.

    This is a backend base op (`OP_TYPE = HomOpType.Conv`) with explicit
    kernel/stride/padding/group configuration and optional bias.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Spatial kernel size `(kh, kw)` or scalar.
        stride: Spatial stride `(sh, sw)` or scalar.
        padding: Spatial zero-padding `(ph, pw)` or scalar. When `None`, uses
            'same' padding, `(k - 1) // 2` on each side.
        groups: Grouped convolution factor.
        bias: Whether this op expects a bias tensor in `set_data`.
        dilation: Spatial dilation `(dh, dw)` or scalar.
        image_hw: Optional input image spatial size hint `(h, w)`.
    """

    OP_TYPE = HomOpType.Conv

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Tuple[int, int],
        stride: int | Tuple[int, int],
        padding: Optional[int | Tuple[int, int]],
        groups: int = 1,
        bias: bool = False,
        dilation: Optional[int | Tuple[int, int]] = (1, 1),
        image_hw: Optional[int | Tuple[int, int]] = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self._set_geometry(kernel_size, stride, padding, dilation, image_hw)
        self.groups = groups
        self.bias = bias
        self.strided = self.stride != (1, 1)
        self.stride_pitch = tuple(s * d for s, d in zip(self.stride, self.dilation))

        if groups <= 0:
            raise ValueError(f"groups must be positive, got {groups}")
        if any(value <= 0 for value in self.stride):
            raise ValueError(f"stride must be positive, got {self.stride}")
        if any(value <= 0 for value in self.dilation):
            raise ValueError(f"dilation must be positive, got {self.dilation}")
        if in_channels % groups != 0:
            raise ValueError(
                f"in_channels ({in_channels}) must be divisible by groups ({groups})"
            )
        if out_channels % groups != 0:
            raise ValueError(
                f"out_channels ({out_channels}) must be divisible by groups ({groups})"
            )

        self.kernel_shape = (
            out_channels,
            in_channels // groups,
            self.kernel_size[0],
            self.kernel_size[1],
        )

    def infer_output_shape(self, input: HomValue, **kwargs) -> HomValue:
        """Infer convolution output tensor shape metadata."""
        return input.make_copy(
            tensor_shape=(self.kernel_shape[0], input.tensor_shape[1]),
            n_axis=-1
        )

    def infer_output_level_and_scale(
        self,
        input: HomValue,
        hom_params=None,
        **kwargs,
    ) -> HomValue:
        """Infer post-conv level/scale via optional modswitch policy."""
        # TODO: add with_modswitch and rows_budget parameters to the constructor
        # of HomConv and use them here.
        return infer_optional_modswitch(
            hom_params,
            input,
            with_modswitch=True,
            rows_budget=None,
            op_scale_up=None,
        )

    def set_data(self, *data: torch.Tensor, **kwargs) -> None:
        """Set convolution weight (and optional bias) tensors with validation."""
        if self.bias:
            assert len(data) == 2, "Conv define with bias=True, but data is missing"
            weight, bias = data
            assert bias.ndim == 1, "bias should be 1D"
            assert weight.shape[0] == len(bias), "bias should be of size C_out"
            super().set_data(weight, bias)
        else:
            weight = data[0]
            assert weight.shape == self.kernel_shape, (
                "Conv weight should be of shape [Cout, Cin // groups, H, W], "
                f"got {tuple(weight.shape)}, expected {self.kernel_shape}"
            )
            super().set_data(weight)

    def _resolve_image_hw(self, input):
        if input.ndim != 2:
            raise ValueError(
                "HomConv clear execution expects an unbatched packed tensor "
                "with shape (channels, packed spatial slots)"
            )

        channels, n_pixels = input.shape
        if channels != self.in_channels:
            raise ValueError(
                "Flattened clear convolution input has an incompatible number "
                f"of channels: got {channels}, expected {self.in_channels}"
            )

        if self.image_hw is not None:
            height, width = self.image_hw
        else:
            height = width = math.isqrt(n_pixels)
            if height * width != n_pixels:
                raise ValueError(
                    "Cannot infer square image dimensions from the packed input; "
                    "provide image_hw explicitly"
                )

        if n_pixels % (height * width):
            raise ValueError(
                "Flattened clear convolution input has an incompatible packed "
                "spatial size for image_hw"
            )
        return height, width


    def forward_clear(self, input):
        height, width = self._resolve_image_hw(input)
        channels, n_pixels = input.shape
        image_size = height * width
        n_tiles = n_pixels // image_size

        # Present packed tiles as the batch dimension expected by conv2d.
        conv_input = input.reshape(channels, n_tiles, height, width).permute(1, 0, 2, 3)
        raw_result = self._clear_conv2d(
            conv_input,
            # Compute the dense full-resolution result first. The strided layout is applied afterward.
            stride=1,
            padding=tuple(p * d for p, d in zip(self.padding, self.dilation)),
            groups=self.groups,
        )

        result = raw_result
        if self.strided:
            # Keep only positions used by the backend's strided output grid.
            rows = slice(None, None, self.stride_pitch[0])
            cols = slice(None, None, self.stride_pitch[1])
            result = torch.zeros_like(raw_result)
            result[..., rows, cols] = raw_result[..., rows, cols]

        # Restore the homomorphic packed layout: channels first, then tiles.
        return result.permute(1, 0, 2, 3).reshape(self.out_channels, -1)


class HomConvBnFused(HomConv):
    """Convolution with BatchNorm fused into weight and bias tensors."""

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
        super().set_data(*self._fold_batch_norm(weight, bias, mean, var, gamma, beta, eps, delta))
