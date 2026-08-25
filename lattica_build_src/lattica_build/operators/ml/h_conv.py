"""See `operators/ml/README.md` for usage details."""

from typing import Optional, Sequence, Tuple

import torch

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


class HomConv(HomOp):
    """Base homomorphic 2D convolution operator.

    This is a backend base op (`OP_TYPE = HomOpType.Conv`) with explicit
    kernel/stride/padding/group configuration and optional bias.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Spatial kernel size `(kh, kw)` or scalar.
        stride: Spatial stride `(sh, sw)` or scalar.
        padding: Spatial zero-padding `(ph, pw)` or scalar. When `None`, uses
            full kernel padding.
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
        self.kernel_size = _normalize_tuple(kernel_size, 2, "kernel_size")
        if padding is None:
            self.padding = (self.kernel_size[0], self.kernel_size[1])
        else:
            self.padding = _normalize_tuple(padding, 2, "padding")
        self.stride = _normalize_tuple(stride, 2, "stride")
        self.groups = groups
        self.bias = bias
        self.dilation = _normalize_tuple(dilation, 2, "dilation")
        self.image_hw = (
            _normalize_tuple(image_hw, 2, "image_hw") if image_hw is not None else None
        )

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
