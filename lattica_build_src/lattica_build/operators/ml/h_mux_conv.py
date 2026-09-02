"""Build-side operators for the multiplexed ("gap") packing convolution and its
relayout companions. Each is a leaf HomOp whose serialized attributes match the
corresponding backend op's constructor (see
``latticabe.homomorphic_operations.conv_mux``):

    HomMuxConv          -> BackendHomMuxConv          (weight-only leaf)
    HomMuxStrideRepack  -> BackendHomMuxStrideRepack  (no data; geometric masks)
    HomMuxBiasAdd       -> BackendHomMuxBiasAdd       (per-channel bias leaf)
    HomMuxGlobalAvgPool -> BackendHomMuxGlobalAvgPool (no data; geometric masks)

LAYOUT CONTRACT: input and output are ONE ciphertext in the mux gap layout, so the
ciphertext shape (external_shape == (n_slots,)) is unchanged by every op -- only the
logical (C, H, W) it encodes changes. Hence ``infer_output_shape`` is the identity
(inherited); only the level bookkeeping differs (each op spends one mult level via
the mask multiply).

See `operators/ml/README.md` for usage details.
"""

import math
from typing import Optional, Tuple, Union

import torch

from lattica_build.base_classes.hom_op import HomOp
from lattica_build.base_classes.hom_value import HomValue
from lattica_build.operators.ml.h_conv import _normalize_tuple
from lattica_build.params.level_and_scale_tracing import infer_optional_modswitch
from lattica_build.serialization.hom_op_pb2 import HomOpType


def _conv_out_dims(image_hw, kernel, stride, padding, dilation):
    """Conv output (H, W) -- same formula as torch.nn.Conv2d / the mux compiler."""
    (h, w), (kh, kw), (sh, sw), (ph, pw), (dh, dw) = image_hw, kernel, stride, padding, dilation
    return ((h + 2 * ph - dh * (kh - 1) - 1) // sh + 1,
            (w + 2 * pw - dw * (kw - 1) - 1) // sw + 1)


class HomMuxConv(HomOp):
    """Aligned convolution on the mux layout (weight only; fold BN bias separately).

    kernel_shape : (C_out, C_in, kh, kw).  image_hw : (H, W) of the INPUT map.
    t_in : input gap (channels per pixel); the aligned case is t_in == C_in.
    """

    OP_TYPE = HomOpType.MuxConv

    def __init__(self, kernel_shape, image_hw, t_in,
                 stride: Union[int, Tuple[int, int]] = (1, 1),
                 padding: Union[int, Tuple[int, int]] = (0, 0),
                 dilation: Union[int, Tuple[int, int]] = (1, 1),
                 with_modswitch: bool = True, n_cosets: int = 1,
                 t_out: Optional[int] = None) -> None:
        super().__init__()
        self.kernel_shape = tuple(kernel_shape)
        self.image_hw = _normalize_tuple(image_hw, 2, 'image_hw')
        self.t_in = t_in
        self.stride = _normalize_tuple(stride, 2, 'stride')
        self.padding = _normalize_tuple(padding, 2, 'padding')
        self.dilation = _normalize_tuple(dilation, 2, 'dilation')
        self.with_modswitch = with_modswitch
        self.n_cosets = n_cosets
        # t_out: single-shot fused transition -- run strided and re-interleave at
        # t_out (=stride*t_in), so the decimation costs no separate repack level.
        if t_out is not None:
            self.t_out = t_out

    def infer_output_level_and_scale(self, input: HomValue, hom_params=None, **kwargs) -> HomValue:
        return infer_optional_modswitch(hom_params, input, with_modswitch=self.with_modswitch,
                                        rows_budget=None, op_scale_up=None)

    def set_data(self, weight: torch.Tensor, **kwargs) -> None:
        assert tuple(weight.shape) == self.kernel_shape, (
            f"mux conv weight should be {self.kernel_shape}, got {tuple(weight.shape)}")
        super().set_data(weight)


class HomMuxConvBn(HomOp):
    """Fused conv + batch-norm on the mux layout, as a composite of a weight-only
    HomMuxConv followed by a HomMuxBiasAdd (the single-tensor leaf serialization
    can't carry weight+bias together). BN is folded into the conv weight/bias
    offline exactly as HomConvBnFused does:

        W_fused = delta * gamma * W / sqrt(var + eps)
        B_fused = delta * (beta + gamma * (bias - mean) / sqrt(var + eps))

    Aligned (stride 1, t_out=None): output keeps the input H/W and gap. Single-shot
    fused transition (stride 2, t_out=stride*t_in): the conv also decimates and
    re-interleaves, so the output is (out_channels, H//s, W//s, t_out) and no
    separate repack level is spent -- the bias op is placed on that output layout.
    """

    def __init__(self, in_channels, out_channels, kernel_size, t_in, image_hw,
                 stride=(1, 1), padding=(0, 0), dilation=(1, 1),
                 t_out: Optional[int] = None, n_cosets: int = 1) -> None:
        super().__init__()
        self.n_cosets = n_cosets
        kh, kw = _normalize_tuple(kernel_size, 2, 'kernel_size')
        stride = _normalize_tuple(stride, 2, 'stride')
        padding = _normalize_tuple(padding, 2, 'padding')
        dilation = _normalize_tuple(dilation, 2, 'dilation')
        image_hw = _normalize_tuple(image_hw, 2, 'image_hw')
        self.conv = HomMuxConv(
            kernel_shape=(out_channels, in_channels, kh, kw), image_hw=image_hw,
            t_in=t_in, stride=stride, padding=padding, dilation=dilation,
            t_out=t_out, n_cosets=n_cosets)
        # bias sits on the conv's OUTPUT layout. For an aligned conv (t_out None)
        # the output gap follows the compiler's gap policy (conv_output_layout):
        # keep t_in if it divides C_out, else fall back to gcd(t_in, C_out) -- e.g.
        # the 64->10 FC head keeps t=gcd(64,10)=2, not t_in=64 (which wouldn't
        # divide C_out and is not a valid mux layout).
        if t_out is None:
            out_hw = image_hw
            out_t = t_in if out_channels % t_in == 0 else math.gcd(t_in, out_channels)
        else:
            out_hw = _conv_out_dims(image_hw, (kh, kw), stride, padding, dilation)
            out_t = t_out
        self.bias_add = HomMuxBiasAdd(channels=out_channels, image_hw=out_hw, t=out_t,
                                      n_cosets=n_cosets)

    def forward(self, x: HomValue) -> HomValue:
        return self.bias_add(self.conv(x))

    def set_data(self, weight, bias, mean, var, gamma, beta,
                 eps: float = 1e-5, delta: float = 1.0, **kwargs) -> None:
        if bias is None:
            bias = torch.zeros(mean.shape)
        if gamma is None:
            gamma = torch.ones(mean.shape)
        if beta is None:
            beta = torch.zeros(mean.shape)
        scale = gamma / (var + eps) ** 0.5
        w_fused = delta * (weight * scale.view(-1, 1, 1, 1))
        b_fused = delta * (beta + (bias - mean) * scale)
        self.conv.set_data(w_fused)
        self.bias_add.set_data(b_fused)


class HomMuxStrideRepack(HomOp):
    """Decimate the spatial grid by ``stride`` and re-interleave at gap ``t_new``."""

    OP_TYPE = HomOpType.MuxStrideRepack

    def __init__(self, channels, image_hw, t_in,
                 stride: Union[int, Tuple[int, int]], t_new: int,
                 with_modswitch: bool = True, n_cosets: int = 1) -> None:
        super().__init__()
        self.channels = channels
        self.image_hw = _normalize_tuple(image_hw, 2, 'image_hw')
        self.t_in = t_in
        self.stride = _normalize_tuple(stride, 2, 'stride')
        self.t_new = t_new
        self.with_modswitch = with_modswitch
        self.n_cosets = n_cosets

    def infer_output_level_and_scale(self, input: HomValue, hom_params=None, **kwargs) -> HomValue:
        return infer_optional_modswitch(hom_params, input, with_modswitch=self.with_modswitch,
                                        rows_budget=None, op_scale_up=None)


class HomMuxBiasAdd(HomOp):
    """Folded-BN bias on the mux layout: bias[co] at every valid output slot.
    Carries the raw per-channel bias (one tensor); the backend expands it to a
    per-slot vector using the output layout. No level cost (packing add).

    channels/image_hw/t are the conv OUTPUT layout (aligned conv: C_out, input
    H/W, t_out)."""

    OP_TYPE = HomOpType.MuxBiasAdd

    def __init__(self, channels, image_hw, t, n_cosets: int = 1) -> None:
        super().__init__()
        self.n_cosets = n_cosets
        self.channels = channels
        self.image_hw = _normalize_tuple(image_hw, 2, 'image_hw')
        self.t = t

    def set_data(self, bias: torch.Tensor, **kwargs) -> None:
        assert bias.ndim == 1 and len(bias) == self.channels, (
            f"mux bias should be 1D of length {self.channels}, got {tuple(bias.shape)}")
        super().set_data(bias)


class HomMuxGlobalAvgPool(HomOp):
    """Global average pool over H x W on the mux layout (group count 1, square,
    power-of-two). Output: channel c's average in slot c."""

    OP_TYPE = HomOpType.MuxGlobalAvgPool

    def __init__(self, channels, image_hw, t_in, with_modswitch: bool = True,
                 n_cosets: int = 1) -> None:
        super().__init__()
        self.channels = channels
        self.image_hw = _normalize_tuple(image_hw, 2, 'image_hw')
        self.t_in = t_in
        self.with_modswitch = with_modswitch
        self.n_cosets = n_cosets

    def infer_output_level_and_scale(self, input: HomValue, hom_params=None, **kwargs) -> HomValue:
        return infer_optional_modswitch(hom_params, input, with_modswitch=self.with_modswitch,
                                        rows_budget=None, op_scale_up=None)
