"""Build-side operators for the multiplexed ("gap") packing convolution and its
relayout companions. Each is a leaf HomOp whose serialized attributes match the
corresponding backend op's constructor (see
``latticabe.homomorphic_operations.conv_mux``):

    HomMuxConv          -> BackendHomMuxConv          (weight-only leaf)
    HomMuxStrideRepack  -> BackendHomMuxStrideRepack  (no data; geometric masks)
    HomMuxBiasAdd       -> BackendHomMuxBiasAdd       (per-channel bias leaf)

LAYOUT CONTRACT: input and output are ONE ciphertext in the mux gap layout, with
external_shape == (n_slots,). The output is packed at its own layout's period, so a
conv that adds channels widens the ciphertext and a stride repack narrows it; the
bias add keeps it. Each conv and repack spends one mult level on the mask multiply.

See `operators/ml/README.md` for usage details.
"""

import math
from typing import Optional, Tuple, Union

import torch

from lattica_build.base_classes.hom_op import HomOp
from lattica_build.base_classes.hom_value import HomValue
from lattica_build.operators.ml.h_conv import _normalize_tuple, conv_output_hw
from lattica_build.params.level_and_scale_tracing import infer_optional_modswitch
from lattica_build.serialization.hom_op_pb2 import HomOpType


def _mux_slots(channels, image_hw):
    """Slots a (channels, H, W) mux layout occupies in one ciphertext, rounded up to
    a power of two: the padded spatial grid times the channel count, whatever the gap."""
    h, w = image_hw
    used = (1 << (h - 1).bit_length()) * (1 << (w - 1).bit_length()) * channels
    return 1 << (used - 1).bit_length()


def _infer_mux_output(input: HomValue, in_slots: int, out_slots: int, internal_n) -> HomValue:
    """The backend op leaves one ciphertext packed at the output layout's period."""
    if internal_n is None:
        return input
    if max(in_slots, out_slots) > internal_n:
        raise NotImplementedError(
            "build-side shape inference covers mux layouts that fit one ciphertext; "
            f"this one needs {max(in_slots, out_slots)} of {internal_n} slots")
    return input.make_copy(tensor_shape=(out_slots,), n_slots=out_slots)


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
                 with_modswitch: bool = True,
                 t_out: Optional[int] = None) -> None:
        super().__init__()
        self.kernel_shape = tuple(kernel_shape)
        self.image_hw = _normalize_tuple(image_hw, 2, 'image_hw')
        self.t_in = t_in
        self.stride = _normalize_tuple(stride, 2, 'stride')
        self.padding = _normalize_tuple(padding, 2, 'padding')
        self.dilation = _normalize_tuple(dilation, 2, 'dilation')
        self.with_modswitch = with_modswitch
        if t_out is not None:
            self.t_out = t_out

    def infer_output_shape(self, input: HomValue, internal_n=None, **kwargs) -> HomValue:
        c_out, c_in, kh, kw = self.kernel_shape
        out_hw = conv_output_hw(self.image_hw, (kh, kw), self.stride, self.padding, self.dilation)
        return _infer_mux_output(input, _mux_slots(c_in, self.image_hw),
                                 _mux_slots(c_out, out_hw), internal_n)

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
                 t_out: Optional[int] = None) -> None:
        super().__init__()
        (kh, kw) = _normalize_tuple(kernel_size, 2, 'kernel_size')
        (sh, sw) = _normalize_tuple(stride, 2, 'stride')
        (ph, pw) = _normalize_tuple(padding, 2, 'padding')
        (dh, dw) = _normalize_tuple(dilation, 2, 'dilation')
        (h, w) = _normalize_tuple(image_hw, 2, 'image_hw')
        self.conv = HomMuxConv(
            kernel_shape=(out_channels, in_channels, kh, kw), image_hw=(h, w),
            t_in=t_in, stride=(sh, sw), padding=(ph, pw), dilation=(dh, dw),
            t_out=t_out)
        if t_out is None:
            out_hw = image_hw
            out_t = t_in if out_channels % t_in == 0 else math.gcd(t_in, out_channels)
        else:
            out_hw = conv_output_hw((h, w), (kh, kw), (sh, sw), (ph, pw), (dh, dw))
            out_t = t_out
        self.bias_add = HomMuxBiasAdd(channels=out_channels, image_hw=out_hw, t=out_t)

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
                 with_modswitch: bool = True) -> None:
        super().__init__()
        self.channels = channels
        self.image_hw = _normalize_tuple(image_hw, 2, 'image_hw')
        self.t_in = t_in
        self.stride = _normalize_tuple(stride, 2, 'stride')
        self.t_new = t_new
        self.with_modswitch = with_modswitch

    def infer_output_shape(self, input: HomValue, internal_n=None, **kwargs) -> HomValue:
        (h, w), (sh, sw) = self.image_hw, self.stride
        return _infer_mux_output(input, _mux_slots(self.channels, (h, w)),
                                 _mux_slots(self.channels, (h // sh, w // sw)), internal_n)

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

    def __init__(self, channels, image_hw, t) -> None:
        super().__init__()
        self.channels = channels
        self.image_hw = _normalize_tuple(image_hw, 2, 'image_hw')
        self.t = t

    def set_data(self, bias: torch.Tensor, **kwargs) -> None:
        assert bias.ndim == 1 and len(bias) == self.channels, (
            f"mux bias should be 1D of length {self.channels}, got {tuple(bias.shape)}")
        super().set_data(bias)
