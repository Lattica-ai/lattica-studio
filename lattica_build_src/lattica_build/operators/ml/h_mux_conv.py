"""Convolution operators on the multiplexed ("gap") packing of Lee et al. 2022.

A (C, H, W) feature map is one ciphertext with ``t`` channels interleaved per
pixel. Each conv spends one level; the bias add spends none.
"""

import math
from typing import Optional, Tuple, Union

import torch

from lattica_build.base_classes.hom_op import HomOp
from lattica_build.base_classes.hom_value import HomValue
from lattica_build.operators.ml.conv_bn_fused import fold_batch_norm
from lattica_build.operators.ml.h_conv import HomConvBase, _normalize_tuple, conv_output_hw
from lattica_build.params.level_and_scale_tracing import infer_optional_modswitch
from lattica_build.serialization.hom_op_pb2 import HomOpType


def _mux_slots(channels, image_hw):
    """Slots a (channels, H, W) mux layout takes, rounded up to a power of two."""
    h, w = image_hw
    used = (1 << (h - 1).bit_length()) * (1 << (w - 1).bit_length()) * channels
    return 1 << (used - 1).bit_length()


def mux_output_gap(t_in: int, out_channels: int) -> int:
    """Channels per pixel a mux conv leaves when it does not set t_out: the input's
    when they divide out_channels, else their gcd."""
    return t_in if out_channels % t_in == 0 else math.gcd(t_in, out_channels)


def _check_gap(channels, t, what):
    if t <= 0 or channels % t:
        raise ValueError(f"{what}: the gap t={t} must be positive and divide the {channels} channels")


def mux_pack(x, t: int, n_slots: int) -> torch.Tensor:
    """A (C, H, W) map as the slot vector a client encrypts for a mux conv: channel c of
    pixel (i, j) at slot ((g * Ht + i) * Wt + j) * t + cc, with g, cc = divmod(c, t) and
    Ht, Wt the sides rounded up to powers of 2."""
    x = torch.as_tensor(x, dtype=torch.float64)
    channels, h, w = x.shape
    _check_gap(channels, t, "mux_pack")
    ht, wt = 1 << (h - 1).bit_length(), 1 << (w - 1).bit_length()
    if channels * ht * wt > n_slots:
        raise ValueError(f"a {channels}x{h}x{w} map needs {channels * ht * wt} slots, got {n_slots}")
    c, i, j = torch.meshgrid(torch.arange(channels), torch.arange(h), torch.arange(w), indexing="ij")
    vec = torch.zeros(n_slots, dtype=torch.float64)
    vec[((c // t * ht + i) * wt + j) * t + c % t] = x
    return vec


def _infer_mux_output(input: HomValue, in_slots: int, out_slots: int, internal_n) -> HomValue:
    """The output ciphertext, packed at the output layout's slot count."""
    if internal_n is None:
        return input
    if max(in_slots, out_slots) > internal_n:
        raise NotImplementedError(
            "mux layouts must fit one ciphertext; "
            f"this one needs {max(in_slots, out_slots)} of {internal_n} slots")
    return input.make_copy(tensor_shape=(out_slots,), n_slots=out_slots)


class HomMuxConv(HomConvBase):
    """Convolution on the mux layout, without bias (see HomMuxConvBn).

    kernel_shape: (C_out, C_in, kh, kw). image_hw: (H, W) of the input.
    t_in: channels per pixel in the input. t_out: channels per pixel in the
    output of a strided conv; by default the conv keeps the input's.
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
        self._set_geometry(self.kernel_shape[2:], stride, padding, dilation, image_hw)
        _check_gap(self.kernel_shape[1], t_in, "HomMuxConv input")
        if t_out is not None:
            _check_gap(self.kernel_shape[0], t_out, "HomMuxConv output")
        self.t_in = t_in
        self.with_modswitch = with_modswitch
        if t_out is not None:
            self.t_out = t_out

    def output_hw(self):
        return conv_output_hw(self.image_hw, self.kernel_size, self.stride, self.padding, self.dilation)

    def infer_output_shape(self, input: HomValue, internal_n=None, **kwargs) -> HomValue:
        c_out, c_in = self.kernel_shape[:2]
        out_hw = self.output_hw()
        return _infer_mux_output(input, _mux_slots(c_in, self.image_hw),
                                 _mux_slots(c_out, out_hw), internal_n)

    def infer_output_level_and_scale(self, input: HomValue, hom_params=None, **kwargs) -> HomValue:
        return infer_optional_modswitch(hom_params, input, with_modswitch=self.with_modswitch,
                                        rows_budget=None, op_scale_up=None)

    def set_data(self, weight: torch.Tensor, **kwargs) -> None:
        assert tuple(weight.shape) == self.kernel_shape, (
            f"mux conv weight should be {self.kernel_shape}, got {tuple(weight.shape)}")
        super().set_data(weight)

    def forward_clear(self, *inputs, **kwargs):
        # without this, the leaf default would return the input unchanged
        raise NotImplementedError("HomMuxConv has no clear execution; the backend's apply_clear has one")


class HomMuxConvBn(HomOp):
    """Conv + batch norm on the mux layout: a HomMuxConv followed by a HomMuxBiasAdd,
    with the batch norm folded into them as in HomConvBnFused. A strided conv with
    t_out (e.g. stride 2, t_out = 2 * t_in) downsamples in the same level.
    """

    def __init__(self, in_channels, out_channels, kernel_size, t_in, image_hw,
                 stride=(1, 1), padding=(0, 0), dilation=(1, 1),
                 t_out: Optional[int] = None) -> None:
        super().__init__()
        self.conv = HomMuxConv(
            kernel_shape=(out_channels, in_channels, *_normalize_tuple(kernel_size, 2, 'kernel_size')),
            image_hw=image_hw, t_in=t_in, stride=stride, padding=padding, dilation=dilation, t_out=t_out)
        out_t = mux_output_gap(t_in, out_channels) if t_out is None else t_out
        self.bias_add = HomMuxBiasAdd(channels=out_channels, image_hw=self.conv.output_hw(), t=out_t)

    def forward(self, x: HomValue) -> HomValue:
        return self.bias_add(self.conv(x))

    def set_data(self, weight, bias, mean, var, gamma, beta,
                 eps: float = 1e-5, delta: float = 1.0, **kwargs) -> None:
        w_fused, b_fused = fold_batch_norm(weight, bias, mean, var, gamma, beta, eps, delta)
        self.conv.set_data(w_fused)
        self.bias_add.set_data(b_fused)


class HomMuxBiasAdd(HomOp):
    """Adds a per-channel bias on the mux layout. channels, image_hw and t describe
    the layout it is added on (the conv's output)."""

    OP_TYPE = HomOpType.MuxBiasAdd

    def __init__(self, channels, image_hw, t) -> None:
        super().__init__()
        _check_gap(channels, t, "HomMuxBiasAdd")
        self.channels = channels
        self.image_hw = _normalize_tuple(image_hw, 2, 'image_hw')
        self.t = t

    def set_data(self, bias: torch.Tensor, **kwargs) -> None:
        assert bias.ndim == 1 and len(bias) == self.channels, (
            f"mux bias should be 1D of length {self.channels}, got {tuple(bias.shape)}")
        super().set_data(bias)

    def forward_clear(self, *inputs, **kwargs):
        # without this, the leaf default would return the input unchanged
        raise NotImplementedError("HomMuxBiasAdd has no clear execution; the backend's apply_clear has one")
