"""Convolution operators on the multiplexed ("gap") packing of Lee et al. 2022.

A (C, H, W) feature map is one ciphertext with all its channels interleaved per
pixel. Each conv spends one level; its bias is a plaintext add (HomConstAdd) of the
bias spread over the output map, which spends none.
"""

from typing import Tuple, Union

import torch

from lattica_build.base_classes.hom_op import HomOp
from lattica_build.base_classes.hom_value import HomValue
from lattica_build.operators.arithmetic.h_const_add import HomConstAdd
from lattica_build.operators.ml.h_conv import HomConvBase, _normalize_tuple, conv_output_hw
from lattica_build.params.level_and_scale_tracing import infer_optional_modswitch
from lattica_build.serialization.hom_op_pb2 import HomOpType


def _mux_slots(channels, image_hw):
    """Slots a (channels, H, W) mux layout takes, rounded up to a power of two."""
    h, w = image_hw
    used = (1 << (h - 1).bit_length()) * (1 << (w - 1).bit_length()) * channels
    return 1 << (used - 1).bit_length()


def _mux_slot_indices(channels, h, w, device=None) -> torch.Tensor:
    """[C, H, W] slot of each map entry: (i * Wt + j) * C + c."""
    wt = 1 << (w - 1).bit_length()
    c, i, j = torch.meshgrid(torch.arange(channels, device=device), torch.arange(h, device=device),
                             torch.arange(w, device=device), indexing="ij")
    return (i * wt + j) * channels + c


def mux_pack(x, n_slots: int) -> torch.Tensor:
    """A (C, H, W) map as the slot vector a client encrypts for a mux conv: channel c of
    pixel (i, j) at slot (i * Wt + j) * C + c, with Wt the width rounded up to a power of 2.
    The map takes C * Ht * Wt slots, Ht the height rounded up the same way."""
    x = torch.as_tensor(x, dtype=torch.float64)
    channels, h, w = x.shape
    ht, wt = 1 << (h - 1).bit_length(), 1 << (w - 1).bit_length()
    if channels * ht * wt > n_slots:
        raise ValueError(f"a {channels}x{h}x{w} map needs {channels * ht * wt} slots, got {n_slots}")
    vec = torch.zeros(n_slots, dtype=torch.float64)
    vec[_mux_slot_indices(channels, h, w)] = x
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
    """Convolution on the mux layout, without bias (see HomMuxConvBias).

    kernel_shape: (C_out, C_in, kh, kw). image_hw: (H, W) of the input. A strided
    conv also re-interleaves its output to the C_out channels, in the same level.
    """

    OP_TYPE = HomOpType.MuxConv

    def __init__(self, kernel_shape, image_hw,
                 stride: Union[int, Tuple[int, int]] = (1, 1),
                 padding: Union[int, Tuple[int, int]] = (0, 0),
                 dilation: Union[int, Tuple[int, int]] = (1, 1),
                 with_modswitch: bool = True) -> None:
        super().__init__()
        self.kernel_shape = tuple(kernel_shape)
        self._set_geometry(self.kernel_shape[2:], stride, padding, dilation, image_hw)
        self.with_modswitch = with_modswitch

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

    def forward_clear(self, input):
        """The conv on slot vectors [..., n]: unpack the input map (the input repeats at its
        length, so a shorter one is tiled), torch's conv2d, pack into the output's slots."""
        weight = self._require_clear_data().to(dtype=input.dtype, device=input.device)
        c_out, c_in = self.kernel_shape[:2]
        h, w = self.image_hw
        in_slots = _mux_slots(c_in, self.image_hw)
        if input.shape[-1] < in_slots:
            input = input.repeat(*([1] * (input.dim() - 1)), in_slots // input.shape[-1])
        batch = input.shape[:-1]
        x = input.reshape(-1, input.shape[-1])[:, _mux_slot_indices(c_in, h, w, input.device)]
        y = torch.nn.functional.conv2d(x, weight, stride=self.stride, padding=self.padding,
                                       dilation=self.dilation)
        out = torch.zeros(y.shape[0], _mux_slots(c_out, self.output_hw()), dtype=input.dtype, device=input.device)
        out[:, _mux_slot_indices(c_out, *self.output_hw(), input.device)] = y
        return out.reshape(*batch, out.shape[-1])


class HomMuxConvBias(HomOp):
    """Conv + bias on the mux layout: a HomMuxConv, then a HomConstAdd of bias[c] at every
    slot of output channel c. A batch norm folds into the weight and bias first
    (conv_bn_fused.fold_batch_norm)."""

    def __init__(self, in_channels, out_channels, kernel_size, image_hw,
                 stride=(1, 1), padding=(0, 0), dilation=(1, 1)) -> None:
        super().__init__()
        self.conv = HomMuxConv(
            kernel_shape=(out_channels, in_channels, *_normalize_tuple(kernel_size, 2, 'kernel_size')),
            image_hw=image_hw, stride=stride, padding=padding, dilation=dilation)
        self._output_slots = _mux_slots(out_channels, self.conv.output_hw())
        self.bias_add = HomConstAdd(dims=(self._output_slots,))

    def forward(self, x: HomValue) -> HomValue:
        return self.bias_add(self.conv(x))

    def set_data(self, weight, bias, **kwargs) -> None:
        out_channels = self.conv.kernel_shape[0]
        bias = torch.as_tensor(bias, dtype=torch.float64)
        assert bias.shape == (out_channels,), (
            f"mux bias should be 1D of length {out_channels}, got {tuple(bias.shape)}")
        self.conv.set_data(weight)
        bias_map = bias[:, None, None].expand(out_channels, *self.conv.output_hw())
        self.bias_add.set_data(mux_pack(bias_map, self._output_slots))
