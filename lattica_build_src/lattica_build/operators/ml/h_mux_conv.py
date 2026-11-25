"""Convolution operator on the multiplexed ("gap") packing.

A (C, H, W) feature map is one ciphertext with all its channels interleaved per
pixel. Each conv spends one level.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch

from lattica_build.base_classes.hom_value import HomValue, to_neg_axis
from lattica_build.operators.ml.h_conv import HomConvBase
from lattica_build.params.level_and_scale_tracing import infer_optional_modswitch
from lattica_build.params.shape_tracing import infer_n_batch
from lattica_build.serialization.hom_op_pb2 import HomOpType


def next_pow2(x: int) -> int:
    """The smallest power of 2 >= x (1 for x <= 1)."""
    return 1 << (max(int(x), 1) - 1).bit_length()


@dataclass(frozen=True)
class MuxLayout:
    """A (C, H, W) map in one ciphertext of n_slots slots.

    Each pixel holds its C channels side by side: channel c of pixel (i, j) sits at slot
    (i * row_gap * Wt + j) * C + c, with Wt the width rounded up to a power of 2. With
    row_gap > 1 there is a gap between the map's rows, row_gap grid rows apart with the
    rows between empty: what a strided conv with keep_gap leaves. n_slots defaults to the
    layout's period.
    """

    channels: int
    height: int
    width: int
    n_slots: Optional[int] = None
    row_gap: int = 1

    def __post_init__(self) -> None:
        if self.n_slots is None:
            object.__setattr__(self, "n_slots", self.period)
        if min(self.channels, self.height, self.width, self.n_slots, self.row_gap) <= 0:
            raise ValueError(f"layout sizes must be positive, got {self}")
        if self.size > self.n_slots:
            raise ValueError(f"a {self.channels}x{self.height}x{self.width} map needs {self.size} "
                             f"slots, the ciphertext has {self.n_slots}")

    @property
    def size(self) -> int:
        """Slots the map spans: C channels on a power-of-2 grid, its empty rows included."""
        return next_pow2(self.height) * self.row_gap * next_pow2(self.width) * self.channels

    @property
    def period(self) -> int:
        """The size rounded up to a power of 2: the least period that holds the map."""
        return next_pow2(self.size)

    def slot(self, channel, row, column):
        """The slot of each (channel, row, column); takes ints or index arrays, numpy or torch."""
        return (row * self.row_gap * next_pow2(self.width) + column) * self.channels + channel

    def slot_map(self) -> torch.Tensor:
        """[C, H, W] slot of each map entry."""
        return self.slot(*torch.meshgrid(torch.arange(self.channels), torch.arange(self.height),
                                         torch.arange(self.width), indexing="ij"))

    def pack(self, x) -> torch.Tensor:
        """A (C, H, W) map as the slot vector a client encrypts."""
        x = torch.as_tensor(x, dtype=torch.float64)
        assert tuple(x.shape) == (self.channels, self.height, self.width), (
            f"pack expects {(self.channels, self.height, self.width)}, got {tuple(x.shape)}")
        vec = torch.zeros(self.n_slots, dtype=torch.float64)
        vec[self.slot_map()] = x
        return vec

    def pack_channels(self, values) -> torch.Tensor:
        """A per-channel vector (C,) at every pixel of its channel, as slots."""
        values = torch.as_tensor(values, dtype=torch.float64)
        assert tuple(values.shape) == (self.channels,), (
            f"pack_channels expects {(self.channels,)}, got {tuple(values.shape)}")
        return self.pack(values[:, None, None].expand(self.channels, self.height, self.width))

    def conv_output(self, channels: int, output_hw, stride: int, keep_gap: bool) -> "MuxLayout":
        """The layout of a conv's (channels, *output_hw) output, at its own period: compact, or with
        keep_gap with the gap the stride opens, its rows `stride` times as far apart as this map's."""
        return MuxLayout(channels, *output_hw, row_gap=self.row_gap * stride if keep_gap else 1)

    def unpack(self, vec) -> torch.Tensor:
        """Slot vectors [..., n] -> maps [..., C, H, W], as the client decodes them."""
        vec = torch.as_tensor(vec)
        return vec[..., self.slot_map()]


def _infer_mux_output(input: HomValue, in_slots: int, out_slots: int, internal_n) -> HomValue:
    """The output ciphertext, packed at the output layout's slot count; axes before the slots are a batch of
    maps and pass through. The maps of a batch of plaintexts are packed as full as their ciphertexts take, so a
    conv whose output maps are smaller merges ciphertexts."""
    if max(in_slots, out_slots) * input.n_batch > internal_n:
        raise NotImplementedError(
            "mux layouts must fit one ciphertext; "
            f"this one needs {max(in_slots, out_slots)} of {internal_n // input.n_batch} slots")
    if input.n_axis is not None and to_neg_axis(input.n_axis, input.tensor_shape) != -1:
        raise ValueError(f"mux conv needs the slots on the last axis; got {tuple(input.tensor_shape)} "
                         f"with slots on axis {input.n_axis}")
    n_batch = input.n_batch
    if input.batch_size is not None:
        n_batch = infer_n_batch(internal_n, out_slots, input.batch_size)
    return input.make_copy(tensor_shape=(*input.tensor_shape[:-1], out_slots), n_slots=out_slots, n_batch=n_batch)


class HomMuxConv(HomConvBase):
    """Convolution on the mux layout.

    kernel_shape: (C_out, C_in, kh, kw). image_hw: (H, W) of the input. A strided
    conv also re-interleaves its output to the C_out channels, in the same level.
    bias: a per-channel bias after the weight, added at every slot of
    its channel.
    row_gap: the input layout's (MuxLayout). keep_gap: a strided conv's output keeps the
    gap its stride opens, its channels in the skipped columns, which takes far fewer
    rotations; the convs after it take its row_gap. Needs a square stride and
    C_out = stride * C_in.
    """

    OP_TYPE = HomOpType.MuxConv

    def __init__(self, kernel_shape, image_hw,
                 stride: Union[int, Tuple[int, int]] = (1, 1),
                 padding: Union[int, Tuple[int, int]] = (0, 0),
                 dilation: Union[int, Tuple[int, int]] = (1, 1),
                 with_modswitch: bool = True,
                 bias: bool = False,
                 row_gap: int = 1,
                 keep_gap: bool = False) -> None:
        super().__init__()
        self.kernel_shape = tuple(kernel_shape)
        self._set_geometry(self.kernel_shape[2:], stride, padding, dilation, image_hw)
        self.with_modswitch = with_modswitch
        self.bias = bias
        self.row_gap = row_gap
        self.keep_gap = keep_gap
        c_out, c_in = self.kernel_shape[:2]
        if keep_gap and self.stride[0] != self.stride[1]:
            raise ValueError(f"keep_gap needs a square stride, got {self.stride}")
        if keep_gap and c_out != self.stride[1] * c_in:
            raise ValueError(f"keep_gap puts the output's channels in the slots of the pixels a stride skips: "
                             f"C_out must be stride * C_in = {self.stride[1] * c_in}, got {c_out}")

    def _layouts(self):
        """The input and output maps' layouts, each at its own period."""
        c_out, c_in = self.kernel_shape[:2]
        input_layout = MuxLayout(c_in, *self.image_hw, row_gap=self.row_gap)
        return input_layout, input_layout.conv_output(c_out, self.output_hw, self.stride[0], self.keep_gap)

    def infer_output_shape(self, input: HomValue, internal_n=None, **kwargs) -> HomValue:
        input_layout, output_layout = self._layouts()
        return _infer_mux_output(input, input_layout.period, output_layout.period, internal_n)

    def infer_output_level_and_scale(self, input: HomValue, hom_params=None, **kwargs) -> HomValue:
        return infer_optional_modswitch(hom_params, input, with_modswitch=self.with_modswitch,
                                        rows_budget=None, op_scale_up=None)

    def set_data(self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None, **kwargs) -> "HomMuxConv":
        assert tuple(weight.shape) == self.kernel_shape, (
            f"mux conv weight should be {self.kernel_shape}, got {tuple(weight.shape)}")
        if not self.bias:
            assert bias is None, "HomMuxConv defined with bias=False, but got a bias"
            return super().set_data(weight)
        assert bias is not None, "HomMuxConv defined with bias=True, but the bias is missing"
        bias = torch.as_tensor(bias, dtype=torch.float64)
        assert bias.shape == (self.kernel_shape[0],), (
            f"mux conv bias should be 1D of length {self.kernel_shape[0]}, got {tuple(bias.shape)}")
        return super().set_data(weight, bias)

    def forward_clear(self, input):
        """The conv on slot vectors [..., n]: unpack the input map (the input repeats at its
        length, so a shorter one is tiled), the clear conv HomConv also runs, bias included,
        pack into the output's slots."""
        input_layout, output_layout = self._layouts()
        if input.shape[-1] < input_layout.n_slots:
            input = input.repeat(*([1] * (input.dim() - 1)), input_layout.n_slots // input.shape[-1])
        batch = input.shape[:-1]
        x = input_layout.unpack(input.reshape(-1, input.shape[-1]))
        y = self._clear_conv2d(x, stride=self.stride, padding=self.padding)
        out = torch.zeros(y.shape[0], output_layout.n_slots, dtype=input.dtype, device=input.device)
        out[:, output_layout.slot_map()] = y
        return out.reshape(*batch, out.shape[-1])


class HomMuxConvBnFused(HomMuxConv):
    """Mux convolution with BatchNorm fused into its weight and bias."""

    def __init__(self, kernel_shape, image_hw,
                 stride: Union[int, Tuple[int, int]] = (1, 1),
                 padding: Union[int, Tuple[int, int]] = (0, 0),
                 dilation: Union[int, Tuple[int, int]] = (1, 1),
                 with_modswitch: bool = True,
                 row_gap: int = 1,
                 keep_gap: bool = False) -> None:
        super().__init__(kernel_shape, image_hw, stride, padding, dilation, with_modswitch, bias=True,
                         row_gap=row_gap, keep_gap=keep_gap)

    def set_data(self, weight, bias, mean, var, gamma, beta, eps: float = 1e-5,
                 delta: float = 1.0, **kwargs) -> "HomMuxConvBnFused":
        """Fuse Conv+BN tensors into the conv's weight and bias; the arguments are
        HomConvBnFused.set_data's."""
        return super().set_data(*self._fold_batch_norm(weight, bias, mean, var, gamma, beta, eps, delta))
