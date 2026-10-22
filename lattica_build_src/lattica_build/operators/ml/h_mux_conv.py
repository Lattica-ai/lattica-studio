"""Convolution operator on the multiplexed ("gap") packing.

A (C, H, W) feature map is one ciphertext with all its channels interleaved per
pixel. Each conv spends one level.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch

from lattica_build.base_classes.hom_value import HomValue
from lattica_build.operators.ml.h_conv import HomConvBase
from lattica_build.params.level_and_scale_tracing import infer_optional_modswitch
from lattica_build.serialization.hom_op_pb2 import HomOpType


def next_pow2(x: int) -> int:
    """The smallest power of 2 >= x (1 for x <= 1)."""
    return 1 << (max(int(x), 1) - 1).bit_length()


@dataclass(frozen=True)
class MuxLayout:
    """A (C, H, W) map in one ciphertext of n_slots slots.

    Each pixel holds its C channels side by side: channel c of pixel (i, j) sits at slot
    (i * Wt + j) * C + c, with Wt the width rounded up to a power of 2. n_slots defaults
    to the layout's period.
    """

    channels: int
    height: int
    width: int
    n_slots: Optional[int] = None

    def __post_init__(self) -> None:
        if self.n_slots is None:
            object.__setattr__(self, "n_slots", self.period)
        if min(self.channels, self.height, self.width, self.n_slots) <= 0:
            raise ValueError(f"layout sizes must be positive, got {self}")
        if self.size > self.n_slots:
            raise ValueError(f"a {self.channels}x{self.height}x{self.width} map needs {self.size} "
                             f"slots, the ciphertext has {self.n_slots}")

    @property
    def size(self) -> int:
        """Slots the map occupies: C channels on a power-of-2 grid."""
        return next_pow2(self.height) * next_pow2(self.width) * self.channels

    @property
    def period(self) -> int:
        """The size rounded up to a power of 2: the least period that holds the map."""
        return next_pow2(self.size)

    def slot(self, channel, row, column):
        """The slot of each (channel, row, column); takes ints or index arrays, numpy or torch."""
        return (row * next_pow2(self.width) + column) * self.channels + channel

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

    def unpack(self, vec) -> torch.Tensor:
        """Slot vectors [..., n] -> maps [..., C, H, W], as the client decodes them."""
        vec = torch.as_tensor(vec)
        return vec[..., self.slot_map()]


def _infer_mux_output(input: HomValue, in_slots: int, out_slots: int, internal_n) -> HomValue:
    """The output ciphertext, packed at the output layout's slot count."""
    if max(in_slots, out_slots) > internal_n:
        raise NotImplementedError(
            "mux layouts must fit one ciphertext; "
            f"this one needs {max(in_slots, out_slots)} of {internal_n} slots")
    return input.make_copy(tensor_shape=(out_slots,), n_slots=out_slots)


class HomMuxConv(HomConvBase):
    """Convolution on the mux layout.

    kernel_shape: (C_out, C_in, kh, kw). image_hw: (H, W) of the input. A strided
    conv also re-interleaves its output to the C_out channels, in the same level.
    bias: a per-channel bias after the weight, added at every slot of
    its channel.
    with_modswitch: rescale after the product. pt_scale: the scale the weight is encoded at;
    None is the rescale's factor (the output keeps the input's scale), or without the rescale
    hom_params.pt_scale.
    drop_rows: the conv first drops the input's first drop_rows rows, the ones rescales would consume next, without
    a rescale (the scale is unchanged): an output used only at a lower level, e.g. a residual block's downsample
    shortcut, is computed there, on fewer limbs, with a plaintext of fewer limbs.
    """

    OP_TYPE = HomOpType.MuxConv

    def __init__(self, kernel_shape, image_hw,
                 stride: Union[int, Tuple[int, int]] = (1, 1),
                 padding: Union[int, Tuple[int, int]] = (0, 0),
                 dilation: Union[int, Tuple[int, int]] = (1, 1),
                 with_modswitch: bool = True,
                 bias: bool = False,
                 pt_scale: Optional[int] = None,
                 drop_rows: int = 0) -> None:
        super().__init__()
        self.kernel_shape = tuple(kernel_shape)
        self._set_geometry(self.kernel_shape[2:], stride, padding, dilation, image_hw)
        self.with_modswitch = with_modswitch
        self.bias = bias
        self.pt_scale = pt_scale
        self.drop_rows = drop_rows

    def _layouts(self):
        """The input and output maps' layouts, each at its own period."""
        c_out, c_in = self.kernel_shape[:2]
        return MuxLayout(c_in, *self.image_hw), MuxLayout(c_out, *self.output_hw)

    def infer_output_shape(self, input: HomValue, internal_n=None, **kwargs) -> HomValue:
        input_layout, output_layout = self._layouts()
        return _infer_mux_output(input, input_layout.period, output_layout.period, internal_n)

    def infer_output_level_and_scale(self, input: HomValue, hom_params=None, **kwargs) -> HomValue:
        if self.drop_rows:
            assert self.drop_rows < len(input.active_rows), (
                f"drop_rows {self.drop_rows} leaves none of the input's {len(input.active_rows)} rows")
            input = input.make_copy(active_rows=input.active_rows[self.drop_rows:],
                                    active_cols=input.active_cols[self.drop_rows:])
        return infer_optional_modswitch(hom_params, input, with_modswitch=self.with_modswitch,
                                        rows_budget=None, op_scale_up=self.pt_scale)

    def set_data(self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None, **kwargs) -> None:
        assert tuple(weight.shape) == self.kernel_shape, (
            f"mux conv weight should be {self.kernel_shape}, got {tuple(weight.shape)}")
        if not self.bias:
            assert bias is None, "HomMuxConv defined with bias=False, but got a bias"
            super().set_data(weight)
            return
        assert bias is not None, "HomMuxConv defined with bias=True, but the bias is missing"
        bias = torch.as_tensor(bias, dtype=torch.float64)
        assert bias.shape == (self.kernel_shape[0],), (
            f"mux conv bias should be 1D of length {self.kernel_shape[0]}, got {tuple(bias.shape)}")
        super().set_data(weight, bias)

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
                 pt_scale: Optional[int] = None,
                 drop_rows: int = 0) -> None:
        super().__init__(kernel_shape, image_hw, stride, padding, dilation, with_modswitch, bias=True,
                         pt_scale=pt_scale, drop_rows=drop_rows)

    def set_data(self, weight, bias, mean, var, gamma, beta, eps: float = 1e-5,
                 delta: float = 1.0, **kwargs) -> None:
        """Fuse Conv+BN tensors into the conv's weight and bias; the arguments are
        HomConvBnFused.set_data's."""
        super().set_data(*self._fold_batch_norm(weight, bias, mean, var, gamma, beta, eps, delta))
