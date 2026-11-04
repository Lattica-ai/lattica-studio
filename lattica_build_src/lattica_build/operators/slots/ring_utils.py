"""How encryption lays out the slot axis of a value, for clear execution of the slot operators.

The slot operators (RotateSum, RunningSum, Expand, SumSlots) act on the slots of the ciphertext, not on the
logical tensor, so their clear execution lays the tensor out as encryption does, computes on that layout, and reads
the result back as decryption does.

A slot axis of at most `n_slots` values is zero-padded to its period `n_slots`, and the period is repeated over the
`internal_n` slots of the ring. A longer slot axis with a period shorter than the ring holds values that repeat at the
period (the output of SumSlots), and the ring repeats its first period. A longer slot axis with the ring as its period
spans whole rings (ciphertexts) of `internal_n` slots, the last one zero-padded. Decryption reads back the slots that
hold the values, dropping the padding and the repeats.
"""

import torch

from lattica_build.base_classes.hom_op import HomOp
from lattica_build.base_classes.hom_value import HomValue
from lattica_build.operators.ml.h_mux_conv import next_pow2


class SlotOp(HomOp):
    """A slot operator: its clear execution computes on the slot layout of its input."""

    # The traced value of the input and its ring size, set by clear execution of a traced graph (run_clear_section)
    # as it sets `data`. Without them, clear execution takes the input's last axis as one whole ring.
    input_value: HomValue | None = None
    internal_n: int | None = None

    def infer_output_shape(self, input: HomValue, internal_n: int, **kwargs) -> HomValue:
        """The input's slot axis takes no zero padding: it holds n_slots values, or whole rings when n_slots is
        internal_n. Then the operator's own inference, `infer_slot_output_shape`."""
        length = input.tensor_shape[input.n_axis]
        whole_rings = input.n_slots == internal_n and length % internal_n == 0
        if length != input.n_slots and not whole_rings:
            raise ValueError(f"{type(self).__name__} takes no padded input: a slot axis of n_slots={input.n_slots} "
                             f"values, or of whole rings when n_slots is internal_n={internal_n}: got {length}")
        return self.infer_slot_output_shape(input, internal_n, **kwargs)

    def infer_slot_output_shape(self, input: HomValue, internal_n: int, **kwargs) -> HomValue:
        raise NotImplementedError


def input_layout(op: SlotOp, x: torch.Tensor) -> tuple[HomValue, int]:
    """The traced value of the slot operator's input and its ring size `internal_n`, as clear execution of a traced
    graph sets them on the op (`op.input_value`, `op.internal_n`). Outside a traced graph (the op run on its own, or
    `forward_clear` without `hom_params`), the input's last axis is one whole ring, of its length rounded up to a
    power of 2."""
    value, internal_n = op.input_value, op.internal_n
    if value is None:
        ring = next_pow2(x.shape[-1])
        return HomValue(tensor_shape=tuple(x.shape), n_axis=x.dim() - 1, n_slots=ring), ring
    return value, internal_n


def to_ring(x: torch.Tensor, value: HomValue, internal_n: int) -> torch.Tensor:
    """`x` as encryption lays it out: the slot axis moved last and split into (rings, internal_n)."""
    if tuple(x.shape) != value.tensor_shape:
        raise ValueError(f"Clear input has shape {tuple(x.shape)}; the traced shape is {value.tensor_shape}")
    length = value.tensor_shape[value.n_axis]
    x = x.movedim(value.n_axis, -1)
    if length <= value.n_slots or value.n_slots < internal_n:
        period = torch.nn.functional.pad(x[..., :value.n_slots], (0, max(value.n_slots - length, 0)))
        ring = period.repeat(*([1] * (x.dim() - 1)), internal_n // value.n_slots)
        return ring.unsqueeze(-2)
    rings = -(-length // internal_n)
    x = torch.nn.functional.pad(x, (0, rings * internal_n - length))
    return x.unflatten(-1, (rings, internal_n))


def from_ring(ring: torch.Tensor, value: HomValue) -> torch.Tensor:
    """What decryption reads from slots laid out as by `to_ring`: the slots that hold the values, dropping the
    padding and the repeats, moved back to the slot axis."""
    return ring.flatten(-2)[..., :value.tensor_shape[value.n_axis]].movedim(-1, value.n_axis)
