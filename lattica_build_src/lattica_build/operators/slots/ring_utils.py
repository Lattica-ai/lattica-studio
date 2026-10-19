"""How encryption lays out the slot axis of a value, for clear execution of the slot operators.

The slot operators (RotateSum, RunningSum, Expand, SumSlots) act on the slots of the ciphertext, not on the
logical tensor, so their clear execution lays the tensor out as encryption does, computes on that layout, and reads
the result back as decryption does.

A slot axis of at most `n_slots` values is zero-padded to its period `n_slots`, and the period is repeated over the
`internal_n` slots of the ring. A longer slot axis spans whole rings (ciphertexts) of `internal_n` slots, the last
one zero-padded. Decryption reads back the slots that hold the values, dropping the padding and the repeats.
"""

import torch

from lattica_build.base_classes.hom_value import HomValue


def input_layout(op, x: torch.Tensor) -> tuple[HomValue, int]:
    """The traced value of the slot operator's input and its ring size `internal_n`, as clear execution of a traced
    graph sets them on the op (`op.input_value`, `op.internal_n`), with its period `n_slots` set (internal_n when the
    value has none). Outside a traced graph, the input's last axis is one whole ring, of its length rounded up to a
    power of 2."""
    value, internal_n = op.input_value, op.internal_n
    if value is not None and value.n_axis is not None and internal_n is not None:
        return (value if value.n_slots is not None else value.make_copy(n_slots=internal_n)), internal_n
    ring = 1 << (x.shape[-1] - 1).bit_length()
    return HomValue(tensor_shape=tuple(x.shape), n_axis=x.dim() - 1, n_slots=ring), ring


def to_ring(x: torch.Tensor, value: HomValue, internal_n: int) -> torch.Tensor:
    """`x` as encryption lays it out: the slot axis moved last and split into (rings, internal_n)."""
    if tuple(x.shape) != value.tensor_shape:
        raise ValueError(f"Clear input has shape {tuple(x.shape)}; the traced shape is {value.tensor_shape}")
    length = value.tensor_shape[value.n_axis]
    x = x.movedim(value.n_axis, -1)
    if length <= value.n_slots:
        period = torch.nn.functional.pad(x, (0, value.n_slots - length))
        ring = period.repeat(*([1] * (x.dim() - 1)), internal_n // value.n_slots)
        return ring.unsqueeze(-2)
    rings = -(-length // internal_n)
    x = torch.nn.functional.pad(x, (0, rings * internal_n - length))
    return x.unflatten(-1, (rings, internal_n))


def from_ring(ring: torch.Tensor, value: HomValue) -> torch.Tensor:
    """What decryption reads from slots laid out as by `to_ring`: the slots that hold the values, dropping the
    padding and the repeats, moved back to the slot axis."""
    return ring.flatten(-2)[..., :value.tensor_shape[value.n_axis]].movedim(-1, value.n_axis)
