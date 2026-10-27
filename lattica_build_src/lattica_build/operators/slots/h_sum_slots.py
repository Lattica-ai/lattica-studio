"""See `operators/slots/README.md` for usage details."""

from typing import Optional, Sequence

from lattica_build.base_classes.hom_value import HomValue
from lattica_build.operators.slots.ring_utils import SlotOp, from_ring, input_layout, to_ring
from lattica_build.serialization.hom_op_pb2 import HomOpType

class HomSumSlots(SlotOp):
    """Base slot-axis reduction operator.

    Use this operator family for slot-axis summation semantics instead of
    tensor-axis reduction ops like `HomAxisSum`.

    In this model, `HomSumSlots` uses rotation/addition semantics and does not
    consume levels.

    Args:
        nb_cols: Optional number of sums, the columns of the period seen as rows
            of nb_cols slots (defaults to 1, the sum of the period): slot j gets
            x[j] + x[j + nb_cols] + ..., the n_slots / nb_cols slots of its column.
        stage_sizes: Optional explicit stage schedule for the summation tree,
            whose product is the n_slots / nb_cols slots each sum adds.
        reduce_output: Whether the output's slot axis holds only the nb_cols
            sums (default). Otherwise it keeps the input's length, each slot
            holding the sum it gets in the ciphertext (the sums repeat at
            nb_cols). A slot axis over several rings needs False.
    """

    OP_TYPE = HomOpType.SumSlots

    def __init__(
        self,
        nb_cols: Optional[int] = None,
        stage_sizes: Sequence[int] | None = None,
        reduce_output: bool = True,
    ) -> None:
        super().__init__()
        self.nb_cols = nb_cols
        self.stage_sizes = stage_sizes
        self.reduce_output = reduce_output

    def infer_output_shape(self, input: HomValue, internal_n=None) -> HomValue:
        """The sum folds the input's period to nb_cols. With reduce_output, the slot axis holds the nb_cols sums."""
        nb_cols = self.nb_cols if self.nb_cols is not None else 1
        if not self.reduce_output or input.n_axis is None:
            return input.make_copy(n_slots=nb_cols)
        return _reduced(self, input, nb_cols, internal_n)

    def forward_clear(self, input):
        """The encrypted sum on the ciphertext's slots: in each period of n_slots slots (zero padding included),
        slot j gets the n_slots / nb_cols slots nb_cols apart. The result repeats at nb_cols and is read at the
        input's positions, or, with reduce_output, at the first nb_cols."""
        value, internal_n = input_layout(self, input)
        n_slots = value.n_slots
        nb_cols = self.nb_cols if self.nb_cols is not None else 1
        ring = to_ring(input, value, internal_n)
        periods = ring.unflatten(-1, (internal_n // n_slots, n_slots // nb_cols, nb_cols))
        sums = periods.sum(dim=-2, keepdim=True).expand(periods.shape)
        if self.reduce_output:
            value = _reduced(self, value, nb_cols, internal_n)
        return from_ring(sums.flatten(-3), value)


def _reduced(op: HomSumSlots, value: HomValue, n_sums: int, internal_n: int | None) -> HomValue:
    """`value` with its slot axis holding the `n_sums` sums: raise ValueError if it spans several rings, which hold
    different sums."""
    length = value.tensor_shape[value.n_axis]
    if internal_n is not None and length > internal_n:
        raise ValueError(f"{type(op).__name__} with reduce_output needs a slot axis in one ring; {length} slots span "
                         f"rings of {internal_n}: use reduce_output=False")
    shape = list(value.tensor_shape)
    shape[value.n_axis] = n_sums
    return value.make_copy(tensor_shape=tuple(shape), n_slots=n_sums)
