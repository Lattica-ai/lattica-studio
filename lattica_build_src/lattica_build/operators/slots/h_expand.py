"""See `operators/slots/README.md` for usage details."""

import math
from typing import Sequence

from lattica_build.base_classes.hom_value import HomValue
from lattica_build.operators.slots.ring_utils import SlotOp, input_layout
from lattica_build.params.level_and_scale_tracing import infer_modswitch_by_num_levels

from lattica_build.params.shape_tracing import infer_insert_axis_output_shape, insert_position
from lattica_build.serialization.hom_op_pb2 import HomOpType

class HomExpand(SlotOp):
    """Base slot-expansion operator that inserts a new axis holding the values of the slot axis.

    The expansion span is the input's slot period rather than a constructor
    argument: the stages expand the n_slots slots of the period, and the
    outputs of the slot axis's L <= n_slots values are kept.

    Args:
        expand_axis: Axis where the expanded dimension is inserted.
        stage_sizes: Optional explicit stage schedule. When provided, it
            defines stage count as `len(stage_sizes)` and should satisfy
            `prod(stage_sizes) == n_slots`.
        stages_per_level: Optional number of stages grouped per consumed level.
            It should divide the selected stage count exactly.
        rows_budget: Optional allowed modulus rows for modswitch inference.
    """

    OP_TYPE = HomOpType.Expand

    def __init__(
        self,
        expand_axis: int = 0,
        stage_sizes: Sequence[int] | None = None,
        stages_per_level: int | None = None,
        rows_budget: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        self.expand_axis = expand_axis
        self.stage_sizes = stage_sizes
        self.stages_per_level = stages_per_level
        self.rows_budget = rows_budget

    def infer_output_shape(self, input: HomValue, internal_n: int, **kwargs) -> HomValue:
        """Infer shape by inserting axis `expand_axis` with one output per value of the slot axis. Each output holds
        one value in every slot of the ring (internal_n on the slot axis), so its period is 1.

        Unlike RotateSum and RunningSum, Expand takes a zero-padded slot axis: L <= n_slots values in a period of
        n_slots. Output i reads slot i only, so the padding feeds only outputs past the L values, which are dropped."""
        length = input.tensor_shape[input.n_axis]
        if length > input.n_slots:
            raise ValueError(f"Expand takes a slot axis of at most n_slots={input.n_slots} values, got {length}")
        # The stages expand the whole period: infer_output_level_and_scale counts them, from the output it gets
        self._n_slots = input.n_slots
        position = insert_position(self.expand_axis, len(input.tensor_shape))
        output = infer_insert_axis_output_shape(input, position, length)
        output_shape = list(output.tensor_shape)
        output_shape[output.n_axis] = internal_n
        return output.make_copy(tensor_shape=output_shape, n_slots=1)

    def infer_output_level_and_scale(
        self,
        input: HomValue,
        hom_params=None,
        **kwargs,
    ) -> HomValue:
        """Compute exact level consumption from stages/log2(n_slots), then apply modswitch.

        Expand stages include masked multiplications, so they consume levels.
        """
        if self.stage_sizes is None:
            # The tracer passes the inferred output, whose values axis holds the L values, not the n_slots slots the
            # stages expand: infer_output_shape recorded n_slots
            num_levels = int(math.log2(self._n_slots))
        else:
            num_levels = len(self.stage_sizes)
        if self.stages_per_level is not None:
            num_levels //= self.stages_per_level
        return infer_modswitch_by_num_levels(
            hom_params,
            input,
            num_levels=num_levels,
            rows_budget=self.rows_budget,
        )

    def forward_clear(self, input):
        """Value i of the input's slot axis becomes an output holding it in every slot of the ring: the values go on
        a new axis inserted as `infer_output_shape` does, and the slot axis has internal_n slots."""
        value, internal_n = input_layout(self, input)
        values = input.movedim(value.n_axis, -1)  # (..., L)
        expanded = values.unsqueeze(-1).expand(*values.shape, internal_n)  # (..., values, slots)

        # Order the axes as the traced output: the input's, with the values axis inserted at expand_axis
        axes = [axis for axis in range(input.dim()) if axis != value.n_axis] + ['values', value.n_axis]
        output_axes = list(range(input.dim()))
        output_axes.insert(insert_position(self.expand_axis, input.dim()), 'values')
        return expanded.permute([axes.index(axis) for axis in output_axes]).clone()
