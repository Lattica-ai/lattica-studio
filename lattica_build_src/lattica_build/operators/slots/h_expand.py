"""See `operators/slots/README.md` for usage details."""

import math
from typing import Sequence

from lattica_build.base_classes.hom_op import HomOp
from lattica_build.base_classes.hom_value import HomValue, to_pos_axis
from lattica_build.params.level_and_scale_tracing import infer_modswitch_by_num_levels

from lattica_build.params.shape_tracing import infer_insert_axis_output_shape
from lattica_build.serialization.hom_op_pb2 import HomOpType

class HomExpand(HomOp):
    """Base slot-expansion operator that inserts a new axis of size `k`.

    Args:
        k: Expansion span in slots (conceptually, produces `k` expanded
            positions/copies per slot pattern).
        expand_axis: Axis where the expanded dimension is inserted.
        stage_sizes: Optional explicit stage schedule. When provided, it
            defines stage count as `len(stage_sizes)` and should satisfy
            `prod(stage_sizes) == k`.
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

    def infer_output_shape(self, input: HomValue, internal_n: int | None = None, **kwargs) -> HomValue:
        """Infer shape by inserting axis `expand_axis` with dimension `k`."""
        axis = to_pos_axis(self.expand_axis, input.tensor_shape)
        output = infer_insert_axis_output_shape(input, axis, input.n_slots)
        if internal_n is None or output.n_axis is None:
            return output

        output_shape = list(output.tensor_shape)
        output_shape[output.n_axis] = internal_n
        return output.make_copy(tensor_shape=output_shape)

    def infer_output_level_and_scale(
        self,
        input: HomValue,
        hom_params=None,
        **kwargs,
    ) -> HomValue:
        """Compute exact level consumption from stages/log2(k), then apply modswitch.

        Expand stages include masked multiplications, so they consume levels.
        """
        if self.stage_sizes is None:
            num_levels = int(math.log2(self.k))
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
