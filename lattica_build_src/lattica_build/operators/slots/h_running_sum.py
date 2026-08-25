"""See `operators/slots/README.md` for usage details."""

import math
from typing import Optional, Sequence

from lattica_build.base_classes.hom_op import HomOp
from lattica_build.base_classes.hom_value import HomValue
from lattica_build.params.level_and_scale_tracing import infer_modswitch_by_num_levels
from lattica_build.serialization.hom_op_pb2 import HomOpType

class HomRunningSum(HomOp):
    """Base slot running-sum operator over configured stage schedule.

    Args:
        cumsum_axis: Optional logical axis hint for cumulative summation.
        blocks_axis_external: Optional external blocking-axis hint.
        k: Optional slot span to aggregate (defaults to full `internal_n`).
        stage_sizes: Optional explicit stage schedule. When provided, it
            defines stage count as `len(stage_sizes)` and should satisfy
            `prod(stage_sizes) == k`.
        stages_per_level: Optional number of stages grouped per consumed level.
            It should divide the selected stage count exactly.
        rows_budget: Optional allowed modulus rows for modswitch inference.
    """

    OP_TYPE = HomOpType.RunningSum

    def __init__(
        self,
        cumsum_axis: Optional[int] = None,
        blocks_axis_external: Optional[int] = None,
        k: Optional[int] = None,
        stage_sizes: Sequence[int] | None = None,
        stages_per_level: int | None = None,
        rows_budget: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        self.cumsum_axis = cumsum_axis
        self.blocks_axis_external = blocks_axis_external
        self.k = k
        self.stage_sizes = stage_sizes
        self.stages_per_level = stages_per_level
        self.rows_budget = rows_budget

    def infer_output_level_and_scale(
        self,
        input: HomValue,
        hom_params=None,
        **kwargs,
    ) -> HomValue:
        """Compute exact level consumption from stages/log2(k), then apply modswitch.

        Running-sum stages include masked multiplications, so they consume
        levels.
        """
        if self.stage_sizes is None:
            k = self.k if self.k is not None else hom_params.internal_n
            num_levels = int(math.log2(k))
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
