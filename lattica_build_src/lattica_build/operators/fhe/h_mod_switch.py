"""See `operators/fhe/README.md` for usage details."""

from typing import Sequence

from lattica_build.base_classes.hom_op import HomOp
from lattica_build.base_classes.hom_value import HomValue
from lattica_build.params.level_and_scale_tracing import _get_drop_config_for_x_levels_in_a_given_row, apply_drop_config
from lattica_build.serialization.hom_op_pb2 import HomOpType

class HomModSwitch(HomOp):
    """Base modswitch operator that drops primes from selected row/columns.

    Args:
        variant: Backend mode selector (`0` drop full row, `1` drop columns).
        relative_row: Row index inside the current active-rows view.
        cols_to_drop: Optional column indices to drop for column mode.
    """

    OP_TYPE = HomOpType.ModSwitch

    def __init__(
        self,
        variant: int = 0,  # 0 for reduce full row, 1 for reduce column
        relative_row: int = 0,
        cols_to_drop: Sequence[int] | None = None,  # used only for variant=1
    ) -> None:
        super().__init__()
        self.variant = variant
        self.relative_row = relative_row
        self.cols_to_drop = cols_to_drop

    def infer_output_level_and_scale(
        self,
        input: HomValue,
        hom_params=None,
        **kwargs,
    ) -> HomValue:
        """Infer rows/cols drop effect and resulting pt-scale reduction."""
        if self.cols_to_drop is None:
            num_cols_to_drop = sum(input.active_cols[self.relative_row])
        else:
            num_cols_to_drop = len(self.cols_to_drop)
        drop_config = _get_drop_config_for_x_levels_in_a_given_row(
            hom_params.mod_chain,
            input.active_rows,
            input.active_cols,
            self.relative_row,
            num_cols_to_drop,
        )
        return apply_drop_config(input, drop_config)
