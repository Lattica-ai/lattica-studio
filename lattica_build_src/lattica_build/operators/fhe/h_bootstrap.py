"""See `operators/fhe/README.md` for usage details."""

from typing import Optional

from lattica_build.base_classes.hom_op import HomOp
from lattica_build.base_classes.hom_value import HomValue
from lattica_build.params.level_and_scale_tracing import init_levels_after_bootstrap
from lattica_build.serialization.hom_op_pb2 import HomOpType

class Bootstrap(HomOp):
    """Base bootstrap operator that refreshes ciphertext level budget.

    Args:
        target_output_scale: Optional output plaintext scale override.
        drop_levels: The last pipeline rows to leave out of the output, for a consumer that needs fewer levels;
            the whole bootstrap then runs without them.
    """

    OP_TYPE = HomOpType.Bootstrap

    def __init__(
        self,
        target_output_scale: Optional[int] = None,
        drop_levels: int = 0,
    ):
        super().__init__()
        self.target_output_scale = target_output_scale
        self.drop_levels = drop_levels

    def infer_output_level_and_scale(
        self,
        input: HomValue,
        hom_params=None,
        **kwargs,
    ) -> HomValue:
        """Reset active rows/cols and optionally override the output pt-scale."""
        res = init_levels_after_bootstrap(input, hom_params, self.drop_levels)
        return res.make_copy(
            pt_scale=(
                input.pt_scale
                if self.target_output_scale is None
                else self.target_output_scale
            )
        )
