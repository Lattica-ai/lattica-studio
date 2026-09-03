"""See `operators/arithmetic/README.md` for usage details."""

from typing import Sequence

from lattica_build.base_classes.hom_op import HomOp
from lattica_build.base_classes.hom_value import HomValue
from lattica_build.params.level_and_scale_tracing import infer_optional_modswitch, infer_adjust_levels_and_scale
from lattica_build.params.shape_tracing import infer_broadcast_output_shape, infer_reduce_axis_output_shape
from lattica_build.serialization.hom_op_pb2 import HomOpType

class HomMul(HomOp):
    """Homomorphic multiplication with optional reduction and modswitch.

    This is a base operator (`OP_TYPE = HomOpType.Mul`). It performs
    broadcast-compatible multiplication and may reduce along an axis (for
    dot-product-like behavior).

    Args:
        axis_sum: Optional axis to reduce after elementwise multiplication.
        keep_axis: Whether the reduced axis is kept with size 1.
        with_modswitch: Whether to apply optional post-op modswitch.
        rows_budget: Optional allowed modulus rows for modswitch inference.
    """

    OP_TYPE = HomOpType.Mul

    def __init__(
        self,
        axis_sum: int = None,
        keep_axis: bool = False,
        with_modswitch: bool = True,
        rows_budget: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        self.axis_sum = axis_sum
        self.keep_axis = keep_axis
        self.with_modswitch = with_modswitch
        self.rows_budget = rows_budget

    def infer_output_shape(
        self,
        input_1: HomValue,
        input_2: HomValue,
        **kwargs,
    ) -> HomValue:
        """Infer broadcast output shape, then optional axis-reduced shape."""
        after_mul = infer_broadcast_output_shape(input_1, input_2)
        if self.axis_sum is None:
            return after_mul
        return infer_reduce_axis_output_shape(
            after_mul,
            self.axis_sum,
            self.keep_axis,
        )

    def infer_output_level_and_scale(
        self,
        input_1: HomValue,
        input_2: HomValue,
        hom_params=None,
        **kwargs,
    ) -> HomValue:
        """Infer multiplication level/scale and optional modswitch adjustment."""
        res = infer_adjust_levels_and_scale(input_1, input_2)
        res = res.make_copy(pt_scale=input_1.pt_scale)
        return infer_optional_modswitch(
            hom_params,
            res,
            with_modswitch=self.with_modswitch,
            rows_budget=self.rows_budget,
            op_scale_up=input_2.pt_scale,
        )

    def forward_clear(self, input_1, input_2):
        result = input_1 * input_2
        if self.axis_sum is not None:
            result = result.sum(dim=self.axis_sum, keepdim=self.keep_axis)
        return result
