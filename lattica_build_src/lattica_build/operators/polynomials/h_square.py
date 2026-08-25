"""See `operators/polynomials/README.md` for usage details."""

from lattica_build.base_classes.hom_op import HomOp
from lattica_build.base_classes.hom_value import HomValue
from lattica_build.operators.arithmetic.h_mul import HomMul
from lattica_build.params.level_and_scale_tracing import infer_optional_modswitch


class HomSquare(HomOp):
    """Square a ciphertext by reusing `HomMul(x, x)`.

    Args:
        *args: Forwarded to `HomMul` (for example `axis_sum`, `keep_axis`,
            `with_modswitch`, `rows_budget`).
        **kwargs: Forwarded to `HomMul`.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.h_mul = HomMul(*args, **kwargs)
        self.with_modswitch = self.h_mul.with_modswitch
        self.rows_budget = self.h_mul.rows_budget

    def forward(self, x):
        """Return elementwise square of `x`."""
        return self.h_mul(x, x)

    def infer_output_level_and_scale(
        self,
        input: HomValue,
        hom_params=None,
        **kwargs,
    ) -> HomValue:
        """Infer square level/scale with optional post-op modswitch."""
        return infer_optional_modswitch(
            hom_params,
            input,
            with_modswitch=self.with_modswitch,
            rows_budget=self.rows_budget,
            op_scale_up=input.pt_scale,
        )
