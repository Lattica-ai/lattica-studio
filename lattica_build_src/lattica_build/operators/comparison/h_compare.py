"""See `operators/comparison/README.md` for usage details."""

from typing import Sequence

from lattica_build.base_classes.hom_op import HomOp
from lattica_build.base_classes.hom_value import HomValue
from lattica_build.operators.polynomials.h_sign import HomSign


class HomCompare(HomOp):
    """Approximate comparison op implemented as a composed sign graph.

    The operator computes `comp(a, b) = (approx_sign(a - b) + 1) / 2`, where
    `approx_sign(...)` is implemented by `HomSign`.

    Output semantics approximate a discrete greater-than transition: values are
    near 1 for `a > b`, near 0 for `a < b`, and transition around equality.
    Transition sharpness is controlled by `x_accuracy` and `y_accuracy`.

    This class is composite: it orchestrates underlying base ops through
    `HomSign` rather than exposing a distinct `OP_TYPE`.

    Notes:
        Comparison-based binary ops currently run elementwise and require
        identical input shapes.

    Args:
        x_accuracy: Requested x-accuracy (transition-gap) target for
            comparison.
        y_accuracy: Requested y-accuracy (approximation-error) target for
            comparison.
        left: Lower bound of the expected input domain.
        right: Upper bound of the expected input domain.
        tol: Numerical tolerance forwarded to polynomial stages.
        rows_budget: Optional allowed modulus rows for internal modswitch.
    """

    def __init__(
        self,
        x_accuracy: int = 10,
        y_accuracy: float = 10,
        left: float = -1.0,
        right: float = 1.0,
        tol: float = 1e-8,
        rows_budget: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        # comp(u,v) = (sgn(u-v)+1)/2

        self.h_sign = HomSign(
            scale=0.5,
            x_accuracy=x_accuracy,
            y_accuracy=y_accuracy - 1,
            left=left - right,
            right=right - left,
            tol=tol,
            rows_budget=rows_budget,
        )

    def forward(self, a: HomValue, b: HomValue) -> HomValue:
        """Return an approximate greater-than indicator for `a` vs `b`."""
        return self.h_sign(a - b) + 0.5
