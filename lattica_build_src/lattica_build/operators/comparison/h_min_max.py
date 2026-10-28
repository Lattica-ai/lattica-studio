"""See `operators/comparison/README.md` for usage details."""

from typing import Sequence

from lattica_build.base_classes.hom_op import HomOp
from lattica_build.base_classes.hom_value import HomValue
from lattica_build.operators.comparison.comparison_utils import max_to_sign_accuracy
from lattica_build.operators.comparison.h_compare import HomCompare


class HomMinMax(HomOp):
    """Elementwise approximate minimum or maximum.

    This class is composite: it reuses `HomCompare` and arithmetic to implement
    min/max without introducing a dedicated base `OP_TYPE`.

    Prefer `HomMin` or `HomMax` in user-facing graphs; use this class directly
    only when dynamically selecting between min/max behavior.

    Current accuracy calibration assumes inputs are in `[0, 1]`.

    Notes:
        Comparison-based binary ops currently run elementwise and require
        identical input shapes.

    Args:
        is_min: `True` for min behavior, `False` for max behavior.
        x_accuracy: Requested x-accuracy target for the min/max decision.
        y_accuracy: Requested y-accuracy target for approximation error.
        left: Lower bound of the expected input domain (currently must be 0).
        right: Upper bound of the expected input domain (currently must be 1).
        tol: Numerical tolerance forwarded to comparison internals.
        rows_budget: Optional allowed modulus rows for internal modswitch.
    """

    def __init__(
        self,
        is_min: bool,
        x_accuracy: int = 10,
        y_accuracy: float = 10,
        left: float = 0.0,
        right: float = 1.0,
        tol: float = 1e-8,
        rows_budget: Sequence[int] | None = None,
    ) -> None:
        super().__init__()

        if left != 0.0 or right != 1.0:
            raise ValueError(
                "The current alpha-to-zeta calibration for HomMinMax "
                "requires inputs in [0, 1]."
            )

        self.is_min = is_min

        # Convert the requested min/max accuracy into the sign accuracy
        # required by the internal comparison.
        sign_x_accuracy = max_to_sign_accuracy(x_accuracy)

        self.h_compare = HomCompare(
            x_accuracy=sign_x_accuracy,
            y_accuracy=y_accuracy,
            left=left,
            right=right,
            tol=tol,
            rows_budget=rows_budget,
        )

    def forward(self, a: HomValue, b: HomValue) -> HomValue:
        """Return elementwise min or max according to `self.is_min`."""
        a_is_greater = self.h_compare(a, b)
        correction = (a - b) * a_is_greater

        return a - correction if self.is_min else b + correction
