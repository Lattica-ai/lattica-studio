"""See `operators/polynomials/README.md` for usage details."""

from typing import Sequence

from lattica_build.base_classes.hom_op import HomOp
from lattica_build.base_classes.hom_value import HomValue

from lattica_build.operators.comparison.comparison_utils import max_to_sign_accuracy
from lattica_build.operators.polynomials.h_sign import HomSign


class HomReLU(HomOp):
    """Approximate ReLU via a sign-approximation composite graph.

    The op computes `(approx_sign(x) + 1/2) * x` with transition quality
    controlled by `x_accuracy` and `y_accuracy`.

    Args:
        x_accuracy: Transition accuracy target on the x-axis.
        y_accuracy: Approximation error target on the y-axis.
        left: Lower bound of the expected input domain (currently must be -1).
        right: Upper bound of the expected input domain (currently must be 1).
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

        if left != -1.0 or right != 1.0:
            raise ValueError(
                "The current alpha-to-zeta calibration for HomReLU "
                "requires inputs in [-1, 1]."
            )

        # Convert the requested min/max accuracy into the sign accuracy
        # required by the internal comparison.
        sign_x_accuracy = max_to_sign_accuracy(x_accuracy)

        self.h_sign = HomSign(
            scale=0.5,
            x_accuracy=sign_x_accuracy,
            y_accuracy=y_accuracy - 1,
            left=left,
            right=right,
            tol=tol,
            rows_budget=rows_budget,
        )

    def forward(self, x: HomValue) -> HomValue:
        """Return ReLU approximation for `x`."""
        return (self.h_sign(x) + 0.5) * x
