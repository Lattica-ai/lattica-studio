"""See `operators/polynomials/README.md` for usage details."""

from typing import Sequence

from lattica_build.base_classes.hom_op import HomOp
from lattica_build.base_classes.hom_value import HomValue
from lattica_build.operators.composite.module_list import ModuleListHomOp
from lattica_build.operators.polynomials.h_poly_eval_base import HomPolyEvalBase
from lattica_build.operators.polynomials.remez_utils import sign_minimax


class HomSign(HomOp):
    """Approximate sign as a composed polynomial graph.

    `HomSign` is composite: it chains several `HomPolyEvalBase` stages produced
    by a minimax fit. It approximates a discrete sign transition, and the
    transition quality is controlled by `x_accuracy` and `y_accuracy`.

    Args:
        scale: Final output scaling factor. Only the last polynomial stage is
            scaled so intermediate stages remain in the fit range.
        x_accuracy: Requested x-accuracy (transition-gap) target.
        y_accuracy: Requested y-accuracy (approximation-error) target.
        left: Left boundary for the first-stage fit interval.
        right: Right boundary for the first-stage fit interval.
        tol: Tolerance passed to polynomial evaluation stages.
        rows_budget: Optional allowed modulus rows for internal modswitch
            inference.
    """

    def __init__(
        self,
        scale: float = 1,
        x_accuracy: int = 10,
        y_accuracy: float = 10,
        left: float = -1,
        right: float = 1.0,
        tol: float = 1e-8,
        rows_budget: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        coefs = sign_minimax(
            x_accuracy=x_accuracy,
            y_accuracy=y_accuracy,
        ).coefs()
        # Scale only the last stage: intermediate stages must keep their
        # output in [-1, 1], the range the next stage was fitted on.
        coefs = [*coefs[:-1], coefs[-1] * scale]
        self.h_poly_ops = ModuleListHomOp(
            HomPolyEvalBase(
                coefs=coef,
                left=left if i == 0 else -1,
                right=right if i == 0 else 1,
                tol=tol,
                rows_budget=rows_budget,
            )
            for i, coef in enumerate(coefs)
        )

    def forward(self, x: HomValue) -> HomValue:
        """Apply the composed polynomial stages and return sign approximation."""
        for h_poly in self.h_poly_ops:
            x = h_poly(x)
        return x
