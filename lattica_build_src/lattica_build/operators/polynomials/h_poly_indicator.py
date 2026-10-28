"""See `operators/polynomials/README.md` for usage details."""

from typing import Sequence

from lattica_build.operators.polynomials.h_poly_eval_base import HomPolyEvalBase
from lattica_build.operators.polynomials.polynomial_evaluation_utils import (
    hermite_indicator_cheb_coeffs, indicator_cheb_coeffs_packed,
)


class HomPolyIndicator(HomPolyEvalBase):
    """Polynomial indicator-bank approximation over a bounded domain.

    The generated polynomial outputs approximate bin-indicator responses
    (one-hot-like activations) over equally spaced bins in `[left, right]`.

    Args:
        degree: Polynomial degree per indicator.
        num_bins: Number of equally spaced indicator bins.
        left: Lower bound of the encoded domain.
        right: Upper bound of the encoded domain.
        tol: Coefficient trimming tolerance.
        plot: Whether to render approximation diagnostics.
        rows_budget: Optional allowed modulus rows for internal modswitch.
    """

    def __init__(
        self,
        degree: int,
        num_bins: int,
        left: float = 0,
        right: float = 1,
        tol: float = 1e-8,
        plot: bool = False,
        rows_budget: Sequence[int] | None = None,
    ) -> None:
        coefs = indicator_cheb_coeffs_packed(
            max=num_bins,
            deg=degree,
            tol=tol,
            plot=plot,
        )
        super().__init__(
            coefs=coefs,
            left=left,
            right=right,
            tol=tol,
            rows_budget=rows_budget,
        )


class HomHermiteIndicator(HomPolyEvalBase):
    """Indicators of exact bin values: for inputs that sit on the num_bins + 1 equally spaced points of [left, right]
    (up to noise), output j is 1 on the j-th point after left and 0 on the others. Hermite interpolation: exact at the
    points with zero slope there, degree 2 * num_bins + 1, far below a fitted indicator (HomPolyIndicator); values
    between the points are unconstrained.

    Args:
        num_bins: Number of indicators; the points are left + k * (right - left) / num_bins, k = 0..num_bins.
        left: The first point, which no indicator selects.
        right: The last point.
        rows_budget: Optional allowed modulus rows for internal modswitch.
    """

    def __init__(
        self,
        num_bins: int,
        left: float = 0,
        right: float = 1,
        rows_budget: Sequence[int] | None = None,
    ) -> None:
        super().__init__(
            coefs=hermite_indicator_cheb_coeffs(num_bins),
            left=left,
            right=right,
            rows_budget=rows_budget,
        )
