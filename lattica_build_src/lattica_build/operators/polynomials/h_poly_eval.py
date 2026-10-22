"""See `operators/polynomials/README.md` for usage details."""

from typing import Callable, Sequence

import numpy as np

from lattica_build.operators.polynomials.h_poly_eval_base import HomPolyEvalBase
from lattica_build.operators.polynomials.polynomial_evaluation_utils import get_cheb_coefs


class HomPolyEval(HomPolyEvalBase):
    """
    Polynomial approximation of a general function over a given domain.

    Args:
        func: The function to approximate.
        degree: The degree of the polynomial.
        left: The left endpoint of the domain.
        right: The right endpoint of the domain.
        tol: coefficients with magnitude less than tol are zeroed.
        plot: Whether to render approximation diagnostics.
        rows_budget: Optional allowed modulus rows for internal modswitch.
    """

    def __init__(
        self,
        func: Callable[[np.ndarray], np.ndarray] | Callable[[float], float],
        degree: int,
        left: float = -1,
        right: float = 1,
        tol: float = 1e-8,
        plot: bool = False,
        rows_budget: Sequence[int] | None = None,
    ) -> None:
        coefs = get_cheb_coefs(
            func=func,
            deg=degree,
            left=left,
            right=right,
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
