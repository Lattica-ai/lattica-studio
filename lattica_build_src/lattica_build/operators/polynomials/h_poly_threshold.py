"""See `operators/polynomials/README.md` for usage details."""

from typing import Sequence

from lattica_build.operators.polynomials.h_poly_eval_base import HomPolyEvalBase
from lattica_build.operators.polynomials.polynomial_evaluation_utils import heaviside_sigmoid_cheb_coeffs, \
    heaviside_peacewise_linear_cheb_coeffs, threshold_cheb_coeffs_minimax


class HomPolyThreshold(HomPolyEvalBase):
    """Polynomial approximation of a threshold transition.

    The target behavior is:
    `f(x) = out_val` for `x > margin[1]`, `0` for `x < margin[0]`, and a
    smooth/approximated transition in between.

    Args:
        degree: Polynomial degree.
        margin: Two-value transition interval `[low, high]`.
        variant: Approximation strategy (`sigmoid`, `piecewise_linear`,
            `minimax`).
        sharpness: Sigmoid steepness (required for `variant="sigmoid"`).
        out_val: Output value above the threshold region.
        domain_start: Left boundary used by the minimax builder.
        tol: Coefficient trimming / fitting tolerance.
        plot: Whether to render approximation diagnostics.
        rows_budget: Optional allowed modulus rows for internal modswitch.
    """

    def __init__(
        self,
        degree: int,
        margin: Sequence[float],
        variant: str = "sigmoid",
        sharpness: int | None = None,
        out_val: float = 1,
        domain_start: float = -1,
        tol: float = 1e-5,
        plot: bool = False,
        rows_budget: Sequence[int] | None = None,
    ) -> None:
        if variant == "sigmoid":
            if sharpness is None:
                raise ValueError("sharpness must be provided for sigmoid variant")
            step = (margin[0] + margin[1]) / 2
            coefs = heaviside_sigmoid_cheb_coeffs(
                step=step,
                deg=degree,
                out_val=out_val,
                sharpness=sharpness,
                plot=plot,
            )
        elif variant == "piecewise_linear":
            coefs = heaviside_peacewise_linear_cheb_coeffs(
                deg=degree,
                margin=margin,
                out_val=out_val,
                tol=tol,
                plot=plot,
            )
        elif variant == "minimax":
            coefs = threshold_cheb_coeffs_minimax(
                deg=degree,
                margin=margin,
                out_val=out_val,
                domain_start=domain_start,
                tol=tol,
                plot=plot,
            )
        else:
            raise ValueError(f"Unknown variant: {variant}")
        super().__init__(coefs=coefs, tol=tol, rows_budget=rows_budget)
