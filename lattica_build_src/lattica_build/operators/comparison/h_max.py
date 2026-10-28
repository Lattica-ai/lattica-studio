"""See `operators/comparison/README.md` for usage details."""

from lattica_build.operators.comparison.h_min_max import HomMinMax


class HomMax(HomMinMax):
    """Elementwise approximate maximum specialization of `HomMinMax`.

    Comparison-based binary ops currently run elementwise and require
    identical input shapes.

    Args:
        **kwargs: Forwarded to `HomMinMax` (`x_accuracy`, `y_accuracy`,
            `left`, `right`, `tol`, `rows_budget`).
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(is_min=False, **kwargs)
