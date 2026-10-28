"""See `operators/polynomials/README.md` for usage details."""

from lattica_build.base_classes.hom_op import HomOp
from lattica_build.operators.arithmetic.h_mul import HomMul


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

    def forward(self, x):
        """Return elementwise square of `x`."""
        return self.h_mul(x, x)
