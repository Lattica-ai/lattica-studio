"""See `operators/arithmetic/README.md` for usage details."""

from lattica_build.operators.arithmetic.h_add import HomAdd


class HomSub(HomAdd):
    """Elementwise homomorphic subtraction with broadcast semantics.

    This class reuses `HomAdd` with subtraction mode enabled.
    """

    def __init__(self) -> None:
        super().__init__(is_sub=True)
