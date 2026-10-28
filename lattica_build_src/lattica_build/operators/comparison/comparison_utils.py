"""Utilities for comparison/min-max approximation calibration.

See `operators/comparison/README.md` for usage details."""

import numpy as np


alpha2zeta: dict[int, float] = {
    4: 5,
    5: 5,
    6: 10,
    7: 11,
    8: 12,
    9: 13,
    10: 13,
    11: 15,
    12: 15,
    13: 16,
    14: 17,
}


def max_to_sign_accuracy(x_accuracy: int) -> int:
    """Convert requested min/max accuracy into internal sign accuracy.

    Args:
        x_accuracy: Target x-accuracy used by min/max operators.

    Returns:
        The x-accuracy to use for `HomSign` inside comparison composition.

    Raises:
        ValueError: If `x_accuracy` is not supported by the calibration table.
    """

    try:
        zeta = alpha2zeta[x_accuracy]
    except KeyError:
        raise ValueError(
            f"x_accuracy must be one of {sorted(alpha2zeta)}, "
            f"got {x_accuracy}"
        ) from None

    return x_accuracy - np.log2(zeta)
