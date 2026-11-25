"""See `operators/ml/README.md` for usage details."""

import functools

import torch

from lattica_build.base_classes.hom_op import HomOp
from lattica_build.base_classes.hom_value import HomValue
from lattica_build.operators.arithmetic.h_axis_sum import HomAxisSum
from lattica_build.operators.arithmetic.h_const_mul import HomConstMul
from lattica_build.operators.arithmetic.h_mul import HomMul
from lattica_build.operators.ml.h_mat_mul import HomMatMul
from lattica_build.operators.shape.h_reshape import HomReshape
from lattica_build.operators.slots.h_rotate_sum import HomRotateSum


class HomCtMatMul(HomOp):
    """Product of two encrypted d x d matrices, each one `HomValue` of shape `(d * d,)`.

    A matrix sits row by row on the slot axis, entry (i, j) at slot `d * i + j`, in a slot period of `d * d`
    (`HomParams.n_slots`), and so does the product. With every index modulo d, let sigma(A)[i, j] = A[i, i + j] and
    tau(B)[i, j] = B[i + j, j], and let phi^k shift the columns of a matrix left by k and psi^k shift its rows up
    by k. Then

        A B = sum_{k < d} phi^k(sigma(A)) * psi^k(tau(B)),

    with `*` the entrywise product: at (i, j) term k is A[i, i + j + k] * B[i + j + k, j], and i + j + k takes each
    value modulo d once. On the slots, psi^k is the rotation by d * k, and phi^k is the rotation by k on the columns
    j < d - k and the rotation by k - d on the other columns.

    The product takes three levels: one for sigma and tau, one for the column masks of phi^k, one for the
    multiplication.

    The d shifts of each kind are made in two stages, the shift k = low * g + b as the rotation for b followed by the
    rotation for low * g, with d = low * high and low <= high. Two stages keep the rotation keys few: each kind needs
    those of low - 1 and of high - 1 rotations, about 2 sqrt(d) in all.

    Args:
        d: Number of rows and of columns, a power of two.
    """

    def __init__(self, d: int) -> None:
        super().__init__()
        n = d * d
        slot = torch.arange(n)
        row, col = slot // d, slot % d

        self.sigma = self._permutation(d * row + (row + col) % d)
        self.tau = self._permutation(d * ((row + col) % d) + col)

        low = 1 << (d.bit_length() - 1) // 2
        high = d // low
        # the input and its rotations, stacked on a new first axis, the input first
        stack = functools.partial(HomRotateSum, perform_sum=False, add_identity_rotation=True)

        # phi^k for k = low * g + b. `wrap` stacks the input and its rotation by -d, so the two stages rotate both by
        # k: entry (g, b, 0) of the stack is the rotation by k and entry (g, b, 1) the rotation by k - d. Each is
        # masked to its columns and the pair is added. At k = 0 the rotation by -d has an all-zero mask.
        self.wrap = stack([n - d])
        self.shift_cols_low = stack(range(1, low))
        self.shift_cols_high = stack(low * g for g in range(1, high))
        keep = torch.stack([(col < d - k).double() for k in range(d)]).reshape(high, low, n)
        self.mask_cols = HomConstMul(dims=(high, low, 2, n)).set_data(torch.stack([keep, 1 - keep], dim=2))
        self.add_pair = HomAxisSum(dim=2)

        # psi^k for k = low * g + b: the rotation by d * k
        self.shift_rows_low = stack(d * b for b in range(1, low))
        self.shift_rows_high = stack(d * low * g for g in range(1, high))

        # entry (g, b) of a stack of shifts becomes entry k = low * g + b
        self.flatten = HomReshape((d, n))
        self.mul = HomMul(axis_sum=0)

    @staticmethod
    def _permutation(source: torch.Tensor) -> HomMatMul:
        """The slot permutation output[slot] = input[source[slot]], as a matrix given by its diagonals."""
        n = len(source)
        slot = torch.arange(n)
        offset = (source - slot) % n
        rotations = offset.unique()
        diagonals = torch.zeros(len(rotations), n, dtype=torch.float64)
        diagonals[torch.searchsorted(rotations, offset), source] = 1
        return HomMatMul(dims=(len(rotations), n), rotations=rotations.tolist()).set_data(diagonals)

    def forward(self, a: HomValue, b: HomValue) -> HomValue:
        a_shifted = self.shift_cols_high(self.shift_cols_low(self.wrap(self.sigma(a))))
        a_shifted = self.flatten(self.add_pair(self.mask_cols(a_shifted)))
        b_shifted = self.flatten(self.shift_rows_high(self.shift_rows_low(self.tau(b))))
        return self.mul(a_shifted, b_shifted)
