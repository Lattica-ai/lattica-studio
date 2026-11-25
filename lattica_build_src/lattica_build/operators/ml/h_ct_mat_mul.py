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
from lattica_build.operators.shape.h_slice import HomSlice
from lattica_build.operators.slots.h_real_imag import HomSplitRealImag
from lattica_build.operators.slots.h_rotate_sum import HomRotateSum


class HomCtMatMul(HomOp):
    """Product of two encrypted real d x d matrices, each one `HomValue` of shape `(d * d,)`.

    A matrix sits row by row on the slot axis, entry (i, j) at slot `d * i + j`, in a slot period of `d * d`
    (`HomParams.n_slots`), and so does the product. With every index modulo d, let sigma(A)[i, j] = A[i, i + j] and
    tau(B)[i, j] = B[i + j, j], and let phi^k shift the columns of a matrix left by k and psi^k shift its rows up
    by k. Then

        A B = sum_{k < d} phi^k(sigma(A)) * psi^k(tau(B)),

    with `*` the entrywise product: at (i, j) term k is A[i, i + j + k] * B[i + j + k, j], and i + j + k takes each
    value modulo d once. On the slots, psi^k is the rotation by d * k, and phi^k is the rotation by k on the columns
    j < d - k and the rotation by k - d on the other columns.

    The slots are complex, so two terms of the sum share one product. With h = d / 2,
    A_c = sigma(A) + i phi^h(sigma(A)) and B_c = tau(B) - i psi^h(tau(B)), the real part of
    phi^k(A_c) * psi^k(B_c) is the sum of terms k and k + h. So

        A B = Re(z),   z = sum_{k < h} phi^k(A_c) * psi^k(B_c),

    which takes h shifts of each kind. The op computes z / 2, the column masks holding 1 / 2 where they keep a slot,
    and returns twice its real part.

    The product takes three levels: one for A_c and B_c, one for the column masks of phi^k, one for the
    multiplication.

    The h shifts of each kind are made in two stages, the shift k = low * g + b as the rotation for b followed by the
    rotation for low * g, with h = low * high and low <= high. Two stages keep the rotation keys few: each kind needs
    those of low - 1 and of high - 1 rotations, about 2 sqrt(h) in all.

    Args:
        d: Number of rows and of columns, a power of two, at least 2.
    """

    def __init__(self, d: int) -> None:
        super().__init__()
        n = d * d
        half = d // 2
        slot = torch.arange(n)
        row, col = slot // d, slot % d

        # A_c and B_c, each the sum of two slot permutations of its input
        self.sigma = self._two_permutations(d * row + (row + col) % d, 1, d * row + (row + col + half) % d, 1j)
        self.tau = self._two_permutations(d * ((row + col) % d) + col, 1, d * ((row + col + half) % d) + col, -1j)

        low = 1 << (half.bit_length() - 1) // 2
        high = half // low
        # the input and its rotations, stacked on a new first axis, the input first
        stack = functools.partial(HomRotateSum, perform_sum=False, add_identity_rotation=True)

        # phi^k for k = low * g + b. `wrap` stacks the input and its rotation by -d, so the two stages rotate both by
        # k: entry (g, b, 0) of the stack is the rotation by k and entry (g, b, 1) the rotation by k - d. Each is
        # masked to its columns, and phi^k is the sum of the pair. At k = 0 the rotation by -d has an all-zero mask.
        self.wrap = stack([n - d])
        self.shift_cols_low = stack(range(1, low))
        self.shift_cols_high = stack(low * g for g in range(1, high))
        # The masks hold 1 / 2 on the slots they keep. They are stored complex, as A_c and B_c are: clear execution
        # takes the weights of a pipeline in one dtype.
        keep = torch.stack([(col < d - k).to(torch.complex128) for k in range(half)]).reshape(high, low, n)
        self.mask_cols = HomConstMul(dims=(high, low, 2, n)).set_data(torch.stack([keep, 1 - keep], dim=2) / 2)

        # psi^k for k = low * g + b: the rotation by d * k
        self.shift_rows_low = stack(d * b for b in range(1, low))
        self.shift_rows_high = stack(d * low * g for g in range(1, high))

        # Entry (g, b) of a stack of shifts becomes entry k = low * g + b. The two masked parts of phi^k both
        # multiply psi^k, so the products are summed over k for each part, and the two sums are added.
        self.cols_by_shift = HomReshape((half, 2, n))
        self.rows_by_shift = HomReshape((half, 1, n))
        self.mul = HomMul(axis_sum=0)
        self.add_pair = HomAxisSum(dim=0)

        # twice the real part of z / 2: the first entry of what HomSplitRealImag returns
        self.real_imag = HomSplitRealImag()
        self.real = HomSlice(dim=0, key=0)

    @staticmethod
    def _two_permutations(source, weight, other_source, other_weight) -> HomMatMul:
        """output[slot] = weight * input[source[slot]] + other_weight * input[other_source[slot]], as a matrix given
        by its diagonals."""
        n = len(source)
        slot = torch.arange(n)
        rotations = torch.cat([(source - slot) % n, (other_source - slot) % n]).unique()
        diagonals = torch.zeros(len(rotations), n, dtype=torch.complex128)
        diagonals[torch.searchsorted(rotations, (source - slot) % n), source] += weight
        diagonals[torch.searchsorted(rotations, (other_source - slot) % n), other_source] += other_weight
        return HomMatMul(dims=(len(rotations), n), rotations=rotations.tolist()).set_data(diagonals)

    def forward(self, a: HomValue, b: HomValue) -> HomValue:
        a_shifted = self.shift_cols_high(self.shift_cols_low(self.wrap(self.sigma(a))))
        a_shifted = self.cols_by_shift(self.mask_cols(a_shifted))
        b_shifted = self.rows_by_shift(self.shift_rows_high(self.shift_rows_low(self.tau(b))))
        z = self.add_pair(self.mul(a_shifted, b_shifted))
        return self.real(self.real_imag(z))
