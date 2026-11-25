"""See `operators/ml/README.md` for usage details."""

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

        # phi^k for k < d: entries k of the stack are the rotations by k, entries d + k the rotations by k - d. The
        # rotation by -d, at k = 0, has an all-zero mask: it only keeps the two halves of the stack the same size.
        self.shift_cols = HomRotateSum(rotations=[*range(1, d), *((k - d) % n for k in range(d))],
                                       perform_sum=False, add_identity_rotation=True)
        keep = torch.stack([(col < d - k).double() for k in range(d)])
        self.mask_cols = HomConstMul(dims=(2 * d, n)).set_data(torch.cat([keep, 1 - keep]))
        self.pair = HomReshape((2, d, n))
        self.add_pair = HomAxisSum(dim=0)

        self.shift_rows = HomRotateSum(rotations=[d * k for k in range(1, d)], perform_sum=False,
                                       add_identity_rotation=True)
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
        a_shifted = self.add_pair(self.pair(self.mask_cols(self.shift_cols(self.sigma(a)))))
        b_shifted = self.shift_rows(self.tau(b))
        return self.mul(a_shifted, b_shifted)
