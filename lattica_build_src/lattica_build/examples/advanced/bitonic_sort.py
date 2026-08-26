import numpy as np
import torch

from lattica_build.base_classes.hom_op import HomOp
from lattica_build.base_classes.hom_pipeline import HomomorphicPipeline
from lattica_build.base_classes.hom_value import HomValue
from lattica_build.operators.arithmetic.h_const_mul import HomConstMul
from lattica_build.operators.client_ops import Repeat
from lattica_build.operators.composite.module_list import ModuleListHomOp
from lattica_build.operators.composite.sequential import SequentialHomOp
from lattica_build.operators.fhe.h_bootstrap import Bootstrap
from lattica_build.operators.polynomials.h_poly_threshold import HomPolyThreshold
from lattica_build.operators.shape.h_squeeze import HomSqueeze
from lattica_build.operators.slots.h_rotate_sum import HomRotateSum
from lattica_build.params.bootstrapping_params import BootstrappingVariant
from lattica_build.params.params import HomParams


ARRAY_LEN = 16       # array length being sorted; must be a power of two
LOG_N = 16           # ring degree 2**LOG_N, i.e. 2**(LOG_N - 1) slots
DEG = 119            # Chebyshev degree of the threshold comparator
MARGIN = 0.04        # comparator don't-care band; entries closer than this may come out unordered

LOG_SCALE = 30
BOOT_EVERY = 1
Q_ROWS = 4
SPECIAL_PRIMES = 6


def _get_masks(array_len: int, n_slots: int, k: int, j: int) -> list[np.ndarray]:
    masks = [np.zeros(array_len) for _ in range(4)]
    for i in range(array_len):
        asc, low = (i & k) == 0, (i & j) == 0
        masks[(0 if low else 1) if asc else (2 if low else 3)][i] = 1
    return [np.tile(m, n_slots // array_len) for m in masks]


def _mask_mul(mask: np.ndarray) -> HomConstMul:
    op = HomConstMul(dims=tuple(mask.shape))
    op.set_data(torch.tensor(mask, dtype=torch.float32))
    return op


def _log_n_subring(array_len: int) -> int:
    """Subring holds the array twice over; floored at 4 for the coefs-to-slots split."""
    return max(4, int(np.log2(array_len)) + 1)


def _rotate(s: int) -> SequentialHomOp:
    """Cyclic rotate by s, rot(x, s)[i] == x[i+s]; the squeeze undoes HomRotateSum's new axis."""
    return SequentialHomOp(HomRotateSum(rotations=[s], perform_sum=False), HomSqueeze(dim=0))


def build_pipeline() -> HomomorphicPipeline:
    """Construct a bitonic homomorphic pipeline."""

    class _Stage(HomOp):
        """One compare-exchange layer of the bitonic sort"""

        def __init__(self, array_len: int, n_slots: int, k: int, j: int):
            super().__init__()
            m_el, m_eh, m_dl, m_dh = _get_masks(array_len, n_slots, k, j)
            self.rot_up = _rotate(+j)
            self.rot_down = _rotate(-j)
            # np.roll(v, -j) is the plaintext mirror of rot(v, +j): out[i] = v[i+j].
            self.mask_swap_low = _mask_mul(m_dl - m_el)
            self.mask_swap_high = _mask_mul(np.roll(m_eh - m_dh, -j))
            self.sel_up, self.sel_down, self.sel_keep = (
                _mask_mul(m_el), _mask_mul(m_eh), _mask_mul(m_dl + m_dh))
            # A band that shrinks with array_len stops being resolvable at DEG, and the stage
            # outputs then leave the [-1, 1] Chebyshev domain and diverge.
            self.step = HomPolyThreshold(
                degree=DEG, margin=[-MARGIN, MARGIN], variant='minimax', tol=1e-5)

        def forward(self, x: HomValue) -> HomValue:
            x_up = self.rot_up(x)
            d = x_up - x
            t = self.step(d)
            p1 = t * self.mask_swap_low(d)
            p2 = t * self.mask_swap_high(d)
            x_lin = self.sel_up(x_up) + self.sel_down(self.rot_down(x)) + self.sel_keep(x)
            return x_lin + p1 + self.rot_down(p2)


    class _BitonicSort(HomOp):
        def __init__(self, array_len: int, n_slots: int, boot_every: int = BOOT_EVERY):
            super().__init__()
            stages = []
            k = 2
            while k <= array_len:
                j = k // 2
                while j > 0:
                    stages.append(_Stage(array_len, n_slots, k, j))
                    j //= 2
                k *= 2
            self.stages = ModuleListHomOp(stages)

            self.bootstrap = Bootstrap(log_n_subring=_log_n_subring(array_len),
                                       target_output_scale=2 ** LOG_SCALE)
            # Refresh every boot_every stages, never after the last.
            self.boot_after = set(range(boot_every - 1, len(self.stages) - 1, boot_every))

        def forward(self, x: HomValue) -> HomValue:
            for i, stage in enumerate(self.stages):
                x = stage(x)
                if i in self.boot_after:
                    x = self.bootstrap(x)
            return x

    return HomomorphicPipeline(
        client_pre=[Repeat()],
        hom=_BitonicSort(ARRAY_LEN, 2 ** (LOG_N - 1)),
        input_shape=(ARRAY_LEN,),
    )


def build_params() -> HomParams:
    return HomParams(
        n=2 ** LOG_N,
        full_q_list_precision=Q_ROWS * ((LOG_SCALE * 2, LOG_SCALE),),
        pt_scale=2 ** LOG_SCALE,
        sk_hw=192,
        num_special_primes=SPECIAL_PRIMES,
    )
