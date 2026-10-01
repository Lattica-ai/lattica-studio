import torch

from lattica_build.base_classes.hom_op import HomOp
from lattica_build.base_classes.hom_pipeline import HomomorphicPipeline
from lattica_build.base_classes.hom_value import HomValue
from lattica_build.base_classes.pipeline_wrapper import PipelineWrapper
from lattica_build.operators.arithmetic.h_const_mul import HomConstMul
from lattica_build.operators.composite.module_list import ModuleListHomOp
from lattica_build.operators.composite.sequential import SequentialHomOp
from lattica_build.operators.fhe.h_bootstrap import Bootstrap
from lattica_build.operators.polynomials.h_poly_threshold import HomPolyThreshold
from lattica_build.operators.shape.h_squeeze import HomSqueeze
from lattica_build.operators.slots.h_rotate_sum import HomRotateSum
from lattica_build.params.params import HomParams


LOG_N = 16           # ring degree 2**LOG_N, i.e. 2**(LOG_N - 1) slots
DEG = 119            # Chebyshev degree of the threshold comparator
MARGIN = 0.04        # comparator don't-care band; entries closer than this may come out unordered
# Value range of the inputs. Inputs must span < 1 so every comparator
# input (a difference of two entries) stays inside its [-1, 1] fit domain, with headroom for
# encryption noise: just past the domain the degree-DEG polynomial blows up.
VAL_LO, VAL_HI = -0.45, 0.45
SORT_TOLERANCE = 0.05        # max per-entry error against the true sort

LOG_SCALE = 30
BOOT_EVERY = 1
Q_ROWS = 4
SPECIAL_PRIMES = 6


def _get_masks(array_len: int, k: int, j: int) -> list[torch.Tensor]:
    i = torch.arange(array_len)
    asc, low = (i & k) == 0, (i & j) == 0
    return [(asc & low).float(), (asc & ~low).float(), (~asc & low).float(), (~asc & ~low).float()]


def _mask_mul(mask: torch.Tensor) -> HomConstMul:
    op = HomConstMul(dims=tuple(mask.shape))
    op.set_data(mask)
    return op

def _rotate(s: int) -> SequentialHomOp:
    """Cyclic rotate by s, rot(x, s)[i] == x[i+s]; the squeeze undoes HomRotateSum's new axis."""
    return SequentialHomOp(HomRotateSum(rotations=[s], perform_sum=False), HomSqueeze(dim=0))


class _Stage(HomOp):
    """One compare-exchange layer of the bitonic sort"""

    def __init__(self, array_len: int, k: int, j: int):
        super().__init__()
        m_el, m_eh, m_dl, m_dh = _get_masks(array_len, k, j)
        self.rot_up = _rotate(+j)
        self.rot_down = _rotate(-j)
        # torch.roll(v, -j) is the plaintext mirror of rot(v, +j): out[i] = v[i+j].
        self.mask_swap_low = _mask_mul(m_dl - m_el)
        self.mask_swap_high = _mask_mul(torch.roll(m_eh - m_dh, -j))
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
    def __init__(self, array_len: int, boot_every: int = BOOT_EVERY):
        super().__init__()
        stages = []
        k = 2
        while k <= array_len:
            j = k // 2
            while j > 0:
                stages.append(_Stage(array_len, k, j))
                j //= 2
            k *= 2
        self.stages = ModuleListHomOp(stages)

        self.bootstrap = Bootstrap(target_output_scale=2 ** LOG_SCALE)
        # Refresh every boot_every stages, never after the last.
        self.boot_after = set(range(boot_every - 1, len(self.stages) - 1, boot_every))

    def forward(self, x: HomValue) -> HomValue:
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i in self.boot_after:
                x = self.bootstrap(x)
        return x


class Pipeline(PipelineWrapper):

    def __init__(self, array_len: int = 1024) -> None:
        max_len = 2 ** (LOG_N - 1)
        if array_len <= 0 or array_len & (array_len - 1) or array_len > max_len:
            raise ValueError(f"array_len must be a power of two up to {max_len}, got {array_len}")
        self.array_len = array_len

    def build_pipeline(self) -> HomomorphicPipeline:
        """Construct a bitonic homomorphic pipeline."""
        hom_pipeline = HomomorphicPipeline(
            input_shape=(self.array_len,),
            hom=_BitonicSort(self.array_len),
        )
        hom_pipeline.verification_data = {
            hom_pipeline.primary_input_name: self._set_example_pt(),
            "accuracy": 2 ** -3,
        }
        return hom_pipeline

    def build_params(self) -> HomParams:
        return HomParams(
            n=2 ** LOG_N,
            full_q_list_precision=Q_ROWS * ((LOG_SCALE * 2, LOG_SCALE),),
            pt_scale=2 ** LOG_SCALE,
            sk_hw=192,
            num_special_primes=SPECIAL_PRIMES,
            n_slots=self.array_len,
        )

    def _set_example_pt(self) -> torch.Tensor:
        return torch.rand(self.array_len) * (VAL_HI - VAL_LO) + VAL_LO

    def compute_expected(self, example_pt: torch.Tensor) -> torch.Tensor:
        assert example_pt.ndim == 1 and example_pt.shape[0] == self.array_len, (
            f"Input must be 1D of length {self.array_len}"
        )
        return torch.sort(example_pt).values

    def verify_results(self, actual: torch.Tensor, expected: torch.Tensor) -> None:
        # Check against the true sort rather than `expected`, the runtime's clear execution.
        true_sorted = self.compute_expected(self.exmpl_pt).to(actual.dtype)
        max_err = (actual[:self.array_len] - true_sorted).abs().max().item()
        assert max_err < SORT_TOLERANCE, f"sort not within tolerance: max error {max_err:.4g}"
