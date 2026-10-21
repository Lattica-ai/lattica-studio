"""See `params/README.md` for usage details."""

import enum
from numbers import Real
from dataclasses import dataclass
from typing import Any, Sequence, Tuple, Union

from lattica_build.params.bootstrapping_params import BootstrappingParams

QListNorm = Tuple[Tuple[int, ...], ...]
QListInput = Sequence[Sequence[int]]
GHSInput = Sequence[int]


class DecompositionType(enum.StrEnum):
    BV = enum.auto()
    HYBRID = enum.auto()

def _is_power_of_two(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0


SPECIAL_PRIME_BITS = 61
# HYBRID digits when dnum is not given: the fastest bootstrap measured with the smallest keys (2^16, 10x40 and
# 5x(60,30) after it); 1-2 put a digit over the fused ModUp's 12 primes or log QP over 128-bit security there.
DEFAULT_DNUM = 3


def _prime_bits(q_list_precision: QListNorm) -> list[int]:
    """Each prime's bits, in chain order: a row lists cumulative precisions, so (60, 30) holds two 30-bit primes."""
    return [bits for row in q_list_precision
            for bits in (*(row[i] - row[i + 1] for i in range(len(row) - 1)), row[-1])]


def _num_digits(prime_bits: Sequence[int], cap: int) -> int:
    """The digits of the toolkit's hybrid split: consecutive primes, greedily, at most cap bits each."""
    digits, acc = 1, 0
    for bits in prime_bits:
        if acc + bits > cap:
            digits, acc = digits + 1, bits
        else:
            acc += bits
    return digits

@dataclass(kw_only=True)
class HomParams:
    n: int
    full_q_list_precision: QListNorm
    pt_scale: int
    err_std: float = 3.19
    sk_hw: int = 0
    bv_gadget_bits: int = 4
    # HYBRID: the digits the whole chain (pipeline and bootstrap) is split into; the special primes follow from it.
    # None: DEFAULT_DNUM with HYBRID, 0 with BV.
    dnum: int | None = None
    decomposition_type: Union[DecompositionType, str] = DecompositionType.HYBRID
    num_init_rows: int | None = None
    # Bootstrapping: real slots (one EvalMod) or complex; StC first (slim) or last.
    real_boot: bool = True
    slim_boot: bool = False
    n_slots: int | None = None

    @property
    def internal_n(self) -> int:
        return self.n // 2

    @property
    def bootstrapping(self) -> bool:
        return self.boot_params is not None

    @property
    def chain_q_list_precision(self) -> QListNorm:
        """The ciphertext primes: the pipeline's, then the bootstrap's (once the pipeline has set boot_params)."""
        if self.boot_params is None:
            return self.full_q_list_precision
        return (self.full_q_list_precision + self.boot_params.cts_q_list_precision
                + self.boot_params.eval_mod_q_list_precision + self.boot_params.stc_q_list_precision
                + ((self.boot_params.q_base_precision,),))

    @property
    def num_special_primes(self) -> int:
        """HYBRID: the fewest special primes P whose digit cap, |P|, splits the chain into at most dnum digits."""
        if self.decomposition_type != DecompositionType.HYBRID:
            return 0
        bits = _prime_bits(self.chain_q_list_precision)
        num = -(-max(bits) // SPECIAL_PRIME_BITS)
        while _num_digits(bits, SPECIAL_PRIME_BITS * num) > self.dnum:
            num += 1
        return num

    @property
    def g_base_bits(self) -> int:
        if self.decomposition_type == DecompositionType.HYBRID:
            return SPECIAL_PRIME_BITS * self.num_special_primes
        return self.bv_gadget_bits

    @property
    def ghs_keyswtich_scale_precision(self) -> tuple[int, ...] | None:
        if self.decomposition_type == DecompositionType.HYBRID:
            return (SPECIAL_PRIME_BITS,) * self.num_special_primes
        return None

    def __post_init__(self) -> None:
        self.boot_params = None
        self.ring_switch_params = None
        self.full_q_list_precision = tuple(tuple(row) for row in self.full_q_list_precision)

        if not isinstance(self.n, int) or not _is_power_of_two(self.n):
            raise ValueError(f"n must be a positive power of two; got {self.n!r}.")
        
        if not isinstance(self.err_std, Real) or self.err_std < 0:
            raise ValueError(f"err_std must be a non-negative number; got {self.err_std!r}.")

        if self.dnum is None:
            self.dnum = 0 if self.decomposition_type == DecompositionType.BV else DEFAULT_DNUM
        if not isinstance(self.dnum, int) or self.dnum < 0:
            raise ValueError(f"dnum must be an int >= 0; got {self.dnum!r}.")

        if self.decomposition_type == DecompositionType.BV and self.dnum > 0:
            raise ValueError(f"dnum must be 0 when using BV decomposition; got {self.dnum}")

        if self.decomposition_type == DecompositionType.HYBRID:
            if self.dnum == 0:
                raise ValueError(f"dnum must be greater than 0 when using HYBRID decomposition; got {self.dnum}")
        elif not isinstance(self.bv_gadget_bits, int) or self.bv_gadget_bits < 1:
            raise ValueError(f"bv_gadget_bits must be an int >= 1; got {self.bv_gadget_bits!r}.")

        if not isinstance(self.sk_hw, int) or self.sk_hw < 0:
            raise ValueError(f"sk_hw must be an int >= 0; got {self.sk_hw!r}.")

        if isinstance(self.decomposition_type, DecompositionType):
            pass  # already valid
        elif isinstance(self.decomposition_type, str):
            try:
                self.decomposition_type = DecompositionType(self.decomposition_type)
            except KeyError as e:
                raise ValueError(f"Invalid decomposition_type string: {self.decomposition_type}") from e
        else:
            raise TypeError(
                f"decomposition_type must be DecompositionType or valid str, got {type(self.decomposition_type).__name__}"
            )

        if not isinstance(self.pt_scale, int) or self.pt_scale < 1:
            raise ValueError(f"pt_scale must be an int >= 1; got {self.pt_scale!r}.")

        if self.n_slots is None:
            self.n_slots = self.internal_n
        else:
            if not (isinstance(self.n_slots, int) and _is_power_of_two(self.n_slots)):
                raise ValueError(
                    f"n_slots must be None or a positive power of two; got {self.n_slots!r}.")
            if self.n_slots > self.internal_n:
                raise ValueError(
                    f"n_slots={self.n_slots} is larger than the {self.internal_n} available slots.")

        if not ((isinstance(self.num_init_rows, int) and self.num_init_rows >= 0) or self.num_init_rows is None):
            raise ValueError(f"num_init_rows must be None or an int >= 0; got {self.num_init_rows!r}.")
        for name in ("real_boot", "slim_boot"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a bool; got {getattr(self, name)!r}.")


    def to_dict(self) -> dict[str, Any]:
        return {
            "full_q_list_precision": self.full_q_list_precision,
            "n": self.n,
            "err_std": self.err_std,
            "sk_hw": self.sk_hw,
            "dnum": self.dnum,
            "g_base_bits": self.g_base_bits,
            "ghs_keyswtich_scale_precision": self.ghs_keyswtich_scale_precision or (),
            "decomposition_type": self.decomposition_type,
            "ring_switch_params": (
                self.ring_switch_params.to_dict()
                if self.ring_switch_params is not None else None
            ),
            "pt_scale": self.pt_scale,
            "bootstrapping": self.bootstrapping,
            "num_init_rows": self.num_init_rows,
            "real_boot": self.real_boot,
            "slim_boot": self.slim_boot,
            "n_slots": self.n_slots,
            # the chain BootstrappingParams.for_slot_counts chose, so from_dict rebuilds it
            "boot_dft_levels": self.boot_params.dft_levels if self.boot_params is not None else None,
            "boot_cts_scale": self.boot_params.cts_scale if self.boot_params is not None else None,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "HomParams":
        d = d.copy() # Avoid modifying the original dict

        ring_switch_params = d.pop("ring_switch_params", None)
        bootstrapping = d.pop("bootstrapping", False)
        boot_dft_levels = d.pop("boot_dft_levels", None)
        boot_cts_scale = d.pop("boot_cts_scale", None)

        # the special primes are derived from dnum and the chain; only BV's gadget bits are an input
        g_base_bits = d.pop("g_base_bits", None)
        if d["decomposition_type"] == DecompositionType.BV and g_base_bits is not None:
            d["bv_gadget_bits"] = g_base_bits
        elif "dnum" not in d:
            raise ValueError("HYBRID params saved without dnum (from before it replaced num_special_primes): recompile "
                             "the pipeline")
        d.pop("ghs_keyswtich_scale_precision", None)

        params = cls(**d)
        if bootstrapping:
            params.boot_params = BootstrappingParams(
                params.real_boot, params.slim_boot, dft_levels=boot_dft_levels, cts_scale=boot_cts_scale)
        if ring_switch_params is not None:
            params.ring_switch_params = cls.from_dict(ring_switch_params)
        return params
