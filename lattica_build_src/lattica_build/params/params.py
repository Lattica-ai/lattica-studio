"""See `params/README.md` for usage details."""

import enum
import functools
import json
import math
import os
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
# Most primes in a hybrid digit: the toolkit's fused ModUp (its HPS head kernel) takes at most this many.
MAX_DIGIT_PRIMES = 16
# Largest log2(QP) at 128-bit security, by (log2 n, secret Hamming weight sk_hw; 0 = uniform ternary): lattice-estimator
# with error sigma 3.19, m = n, Arora-GB and BKW excluded, the cheapest attack. dnum unset is derived from it.
MAX_LOG_QP_128 = {
    (14, 192): 402,
    (15, 192): 809,
    (16, 192): 1638,
}


@functools.cache
def _max_prime_log2(bits: int) -> float:
    """The largest log2 among the good primes of this size: a pool's primes can exceed their nominal size (the 26-bit
    ones reach 26.73 bits: few NTT-friendly primes that small), so the security bound counts them at this size."""
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "good_primes.json"), encoding="utf-8") as f:
        pool = json.load(f).get(str(bits))
    return max(math.log2(q) for q in pool) if pool else float(bits)


def _prime_bits(q_list_precision: QListNorm) -> list[int]:
    """Each prime's bits, in chain order: a row lists cumulative precisions, so (60, 30) holds two 30-bit primes."""
    return [bits for row in q_list_precision
            for bits in (*(row[i] - row[i + 1] for i in range(len(row) - 1)), row[-1])]


def _digit_sizes(prime_bits: Sequence[int], cap: int) -> list[int]:
    """The primes in each digit of the toolkit's hybrid split: consecutive primes, greedily, at most cap bits each."""
    sizes, acc = [0], 0
    for bits in prime_bits:
        if acc + bits > cap:
            sizes, acc = sizes + [0], 0
        sizes[-1] += 1
        acc += bits
    return sizes


def _num_special_primes(prime_bits: Sequence[int], dnum: int) -> int:
    """The fewest special primes P whose digit cap, |P|, splits the chain into at most dnum digits."""
    num = -(-max(prime_bits) // SPECIAL_PRIME_BITS)
    while len(_digit_sizes(prime_bits, SPECIAL_PRIME_BITS * num)) > dnum:
        num += 1
    return num

@dataclass(kw_only=True)
class HomParams:
    n: int
    full_q_list_precision: QListNorm
    pt_scale: int
    err_std: float = 3.19
    sk_hw: int = 0
    bv_gadget_bits: int = 4
    # HYBRID: the digits the whole chain (pipeline and bootstrap) is split into; the special primes follow from it.
    # None: derived (resolved_dnum), the fewest 128-bit-secure digits.
    dnum: int | None = None
    decomposition_type: Union[DecompositionType, str] = DecompositionType.HYBRID
    # With a bootstrap: fresh ciphertexts hold this many pipeline rows, counted from the bottom of the chain (the
    # last pipeline rows), plus the bootstrap base. None: all the pipeline rows.
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
    def resolved_dnum(self) -> int:
        """dnum, or if unset the fewest digits whose special primes keep the whole chain's log2(QP) within
        MAX_LOG_QP_128 and whose digits fit MAX_DIGIT_PRIMES. It follows the chain, so the bootstrap's once
        `serialize` has chosen it. 0 with BV."""
        if self.dnum is not None or self.decomposition_type != DecompositionType.HYBRID:
            return self.dnum or 0
        log_n = self.n.bit_length() - 1
        max_log_qp = MAX_LOG_QP_128.get((log_n, self.sk_hw))
        if max_log_qp is None:
            raise ValueError(f"No 128-bit bound for n = 2^{log_n} with sk_hw = {self.sk_hw} (MAX_LOG_QP_128): set dnum.")
        bits = _prime_bits(self.chain_q_list_precision)
        for dnum in range(1, len(bits) + 1):
            num = _num_special_primes(bits, dnum)
            if (max(_digit_sizes(bits, SPECIAL_PRIME_BITS * num)) <= MAX_DIGIT_PRIMES
                    and sum(map(_max_prime_log2, bits)) + _max_prime_log2(SPECIAL_PRIME_BITS) * num <= max_log_qp):
                return dnum
        raise ValueError(f"No dnum keeps log2(QP) of this chain ({sum(bits)} bits of Q) within {max_log_qp} bits, the "
                         f"128-bit bound for n = 2^{log_n} with sk_hw = {self.sk_hw}: shorten the chain or set dnum.")

    @property
    def num_special_primes(self) -> int:
        """HYBRID: the fewest special primes P whose digit cap, |P|, splits the chain into at most resolved_dnum
        digits."""
        if self.decomposition_type != DecompositionType.HYBRID:
            return 0
        return _num_special_primes(_prime_bits(self.chain_q_list_precision), self.resolved_dnum)

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
        # the pipeline rows in the order the pipeline consumes them, set by serialize from the level tracing; the
        # toolkit fills the key-switching digits along it
        self.consumption_order: tuple[int, ...] | None = None
        self.full_q_list_precision = tuple(tuple(row) for row in self.full_q_list_precision)

        if not isinstance(self.n, int) or not _is_power_of_two(self.n):
            raise ValueError(f"n must be a positive power of two; got {self.n!r}.")
        
        if not isinstance(self.err_std, Real) or self.err_std < 0:
            raise ValueError(f"err_std must be a non-negative number; got {self.err_std!r}.")

        if self.dnum is not None and (not isinstance(self.dnum, int) or self.dnum < 0):
            raise ValueError(f"dnum must be an int >= 0 or None; got {self.dnum!r}.")

        if self.decomposition_type == DecompositionType.BV and self.dnum:
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
            "consumption_order": self.consumption_order,
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
        consumption_order = d.pop("consumption_order", None)

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
        if consumption_order is not None:
            params.consumption_order = tuple(int(row) for row in consumption_order)
        return params
