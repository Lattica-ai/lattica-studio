"""See `params/README.md` for usage details."""

import enum
from numbers import Real
from dataclasses import dataclass
from typing import Any, Sequence, Tuple, Union

from lattica_build.params.bootstrapping_params import BootstrappingVariant, BootstrappingParams

QListNorm = Tuple[Tuple[int, ...], ...]
QListInput = Sequence[Sequence[int]]
GHSInput = Sequence[int]


class DecompositionType(enum.StrEnum):
    BV = enum.auto()
    HYBRID = enum.auto()

def _is_power_of_two(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0

@dataclass(kw_only=True)
class HomParams:
    n: int
    full_q_list_precision: QListNorm
    pt_scale: int
    err_std: float = 3.19
    sk_hw: int = 0
    bv_gadget_bits: int = 4
    num_special_primes: int = 0
    decomposition_type: Union[DecompositionType, str] = DecompositionType.HYBRID
    num_init_rows: int | None = None
    bootstrapping_variant: BootstrappingVariant = BootstrappingVariant.REAL
    n_slots: int | None = None

    @property
    def internal_n(self) -> int:
        return self.n // 2

    @property
    def bootstrapping(self) -> bool:
        return self.boot_params is not None

    def __post_init__(self) -> None:
        self.boot_params = None
        self.ring_switch_params = None
        self.full_q_list_precision = tuple(tuple(row) for row in self.full_q_list_precision)

        if not isinstance(self.n, int) or not _is_power_of_two(self.n):
            raise ValueError(f"n must be a positive power of two; got {self.n!r}.")
        
        if not isinstance(self.err_std, Real) or self.err_std < 0:
            raise ValueError(f"err_std must be a non-negative number; got {self.err_std!r}.")

        if not isinstance(self.num_special_primes, int) or self.num_special_primes < 0:
            raise ValueError(f"num_special_primes must be an int >= 0; got {self.num_special_primes!r}.")

        if self.decomposition_type == DecompositionType.BV and self.num_special_primes > 0:
            raise ValueError(
                "num_special_primes must be 0 when using BV decomposition; "
                f"got {self.num_special_primes}"
            )
        
        if self.decomposition_type == DecompositionType.HYBRID: 
            if self.num_special_primes == 0:
                raise ValueError(
                    "num_special_primes must be greater than 0 when using HYBRID decomposition; "
                    f"got {self.num_special_primes}"
                )
            self.ghs_keyswtich_scale_precision = (61,) * self.num_special_primes
            self.g_base_bits = 61 * self.num_special_primes
        else:
            if not isinstance(self.bv_gadget_bits, int) or self.bv_gadget_bits < 1:
                raise ValueError(f"bv_gadget_bits must be an int >= 1; got {self.bv_gadget_bits!r}.")
            self.g_base_bits = self.bv_gadget_bits
            self.ghs_keyswtich_scale_precision = None

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
            if self.n_slots is not None and not (
                    isinstance(self.n_slots, int) and _is_power_of_two(self.n_slots)):
                raise ValueError(
                    f"n_slots must be None or a positive power of two; got {self.n_slots!r}.")
            if self.n_slots > self.internal_n:
                raise ValueError(
                    f"n_slots={self.n_slots} is larger than the {self.internal_n} available slots.")

        if not ((isinstance(self.num_init_rows, int) and self.num_init_rows >= 0) or self.num_init_rows is None):
            raise ValueError(f"num_init_rows must be None or an int >= 0; got {self.num_init_rows!r}.")
        if isinstance(self.bootstrapping_variant, int):
            try:
                self.bootstrapping_variant = BootstrappingVariant(self.bootstrapping_variant)
            except ValueError as e:
                raise ValueError(f"Invalid bootstrapping_variant value: {self.bootstrapping_variant}") from e
        elif not isinstance(self.bootstrapping_variant, BootstrappingVariant):
            raise ValueError(f"bootstrapping_variant must be a BootstrappingVariant; got {self.bootstrapping_variant!r}.")


    def to_dict(self) -> dict[str, Any]:
        return {
            "full_q_list_precision": self.full_q_list_precision,
            "n": self.n,
            "err_std": self.err_std,
            "sk_hw": self.sk_hw,
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
            "bootstrapping_variant": self.bootstrapping_variant.value,
            "n_slots": self.n_slots,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "HomParams":
        d = d.copy() # Avoid modifying the original dict

        ring_switch_params = d.pop("ring_switch_params", None)
        bootstrapping = d.pop("bootstrapping", False)

        if "g_base_bits" in d:
            if d["decomposition_type"]==DecompositionType.BV:
                d["bv_gadget_bits"] = d.pop("g_base_bits")
            else:
                d["num_special_primes"] = d.pop("g_base_bits") // 61
        d.pop("ghs_keyswtich_scale_precision", None)

        params = cls(**d)
        if bootstrapping:
            params.boot_params = BootstrappingParams(params.bootstrapping_variant, params.sk_hw)
        if ring_switch_params is not None:
            params.ring_switch_params = cls.from_dict(ring_switch_params)
        return params
