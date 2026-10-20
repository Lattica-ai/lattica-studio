"""See `params/README.md` for usage details."""

import enum

class BootstrappingVariant(enum.Enum):
    SLIM = 0
    COMPLEX = 1
    REAL = 2


BOOT_CONFIGURATION = {
    # Prime sizes (bits) and level counts of the CtS, EvalMod and StC sections.
    'cts_scale' : 48,
    'n_cts_levels' : 3,
    # Also the size of q0: EvalMod running at scale q0 makes the division by
    # delta a free scale relabel.
    'eval_mod_scale' : 50,
    'stc_scale' : 30,
    'n_stc_levels' : 3,

    'n_cosine_levels' : 5,
    'n_double_angle' : 3,
    'n_arcsine_levels' : 0,
    # Sparse-secret encapsulation: ModRaise switches q0 to an ephemeral secret
    # of this hamming weight, raises the modulus, and switches back, so the
    # wrap count |I| depends on it rather than on sk_hw. ModRaise always
    # encapsulates, so it must be nonzero.
    'sparse_secret_hw' : 32,
    # Bound on the ModRaise wrap count |I|. With the H=32 ephemeral secret, K=16 fails with
    # probability 2^-138.7 for 2^15 slots (eprint 2022/024). Han-Ki's cosine
    # needs degree 2K-2 = 30, the largest even degree in 5 levels.
    'k' : 16,

    # Headroom q0/|m| for messages bounded by 1:
    # the slots for SLIM, the polynomial coefficients for REAL/COMPLEX. Costs
    # precision 1:1 but suppresses the sine's uncorrected cubic term. SLIM
    # peaks at 12; REAL's coefficients are ~sqrt(2n) below its n slots, so 6
    # suffices unless the plaintext has few slots or is near-constant.
    'log_delta' : 6,
    'log_delta_slim' : 12,

    # SubSum stage size cap and the DFTs' baby-step/giant-step ratio.
    'subsum_max_stage_size' : 8,
    'baby_giant_ratio' : 2,
}


class BootstrappingParams:

    @staticmethod
    def get_q_list_precision(scale: int, levels: int):
        if scale < 25 or scale > 62:
            raise ValueError(f"scale must be in the range [25, 62]; got {scale}.")
        if levels <= 0:
            raise ValueError(f"levels must be > 0; got {levels}.")
        if scale < 32:
            half_levels = levels // 2
            extra_level = levels % 2
            return tuple((2 * scale, scale,) for _ in range(half_levels)) + tuple((scale,) for _ in range(extra_level))
        return tuple((scale,) for _ in range(levels))

    @staticmethod
    def get_q_base_precision() -> int:
        """Size of q0, the ModRaise base modulus: the EvalMod prime size."""
        return BOOT_CONFIGURATION['eval_mod_scale']
    
    def __init__(self, bootstrapping_variant: BootstrappingVariant = BootstrappingVariant.REAL) -> None:

        self.variant = BootstrappingVariant(bootstrapping_variant)
        self.stc_scale = BOOT_CONFIGURATION['stc_scale']
        self.eval_mod_scale = BOOT_CONFIGURATION['eval_mod_scale']
        self.cts_scale = BOOT_CONFIGURATION['cts_scale']
        self.n_double_angle = BOOT_CONFIGURATION['n_double_angle']
        self.subsum_max_stage_size = BOOT_CONFIGURATION['subsum_max_stage_size']
        self.baby_giant_ratio = BOOT_CONFIGURATION['baby_giant_ratio']
        self.arcsine_correction = BOOT_CONFIGURATION['n_arcsine_levels'] > 0

        self.n_stc_levels = BOOT_CONFIGURATION['n_stc_levels']
        self.n_cosine_levels = BOOT_CONFIGURATION['n_cosine_levels']
        self.n_arcsine_levels = BOOT_CONFIGURATION['n_arcsine_levels']
        self.n_cts_levels = BOOT_CONFIGURATION['n_cts_levels']
        self.stc_q_list_precision = self.get_q_list_precision(self.stc_scale, self.n_stc_levels)
        self.cts_q_list_precision = self.get_q_list_precision(self.cts_scale, self.n_cts_levels)
        if self.variant != BootstrappingVariant.SLIM:
            self.log_delta = BOOT_CONFIGURATION['log_delta']
        else:
            self.log_delta = BOOT_CONFIGURATION['log_delta_slim']
        self.eval_mod_q_list_precision = self.get_q_list_precision(self.eval_mod_scale, self.n_cosine_levels + self.n_double_angle + self.n_arcsine_levels)
        self.k = BOOT_CONFIGURATION['k']
        self.sparse_secret_hw = BOOT_CONFIGURATION['sparse_secret_hw']
        self.q_list_precision = self.cts_q_list_precision +  self.eval_mod_q_list_precision + self.stc_q_list_precision
        self.q_base_precision = self.get_q_base_precision()
        self.modulus = sum(row[0] for row in self.q_list_precision) + self.q_base_precision
        # The tree evaluates degree d in ceil(log2(d + 1)) levels: n levels fit degree 2^n - 1.
        # The cosine is even (in y = x - 1/(4K), evaluated through z = T_2(y)), so its top degree is 2^n - 2.
        self.cosine_deg = 2 ** self.n_cosine_levels - 2
        # Han-Ki silently raises a lower degree to 2K-2, deeper than provisioned.
        if 2 * self.k - 2 > self.cosine_deg:
            raise ValueError(
                f"k={self.k} needs a cosine of degree {2 * self.k - 2}; "
                f"n_cosine_levels={self.n_cosine_levels} gives degree {self.cosine_deg}.")
        self.arcsine_deg = 2 ** self.n_arcsine_levels - 1   # 0 without an arcsine
