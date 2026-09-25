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
    # delta a free scale relabel (see h_bootstrapping.py).
    'eval_mod_scale' : 50,
    'stc_scale' : 30,
    'n_stc_levels' : 3,

    'n_cosine_levels' : 6,
    'n_double_angle' : 3,
    'n_arcsine_levels' : 0,
    # Bound on the ModRaise wrap count |I|. Han-Ki needs degree 2K-2, so depth 6
    # allows K <= 30. At sk_hw=192, |I| >= 30 somewhere in a
    # ciphertext has probability 2**-27.6, and corrupts it silently;
    'k' : 30,

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
    def optimal_poly_deg_per_depth(depth):
        optimal_degs = [2, 3, 5, 13, 27, 59, 119, 243]
        if depth < 0 or depth > 8:
            raise ValueError("depth must be in the range [0, 8]")
        if depth == 0:
            return 0
        return optimal_degs[depth - 1]

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
        self.q_list_precision = self.cts_q_list_precision +  self.eval_mod_q_list_precision + self.stc_q_list_precision
        self.q_base_precision = self.get_q_base_precision()
        self.modulus = sum(row[0] for row in self.q_list_precision) + self.q_base_precision
        self.cosine_deg = self.optimal_poly_deg_per_depth(self.n_cosine_levels)
        # Han-Ki silently raises a lower degree to 2K-2, deeper than provisioned.
        if 2 * self.k - 2 > self.cosine_deg:
            raise ValueError(
                f"k={self.k} needs a cosine of degree {2 * self.k - 2}; "
                f"n_cosine_levels={self.n_cosine_levels} gives degree {self.cosine_deg}.")
        self.arcsine_deg = self.optimal_poly_deg_per_depth(self.n_arcsine_levels)
