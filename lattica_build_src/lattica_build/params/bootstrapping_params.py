"""See `params/README.md` for usage details."""


BOOT_CONFIGURATION = {
    # Prime sizes (bits) and level counts of the CtS, EvalMod and StC sections.
    # 4 DFT levels rather than 3: about half the diagonals per factor (16-31 vs
    # 32-63 at 2**15 slots), for one more prime each.
    'cts_scale' : 48,
    'n_cts_levels' : 4,
    # Also the size of q0: EvalMod running at scale q0 makes the division by
    # delta a free scale relabel.
    'eval_mod_scale' : 50,
    'stc_scale' : 30,
    'n_stc_levels' : 4,

    'n_cosine_levels' : 5,
    'n_double_angle' : 3,
    'n_arcsine_levels' : 0,
    # Sparse-secret encapsulation: ModRaise switches q0 to an ephemeral secret
    # of this hamming weight, raises the modulus, and switches back, so the
    # wrap count |I| depends on it rather than on sk_hw. ModRaise always
    # encapsulates, so it must be nonzero.
    'sparse_secret_hw' : 32,
    # Bound on the ModRaise wrap count |I|: Han-Ki needs degree 2K-2, so depth 5
    # (degree 27) allows K <= 14. At H=32, |I| >= 14 has probability 2**-38.8.
    'k' : 14,

    # Headroom q0/|m| for messages bounded by 1, set by what EvalMod sees: the
    # slots with slim_boot (StC first), the polynomial coefficients otherwise.
    # Costs precision 1:1 but suppresses the sine's uncorrected cubic term, an
    # error of ~6.6 * m**3 / 2**(2 * log_delta). Slots peak at 12. Coefficients
    # are ~sqrt(2n) below the n slots, except the constant one, which carries
    # the slots' mean: a mean of 0.5 costs 2e-4 at 6, 1.3e-5 at 8. Above 8,
    # EvalMod's own error, summed over the coefficients by StC, takes over.
    'log_delta' : 8,
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
    
    def __init__(self, real_boot: bool = True, slim_boot: bool = False) -> None:
        """real_boot: the slots hold real values, so one EvalMod over the real part does
        (complex slots take two, on the real and imaginary parts). slim_boot: StC first,
        at the bottom of the chain, then ModRaise, CtS and EvalMod."""
        for name, value in (("real_boot", real_boot), ("slim_boot", slim_boot)):
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be a bool; got {value!r}.")
        self.real_boot = real_boot
        self.slim_boot = slim_boot
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
        if self.slim_boot:
            self.log_delta = BOOT_CONFIGURATION['log_delta_slim']
        else:
            self.log_delta = BOOT_CONFIGURATION['log_delta']
        self.eval_mod_q_list_precision = self.get_q_list_precision(self.eval_mod_scale, self.n_cosine_levels + self.n_double_angle + self.n_arcsine_levels)
        self.k = BOOT_CONFIGURATION['k']
        self.sparse_secret_hw = BOOT_CONFIGURATION['sparse_secret_hw']
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
