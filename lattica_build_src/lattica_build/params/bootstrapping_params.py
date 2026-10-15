"""See `params/README.md` for usage details."""

BOOT_CONFIGURATION = {
    # EvalMod and StC prime sizes (bits); the CtS ones and the DFT level counts come
    # per subring from BOOT_PARAMS_PER_SUBRING.
    # Also the size of q0: EvalMod running at scale q0 makes the division by
    # delta a free scale relabel. 53 is the smallest without losing accuracy
    # (at 2^12 slots, log_delta 9: 52 errs ~5% more, 50 ~1.7x; same time).
    'eval_mod_scale' : 53,
    'stc_scale' : 30,

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

    # SLIM's headroom q0/|m| (its EvalMod sees the slots); the others take theirs from
    # BOOT_PARAMS_PER_SUBRING.
    'log_delta_slim' : 12,

    # SubSum stage size cap and the DFTs' baby-step/giant-step ratio. 4 puts the 15-16
    # diagonal factors on 8x2 grids (one giant step) rather than 4x4 (three).
    'subsum_max_stage_size' : 8,
    'baby_giant_ratio' : 4,
}

# Per subring dimension log2(2 * n_slots): the StC split of the log2(n_slots) FFT stages
# (component sizes, stage 0 first; CtS uses the same components in reverse order, so both
# share their rotation keys) and the smallest CtS prime, in bits, that keeps the bootstrap
# at its error floor with that split. A component of k stages has 2**(k+1) - 1 diagonals,
# 2**k if it holds the top stage, and costs about its giant steps; the splits are the
# fastest measured (RTX 5090, logN 16, REAL, ratio 4). One level needs a wider CtS prime:
# its single factor takes the whole CtS normalization. Last, log_delta, the headroom q0/|m|:
# each bit divides the sine's cubic error (~6.6 m**3 / 4**log_delta, m: for REAL the slots'
# mean) by 4 but doubles EvalMod's noise, and fewer slots leave StC less noise to sum, so they
# afford more (measured at 2**15, 2**14, 2**13, 2**12, 2**10, 2**8, 2**6 and 2**4 slots, the
# others in between). An input with only q0 left must be at most 2 * q0 / 2**log_delta.
BOOT_PARAMS_PER_SUBRING = {
    2:  ((1,), 48, 10),
    3:  ((2,), 49, 10),
    4:  ((3,), 50, 10),
    5:  ((4,), 54, 10),
    6:  ((5,), 55, 9),
    7:  ((3, 3), 44, 9),
    8:  ((3, 4), 45, 9),
    9:  ((3, 5), 46, 9),
    10: ((4, 5), 47, 9),
    11: ((3, 3, 4), 45, 9),
    12: ((3, 3, 5), 46, 9),
    13: ((3, 3, 3, 3), 45, 9),
    14: ((3, 3, 3, 4), 46, 9),
    15: ((3, 3, 3, 5), 47, 8),
    16: ((3, 3, 4, 5), 47, 8),
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
    
    @classmethod
    def for_slot_counts(cls, real_boot: bool, slim_boot: bool, slot_counts) -> "BootstrappingParams":
        """The chain for bootstraps of these slot counts (one per BOOT_PARAMS_PER_SUBRING entry): as many
        CtS and StC levels as the deepest split, and CtS primes as wide as the widest need.
        A bootstrap with a shallower split skips the levels it does not use."""
        entries = []
        for n_slots in slot_counts:
            log_n_subring = (2 * n_slots).bit_length() - 1
            if log_n_subring not in BOOT_PARAMS_PER_SUBRING:
                raise ValueError(f"no bootstrap parameters for {n_slots} slots (subring 2**{log_n_subring})")
            entries.append(BOOT_PARAMS_PER_SUBRING[log_n_subring])
        if not entries:
            raise ValueError("slot_counts must not be empty")
        return cls(real_boot, slim_boot,
                   dft_levels=max(len(split) for split, _, _ in entries),
                   cts_scale=max(bits for _, bits, _ in entries))

    def __init__(self, real_boot: bool, slim_boot: bool, dft_levels: int, cts_scale: int) -> None:
        """real_boot: the slots hold real values, so one EvalMod over the real part does
        (complex slots take two, on the real and imaginary parts). slim_boot: StC first,
        at the bottom of the chain, then ModRaise, CtS and EvalMod. dft_levels, cts_scale:
        the CtS and StC level count and the CtS prime size; for_slot_counts picks them."""
        for name, value in (("real_boot", real_boot), ("slim_boot", slim_boot)):
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be a bool; got {value!r}.")
        if not isinstance(dft_levels, int) or dft_levels < 1:
            raise ValueError(f"dft_levels must be an int >= 1; got {dft_levels!r}.")
        self.real_boot = real_boot
        self.slim_boot = slim_boot
        self.stc_scale = BOOT_CONFIGURATION['stc_scale']
        self.eval_mod_scale = BOOT_CONFIGURATION['eval_mod_scale']
        self.cts_scale = cts_scale
        self.n_double_angle = BOOT_CONFIGURATION['n_double_angle']
        self.subsum_max_stage_size = BOOT_CONFIGURATION['subsum_max_stage_size']
        self.baby_giant_ratio = BOOT_CONFIGURATION['baby_giant_ratio']
        self.arcsine_correction = BOOT_CONFIGURATION['n_arcsine_levels'] > 0

        self.dft_levels = dft_levels
        self.n_stc_levels = dft_levels
        self.n_cosine_levels = BOOT_CONFIGURATION['n_cosine_levels']
        self.n_arcsine_levels = BOOT_CONFIGURATION['n_arcsine_levels']
        self.n_cts_levels = dft_levels
        self.stc_q_list_precision = self.get_q_list_precision(self.stc_scale, self.n_stc_levels)
        self.cts_q_list_precision = self.get_q_list_precision(self.cts_scale, self.n_cts_levels)
        # SLIM's headroom; the other variants' comes per subring from BOOT_PARAMS_PER_SUBRING
        self.log_delta = BOOT_CONFIGURATION['log_delta_slim'] if self.slim_boot else None
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
