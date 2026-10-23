"""See `params/README.md` for usage details."""


BOOT_CONFIGURATION = {
    # EvalMod and StC prime sizes (bits); the CtS ones and the DFT level counts come
    # per slot count from BOOT_PARAMS_PER_N_SLOTS.
    # Also the size of q0: EvalMod running at scale q0 makes the division by
    # delta a free scale relabel; 53 is the smallest without accuracy loss.
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
    # Bound on the ModRaise wrap count |I|. With the H=32 ephemeral secret, K=16 fails with
    # probability 2^-138.7 for 2^15 slots (eprint 2022/024). Han-Ki's cosine
    # needs degree 2K-2 = 30, the largest even degree in 5 levels.
    'k' : 16,

    # SLIM's headroom q0/|m| (its EvalMod sees the slots, and no StC sums its noise): 12 is the best at
    # every slot count measured (2**15 to 2**9); the others take theirs from BOOT_PARAMS_PER_N_SLOTS.
    'log_delta_slim' : 12,

    # SubSum stage size cap and the DFTs' baby-step/giant-step ratio. 4 puts the 15-16
    # diagonal factors on 8x2 grids (one giant step) rather than 4x4 (three).
    'subsum_max_stage_size' : 8,
    'baby_giant_ratio' : 4,
}

# Per number of slots, by log2(n_slots): the StC split of the log2(n_slots) FFT stages
# (component sizes, stage 0 first; CtS uses the same components in reverse order, so both
# share their rotation keys) and the smallest CtS prime, in bits, that keeps the bootstrap
# at its error floor with that split. One level needs a wider CtS prime:
# its single factor takes the whole CtS normalization. Last, log_delta, the headroom q0/|m|:
# each bit divides the sine's cubic error (~6.6 m**3 / 4**log_delta, m: a coefficient EvalMod
# sees; for REAL the slots' mean lands on the constant one) by 4 but doubles EvalMod's noise,
# which StC sums over the coefficients. Fewer slots leave less noise to sum and give larger
# coefficients, so they take more. Tuned per slot count on zero-mean amplitude-1 data. 
# An input with only q0 left must be at most 2 * q0 / 2**log_delta.
BOOT_PARAMS_PER_N_SLOTS = {
    5:  ((5,), 55, 9),
    6:  ((3, 3), 44, 8),
    7:  ((3, 4), 45, 8),
    8:  ((3, 5), 46, 7),
    9:  ((4, 5), 47, 7),
    10: ((3, 3, 4), 45, 7),
    11: ((3, 3, 5), 46, 6),
    12: ((3, 3, 3, 3), 45, 6),
    13: ((3, 3, 3, 4), 46, 5),
    14: ((3, 3, 3, 5), 47, 5),
    15: ((3, 3, 4, 5), 47, 4),
}


# The most terms (key products) a level-conserving first CtS level takes: the toolkit's lcr_rotations_sum kernel stages
# each term's source indices in 48 KB of shared memory (256 bytes per term). A sparse bootstrap's level folds the
# SubSum trace in, so its terms are the first CtS factor's diagonals times the N / (2 n_slots) trace rotations.
LCR_MAX_TERMS = 192


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
    
    @classmethod
    def for_slot_counts(cls, real_boot: bool, slim_boot: bool, slot_counts, n: int | None = None,
                        lcr_first_cts: bool = True) -> "BootstrappingParams":
        """The chain for bootstraps of these slot counts (each needs a BOOT_PARAMS_PER_N_SLOTS entry): as
        many CtS and StC levels as the deepest split, and CtS primes as wide as the widest need.
        A bootstrap with a shallower split skips the levels it does not use.
        n: the ring dimension. The chain has one CtS prime fewer when every slot count's bootstrap can run its first
        CtS level level-conserving (lcr_first_cts_fits, which needs n); lcr_first_cts=False keeps it regular."""
        slot_counts = list(slot_counts)
        entries = [cls._n_slots_entry(n_slots) for n_slots in slot_counts]
        dft_levels = max(len(split) for split, _, _ in entries)
        cts_scale = max(bits for _, bits, _ in entries)
        lcr = False
        if lcr_first_cts and n is not None:
            probe = cls(real_boot, slim_boot, dft_levels=dft_levels, cts_scale=cts_scale)
            lcr = all(probe.lcr_first_cts_fits(n_slots, n) for n_slots in slot_counts)
        return cls(real_boot, slim_boot, dft_levels=dft_levels, cts_scale=cts_scale, lcr_first_cts=lcr)

    def lcr_first_cts_terms(self, n_slots: int, n: int) -> int:
        """The terms of the level-conserving first CtS level of a bootstrap of n_slots slots in a ring of dimension n,
        at most: the first CtS factor collapses the last split component's s stages, whose diagonals are the 2**s
        multiples of n_slots / 2**s (as many on a complex bootstrap's repacked factors), and a sparse bootstrap's
        level sums each over the N / (2 n_slots) rotations of the SubSum trace."""
        return 2 ** self.dft_split(n_slots)[-1] * (n // 2 // n_slots)

    def lcr_first_cts_fits(self, n_slots: int, n: int) -> bool:
        """Whether that level fits the toolkit's kernel (LCR_MAX_TERMS), with another CtS level after it: 2**12 slots
        and up at N = 2**16, at most 8 trace rotations. Where it fits, it is also faster: RTX 5090, complex, 2 to 8
        trace rotations -3..-8%; 16 (a 2**11-slot split with a 3-stage last component) was +8%."""
        return len(self.dft_split(n_slots)) > 1 and self.lcr_first_cts_terms(n_slots, n) <= LCR_MAX_TERMS

    @staticmethod
    def _n_slots_entry(n_slots: int):
        log_n_slots = n_slots.bit_length() - 1
        if log_n_slots not in BOOT_PARAMS_PER_N_SLOTS:
            raise ValueError(f"no bootstrap parameters for {n_slots} slots")
        return BOOT_PARAMS_PER_N_SLOTS[log_n_slots]

    def dft_split(self, n_slots: int) -> tuple[int, ...]:
        """The StC split (component sizes, stage 0 first; CtS takes it reversed) of a bootstrap of n_slots slots.
        BOOT_PARAMS_PER_N_SLOTS's split, which the chain must fit: as many levels and CtS primes as wide (for_slot_counts
        with n_slots among its slot counts does). SLIM takes the same split: its StC and CtS are the same transforms,
        at other levels."""
        n_levels = self.dft_levels
        split, cts_bits, _ = self._n_slots_entry(n_slots)
        if len(split) > n_levels or cts_bits > self.cts_scale:
            # fewer levels than the split would mean far more diagonals per DFT factor, and a CtS prime
            # narrower than the split needs loses precision
            raise ValueError(
                f"the bootstrap chain ({n_levels} DFT levels, {self.cts_scale}-bit CtS primes) does not fit the "
                f"{n_slots}-slot bootstrap's split {split} ({cts_bits}-bit CtS primes): build it with "
                f"for_slot_counts including {n_slots}")
        return tuple(split)

    def chain_sections(self, pipeline_q_list_precision) -> list[tuple[str, tuple]]:
        """The chain's ciphertext sections, (name, q_list_precision), in row order.

        A ciphertext's limbs are its active rows in row order, and a rescale drops the first active row of its rows
        budget. So each section comes before every row still active while it is consumed: every rescale then
        drops the leading limb, and a level drop is a slice view rather than a gather. After ModRaise the
        bootstrap consumes its CtS rows, then its EvalMod rows, then (StC last) its StC rows; the pipeline's are
        consumed after that, and q0 stays last. SLIM's StC runs on the input, after the pipeline's rows."""
        pipeline = ("pipeline", pipeline_q_list_precision)
        stc = ("boot_stc", self.stc_q_list_precision)
        consumed = [pipeline, stc] if self.slim_boot else [stc, pipeline]
        return ([("boot_cts", self.cts_q_list_precision), ("boot_eval", self.eval_mod_q_list_precision)]
                + consumed + [("boot_base", ((self.q_base_precision,),))])

    def log_delta_for(self, n_slots: int) -> int:
        """The headroom q0/|m| of a bootstrap of n_slots slots: SLIM's own (its EvalMod sees the slots),
        else its slot count's (fewer slots take more)."""
        if self.slim_boot:
            return BOOT_CONFIGURATION['log_delta_slim']
        return self._n_slots_entry(n_slots)[2]

    def __init__(self, real_boot: bool, slim_boot: bool, dft_levels: int, cts_scale: int,
                 lcr_first_cts: bool = False) -> None:
        """real_boot: the slots hold real values, so one EvalMod over the real part does
        (complex slots take two, on the real and imaginary parts). slim_boot: StC first,
        at the bottom of the chain, then ModRaise, CtS and EvalMod. dft_levels, cts_scale:
        the CtS and StC level count and the CtS prime size; for_slot_counts picks them.
        lcr_first_cts: one CtS prime fewer, for bootstraps whose first CtS level is level-conserving; for_slot_counts
        picks it."""
        for name, value in (("dft_levels", dft_levels), ("cts_scale", cts_scale)):
            if not isinstance(value, int) or value < 1:
                raise ValueError(
                    f"the bootstrap chain needs {name}, an int >= 1; got {value!r}. A pipeline's serialize sets it "
                    f"(HomParams.to_dict records it); for params built directly, for_slot_counts with the slot counts "
                    f"they bootstrap gives it.")
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
        # The level-conserving first CtS level (toolkit BackendHomLcrMatMul) consumes no prime, so the chain carries one
        # CtS prime fewer. Only for pipelines whose bootstraps all have it (for_slot_counts).
        # (a one-level split keeps its CtS prime: its single factor has every diagonal)
        self.lcr_first_cts = bool(lcr_first_cts) and dft_levels > 1
        self.n_cosine_levels = BOOT_CONFIGURATION['n_cosine_levels']
        self.n_arcsine_levels = BOOT_CONFIGURATION['n_arcsine_levels']
        self.stc_q_list_precision = self.get_q_list_precision(self.stc_scale, self.dft_levels)
        self.cts_q_list_precision = self.get_q_list_precision(self.cts_scale, self.dft_levels - int(self.lcr_first_cts))
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
