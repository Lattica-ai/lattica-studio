"""See `params/README.md` for usage details."""

from dataclasses import dataclass

BOOT_CONFIGURATION = {
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

    # SubSum stage size cap and the DFTs' baby-step/giant-step ratio. On an RTX 5090, 8 is
    # 2.8% faster than 4 at 2**15 slots and 0.7% at 2**14 for 0.66 and 0.33 GiB more peak
    # memory, the same at 2**13 and 2**10, and 5% slower at 2**12.
    'subsum_max_stage_size' : 8,
    'baby_giant_ratio' : 8,
}

# Per number of slots, by log2(n_slots): the StC split of the log2(n_slots) FFT stages
# (component sizes, stage 0 first; CtS uses the same components in reverse order, so both
# share their rotation keys) and the smallest CtS prime, in bits, that keeps the bootstrap
# at its error floor with that split. One level needs a wider CtS prime:
# its single factor takes the whole CtS normalization. Last, log_delta, the headroom q0/|m|:
# each bit divides the sine's cubic error (~6.6 m**3 / 4**log_delta, m: a coefficient EvalMod
# sees; for REAL the slots' mean lands on the constant one) by 4 but doubles EvalMod's noise,
# which StC sums over the coefficients. Fewer slots leave less noise to sum and give larger
# coefficients, so they take more. Tuned per slot count on zero-mean amplitude-1 data; from 2**12 slots up, together
# with the exit below, on the level-conserving chain at output scale 2**30.
# An input with only q0 left must be at most 2 * q0 / 2**log_delta.
# Last, on a level-conserving chain (lcr_first_cts), EvalMod's exit scale in bits. Double angles 1 and 3 share one
# word there (LCR_EVAL_MOD_ROWS), so the double angles divide the scale by less than their squarings multiply it, and
# the cosine is aimed so that EvalMod ends at 2**exit. The StC then brings exit - log_delta bits down to the output
# scale: a higher exit keeps more precision in the double angles and leaves fewer bits to the StC's plaintexts.
# Tuned per slot count at output scale 2**30, between the best REAL and the best COMPLEX exit. None: the slot count
# has no such chain.
BOOT_PARAMS_PER_N_SLOTS = {
    5:  ((5,), 55, 9, None),
    6:  ((3, 3), 44, 8, None),
    7:  ((3, 4), 45, 8, None),
    8:  ((3, 5), 46, 7, None),
    9:  ((4, 5), 47, 7, None),
    10: ((3, 3, 4), 45, 7, None),
    11: ((3, 3, 5), 46, 6, None),
    12: ((3, 3, 3, 3), 45, 5, 64),
    13: ((3, 3, 3, 4), 46, 5, 63),
    14: ((3, 3, 4, 4), 47, 4, 62),
    15: ((3, 4, 4, 4), 47, 3, 60),
}


# The most terms (key products) a level-conserving first CtS level takes. The level runs as one kernel that keeps each
# term's source indices in 48 KB of shared memory, 256 bytes per term. In a sparse bootstrap the level also computes
# the SubSum trace, so its terms are the first CtS factor's diagonals times the N / (2 n_slots) trace rotations.
LCR_MAX_TERMS = 192
# The most bits of log QP any key may have for 128-bit security under the secret key (hamming weight 1024), by ring
# dimension. 2**15: the lattice estimator crosses 128 bits at 866 (cheapest attack bdd; the sparse-secret hybrids are
# about 2 bits costlier at this weight), less 2 bits of margin. 2**14 and 2**16 keep the hamming-weight-192 limits,
# which are below the hamming-weight-1024 ones.
MAX_LOG_QP_128 = {2 ** 14: 402, 2 ** 15: 864, 2 ** 16: 1638}
SPECIAL_PRIME_BITS = 61
# A level-conserving chain's EvalMod rows, one word fewer than one per level: the cosine's rows, double angle 2's
# row, then one word that double angles 1 and 3 share (a 38-bit and a 23-bit prime), last so that every row dropped
# whole comes before the ones still active. A double angle's precision is the scale it keeps after its rescale, and
# each later double angle doubles its error. Double angle 3's error is doubled by none, so it takes the small prime;
# double angle 2 then keeps the scale (exit + 23) / 2, and double angle 1 gets its scale back from double angle 2's
# 61-bit prime. LCR_DOUBLE_ANGLE_ROWS: the EvalMod row each double angle drops a prime of.
LCR_EVAL_MOD_ROWS = ((53,),) * 5 + ((61,), (61, 23))
LCR_DOUBLE_ANGLE_ROWS = (6, 5, 6)
# A level-conserving chain's StC scale: its words hold two primes just below 2**31 (62-bit rows, the kernels' limit)
# rather than 2**30, so the StC has room for the bits the higher EvalMod exit leaves it.
LCR_STC_SCALE = 31


@dataclass(frozen=True)
class BootPrecision:
    """How precise a bootstrap is, and the chain it needs for that.
    eval_mod_scale: the EvalMod prime size in bits, which is also q0's: EvalMod running at scale q0 makes the division
    by delta a free scale relabel. stc_scale: the StC prime size (a level-conserving chain with exit scales uses
    LCR_STC_SCALE). per_n_slots: by log2(n_slots), the split, the CtS prime size, log_delta and the exit scale
    (BOOT_PARAMS_PER_N_SLOTS explains them); the profile bootstraps only these slot counts. lcr_min_special_bits: the
    level-conserving CtS keys need special primes P' of more than this many bits in all, and the backend gives them the
    fewest that reach it. Their key noise, about T N |m| q0 e / P' for T terms of plaintexts m at the CtS scale, must
    stay below the bootstrap's own error."""
    eval_mod_scale: int
    stc_scale: int
    per_n_slots: dict[int, tuple[tuple[int, ...], int, int, int | None]]
    lcr_min_special_bits: int


BOOT_PRECISIONS = {
    # The general-purpose bootstrap, about 18 to 21 bits. q0 = 53 bits is the smallest without accuracy loss. The
    # level-conserving keys' noise at N = 2**16, q0 = 2**53, 192 terms and plaintexts up to 2**60 is about 2**-42 with
    # three 61-bit special primes, well below the ModDown's rounding, but about 2**19 with two.
    "full": BootPrecision(eval_mod_scale=53, stc_scale=30, per_n_slots=BOOT_PARAMS_PER_N_SLOTS,
                          lcr_min_special_bits=170),
    # About 10 bits, for computations that tolerate it. Its chain is small enough for N = 2**15 at 128 bits, with one
    # DFT level fewer than "full" on each side. The level-conserving keys' noise is about 2**-24 with two special
    # primes, far below this bootstrap's error. Inputs need |x| <= 0.75 and a slot mean of at most about 0.1: with
    # log_delta 2, the sine's cubic error on the largest coefficient EvalMod sees, the mean, is about 2**-10, and a
    # larger mean loses precision. It has no exit scales, so its chain keeps one EvalMod row per level.
    "low": BootPrecision(eval_mod_scale=36, stc_scale=26, per_n_slots={
        12: ((4, 4, 4), 34, 2, None),
        13: ((4, 4, 5), 34, 2, None),
        14: ((4, 5, 5), 34, 2, None),
    }, lcr_min_special_bits=100),
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

    @classmethod
    def for_slot_counts(cls, real_boot: bool, slim_boot: bool, slot_counts, n: int | None = None,
                        special_bits: int = 0, precision: str = "full") -> "BootstrappingParams":
        """The chain for bootstraps of these slot counts (each needs an entry in the precision's per_n_slots): as
        many CtS and StC levels as the deepest split, and CtS primes as wide as the widest need.
        A bootstrap with a shallower split skips the levels it does not use.
        The chain gets one CtS prime fewer, and lcr_first_cts is set, when every bootstrap can run its first CtS level
        level-conserving. That needs the ring dimension n, special primes of more than the precision's
        lcr_min_special_bits in all (special_bits, which is 0 under BV), and a level that fits the kernel for every
        slot count (lcr_first_cts_fits)."""
        slot_counts = list(slot_counts)
        entries = [cls._n_slots_entry(n_slots, precision) for n_slots in slot_counts]
        dft_levels = max(len(entry[0]) for entry in entries)
        cts_scale = max(entry[1] for entry in entries)
        params = cls(real_boot, slim_boot, dft_levels=dft_levels, cts_scale=cts_scale, precision=precision)
        if (n is None or special_bits <= params.lcr_min_special_bits
                or not all(params.lcr_first_cts_fits(n_slots, n) for n_slots in slot_counts)):
            return params
        return cls(real_boot, slim_boot, dft_levels=dft_levels, cts_scale=cts_scale, lcr_first_cts=True,
                   precision=precision)

    def lcr_first_cts_fits(self, n_slots: int, n: int) -> bool:
        """Returns whether an n_slots bootstrap in a ring of dimension n can run its first CtS level level-conserving.
        Its DFT split must have another CtS level after that one, and the level's terms must fit the kernel
        (LCR_MAX_TERMS). The first CtS factor combines the last s stages of the split, so it has 2**s diagonals. In a
        sparse bootstrap the level uses each diagonal once per rotation of the SubSum trace, and there are
        N / (2 n_slots) of those."""
        split = self.dft_split(n_slots)
        return len(split) > 1 and 2 ** split[-1] * (n // 2 // n_slots) <= LCR_MAX_TERMS

    def one_digit_relin_key(self, chain_sections, num_special_primes: int, n: int) -> tuple[int, int] | None:
        """The one-digit relinearization key of EvalMod's lower levels, as (e, s): the relinearizations after e of
        EvalMod's rescales switch with one digit and s special primes, rather than with the square key's two or more
        digits, which read the key once per digit and add a ModUp. The key holds the q rows left at level e and the
        fewest special primes with more bits than those rows, so its key switch noise, about N * sigma * Q / P, is a
        few bits that the rescale after the relinearization divides away. e is the first level where the rows and
        primes fit MAX_LOG_QP_128. The chain's special primes past num_special_primes are only this key's.
        chain_sections: the chain's ciphertext sections (chain_sections). None if no level both fits and needs
        more than one digit of the square key, or if n has no bound."""
        if n not in MAX_LOG_QP_128 or num_special_primes == 0:
            return None
        names = [name for name, _ in chain_sections]
        first_below = names.index("boot_eval") + 1
        eval_rows = chain_sections[first_below - 1][1]
        below_bits = sum(row[0] for _, q_list in chain_sections[first_below:] for row in q_list)
        for e in range(len(eval_rows)):
            q_bits = below_bits + sum(row[0] for row in eval_rows[e:])
            special = q_bits // SPECIAL_PRIME_BITS + 1
            if special <= num_special_primes:
                return None   # the square key switches this level, and the ones after it, in one digit already
            if q_bits + special * SPECIAL_PRIME_BITS <= MAX_LOG_QP_128[n]:
                return e, special
        return None

    @staticmethod
    def _n_slots_entry(n_slots: int, precision: str):
        log_n_slots = n_slots.bit_length() - 1
        per_n_slots = BOOT_PRECISIONS[precision].per_n_slots
        if log_n_slots not in per_n_slots:
            raise ValueError(f"no {precision!r}-precision bootstrap parameters for {n_slots} slots")
        return per_n_slots[log_n_slots]

    def dft_split(self, n_slots: int) -> tuple[int, ...]:
        """The StC split (component sizes, stage 0 first; CtS takes it reversed) of a bootstrap of n_slots slots.
        Its precision's split, which the chain must fit: as many levels and CtS primes as wide (for_slot_counts
        with n_slots among its slot counts does). SLIM: equal parts over all the chain's levels, since its StC runs on
        the input's own primes."""
        n_levels = self.dft_levels
        if not self.slim_boot:
            split, cts_bits = self._n_slots_entry(n_slots, self.precision)[:2]
            if len(split) > n_levels or cts_bits > self.cts_scale:
                # fewer levels than the split would mean far more diagonals per DFT factor, and a CtS prime
                # narrower than the split needs loses precision
                raise ValueError(
                    f"the bootstrap chain ({n_levels} DFT levels, {self.cts_scale}-bit CtS primes) does not fit the "
                    f"{n_slots}-slot bootstrap's split {split} ({cts_bits}-bit CtS primes): build it with "
                    f"for_slot_counts including {n_slots}")
            return tuple(split)
        log_n_slots = n_slots.bit_length() - 1
        if not 1 <= n_levels <= log_n_slots:
            raise ValueError(
                f"bootstrapping {n_slots} slots needs log2(n_slots) >= the {self.dft_levels} DFT levels")
        q, r = divmod(log_n_slots, n_levels)
        return (q + 1,) * r + (q,) * (n_levels - r)

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
        return self._n_slots_entry(n_slots, self.precision)[2]

    def exit_scale_for(self, n_slots: int) -> int | None:
        """EvalMod's exit scale, in bits, of a bootstrap of n_slots slots on a level-conserving chain, whose double
        angles 1 and 3 share a word (LCR_EVAL_MOD_ROWS); None on the other chains, where EvalMod ends at its own
        scale."""
        if self.double_angle_rows is None:
            return None
        return self._n_slots_entry(n_slots, self.precision)[3]

    def __init__(self, real_boot: bool, slim_boot: bool, dft_levels: int, cts_scale: int,
                 lcr_first_cts: bool = False, precision: str = "full") -> None:
        """real_boot: the slots hold real values, so one EvalMod over the real part does
        (complex slots take two, on the real and imaginary parts). slim_boot: StC first,
        at the bottom of the chain, then ModRaise, CtS and EvalMod. dft_levels, cts_scale:
        the CtS and StC level count and the CtS prime size; for_slot_counts picks them.
        lcr_first_cts: the first CtS level is level-conserving, so it consumes no prime and the chain has one CtS
        prime fewer; for_slot_counts picks it. A one-level split keeps its prime: its single factor has every
        diagonal. precision: the BOOT_PRECISIONS profile."""
        for name, value in (("dft_levels", dft_levels), ("cts_scale", cts_scale)):
            if not isinstance(value, int) or value < 1:
                raise ValueError(
                    f"the bootstrap chain needs {name}, an int >= 1; got {value!r}. A pipeline's serialize sets it "
                    f"(HomParams.to_dict records it); for params built directly, for_slot_counts with the slot counts "
                    f"they bootstrap gives it.")
        self.real_boot = real_boot
        self.slim_boot = slim_boot
        self.precision = precision
        profile = BOOT_PRECISIONS[precision]
        self.stc_scale = profile.stc_scale
        self.eval_mod_scale = profile.eval_mod_scale
        self.lcr_min_special_bits = profile.lcr_min_special_bits
        self.cts_scale = cts_scale
        self.n_double_angle = BOOT_CONFIGURATION['n_double_angle']
        self.subsum_max_stage_size = BOOT_CONFIGURATION['subsum_max_stage_size']
        self.baby_giant_ratio = BOOT_CONFIGURATION['baby_giant_ratio']
        self.arcsine_correction = BOOT_CONFIGURATION['n_arcsine_levels'] > 0

        self.dft_levels = dft_levels
        self.lcr_first_cts = lcr_first_cts and dft_levels > 1
        self.n_cosine_levels = BOOT_CONFIGURATION['n_cosine_levels']
        self.n_arcsine_levels = BOOT_CONFIGURATION['n_arcsine_levels']
        self.cts_q_list_precision = self.get_q_list_precision(self.cts_scale, self.dft_levels - self.lcr_first_cts)
        # The shared double-angle word needs every slot count's exit scale. A chain's DFT level count is its slot
        # counts' split length, and only "full"'s 4-level splits (2**12 to 2**15 slots) have exits: the other chains
        # keep one EvalMod row per level.
        if self.lcr_first_cts and not self.slim_boot and all(
                entry[3] is not None for entry in profile.per_n_slots.values() if len(entry[0]) == dft_levels):
            self.stc_scale = LCR_STC_SCALE
            self.eval_mod_q_list_precision = LCR_EVAL_MOD_ROWS
            self.double_angle_rows = LCR_DOUBLE_ANGLE_ROWS
        else:
            self.eval_mod_q_list_precision = self.get_q_list_precision(self.eval_mod_scale, self.n_cosine_levels + self.n_double_angle + self.n_arcsine_levels)
            self.double_angle_rows = None
        self.stc_q_list_precision = self.get_q_list_precision(self.stc_scale, self.dft_levels)
        self.k = BOOT_CONFIGURATION['k']
        self.sparse_secret_hw = BOOT_CONFIGURATION['sparse_secret_hw']
        self.q_list_precision = self.cts_q_list_precision +  self.eval_mod_q_list_precision + self.stc_q_list_precision
        # q0, the ModRaise base modulus, has the EvalMod prime size
        self.q_base_precision = self.eval_mod_scale
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
