"""See `params/README.md` for usage details."""

import enum
import math

class BootstrappingVariant(enum.Enum):
    SLIM = 0
    COMPLEX = 1
    REAL = 2


BOOT_CONFIGURATION = {
    # ---------------------------------------------------------------- modulus
    # Prime sizes in bits and level counts of the three bootstrapping sections.
    'cts_scale' : 48,
    'n_cts_levels' : 3,
    # Size of each EvalMod prime, and of q0 (the ModRaise base modulus) -- one
    # knob, because they must be equal. "x mod 1" means dividing the mod-raised
    # value by q0, and a ciphertext is divided by relabelling its scale, so if
    # the scale EvalMod runs at *is* q0 the whole division by delta is free
    # bookkeeping. Split them and the shortfall q0/eval_mod_prime has to be
    # multiplied into the CtS matrices instead, which costs precision: the old
    # 45-vs-50 pairing put all of delta there and capped REAL around 16 bits.
    #
    # Raising it is not free for REAL: its trailing StC has to shrink the value
    # from the EvalMod scale down to the output scale, and h_dft.py spreads that
    # evenly over the StC levels, encoding each one's matrices
    # (eval_mod_scale - output_scale)/n_stc_levels bits coarser. Measured for
    # REAL at 55 against 50, before the 1/(2*pi) fold: -1.4 bits, recovered by
    # stc_scale 32. With the fold, 55 and stc_scale 32 gave 19.8 bits.
    'eval_mod_scale' : 50,
    'stc_scale' : 30,
    'n_stc_levels' : 3,

    # ---------------------------------------------------------------- EvalMod
    # One recipe for every variant, as in Lattigo: its SlotsToCoeffs-first
    # example and its standard N16QP1553H192H32 set use the same Mod1
    # parameters. A depth-5 Han-Ki cosine plus three double angles and no
    # arcsine, 5 + 3 = 8 EvalMod levels. The sine's 1/(2*pi) is folded into the
    # cosine and the double angles (see h_bootstrapping.py), so EvalMod outputs
    # ~t rather than ~2*pi*t. Measured at LogN=16, 2**15 slots, eval_mod_scale
    # 50, stc_scale 30, output scale 2**30 (mean / worst bits):
    #   REAL  this recipe, log_delta 6                   19.65 / 16.52
    #   SLIM  this recipe, log_delta 12                  21.07 / 17.74   8 levels
    #   SLIM  lsq degree 119, 1 double angle, arcsine    21.22 / 17.8   10 levels
    #   SLIM  this cosine and double angles, arcsine     21.21 / 17.84  10 levels
    # so SLIM gives up ~0.15 bits for two levels by dropping its arcsine.
    #
    # cosine_fit: 'hanki' (Han and Ki, eprint 2019/688; Lattigo's CosDiscrete)
    # or 'lsq' (cosine_cheb_coeffs, a least-squares fit over a dense point set).
    # At low degree Han-Ki is worth a level: at k=16, 3 double angles and
    # half-width 2**-7, over the |I| <= 8 a Hamming-weight-32 secret reaches,
    # lsq gives 2**-24.2 at degree 27 and 2**-44.4 at degree 59, Han-Ki 2**-47.9
    # at degree 30. It needs one node per cluster, 2k-1 of them.
    #
    # n_arcsine_levels > 0 turns the arcsine back on, and the 1/(2*pi) fold off:
    # the arcsine inverts the sine, so it needs it unscaled. Without it the
    # sine's (2*pi)**2 t**3 / 6 term stays uncorrected; log_delta is what keeps
    # that small -- see below.
    'cosine_fit' : 'hanki',
    'n_cosine_levels' : 5,
    'n_double_angle' : 3,
    'n_arcsine_levels' : 0,
    # Pins K instead of deriving it from sk_hw. None = derive. K must bound the
    # ModRaise wrap count |I|, whose std is sqrt((h+1)/12) for a ternary secret
    # of Hamming weight h, so it is only meaningful together with sk_hw. Exact
    # Irwin-Hall tails, summed over the N=2**16 coefficients of one ciphertext:
    #   h=192 K=25 (1.8*sqrt(h))   6.2 sigma   2**-15.6
    #   h=32  K=16 (Lattigo + SSE) 9.7 sigma   2**-138.7
    #   h=32  K=14                 8.4 sigma   2**-62.1
    #   h=192 K=16                 4.0 sigma   2**+2.0   <- ~4 wrapped coefs, garbage
    # Lattigo only reaches K=16 because sparse-secret encapsulation switches to
    # an H=32 ephemeral secret for the ModUp; pinning K=16 without that, at
    # sk_hw=192, corrupts every ciphertext.
    #
    # 14 is the largest K a depth-5 cosine can take: Han-Ki needs one node per
    # cluster, 2K-1 of them, and depth 5 gives 28 coefficients (degree 27).
    'k_override' : 14,

    # ------------------------------------------------------- per variant
    # The only per-variant knobs, because they describe *what EvalMod reduces*:
    # the slot values for SLIM, which runs SlotsToCoefs first, but the encoded
    # polynomial's coefficients for REAL and COMPLEX. COMPLEX runs the same
    # CtS-first circuit as REAL; nothing constructs it today (it is only the
    # DFT classes' default), so its values are unvalidated.
    #
    # message_bound_log2: log2 of the bound on those values. It sizes the
    # cosine's fit clusters and the arcsine's range when enabled; it does not
    # touch q_base. It is an assertion about the plaintexts, not something
    # derivable from the ring dimension. Exceeding it evaluates the cosine
    # outside its fit clusters, which with this recipe is not what limits
    # precision: a constant 0.9 input to REAL gives 9.7 bits at bound 2**-1 and
    # at 2**0 alike -- the limit there is the cubic term, see log_delta. With
    # this recipe REAL's 2**-1 buys nothing over 2**0 on random slots either
    # (19.62 vs 19.62 mean).
    #
    # For SLIM it is a hard bound on the slots. For REAL it is statistical: a
    # generic slot vector's coefficients are ~sqrt(N) smaller than its slots
    # (slots ~U(-1/2, 1/2) at N=2**16 give a coefficient std of 2**-9.8 and a
    # max of ~2**-7.6), but a *constant* slot vector encodes to a constant
    # polynomial whose coefficient equals the slot value. A caller that may feed
    # near-constant plaintexts to REAL needs SLIM's log_delta or the arcsine:
    # at log_delta 6 the cubic term caps such inputs at ~9.7 bits.
    'message_bound_log2' : {
        BootstrappingVariant.SLIM: 0,
        BootstrappingVariant.REAL: -1,
        BootstrappingVariant.COMPLEX: 0,
    },
    # log_delta: headroom ratio q_base/|m|, Lattigo's LogMessageRatio. It bounds
    # how much of q_base a message may occupy (1/2**log_delta); only past
    # q_base/2 does ModRaise wrap on the wrong multiple, silently. The cosine
    # only sees message_bound / 2**log_delta, the half-width of its fit clusters.
    #
    # The input is scaled *up* to q_base / 2**log_delta (2**38 for SLIM here) but
    # never down, so a ciphertext arriving above that scale gets a smaller
    # delta than configured, silently. SLIM measured at input scale 2**30:
    # 21.08, 2**40: 20.16 (delta 2**10), 2**44: 13.35 (delta 2**6).
    #
    # It costs precision 1:1 -- EvalMod's error is an additive floor at q0, and
    # the message sits 2**log_delta below it -- but without an arcsine it is also
    # what suppresses the cubic term, which falls as 1/delta**2 and only matters
    # when EvalMod sees full-size values. Hence the split:
    #   SLIM reduces the slots themselves. Mean bits by log_delta, this recipe:
    #     6 -> 13.4, 8 -> 17.0, 10 -> 19.9, 11 -> 20.9, 12 -> 21.07,
    #     13 -> 20.7, 14 -> 20.1, 16 -> 18.2.
    #   REAL's coefficients are ~sqrt(N) smaller, so the cubic term is already
    #   negligible at 6 and the 1:1 cost is all that is left: 3 -> 21.6,
    #   5 -> 19.6, 8 -> 16.6 (measured at output scale 2**40, before the fold).
    # Lattigo makes the same split: LogMessageRatio 10 in its SLIM example, 8
    # for N16QP1553H192H32.
    'log_delta' : {
        BootstrappingVariant.SLIM: 12,
        BootstrappingVariant.REAL: 6,
        BootstrappingVariant.COMPLEX: 6,
    },

    # --------------------------------------------------------------- numerics
    'subsum_max_stage_size' : 8,
    'baby_giant_ratio' : 2,
    # Two unrelated roles that used to share one threshold:
    #   tol      - structural. Detects zero leading terms in the Paterson-
    #              Stockmeyer long division, so it sets the *effective degree*
    #              and guards the divide. Belongs at the float64 noise floor.
    #   coef_tol - approximation. Zeroes Chebyshev coefficients below it, so it
    #              sets *how accurate the polynomial is*. Belongs at the
    #              plaintext encoding resolution, 0.5 / 2**eval_mod_scale
    #              (~2**-51), or None to keep every coefficient.
    # They were both 1e-7, which is ~25 bits coarser than the encoding can
    # represent: at K=25/1 double angle/degree 119 it zeroed the top 16 of 120
    # coefficients and cost 23 bits of EvalMod accuracy for no runtime saving.
    # Measured at cts53/evalmod50/7 cosine levels/1 double angle, LogN=15:
    #   tol 1e-7  -> 17.04 bits;  tol <= 1e-10 -> 20.11 bits (flat down to 0.0).
    # 1e-7 was trimming real content out of the long division, not just zeros.
    'tol' : 1e-12,
    'coef_tol' : None,
}

# Widest cosine fit cluster the EvalMod approximation has been validated at, as
# log2 of the half-width in integer units; message_bound_log2 - log_delta must
# not exceed it. The width sets the degree the cosine needs: for the earlier
# lsq recipe at K=25 with one double angle, degree 119 measured 2**-47 over
# 2**-6 clusters, degree 59 2**+20 and degree 49 diverged outright. The Han-Ki
# depth-5 recipe has run from 2**-16 to 2**-6 (the SLIM log_delta sweep)
# without the fit being the limit. Widening past this without re-validating
# the degree can destroy EvalMod silently.
COSINE_VALIDATED_HALF_WIDTH_LOG2 = -6

# Minimum K, in units of the ModRaise wrap-count std sqrt((h+1)/12), that an
# explicit k_override is allowed to use. 6.23 is what compute_k_from_sk_hw's
# 1.8*sqrt(h) yields, so this just holds overrides to the same margin.
MIN_K_SIGMA = 6.0


class BootstrappingParams:

    @staticmethod
    def compute_k_from_sk_hw(sk_hw: int) -> int:
        if sk_hw < 0:
            raise ValueError(f"sk_hw must be non-negative; got {sk_hw}.")
        return round(1.8 * sk_hw ** 0.5) 
    
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

    def _per_variant(self, key: str):
        """Read a BOOT_CONFIGURATION entry that is tuned per bootstrapping variant
        (message_bound_log2 and log_delta -- see their comments).

        Plain scalars are still accepted so a caller can pin one value for every
        variant without having to spell out the whole mapping.
        """
        value = BOOT_CONFIGURATION[key]
        if not isinstance(value, dict):
            return value
        if self.variant not in value:
            raise ValueError(
                f"BOOT_CONFIGURATION[{key!r}] has no entry for {self.variant}; "
                f"got keys {sorted(k.name for k in value)}.")
        return value[self.variant]

    @staticmethod
    def get_q_base_precision() -> int:
        """Size of q0, the ModRaise base modulus. Equal to the EvalMod prime size
        by construction -- see the 'eval_mod_scale' comment above."""
        return BOOT_CONFIGURATION['eval_mod_scale']
    
    def __init__(self, bootstrapping_variant: BootstrappingVariant = BootstrappingVariant.REAL, sk_hw: int = 0) -> None:

        self.variant = BootstrappingVariant(bootstrapping_variant)
        self.stc_scale = BOOT_CONFIGURATION['stc_scale']
        self.eval_mod_scale = BOOT_CONFIGURATION['eval_mod_scale']
        self.cts_scale = BOOT_CONFIGURATION['cts_scale']
        self.n_double_angle = BOOT_CONFIGURATION['n_double_angle']
        self.sk_hw = sk_hw
        self.subsum_max_stage_size = BOOT_CONFIGURATION['subsum_max_stage_size']
        self.baby_giant_ratio = BOOT_CONFIGURATION['baby_giant_ratio']
        self.arcsine_correction = BOOT_CONFIGURATION['n_arcsine_levels'] > 0
        self.tol = BOOT_CONFIGURATION['tol']
        self.coef_tol = BOOT_CONFIGURATION['coef_tol']

        self.n_stc_levels = BOOT_CONFIGURATION['n_stc_levels']
        self.n_cosine_levels = BOOT_CONFIGURATION['n_cosine_levels']
        self.n_arcsine_levels = BOOT_CONFIGURATION['n_arcsine_levels']
        self.n_cts_levels = BOOT_CONFIGURATION['n_cts_levels']
        self.cosine_fit = BOOT_CONFIGURATION['cosine_fit']
        self.stc_q_list_precision = self.get_q_list_precision(self.stc_scale, self.n_stc_levels)
        self.cts_q_list_precision = self.get_q_list_precision(self.cts_scale, self.n_cts_levels)
        self.log_delta = self._per_variant('log_delta')
        self.eval_mod_q_list_precision = self.get_q_list_precision(self.eval_mod_scale, self.n_cosine_levels + self.n_double_angle + self.n_arcsine_levels)
        self.n_cts_rows = len(self.cts_q_list_precision)
        self.n_stc_rows = len(self.stc_q_list_precision)
        self.n_evalmod_rows = len(self.eval_mod_q_list_precision)
        k_override = BOOT_CONFIGURATION['k_override']
        self.k = self.compute_k_from_sk_hw(self.sk_hw) if k_override is None else int(k_override)
        if k_override is not None and self.sk_hw > 0:
            # K bounds the ModRaise wrap count I, whose std is sqrt((h+1)/12).
            # compute_k_from_sk_hw's 1.8*sqrt(h) is 6.23 sigma, so hold an
            # explicit K to the same margin -- below it ModRaise wraps on the
            # wrong multiple of q and corrupts the plaintext with no error
            # raised anywhere. K=16 at sk_hw=192 is 3.99 sigma, which is ~4
            # corrupted coefficients in every 2**16, i.e. every ciphertext.
            sigma_i = math.sqrt((self.sk_hw + 1) / 12.0)
            if self.k < MIN_K_SIGMA * sigma_i:
                raise ValueError(
                    f"k_override={self.k} is only {self.k / sigma_i:.2f} sigma of the ModRaise "
                    f"wrap count at sk_hw={self.sk_hw} (sigma={sigma_i:.2f}); need at least "
                    f"{MIN_K_SIGMA} sigma, i.e. k >= {math.ceil(MIN_K_SIGMA * sigma_i)}. "
                    f"Lattigo reaches k=16 by switching to an H=32 ephemeral secret for the "
                    f"ModUp (sparse-secret encapsulation); without that, lower sk_hw to match "
                    f"or drop k_override.")
        self.q_list_precision = self.cts_q_list_precision +  self.eval_mod_q_list_precision + self.stc_q_list_precision
        # Bound on the values EvalMod reduces. It sizes the cosine's fit
        # clusters (and the arcsine's range when enabled), so it must hold for
        # every message that will be bootstrapped -- an overshoot makes ModRaise
        # wrap on the wrong multiple of q and corrupts the result outright.
        #
        # For REAL the bound applies to the encoded polynomial's coefficients,
        # not the slots. For a *generic* slot vector those are ~sqrt(n) smaller,
        # and lowering the bound accordingly buys ~log2(sqrt(n)) bits of
        # precision (EvalMod's error is an additive floor that does not shrink
        # with the input). But that relation is statistical, not a bound: a
        # constant slot vector encodes to a constant polynomial whose leading
        # coefficient equals the slot value. So this stays an explicit assertion
        # by the caller about its own plaintexts, defaulting to the safe 1/2.
        self.message_bound_log2 = self._per_variant('message_bound_log2')
        self.message_bound = 2.0 ** self.message_bound_log2
        if not isinstance(self.message_bound_log2, int):
            raise ValueError(
                f"message_bound_log2 must be an int; got {self.message_bound_log2!r}.")
        # No independent cap on message_bound itself: what EvalMod needs is
        # |t| = message_bound/delta < 1/2 so that "x mod 1" picks the right
        # integer, and that is the half-width check below (which is far
        # stricter). Capping message_bound at 1/2 on its own was rejecting
        # honest configurations -- SLIM reduces the slots, so a caller with
        # slots in [-1, 1] must declare 1, and pay for it in log_delta.
        if not isinstance(self.log_delta, int) or self.log_delta < 1:
            raise ValueError(f"log_delta must be an int >= 1; got {self.log_delta!r}.")
        # The pair is only as good as its ratio: see the comments on the two keys.
        self.cluster_half_width_log2 = self.message_bound_log2 - self.log_delta
        if self.cluster_half_width_log2 > COSINE_VALIDATED_HALF_WIDTH_LOG2:
            raise ValueError(
                f"message_bound_log2 - log_delta = {self.cluster_half_width_log2} for "
                f"{self.variant.name}, which fits the cosine over clusters of half-width "
                f"2**{self.cluster_half_width_log2}; the approximation is only validated up "
                f"to 2**{COSINE_VALIDATED_HALF_WIDTH_LOG2}. Raise log_delta (1 bit of "
                f"precision per bit) or lower message_bound_log2.")
        self.q_base_precision = self.get_q_base_precision()
        self.modulus = sum(row[0] for row in self.q_list_precision) + self.q_base_precision
        self.cosine_deg = self.optimal_poly_deg_per_depth(self.n_cosine_levels)
        self.arcsine_deg = self.optimal_poly_deg_per_depth(self.n_arcsine_levels)
        