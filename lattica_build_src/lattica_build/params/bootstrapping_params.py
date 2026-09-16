"""See `params/README.md` for usage details."""

import enum
import math

class BootstrappingVariant(enum.Enum):
    SLIM = 0
    COMPLEX = 1
    REAL = 2


BOOT_CONFIGURATION = {
    'cts_scale' : 48,
    # Size of each EvalMod prime, and of q0 (the ModRaise base modulus) -- one
    # knob, because they must be equal. Same units as cts_scale/stc_scale: bits. "x mod 1" means dividing the mod-raised
    # value by q0, and a ciphertext is divided by relabelling its scale, so if
    # the scale EvalMod runs at *is* q0 the whole division by delta is free
    # bookkeeping. Split them and the shortfall q0/eval_mod_prime has to be
    # multiplied into the CtS matrices instead, which costs precision: the old
    # 45-vs-50 pairing put all of delta there and capped REAL around 16 bits.
    'eval_mod_scale' : 50,
    'stc_scale' : 30,
    'n_stc_levels' : 3,
    # 7 cosine levels (degree 119) with a single double angle keeps the EvalMod
    # row count at 7+1+2 = 10, exactly as the old 6+2+2 did, so depth and the
    # level budget are unchanged. Degree 119 clears the point where the
    # Chebyshev coefficients of cos(2*pi*(K*x-1/4)/2**r) start to decay (n ~ K*pi
    # ~ 78 at K=25); degree 59 sits below it, which capped EvalMod at 2**-26.8
    # however the rest was tuned. Above the knee the fit reaches 2**-47.3.
    'n_cosine_levels' : 7,
    'n_double_angle' : 1,
    'n_arcsine_levels' : 2,
    'n_cts_levels' : 3,
    # Headroom ratio q_base/|m|; Lattigo's LogMessageRatio. It bounds how much of
    # q_base a message may occupy (1/2**log_delta), and exceeding that makes
    # ModRaise wrap on the wrong multiple -- silently. Precision is flat from 3
    # to 5 and then costs ~0.85 bits per increment (measured at LogN=15, q_base
    # held at 50: 3 -> 20.13, 5 -> 20.07, 6 -> 19.69, 8 -> 18.02), so 5 sits at
    # the edge of the free region; raise it to 6 for 2x the margin at ~0.4 bits
    # if callers may push the message bound. Lattigo uses 8, buying a documented
    # 2**-138.7 failure probability at ~2 bits.
    'log_delta' : 5,
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
    # log2 of the bound on the values EvalMod reduces, i.e. |value| <= 1/2.
    # It applies to whatever EvalMod actually reduces: the *slot* values for
    # SLIM, which runs SlotsToCoefs first, but the encoded polynomial's
    # *coefficients* for REAL. It sizes the cosine's fit clusters and the range
    # the arcsine is fitted over; it does not touch q_base. It is an assertion
    # about the plaintexts, not something derivable from the ring dimension: for
    # a generic slot vector the coefficients are ~sqrt(n) smaller, yet a
    # constant slot vector encodes to a constant polynomial whose coefficient
    # equals the slot value. Getting it wrong makes ModRaise wrap on the wrong
    # multiple, silently. Hence the safe 1/2. Raising it above -1 is rejected:
    # the cosine's fit region would widen with it and the approximation has not
    # been validated there.
    'message_bound_log2' : -1,
}


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
        self.stc_q_list_precision = self.get_q_list_precision(self.stc_scale, self.n_stc_levels)
        self.cts_q_list_precision = self.get_q_list_precision(self.cts_scale, self.n_cts_levels)
        self.log_delta = BOOT_CONFIGURATION['log_delta']
        self.eval_mod_q_list_precision = self.get_q_list_precision(self.eval_mod_scale, self.n_cosine_levels + self.n_double_angle + self.n_arcsine_levels)
        self.n_cts_rows = len(self.cts_q_list_precision)
        self.n_stc_rows = len(self.stc_q_list_precision)
        self.n_evalmod_rows = len(self.eval_mod_q_list_precision)
        self.k = self.compute_k_from_sk_hw(self.sk_hw)
        self.q_list_precision = self.cts_q_list_precision +  self.eval_mod_q_list_precision + self.stc_q_list_precision
        # Bound on the values EvalMod reduces. It sets q_base and the range the
        # arcsine correction is fitted over, so it must hold for every message
        # that will be bootstrapped -- an overshoot makes ModRaise wrap on the
        # wrong multiple of q and corrupts the result outright.
        #
        # For REAL the bound applies to the encoded polynomial's coefficients,
        # not the slots. For a *generic* slot vector those are ~sqrt(n) smaller,
        # and lowering the bound accordingly buys ~log2(sqrt(n)) bits of
        # precision (EvalMod's error is an additive floor that does not shrink
        # with the input). But that relation is statistical, not a bound: a
        # constant slot vector encodes to a constant polynomial whose leading
        # coefficient equals the slot value. So this stays an explicit assertion
        # by the caller about its own plaintexts, defaulting to the safe 1/2.
        self.message_bound_log2 = BOOT_CONFIGURATION['message_bound_log2']
        self.message_bound = 2.0 ** self.message_bound_log2
        if not isinstance(self.message_bound_log2, int) or self.message_bound_log2 > -1:
            raise ValueError(
                f"message_bound_log2 must be an int <= -1; got {self.message_bound_log2!r}.")
        self.q_base_precision = self.get_q_base_precision()
        self.modulus = sum(row[0] for row in self.q_list_precision) + self.q_base_precision
        self.cosine_deg = self.optimal_poly_deg_per_depth(self.n_cosine_levels)
        self.arcsine_deg = self.optimal_poly_deg_per_depth(self.n_arcsine_levels)
        