"""See `params/README.md` for usage details."""

import enum

class BootstrappingVariant(enum.Enum):
    SLIM = 0
    COMPLEX = 1
    REAL = 2


BOOT_CONFIGURATION = {
    'cts_scale' : 42,
    'eval_mod_scale' : 45,
    'stc_scale' : 30,
    'n_stc_levels' : 3,
    'n_cosine_levels' : 6,
    'n_double_angle' : 2,
    'n_arcsine_levels' : 2,
    'n_cts_levels' : 3,
    'log_delta' : 5,
    'subsum_max_stage_size' : 8,
    'tol' : 1e-7,
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
    def get_q_base_precision():
        return BOOT_CONFIGURATION['eval_mod_scale'] + BOOT_CONFIGURATION['log_delta']
    
    def __init__(self, bootstrapping_variant: BootstrappingVariant = BootstrappingVariant.REAL, sk_hw: int = 0) -> None:

        self.variant = BootstrappingVariant(bootstrapping_variant)
        self.stc_scale = BOOT_CONFIGURATION['stc_scale']
        self.eval_mod_scale = BOOT_CONFIGURATION['eval_mod_scale']
        self.cts_scale = BOOT_CONFIGURATION['cts_scale']
        self.n_double_angle = BOOT_CONFIGURATION['n_double_angle']
        self.sk_hw = sk_hw
        self.subsum_max_stage_size = BOOT_CONFIGURATION['subsum_max_stage_size']
        self.arcsine_correction = BOOT_CONFIGURATION['n_arcsine_levels'] > 0 
        self.tol = BOOT_CONFIGURATION['tol']

        self.n_stc_levels = BOOT_CONFIGURATION['n_stc_levels']
        self.n_cosine_levels = BOOT_CONFIGURATION['n_cosine_levels']
        self.n_arcsine_levels = BOOT_CONFIGURATION['n_arcsine_levels']
        self.n_cts_levels = BOOT_CONFIGURATION['n_cts_levels']
        self.stc_q_list_precision = self.get_q_list_precision(self.stc_scale, self.n_stc_levels)
        self.cts_q_list_precision = self.get_q_list_precision(self.cts_scale, self.n_cts_levels)
        self.eval_mod_q_list_precision = self.get_q_list_precision(self.eval_mod_scale, self.n_cosine_levels + self.n_double_angle + self.n_arcsine_levels)
        self.n_cts_rows = len(self.cts_q_list_precision)
        self.n_stc_rows = len(self.stc_q_list_precision)
        self.n_evalmod_rows = len(self.eval_mod_q_list_precision)
        self.k = self.compute_k_from_sk_hw(self.sk_hw)
        self.q_list_precision = self.cts_q_list_precision +  self.eval_mod_q_list_precision + self.stc_q_list_precision
        self.q_base_precision = self.get_q_base_precision()
        self.modulus = sum(row[0] for row in self.q_list_precision) + self.q_base_precision
        self.cosine_deg = self.optimal_poly_deg_per_depth(self.n_cosine_levels)
        self.arcsine_deg = self.optimal_poly_deg_per_depth(self.n_arcsine_levels)
        