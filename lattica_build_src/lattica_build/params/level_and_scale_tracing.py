"""See `params/README.md` for usage details."""

import copy

import torch

from lattica_build.base_classes.hom_value import HomValue
from typing import Sequence, Tuple
from decimal import Decimal

from lattica_build.params.bootstrapping_params import BootstrappingVariant
from lattica_build.params.primes_utils import get_primes_from_precisions_list


class ModulusChain():

    def __init__(self, hom_params):
        self._set_pipeline_sections(hom_params)
        self.post_init()

    def _set_pipeline_sections(self, hom_params):
        self.mod_chain_sections = []
        self.mod_chain_sections.append(
            ["pipeline", hom_params.full_q_list_precision]
        )
        boot_params = hom_params.boot_params
        if boot_params is not None:
            self.mod_chain_sections.extend([
                ["boot_cts",  boot_params.cts_q_list_precision],
                ["boot_eval", boot_params.eval_mod_q_list_precision],
                ["boot_stc",  boot_params.stc_q_list_precision],
                ["boot_base", ((boot_params.q_base_precision,),)]
            ])
        self.mod_chain_sections.append(
            ["special_primes", ((61,),) * hom_params.num_special_primes],
        )

    def post_init(self):
        # Compute rows range for each section.
        self.section_rows = {}
        current_row = 0
        for section_name, section_q_list in self.mod_chain_sections:
            num_rows = len(section_q_list)
            self.section_rows[section_name] = list(range(current_row, current_row + num_rows))
            current_row += num_rows

        # Compute full q_list
        self.full_q_list = []
        for _, section_q_list in self.mod_chain_sections:
            self.full_q_list.extend(section_q_list)

        # Compute full active rows and active cols
        num_cols_per_row = torch.tensor([len(row) for row in self.full_q_list])
        self.max_num_cols = num_cols_per_row.max().item()
        self.num_rows = len(self.full_q_list)
        self.full_active_rows = torch.arange(0, self.num_rows)
        self.full_active_cols = torch.zeros(size=(self.num_rows, self.max_num_cols), dtype=torch.int32)
        for row_idx, num_cols in enumerate(num_cols_per_row):
            self.full_active_cols[row_idx, :num_cols] = 1

        # Sample primes
        self.factors_per_row = get_primes_from_precisions_list(self.full_q_list)

    def get_rows_of_sections(self, section_names):
        rows = []
        for section_name in section_names:
            rows += self.section_rows[section_name]
        return rows

    def get_all_rows(self):
        sections = list(self.section_rows.keys())
        return self.get_rows_of_sections(sections)

    def get_init_rows(self, params):
        active_rows = copy.deepcopy(self.section_rows["pipeline"])
        if not params.bootstrapping:
            return active_rows

        if params.num_init_rows is not None:
            active_rows = active_rows[:params.num_init_rows]
        active_rows += self.section_rows["boot_base"]
        if params.bootstrapping_variant == BootstrappingVariant.SLIM:
            active_rows += self.section_rows["boot_stc"]
        return active_rows


def init_active_rows_cols(hom_params):
    """Initialize the active rows and cols for a fresh ciphertext."""
    mod_chain = hom_params.mod_chain
    rows = mod_chain.get_init_rows(hom_params)
    return (
        mod_chain.full_active_rows[rows].tolist(),
        mod_chain.full_active_cols[rows].tolist()
    )

def _get_drop_row_config(mod_chain, active_rows, relative_row_idx, active_cols_in_chosen_row, **kwargs):
    factors_of_active_rows = [row_factors for i, row_factors in enumerate(mod_chain.factors_per_row) if i in active_rows]
    factors_of_chosen_row = factors_of_active_rows[relative_row_idx]
    scale_down_by = torch.tensor(
        [p for i, p in enumerate(factors_of_chosen_row) if active_cols_in_chosen_row[i] == 1],
        dtype=torch.int64
    ).prod().item()
    return [relative_row_idx, None, scale_down_by]

def _get_drop_cols_config(mod_chain, active_rows, relative_row_idx, cols_to_drop):
    factors_of_active_rows = [row_factors for i, row_factors in enumerate(mod_chain.factors_per_row) if i in active_rows]
    factors_of_chosen_row = factors_of_active_rows[relative_row_idx]
    scale_down_by = torch.tensor(factors_of_chosen_row, dtype=torch.int64)[cols_to_drop].prod().item()
    return [relative_row_idx, cols_to_drop, scale_down_by]

def _get_drop_config_for_x_levels_in_a_given_row(
        mod_chain, active_rows, active_cols, relative_row_idx, num_cols_to_drop
):
    active_cols_in_chosen_row = active_cols[relative_row_idx]
    if sum(active_cols_in_chosen_row) == num_cols_to_drop:
        # Drop all row
        return _get_drop_row_config(mod_chain, active_rows, relative_row_idx, active_cols_in_chosen_row)
    else:
        # Drop some cols
        cols_to_drop = []
        for col_idx, is_active in enumerate(active_cols_in_chosen_row):
            if is_active:
                cols_to_drop.append(col_idx)
            if len(cols_to_drop) == num_cols_to_drop:
                break
        return _get_drop_cols_config(mod_chain, active_rows, relative_row_idx, cols_to_drop)

def _get_config_for_row_or_col(mod_chain, active_rows, active_cols, rows_budget: Sequence[int] | None):
    """Choose the first available col to drop from the rows_budget."""
    for relative_row_idx, abs_row_idx in enumerate(active_rows):
        if rows_budget is not None and abs_row_idx not in rows_budget:
            continue
        return _get_drop_config_for_x_levels_in_a_given_row(
            mod_chain, active_rows, active_cols, relative_row_idx, 1
        )
    raise ValueError(f"Trying to drop level but no valid level found."
                     f" {active_rows=}, {active_cols=}, {rows_budget=}")


def apply_drop_config(
        input: HomValue,
        drop_config: Tuple[int, Sequence[int] | None, int],
):
    relative_row, cols_to_drop, scale_down_by = drop_config
    if cols_to_drop is None:
        # Drop the whole row.
        new_cols = input.active_cols[:relative_row] + input.active_cols[relative_row + 1:]
        new_rows = input.active_rows[:relative_row] + input.active_rows[relative_row + 1:]
    else:
        # Drop cols_to_drop columns from the row at relative_row.
        new_cols = copy.deepcopy(input.active_cols)
        for col in cols_to_drop:
            new_cols[relative_row][col] = 0
        new_rows = input.active_rows

    new_scale = Decimal(input.pt_scale) / Decimal(scale_down_by)

    return input.make_copy(
        active_rows=new_rows,
        active_cols=new_cols,
        pt_scale=new_scale,
    )

def infer_optional_modswitch(hom_params: 'HomParams', input: HomValue, with_modswitch: bool, rows_budget: Sequence[int], op_scale_up: int | None) -> HomValue:
    """Optionally drop a single row/col based on an optional with_modswitch flag.

    if op_scale_up is None:
        if with_modswitch=True, scale-up by the same prime we later scale-down by (so overall scale remains the same).
        if with_modswitch=False, scale-up by the default pt_scale.
    if op_scale_up is a number, scale-up by that number.

    """
    if not with_modswitch:
        if op_scale_up is None:
            op_scale_up = hom_params.pt_scale
        return input.make_copy(pt_scale=input.pt_scale * op_scale_up)

    drop_config = _get_config_for_row_or_col(
        hom_params.mod_chain, input.active_rows, input.active_cols, rows_budget=rows_budget
    )
    if op_scale_up is None:
        op_scale_up = drop_config[2]
    res = input.make_copy(pt_scale=input.pt_scale * op_scale_up)

    return apply_drop_config(res, drop_config)

def infer_modswitch_by_num_levels(hom_params: 'HomParams', input: HomValue, num_levels: int, rows_budget: Sequence[int]) -> HomValue:
    """Drop multiple rows/cols according to the `num_levels` param."""
    for _ in range(num_levels):
        input = infer_optional_modswitch(hom_params, input, True, rows_budget, op_scale_up=None)
    return input

def _get_relative_new_indices(active_rows, target_rows):
    """ Get the relative indices in active_rows for each target_row."""
    # Compare every target against active_rows (broadcasts to [m, n])
    matches = torch.tensor(target_rows)[:, None] == torch.tensor(active_rows)[None, :]
    # Get the index in active_rows for each target
    relative_new_indices = matches.int().argmax(dim=1)
    relative_new_indices.sort()
    return relative_new_indices

def _verify_rows_cols_are_subset(input_1, input_2):
    if not set(input_1.active_rows).issubset(set(input_2.active_rows)):
        return False
    active_cols_1 = torch.tensor(input_1.active_cols)
    rows_of_one_in_two = _get_relative_new_indices(input_2.active_rows, input_1.active_rows)
    active_cols_2 = torch.tensor(input_2.active_cols)[rows_of_one_in_two]
    return ((1 - active_cols_1) + (active_cols_1 * active_cols_2)).all()

def infer_adjust_levels_and_scale(input_1: HomValue, input_2: HomValue) -> HomValue:
    """Infer a compatible level/scale metadata state for binary ops.

    The two inputs are compatible if one input's active rows/cols are a subset
    of the other's active rows/cols. In that case, the function returns a copy
    of the subset-compatible input metadata (shape/rows/cols/scale state).

    If neither input is a subset of the other, a `ValueError` is raised.

    Operation-specific nuance:
        - Add/Sub path: if inputs are already at the same level (same rows/cols)
          then scales are expected to match; otherwise add/sub is invalid. When
          levels differ, callers first align to the lower level and can use that
          step to reconcile scale as needed.
        - Mul path: callers align to the lower level but keep multiplication
          scale policy separate.

    This helper performs the shared level-alignment subset check and returns the
    aligned metadata representative used by those caller policies.
    """
    one_sub_of_two = _verify_rows_cols_are_subset(input_1, input_2)
    two_sub_of_one = _verify_rows_cols_are_subset(input_2, input_1)

    if one_sub_of_two and two_sub_of_one:
        return input_1

    if not (one_sub_of_two or two_sub_of_one):
        raise ValueError(
            f'Cannot adjust levels of ciphertexts with different active rows and cols:'
            f'{input_1.active_rows=} {input_2.active_rows=}'
            f'{input_1.active_cols=} {input_2.active_cols=}. One must be a subset of the other.')

    unchanged_input = input_1 if one_sub_of_two else input_2

    return unchanged_input.make_copy()

def init_levels_after_bootstrap(input, hom_params):
    """Initialize the active rows and cols for a fresh ciphertext."""
    mod_chain = hom_params.mod_chain
    rows = copy.deepcopy(mod_chain.section_rows["pipeline"])
    if hom_params.bootstrapping_variant == BootstrappingVariant.SLIM:
        rows += mod_chain.section_rows["boot_stc"]
    rows += mod_chain.section_rows["boot_base"]
    return input.make_copy(
        active_rows=mod_chain.full_active_rows[rows].tolist(),
        active_cols=mod_chain.full_active_cols[rows].tolist()
    )
