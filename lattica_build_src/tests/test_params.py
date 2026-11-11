import math

import pytest
import numpy as np

from lattica_build.base_classes.hom_pipeline import HomomorphicPipeline
from lattica_build.operators.composite.sequential import SequentialHomOp
from lattica_build.operators.fhe.h_bootstrap import Bootstrap
from lattica_build.operators.polynomials.h_poly_eval import HomPolyEval
from lattica_build.operators.slots.h_sum_slots import HomSumSlots
from lattica_build.params.bootstrapping_params import MAX_LOG_QP_128, BootstrappingParams
from lattica_build.params.level_and_scale_tracing import ModulusChain, _get_config_for_row_or_col
from lattica_build.params.params import HomParams


def test_chain_serves_the_slot_counts_the_pipeline_bootstraps():
    """The input fills the ring's 2^15 slots; HomSumSlots(nb_cols=32) folds them to 32, which the bootstrap sees:
    the chain is built for 32."""
    pipeline = HomomorphicPipeline(hom=SequentialHomOp(HomSumSlots(nb_cols=32), Bootstrap()), input_shape=(2 ** 15,))
    params = HomParams(n=2 ** 16, full_q_list_precision=((60,),) * 4, pt_scale=2 ** 40, num_special_primes=7)
    pipeline.serialize(params)
    expected = BootstrappingParams.for_slot_counts(params.real_boot, params.slim_boot, {32})
    loaded = HomParams.from_dict(params.to_dict())
    assert (loaded.boot_params.dft_levels, loaded.boot_params.cts_scale) == (expected.dft_levels, expected.cts_scale)


def test_serialize_records_the_consumption_order_with_row_budgets():
    """The first polynomial may only rescale with rows 3 and 4: they are consumed first."""
    pipeline = HomomorphicPipeline(
        hom=SequentialHomOp(HomPolyEval(func=np.abs, degree=3, rows_budget=[3, 4]),
                            HomPolyEval(func=np.abs, degree=3)),
        input_shape=(8,))
    params = HomParams(n=256, full_q_list_precision=((60,),) * 5, pt_scale=2 ** 30, num_special_primes=3, n_slots=8)
    pipeline.serialize(params)
    assert params.consumption_order[:2] == (3, 4)
    assert set(params.consumption_order) == {0, 1, 2, 3, 4}
    assert HomParams.from_dict(params.to_dict()).consumption_order == params.consumption_order


@pytest.mark.parametrize("slot_counts, special_bits, lcr", [
    ({2 ** 15}, 3 * 61, True),
    ({2 ** 14, 2 ** 13, 2 ** 12}, 3 * 61, True),     # ResNet-20's bootstraps: 32 to 64 terms
    ({2 ** 14, 2 ** 11}, 3 * 61, False),             # 2**11: 32 diagonals x 16 trace rotations
    ({2 ** 15}, 2 * 61, False),                      # two special primes: the keys' noise is too large
], ids=["full", "resnet", "too-sparse", "two-special-primes"])
def test_lcr_first_cts_chain_drops_a_cts_prime_when_every_bootstrap_has_it(slot_counts, special_bits, lcr):
    """for_slot_counts drops one CtS prime only if the special primes are wide enough for the level-conserving first
    CtS level's keys and every slot count's level (its diagonals times the SubSum trace's rotations) fits the kernel,
    and to_dict carries the choice."""
    n = 2 ** 16
    plain = BootstrappingParams.for_slot_counts(True, False, slot_counts)
    boot = BootstrappingParams.for_slot_counts(True, False, slot_counts, n=n, special_bits=special_bits)
    assert boot.lcr_first_cts == lcr and not plain.lcr_first_cts
    assert len(boot.cts_q_list_precision) == len(plain.cts_q_list_precision) - int(lcr)
    # 16, 32, 64 and 64 terms fit; 2**11's 32 diagonals x 16 trace rotations do not
    assert [boot.lcr_first_cts_fits(s, n) for s in (2 ** 15, 2 ** 14, 2 ** 13, 2 ** 12, 2 ** 11)] == \
        [True, True, True, True, False]
    params = HomParams(n=n, full_q_list_precision=((30,),) + ((60, 30),) * 3, pt_scale=2 ** 30, num_special_primes=3,
                       n_slots=min(slot_counts))
    params.boot_params = boot
    d = params.to_dict()
    assert d["boot_lcr_first_cts"] == lcr
    back = HomParams.from_dict({**d, "bootstrapping": True})
    assert back.boot_params.lcr_first_cts == lcr


@pytest.mark.parametrize("n, num_special_primes, key", [
    (2 ** 16, 9, (2, 13)),   # 758 bits of q left after 2 EvalMod rescales, 13 special primes: 1551 bits
    (2 ** 16, 13, None),     # the square key's digit already holds every level past the first
    (2 ** 10, 9, None),      # no 128-bit bound known
], ids=["2^16", "wide-square-key", "no-bound"])
def test_one_digit_relin_key_fits_the_128_bit_bound(n, num_special_primes, key):
    """The REAL 2**15-slot chain with 5 x (60, 30) after the bootstrap: the one-digit key starts at the first EvalMod
    level whose rows and special primes fit MAX_LOG_QP_128, and the chain carries its special primes after the square
    key's."""
    boot = BootstrappingParams.for_slot_counts(True, False, {2 ** 15}, n=2 ** 16, special_bits=9 * 61)
    params = HomParams(n=n, full_q_list_precision=((60, 30),) * 5, pt_scale=2 ** 30,
                       num_special_primes=num_special_primes)
    params.boot_params = boot
    assert boot.one_digit_relin_key(boot.chain_sections(params.full_q_list_precision), num_special_primes, 61, n) == key
    special = ModulusChain(params).section_rows["special_primes"]
    assert len(special) == (num_special_primes if key is None else key[1])


@pytest.mark.parametrize("q_list, num_special_primes, g_base_bits", [
    (((61,), (61,)), 1, 61),        # one special prime: a digit holds one 61-bit prime
    (((60, 30),) * 5, 9, 9 * 61 - 8),
], ids=["one-special-prime", "nine-special-primes"])
def test_digits_hold_a_prime_and_keep_their_headroom_below_p(q_list, num_special_primes, g_base_bits):
    """The digits stay DIGIT_HEADROOM_BITS below the special primes' product, unless that is less than one prime."""
    params = HomParams(n=2 ** 10, full_q_list_precision=q_list, pt_scale=2 ** 30, num_special_primes=num_special_primes)
    assert params.g_base_bits == g_base_bits


def test_lcr_chain_shares_a_double_angle_word_and_stays_below_the_word_limit():
    """The REAL 2**15-slot level-conserving chain: double angles 1 and 3 share EvalMod's last word (7 rows for 8
    levels), and every row's modulus stays below the kernels' 2**62, the StC's two-prime 62-bit words included."""
    boot = BootstrappingParams.for_slot_counts(True, False, {2 ** 15}, n=2 ** 16, special_bits=9 * 61)
    params = HomParams(n=2 ** 16, full_q_list_precision=((60, 30),) * 5, pt_scale=2 ** 30, num_special_primes=9)
    params.boot_params = boot
    chain = ModulusChain(params)
    assert len(chain.section_rows["boot_eval"]) == 7
    assert all(math.prod(primes) < 2 ** 62 for primes in chain.factors_per_row)
    assert all(math.prod(chain.factors_per_row[row]) > 2 ** 61.9 for row in chain.section_rows["boot_stc"])


def test_low_precision_chain_fits_n_2_15_at_128_bits():
    """The "low" profile's chain for ResNet-20 at N = 2**15 (bootstraps at 2**12 to 2**14 slots, resnet20_n15's rows
    after them, 4 special primes of 57 bits) runs EvalMod on the profile's rows, its base word holds a 26-bit prime and
    q0 below 2**62, every key fits MAX_LOG_QP_128, and serialization keeps the profile and the special primes."""
    n = 2 ** 15
    boot = BootstrappingParams.for_slot_counts(True, False, {2 ** 12, 2 ** 13, 2 ** 14}, n=n, special_bits=4 * 57,
                                               precision="low")
    params = HomParams(n=n, full_q_list_precision=((26,), (52, 26), (52, 26)), pt_scale=2 ** 26, n_slots=2 ** 14,
                       num_special_primes=4, special_prime_bits=57, boot_precision="low")
    params.boot_params = boot
    assert boot.lcr_first_cts and boot.exit_scale_for(2 ** 12) == 38
    chain = ModulusChain(params)
    base = chain.factors_per_row[chain.section_rows["boot_base"][0]]
    assert [int(math.log2(p)) for p in base] == [26, 35] and math.prod(base) < 2 ** 62
    q_rows = [row for name, rows in chain.section_rows.items() if name != "special_primes" for row in rows]
    special = chain.section_rows["special_primes"]
    log_qp = sum(math.log2(math.prod(chain.factors_per_row[row])) for row in q_rows + special[:4])
    assert log_qp <= MAX_LOG_QP_128[n]
    back = HomParams.from_dict({**params.to_dict(), "bootstrapping": True})
    assert back.boot_params.q_list_precision == boot.q_list_precision
    assert (back.num_special_primes, back.special_prime_bits) == (4, 57)


def test_unknown_boot_precision_is_refused():
    with pytest.raises(ValueError, match="boot_precision must be one of"):
        HomParams(n=2 ** 15, full_q_list_precision=((60,),), pt_scale=2 ** 30, num_special_primes=3,
                  boot_precision="medium")


def _boot_params(q_list, real_boot, slim_boot):
    boot = BootstrappingParams.for_slot_counts(real_boot, slim_boot, {2 ** 15})
    return HomParams.from_dict({**HomParams(n=2 ** 16, full_q_list_precision=q_list, pt_scale=2 ** 30,
                                            num_special_primes=7, real_boot=real_boot, slim_boot=slim_boot,
                                            n_slots=2 ** 15).to_dict(),
                                'bootstrapping': True, 'boot_dft_levels': boot.dft_levels,
                                'boot_cts_scale': boot.cts_scale, 'boot_lcr_first_cts': False})


def _drop_levels(mod_chain, rows, sections, dropped):
    """Rescale rows (the active rows, in limb order) by every prime of sections, each under its own budget (None
    for the pipeline's, as the pipeline's ops rescale), recording the limb each drop takes: (relative row, the
    number of limbs). Returns the rows left."""
    cols = [mod_chain.full_active_cols[row].tolist() for row in rows]
    for section in sections:
        budget = None if section == "pipeline" else mod_chain.section_rows[section]
        for _ in range(sum(len(mod_chain.full_q_list[row]) for row in mod_chain.section_rows[section])):
            rel_row, drop_cols, _ = _get_config_for_row_or_col(mod_chain, rows, cols, budget)
            dropped.append((rel_row, len(rows)))
            if drop_cols is None:
                rows, cols = rows[:rel_row] + rows[rel_row + 1:], cols[:rel_row] + cols[rel_row + 1:]
            else:
                cols[rel_row] = [0 if col in drop_cols else on for col, on in enumerate(cols[rel_row])]
    return rows


@pytest.mark.parametrize("real_boot", [True, False], ids=["real", "complex"])
@pytest.mark.parametrize("q_list", [((60, 30),) * 3 + ((30,),), ((40,),) * 4])
def test_every_rescale_of_a_bootstrap_and_after_it_drops_the_leading_limb(q_list, real_boot):
    """A ciphertext's limbs are its active rows in chain order. ModRaise raises to every row but the special primes;
    CtS, EvalMod and StC rescale within their own rows, then the pipeline's ops within any: each drop is of the
    first active row, so a level drop is a slice of the limb axis, not a gather."""
    params = _boot_params(q_list, real_boot, slim_boot=False)
    mod_chain = ModulusChain(params)
    raised = [row for name, rows in mod_chain.section_rows.items() if name != "special_primes" for row in rows]
    assert raised == sorted(raised) and raised[-1] == mod_chain.section_rows["boot_base"][0]
    dropped = []
    left = _drop_levels(mod_chain, raised, ["boot_cts", "boot_eval", "boot_stc", "pipeline"], dropped)
    assert left == mod_chain.section_rows["boot_base"]
    assert all(rel_row == 0 for rel_row, _ in dropped)


def test_slim_bootstrap_rescales_drop_the_leading_limb():
    """SLIM: the input holds the pipeline's rows, the StC's and q0; the pipeline's ops consume theirs, then the StC
    its own, all leading limbs; after ModRaise, CtS and EvalMod theirs, then the pipeline's ops theirs."""
    params = _boot_params(((40,),) * 4, real_boot=True, slim_boot=True)
    mod_chain = ModulusChain(params)
    dropped = []
    left = _drop_levels(mod_chain, mod_chain.get_init_rows(params), ["pipeline", "boot_stc"], dropped)
    assert left == mod_chain.section_rows["boot_base"]
    raised = [row for name, rows in mod_chain.section_rows.items() if name != "special_primes" for row in rows]
    _drop_levels(mod_chain, raised, ["boot_cts", "boot_eval", "pipeline"], dropped)
    assert all(rel_row == 0 for rel_row, _ in dropped)
