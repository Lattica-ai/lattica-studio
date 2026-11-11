import pytest
import numpy as np

from lattica_build.base_classes.hom_pipeline import HomomorphicPipeline
from lattica_build.operators.composite.sequential import SequentialHomOp
from lattica_build.operators.fhe.h_bootstrap import Bootstrap
from lattica_build.operators.polynomials.h_poly_eval import HomPolyEval
from lattica_build.operators.slots.h_sum_slots import HomSumSlots
from lattica_build.params.bootstrapping_params import BootstrappingParams
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
