import pytest

from lattica_build.base_classes.hom_pipeline import HomomorphicPipeline
from lattica_build.base_classes.hom_value import HomValue
from lattica_build.operators.composite.sequential import SequentialHomOp
from lattica_build.operators.fhe.h_bootstrap import Bootstrap
from lattica_build.operators.slots.h_sum_slots import HomSumSlots
from lattica_build.params.bootstrapping_params import BootstrappingParams
from lattica_build.params.level_and_scale_tracing import ModulusChain, _get_config_for_row_or_col, infer_modswitch_by_num_levels
from lattica_build.params.params import DecompositionType, HomParams


def test_chain_serves_the_slot_counts_the_pipeline_bootstraps():
    """The input fills the ring's 2^15 slots; HomSumSlots(k=2^10) folds them to 32, which the bootstrap sees:
    the chain is built for 32."""
    pipeline = HomomorphicPipeline(hom=SequentialHomOp(HomSumSlots(k=2 ** 10), Bootstrap()), input_shape=(2 ** 15,))
    params = HomParams(n=2 ** 16, full_q_list_precision=((60,),) * 4, pt_scale=2 ** 40, dnum=2)
    pipeline.serialize(params)
    expected = BootstrappingParams.for_slot_counts(params.real_boot, params.slim_boot, {32})
    loaded = HomParams.from_dict(params.to_dict())
    assert (loaded.boot_params.dft_levels, loaded.boot_params.cts_scale) == (expected.dft_levels, expected.cts_scale)


@pytest.mark.parametrize("dnum, num_special_primes", [(4, 1), (2, 2), (1, 4)])
def test_special_primes_are_the_fewest_that_split_the_chain_into_dnum_digits(dnum, num_special_primes):
    """Four 60-bit primes: one per 61-bit digit, two per 122-bit digit, all four in one 244-bit digit."""
    params = HomParams(n=2 ** 16, full_q_list_precision=((60,),) * 4, pt_scale=2 ** 40, dnum=dnum)
    assert params.num_special_primes == num_special_primes
    assert params.g_base_bits == 61 * num_special_primes
    loaded = HomParams.from_dict(params.to_dict())
    assert (loaded.dnum, loaded.num_special_primes) == (dnum, num_special_primes)


def test_hybrid_params_saved_without_dnum_are_refused():
    d = HomParams(n=2 ** 16, full_q_list_precision=((60,),) * 4, pt_scale=2 ** 40, dnum=2).to_dict()
    del d["dnum"]
    with pytest.raises(ValueError, match="recompile"):
        HomParams.from_dict(d)


@pytest.mark.parametrize("q_list, dnum", [(((60, 30),) * 5, 3), (((40,),) * 10, 3)])
def test_unset_dnum_is_the_fewest_128_bit_secure_digits(q_list, dnum):
    """The 2^16 bootstrap chains: both need 3 digits to keep QP within 1638 bits. 2 would take 10 special primes for
    5x(60,30) in chain order, the bootstrap's primes first (1695 bits), and 1795 bits for 10x40."""
    boot = BootstrappingParams.for_slot_counts(True, False, {2 ** 15})
    params = HomParams.from_dict({**HomParams(n=2 ** 16, full_q_list_precision=q_list, pt_scale=2 ** 30, sk_hw=192,
                                              n_slots=2 ** 15).to_dict(),
                                  'bootstrapping': True, 'boot_dft_levels': boot.dft_levels,
                                  'boot_cts_scale': boot.cts_scale})
    assert params.dnum is None and params.resolved_dnum == dnum


def test_unset_dnum_without_a_bootstrap_is_one_digit():
    params = HomParams(n=2 ** 16, full_q_list_precision=((60,),) * 4, pt_scale=2 ** 40, sk_hw=192)
    assert (params.resolved_dnum, params.num_special_primes) == (1, 4)


def test_unset_dnum_without_a_security_bound_is_refused():
    params = HomParams(n=2 ** 12, full_q_list_precision=((60,),) * 4, pt_scale=2 ** 40)
    with pytest.raises(ValueError, match="No 128-bit bound"):
        params.num_special_primes


def test_unset_dnum_with_bv_is_zero():
    params = HomParams(n=2 ** 16, full_q_list_precision=((60,),) * 4, pt_scale=2 ** 40, decomposition_type=DecompositionType.BV)
    assert (params.resolved_dnum, params.num_special_primes) == (0, 0)


@pytest.mark.parametrize("q_list, drop_levels, levels_of_dropped_rows", [
    (((30,),) + ((60, 30),) * 3, 1, 1),   # ResNet-20's chain: its (30,) row
    (((60, 30),) * 4, 2, 4),
])
def test_bootstrap_drop_levels_leaves_out_the_rows_the_full_output_consumes_first(
        q_list, drop_levels, levels_of_dropped_rows):
    """Bootstrap(drop_levels) outputs every pipeline row but the first drop_levels: the full output once its rescales
    have consumed those rows, so the lighter one's limbs are a contiguous run of the full one's."""
    params = HomParams(n=2 ** 16, full_q_list_precision=q_list, pt_scale=2 ** 30, dnum=2, n_slots=2 ** 14)
    params.boot_params = BootstrappingParams.for_slot_counts(params.real_boot, params.slim_boot, {2 ** 14})
    params.mod_chain = ModulusChain(params)
    pipeline = params.mod_chain.section_rows["pipeline"]
    value = HomValue(tensor_shape=(2 ** 14,), n_axis=0, pt_scale=2 ** 30, n_slots=2 ** 14)
    full = Bootstrap().infer_output_level_and_scale(value, hom_params=params)
    lighter = Bootstrap(drop_levels=drop_levels).infer_output_level_and_scale(value, hom_params=params)

    assert lighter.active_rows == pipeline[drop_levels:] + params.mod_chain.section_rows["boot_base"]
    consumed = infer_modswitch_by_num_levels(params, full, levels_of_dropped_rows, rows_budget=None)
    assert (consumed.active_rows, consumed.active_cols) == (lighter.active_rows, lighter.active_cols)


def _boot_params(q_list, real_boot, slim_boot):
    boot = BootstrappingParams.for_slot_counts(real_boot, slim_boot, {2 ** 15})
    return HomParams.from_dict({**HomParams(n=2 ** 16, full_q_list_precision=q_list, pt_scale=2 ** 30, dnum=3,
                                            real_boot=real_boot, slim_boot=slim_boot, n_slots=2 ** 15).to_dict(),
                                'bootstrapping': True, 'boot_dft_levels': boot.dft_levels,
                                'boot_cts_scale': boot.cts_scale})


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


@pytest.mark.parametrize("slot_counts, lcr", [
    ({2 ** 15}, True),
    ({2 ** 14, 2 ** 13, 2 ** 12}, True),     # ResNet-20's bootstraps: 64 terms each
    ({2 ** 14, 2 ** 11}, False),             # 2**11: 32 diagonals x 16 trace rotations
], ids=["full", "resnet", "too-sparse"])
def test_lcr_first_cts_chain_drops_a_cts_prime_when_every_bootstrap_has_it(monkeypatch, slot_counts, lcr):
    """With LATTICA_LCR_FIRST_CTS=1, for_slot_counts drops one CtS prime only if every slot count's level-conserving
    first CtS level (its diagonals times the SubSum trace's rotations) fits the kernel, and to_dict carries it."""
    n = 2 ** 16
    monkeypatch.setenv("LATTICA_LCR_FIRST_CTS", "0")
    plain = BootstrappingParams.for_slot_counts(True, False, slot_counts, n=n)
    monkeypatch.setenv("LATTICA_LCR_FIRST_CTS", "1")
    boot = BootstrappingParams.for_slot_counts(True, False, slot_counts, n=n)
    assert boot.lcr_first_cts == lcr and not plain.lcr_first_cts
    assert len(boot.cts_q_list_precision) == len(plain.cts_q_list_precision) - int(lcr)
    assert [boot.lcr_first_cts_terms(s, n) for s in (2 ** 15, 2 ** 14, 2 ** 13, 2 ** 12, 2 ** 11)] == \
        [32, 64, 64, 64, 512]
    params = HomParams(n=n, full_q_list_precision=((30,),) + ((60, 30),) * 3, pt_scale=2 ** 30, dnum=3,
                       n_slots=min(slot_counts))
    params.boot_params = boot
    d = params.to_dict()
    assert d["boot_lcr_first_cts"] == lcr
    monkeypatch.setenv("LATTICA_LCR_FIRST_CTS", "0")
    back = HomParams.from_dict({**d, "bootstrapping": True})
    assert back.boot_params.lcr_first_cts == lcr
