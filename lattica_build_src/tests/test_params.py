import pytest

from lattica_build.base_classes.hom_pipeline import HomomorphicPipeline
from lattica_build.base_classes.hom_value import HomValue
from lattica_build.operators.composite.sequential import SequentialHomOp
from lattica_build.operators.fhe.h_bootstrap import Bootstrap
from lattica_build.operators.slots.h_sum_slots import HomSumSlots
from lattica_build.params.bootstrapping_params import BootstrappingParams
from lattica_build.params.level_and_scale_tracing import ModulusChain, infer_modswitch_by_num_levels
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


@pytest.mark.parametrize("q_list, dnum", [(((60, 30),) * 5, 2), (((40,),) * 10, 3)])
def test_unset_dnum_is_the_fewest_128_bit_secure_digits(q_list, dnum):
    """The 2^16 bootstrap chains: 2 digits keep 5x(60,30) within 1638 bits of QP; 10x40 needs 3 (2 would be 1795)."""
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
