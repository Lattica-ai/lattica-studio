import math

import pytest

from lattica_build.base_classes.hom_pipeline import HomomorphicPipeline
from lattica_build.operators.composite.sequential import SequentialHomOp
from lattica_build.operators.fhe.h_bootstrap import Bootstrap
from lattica_build.operators.slots.h_sum_slots import HomSumSlots
from lattica_build.params.bootstrapping_params import BootstrappingParams
from lattica_build.params.level_and_scale_tracing import ModulusChain
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


def test_unset_dnum_counts_the_primes_at_their_actual_size():
    """26-bit good primes reach 26.73 bits (few NTT-friendly primes that small): 12 rows of (52, 26) at 2^15 fit 809 bits
    at dnum 4 by their nominal size but not by their drawn primes, so the derivation takes 6 digits."""
    params = HomParams(n=2 ** 15, full_q_list_precision=((52, 26),) * 12, pt_scale=2 ** 26, sk_hw=192)
    log_qp = sum(math.log2(int(q)) for row in ModulusChain(params).factors_per_row for q in row)
    assert params.resolved_dnum == 6 and log_qp <= 809


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
