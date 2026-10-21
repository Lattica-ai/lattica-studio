import pytest

from lattica_build.base_classes.hom_pipeline import HomomorphicPipeline
from lattica_build.operators.composite.sequential import SequentialHomOp
from lattica_build.operators.fhe.h_bootstrap import Bootstrap
from lattica_build.operators.slots.h_sum_slots import HomSumSlots
from lattica_build.params.bootstrapping_params import BootstrappingParams
from lattica_build.params.params import DEFAULT_DNUM, DecompositionType, HomParams


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


@pytest.mark.parametrize("decomposition_type, dnum", [(DecompositionType.HYBRID, DEFAULT_DNUM), (DecompositionType.BV, 0)])
def test_dnum_defaults_by_decomposition_type(decomposition_type, dnum):
    params = HomParams(n=2 ** 16, full_q_list_precision=((60,),) * 4, pt_scale=2 ** 40,
                       decomposition_type=decomposition_type)
    assert params.dnum == dnum
