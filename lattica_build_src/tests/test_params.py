import json

import pytest

from lattica_build.base_classes.hom_pipeline import HomomorphicPipeline
from lattica_build.operators.composite.sequential import SequentialHomOp
from lattica_build.operators.fhe.h_bootstrap import Bootstrap
from lattica_build.operators.slots.h_sum_slots import HomSumSlots
from lattica_build.params.bootstrapping_params import BootstrappingParams
from lattica_build.params.params import DecompositionType, HomParams


def test_chain_serves_the_slot_counts_the_pipeline_bootstraps():
    """The input fills the ring's 2^15 slots; HomSumSlots(k=2^10) folds them to 32, which the bootstrap sees:
    the chain is built for 32."""
    pipeline = HomomorphicPipeline(hom=SequentialHomOp(HomSumSlots(k=2 ** 10), Bootstrap()), input_shape=(2 ** 15,))
    params = HomParams(n=2 ** 16, full_q_list_precision=((60,),) * 4, pt_scale=2 ** 40, num_special_primes=7)
    pipeline.serialize(params)
    expected = BootstrappingParams.for_slot_counts(params.real_boot, params.slim_boot, {32})
    loaded = HomParams.from_dict(params.to_dict())
    assert (loaded.boot_params.dft_levels, loaded.boot_params.cts_scale) == (expected.dft_levels, expected.cts_scale)


@pytest.mark.parametrize('options, precisions, gadget_bits', [
    ({'decomposition_type': DecompositionType.BV, 'bv_gadget_bits': 20}, (), 20),
    ({'num_special_primes': 2}, (61, 61), 122),
    ({'ghs_keyswtich_scale_precision': (29,) * 4}, (29,) * 4, 116),
    ({'num_special_primes': 3, 'ghs_keyswtich_scale_precision': (29, 28, 29)}, (29, 28, 29), 86),
])
def test_parameter_round_trip_preserves_special_primes(options, precisions, gadget_bits):
    params = HomParams(n=64, full_q_list_precision=((29,),) * 4, pt_scale=2**24, **options)
    serialized = json.loads(json.dumps(params.to_dict()))
    loaded = HomParams.from_dict(serialized)
    assert tuple(loaded.ghs_keyswtich_scale_precision or ()) == precisions
    assert loaded.g_base_bits == gadget_bits
    assert loaded == params


def test_hybrid_round_trip_preserves_serialized_gadget_bits():
    params = HomParams(n=64, full_q_list_precision=((29,),) * 4, pt_scale=2**24,
                       ghs_keyswtich_scale_precision=(29,) * 4)
    serialized = params.to_dict()
    serialized['g_base_bits'] = 58
    loaded = HomParams.from_dict(serialized)
    assert loaded.to_dict() == serialized
