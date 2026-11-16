import json

import numpy as np
import pytest

from lattica_build.base_classes.hom_op import HomOp
from lattica_build.base_classes.hom_pipeline import HomomorphicPipeline
from lattica_build.operators.arithmetic.h_add import HomAdd
from lattica_build.operators.arithmetic.h_mul import HomMul
from lattica_build.operators.composite.sequential import SequentialHomOp
from lattica_build.operators.fhe.h_bootstrap import Bootstrap
from lattica_build.operators.polynomials.h_poly_eval import HomPolyEval
from lattica_build.operators.slots.h_sum_slots import HomSumSlots
from lattica_build.params.bootstrapping_params import BootstrappingParams
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


class _KeptSumCombinedWithCustom(HomOp):
    """sum(x) kept in every slot (period 1), combined with the additional input z."""

    def __init__(self, combine: HomOp, keep_sum: bool = True):
        super().__init__()
        self.keep, self.combine, self.keep_sum = HomSumSlots(reduce_output=False), combine, keep_sum

    def forward(self, x, z):
        return self.combine(self.keep(x) if self.keep_sum else x, z)


def _serialized_with_custom_input(custom_n_slots: dict, hom: HomOp | None = None) -> dict:
    """The graph of a pipeline on x (32 slots of a 256-slot ring) and an additional input z."""
    params = HomParams(n=2 ** 9, full_q_list_precision=((60, 30),) * 4, pt_scale=2 ** 30, num_special_primes=1,
                       n_slots=32)
    pipeline = HomomorphicPipeline(hom=hom or _KeptSumCombinedWithCustom(HomMul()),
                                   input_shape={'x': (32,), 'z': (32,)}, custom_n_slots=custom_n_slots)
    return json.loads(pipeline.serialize(params)[0])


@pytest.mark.parametrize('custom_n_slots, expected', [({}, 256), ({'z': None}, 256), ({'z': 32}, 32)],
                         ids=['left_out', 'none', 'given'])
def test_custom_input_n_slots_is_an_int_from_serialization_on(custom_n_slots, expected):
    """Left out or None is the full ring, for the trace (the product's period) and the backend alike."""
    graph = _serialized_with_custom_input(custom_n_slots)
    assert graph['custom_n_slots'] == {'z': expected}
    assert graph['pipeline_sections']['hom']['body_output']['n_slots'] == expected


@pytest.mark.parametrize('n_slots, message', [(3, 'power of two'), (512, 'larger than the 256 available slots')])
def test_custom_input_n_slots_is_checked_when_serializing(n_slots, message):
    with pytest.raises(ValueError, match=message):
        _serialized_with_custom_input({'z': n_slots})


@pytest.mark.parametrize('combine', [HomAdd, HomMul])
def test_primary_input_does_not_combine_with_a_full_ring_custom_input(combine):
    """x on 32 slots and z left on the full ring: rejected when the graph is built."""
    with pytest.raises(ValueError, match=r'Cannot combine ciphertexts of periods \(n_slots\) 32, 256'):
        _serialized_with_custom_input({}, _KeptSumCombinedWithCustom(combine(), keep_sum=False))
