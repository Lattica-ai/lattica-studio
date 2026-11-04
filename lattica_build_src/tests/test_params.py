import pytest
import numpy as np

from lattica_build.base_classes.hom_pipeline import HomomorphicPipeline
from lattica_build.operators.composite.sequential import SequentialHomOp
from lattica_build.operators.fhe.h_bootstrap import Bootstrap
from lattica_build.operators.polynomials.h_poly_eval import HomPolyEval
from lattica_build.operators.slots.h_sum_slots import HomSumSlots
from lattica_build.params.bootstrapping_params import BootstrappingParams
from lattica_build.params.params import HomParams


def test_chain_serves_the_slot_counts_the_pipeline_bootstraps():
    """The input fills the ring's 2^15 slots; HomSumSlots(k=2^10) folds them to 32, which the bootstrap sees:
    the chain is built for 32."""
    pipeline = HomomorphicPipeline(hom=SequentialHomOp(HomSumSlots(k=2 ** 10), Bootstrap()), input_shape=(2 ** 15,))
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
    ({2 ** 14, 2 ** 13, 2 ** 12}, 3 * 61, True),     # ResNet-20's bootstraps: 64 terms each
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
    # 32, 64, 64 and 64 terms fit; 2**11's 32 diagonals x 16 trace rotations do not
    assert [boot.lcr_first_cts_fits(s, n) for s in (2 ** 15, 2 ** 14, 2 ** 13, 2 ** 12, 2 ** 11)] == \
        [True, True, True, True, False]
    params = HomParams(n=n, full_q_list_precision=((30,),) + ((60, 30),) * 3, pt_scale=2 ** 30, num_special_primes=3,
                       n_slots=min(slot_counts))
    params.boot_params = boot
    d = params.to_dict()
    assert d["boot_lcr_first_cts"] == lcr
    back = HomParams.from_dict({**d, "bootstrapping": True})
    assert back.boot_params.lcr_first_cts == lcr
