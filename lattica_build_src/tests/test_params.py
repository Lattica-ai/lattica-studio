import json

import pytest

from lattica_build.base_classes.hom_pipeline import HomomorphicPipeline
from lattica_build.operators.composite.sequential import SequentialHomOp
from lattica_build.operators.fhe.h_bootstrap import Bootstrap
from lattica_build.operators.slots.h_sum_slots import HomSumSlots
from lattica_build.params.bootstrapping_params import BOOT_CONFIGURATION, BOOT_PARAMS_PER_SUBRING, BootstrappingParams
from lattica_build.params.level_and_scale_tracing import ModulusChain
from lattica_build.params.params import HomParams


def _boot_params_dict(n=1024, **extra):
    """Bootstrapping params with the chain for bootstraps on the whole ring."""
    boot = BootstrappingParams.for_slot_counts(True, False, {n // 2})
    return dict(n=n, full_q_list_precision=((60,),), pt_scale=2 ** 30, num_special_primes=7, bootstrapping=True,
                boot_dft_levels=boot.dft_levels, boot_cts_scale=boot.cts_scale, **extra)


def test_bootstrapping_params_without_a_chain_are_refused():
    d = _boot_params_dict()
    del d["boot_dft_levels"], d["boot_cts_scale"]
    with pytest.raises(ValueError, match="the bootstrap chain needs dft_levels"):
        HomParams.from_dict(d)


@pytest.mark.parametrize("flags", [(True, True), (False, False), (True, False)])
def test_boot_flags_round_trip(flags):
    """real_boot and slim_boot load into the params and their boot params, and round-trip."""
    params = HomParams.from_dict(_boot_params_dict(real_boot=flags[0], slim_boot=flags[1]))
    assert (params.real_boot, params.slim_boot) == flags
    assert (params.boot_params.real_boot, params.boot_params.slim_boot) == flags
    assert HomParams.from_dict(params.to_dict()).to_dict() == params.to_dict()


@pytest.mark.parametrize("flags", [("false", False), (True, 0), (None, False)])
def test_boot_flags_must_be_bools(flags):
    with pytest.raises(ValueError):
        BootstrappingParams(*flags, dft_levels=4, cts_scale=47)


def test_dft_table_splits_cover_the_fft_stages():
    """Every entry splits its log2(n_slots) = log2(subring) - 1 stages into levels."""
    for log_n_subring, (split, cts_bits, log_delta) in BOOT_PARAMS_PER_SUBRING.items():
        assert sum(split) == log_n_subring - 1 and min(split) >= 1
        BootstrappingParams.get_q_list_precision(cts_bits, len(split))   # a valid prime size
        assert 1 <= log_delta < BOOT_CONFIGURATION['eval_mod_scale']


@pytest.mark.parametrize("slot_counts, levels, cts_scale", [
    ({2 ** 15}, 4, 47),
    ({2 ** 15, 2 ** 5}, 4, 55),     # the 32-slot bootstrap's one level needs a wider prime
    ({2 ** 9, 2 ** 4}, 2, 54),
])
def test_chain_serves_every_slot_count(slot_counts, levels, cts_scale):
    """for_slot_counts: the deepest split's levels, the widest CtS prime; the chain round-trips."""
    boot = BootstrappingParams.for_slot_counts(True, False, slot_counts)
    assert (boot.dft_levels, boot.cts_scale) == (levels, cts_scale)
    params = HomParams.from_dict(_boot_params_dict(n=2 ** 16, n_slots=max(slot_counts)))
    params.boot_params = boot
    loaded = HomParams.from_dict(params.to_dict())
    assert ModulusChain(loaded).full_q_list == ModulusChain(params).full_q_list


def test_slot_count_without_bootstrap_parameters_is_refused():
    with pytest.raises(ValueError, match="no bootstrap parameters"):
        BootstrappingParams.for_slot_counts(True, False, {2 ** 20})


@pytest.mark.parametrize("chain_slots, boot_slots", [
    ({2 ** 6}, 2 ** 15),    # (3, 3) at 44 bits: the 2^15 split needs 4 levels
    ({2 ** 6}, 2 ** 7),     # same level count, but the 2^7 split needs a 45-bit CtS prime
    ({2 ** 15}, 2 ** 5),    # 4 levels at 47 bits: the 32-slot one-level split needs 55
])
def test_dft_split_refuses_a_chain_that_does_not_fit(chain_slots, boot_slots):
    """A bootstrap whose own split the chain does not fit is refused, not squeezed into the chain's levels."""
    boot = BootstrappingParams.for_slot_counts(True, False, chain_slots)
    with pytest.raises(ValueError, match="does not fit"):
        boot.dft_split(boot_slots)
    assert BootstrappingParams.for_slot_counts(True, False, chain_slots | {boot_slots}).dft_split(boot_slots) \
        == BOOT_PARAMS_PER_SUBRING[(2 * boot_slots).bit_length() - 1][0]


def test_chain_serves_the_slot_counts_the_pipeline_bootstraps():
    """The chain comes from the slot counts the Bootstrap ops see: here the input has 2^15 slots, but the bootstrap
    sees 32 after HomSumSlots, whose one-level split needs 55-bit CtS primes. The input's 2^15, never bootstrapped,
    adds no levels."""
    pipeline = HomomorphicPipeline(hom=SequentialHomOp(HomSumSlots(k=2 ** 10), Bootstrap()), input_shape=(32,))
    params = HomParams(n=2 ** 16, full_q_list_precision=((60,),) * 4, pt_scale=2 ** 40, num_special_primes=7)
    graph, _ = pipeline.serialize(params)
    expected = BootstrappingParams.for_slot_counts(params.real_boot, params.slim_boot, {32})
    assert (params.boot_params.dft_levels, params.boot_params.cts_scale) == (expected.dft_levels, expected.cts_scale)
    assert params.boot_params.dft_split(32) == BOOT_PARAMS_PER_SUBRING[6][0]
    assert json.loads(graph)["modulus_chain"]["num_rows"] == ModulusChain(params).num_rows
