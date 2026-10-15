import pytest

from lattica_build.params.bootstrapping_params import BOOT_CONFIGURATION, BOOT_PARAMS_PER_SUBRING, BootstrappingParams
from lattica_build.params.level_and_scale_tracing import ModulusChain
from lattica_build.params.params import HomParams


def _boot_params_dict(n=1024, **extra):
    return dict(n=n, full_q_list_precision=((60,),), pt_scale=2 ** 30, num_special_primes=7,
                bootstrapping=True, **extra)


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
    """Every entry splits its log2(n_slots) = log2(subring) - 1 stages into at most 4 levels."""
    for log_n_subring, (split, cts_bits, log_delta) in BOOT_PARAMS_PER_SUBRING.items():
        assert sum(split) == log_n_subring - 1 and 1 <= len(split) <= 4
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
    assert (boot.n_cts_levels, boot.n_stc_levels, boot.cts_scale) == (levels, levels, cts_scale)
    params = HomParams.from_dict(_boot_params_dict(n=2 ** 16, n_slots=max(slot_counts)))
    params.boot_params = boot
    loaded = HomParams.from_dict(params.to_dict())
    assert ModulusChain(loaded).full_q_list == ModulusChain(params).full_q_list


def test_slot_count_without_bootstrap_parameters_is_refused():
    with pytest.raises(ValueError, match="no bootstrap parameters"):
        BootstrappingParams.for_slot_counts(True, False, {2 ** 20})
