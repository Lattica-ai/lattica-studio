import pytest

from lattica_build.params.bootstrapping_params import BOOT_CONFIGURATION, BootstrappingParams
from lattica_build.params.level_and_scale_tracing import ModulusChain
from lattica_build.params.params import HomParams


def _boot_params_dict(**extra):
    return dict(n=1024, full_q_list_precision=((60,),), pt_scale=2 ** 30, num_special_primes=7,
                bootstrapping=True, **extra)


@pytest.mark.parametrize("variant, flags", [(0, (True, True)), (1, (False, False)), (2, (True, False))])
def test_legacy_boot_variant_maps_to_the_flags(variant, flags):
    """Params saved with bootstrapping_variant (SLIM, COMPLEX, REAL) load as the two flags, and round-trip."""
    params = HomParams.from_dict(_boot_params_dict(bootstrapping_variant=variant))
    assert (params.real_boot, params.slim_boot) == flags
    assert (params.boot_params.real_boot, params.boot_params.slim_boot) == flags
    assert HomParams.from_dict(params.to_dict()).to_dict() == params.to_dict()


@pytest.mark.parametrize("flags", [("false", False), (True, 0), (None, False)])
def test_boot_flags_must_be_bools(flags):
    with pytest.raises(ValueError):
        BootstrappingParams(*flags)


def test_too_few_slots_for_the_dft_levels():
    """log2(n_slots) FFT stages can't fill more DFT levels: the chain is refused, not left with unused primes."""
    n_levels = max(BOOT_CONFIGURATION["n_cts_levels"], BOOT_CONFIGURATION["n_stc_levels"])
    for n_slots, ok in ((2 ** n_levels, True), (2 ** (n_levels - 1), False)):
        params = HomParams.from_dict(_boot_params_dict(n_slots=n_slots))
        if ok:
            ModulusChain(params)
        else:
            with pytest.raises(ValueError, match="DFT levels"):
                ModulusChain(params)
