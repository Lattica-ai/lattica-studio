"""See `operators/fhe/README.md` for usage details."""

from typing import Optional

from lattica_build.base_classes.hom_op import HomOp
from lattica_build.base_classes.hom_value import HomValue
from lattica_build.params.level_and_scale_tracing import init_levels_after_bootstrap
from lattica_build.serialization.hom_op_pb2 import HomOpType

class HomRingSwitch(HomOp):
    """Base ring-switch operator for subring-packed ciphertexts.

    Args:
        log_n_subring: Log dimension of the subring where the input ciphertext
            is encrypted.
        log_n_boot_subring: Log dimension of the subring used for plaintext
            packing. If omitted, backend defaults to `log_n_subring`.
    """

    OP_TYPE = HomOpType.RingSwitch

    def __init__(self, log_n_subring: int, log_n_boot_subring: Optional[int] = None) -> None:
        super().__init__()
        self.log_n_subring = log_n_subring
        self.log_n_boot_subring = log_n_boot_subring

    def infer_output_shape(
        self,
        input: HomValue,
        internal_n: int | None = None,
        **kwargs,
    ) -> HomValue:
        """Replace packed axis size with full-ring `internal_n` shape."""
        if internal_n is None:
            raise ValueError("RingSwitch output shape requires internal_n.")
        if input.n_axis is None:
            raise ValueError("RingSwitch requires an input with an n-axis.")
        axis = input.n_axis
        output_shape = input.tensor_shape[:axis] + (internal_n,) + input.tensor_shape[axis + 1:]
        return input.make_copy(tensor_shape=output_shape)

    def infer_output_level_and_scale(
        self,
        input: HomValue,
        hom_params=None,
        **kwargs,
    ) -> HomValue:
        """Refresh level state after switch while preserving input pt-scale."""
        # The switch bootstraps the sub-ring ciphertext, so level consumption restarts
        # from the refreshed chain. The bootstrap targets the input scale, so pt_scale
        # carries over unchanged.
        return init_levels_after_bootstrap(input, hom_params)
