"""See `operators/arithmetic/README.md` for usage details."""

from typing import Sequence

from lattica_build.base_classes.hom_op import HomOp
from lattica_build.base_classes.hom_value import HomValue, TensorShape
from lattica_build.params.level_and_scale_tracing import infer_optional_modswitch
from lattica_build.params.shape_tracing import infer_broadcast_output_shape
from lattica_build.serialization.hom_op_pb2 import HomOpType

class HomConstMul(HomOp):
    """Multiply a `HomValue` by a plaintext constant tensor.

    This is a base operator (`OP_TYPE = HomOpType.ConstMul`). The op can
    optionally account for plaintext scale-up and post-op modswitch.

    Args:
        dims: Plaintext constant tensor dimensions used for broadcast inference.
        with_modswitch: Whether to apply optional post-op modswitch.
        rows_budget: Optional allowed modulus rows for modswitch inference.
        pt_scale: Plaintext scale-up contributed by the constant.
    """

    OP_TYPE = HomOpType.ConstMul

    def __init__(
        self,
        dims: TensorShape,
        with_modswitch: bool = True,
        rows_budget: Sequence[int] | None = None,
        pt_scale: int = None,
    ) -> None:
        super().__init__()
        self.dims = dims
        self.with_modswitch = with_modswitch
        self.rows_budget = rows_budget
        self.pt_scale = pt_scale

    def infer_output_shape(self, input: HomValue, **kwargs) -> HomValue:
        """Infer output shape from broadcasting `input` with constant dims."""
        return infer_broadcast_output_shape(input, dims=self.dims)

    def infer_output_level_and_scale(
        self,
        input: HomValue,
        hom_params=None,
        **kwargs,
    ) -> HomValue:
        """Infer optional modswitch and scale-up effects for the constant mul."""
        return infer_optional_modswitch(
            hom_params,
            input,
            with_modswitch=self.with_modswitch,
            rows_budget=self.rows_budget,
            op_scale_up=self.pt_scale,
        )

    def forward_clear(self, input):
        data = self._require_clear_data().to(device=input.device, dtype=input.dtype)
        return input * data
