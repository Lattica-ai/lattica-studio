"""See `operators/ml/README.md` for usage details."""

from typing import Sequence

import torch

from lattica_build.base_classes.hom_op import HomOp
from lattica_build.base_classes.hom_value import HomValue, TensorShape
from lattica_build.params.level_and_scale_tracing import infer_optional_modswitch

from lattica_build.params.shape_tracing import infer_broadcast_output_shape, infer_remove_axis_output_shape, \
    to_pos_axis, to_neg_axis
from lattica_build.serialization.hom_op_pb2 import HomOpType

class HomMatMul(HomOp):
    """Base homomorphic matmul-like operator over one multiplication axis.

    The op broadcasts input shape against `dims`, removes `mul_axis`, and
    updates `n_axis` when needed.

    Args:
        dims: Shape of the plaintext multiplicand tensor.
        mul_axis: Axis reduced by the multiply-accumulate.
        new_n_axis: Optional explicit output `n_axis` candidate when
            `mul_axis` matches the input `n_axis`.
        with_modswitch: Whether to apply optional post-op modswitch.
        rows_budget: Optional allowed modulus rows for modswitch inference.
        pt_scale: Optional plaintext scale-up used during inference.
        num_steps: Optional backend hint for staged implementations.
    """

    OP_TYPE = HomOpType.MatMul

    def __init__(
        self,
        dims: TensorShape,
        mul_axis: int = -1,
        new_n_axis: int | None = None,
        with_modswitch: bool = True,
        rows_budget: Sequence[int] | None = None,
        pt_scale: int = None,
        num_steps: int | None = None,
    ) -> None:
        super().__init__()
        self.dims = dims
        self.mul_axis = mul_axis
        self.new_n_axis = new_n_axis
        self.with_modswitch = with_modswitch
        self.rows_budget = rows_budget
        self.pt_scale = pt_scale
        self.num_steps = num_steps

    @staticmethod
    def _choose_matmul_new_n_axis(
            input_tensor_shape: TensorShape,
            dims: TensorShape,
            mul_axis: int,
            new_n_axis: int | None = None,
    ) -> tuple[int, str]:
        """Choose the n-axis surviving a HomMatMul whose mul_axis is the input's n-axis.
        Returns the chosen axis (negative, relative to the broadcasted shape) and the
        implementation kind ('diags' or 'masks') that supports it.
        """
        input_tensor_shape = tuple(input_tensor_shape)
        dims = tuple(dims)
        broadcasted_shape = tuple(torch.broadcast_shapes(input_tensor_shape, dims))
        rank = len(broadcasted_shape)
        padded_input = (1,) * (rank - len(input_tensor_shape)) + input_tensor_shape
        padded_dims = (1,) * (rank - len(dims)) + dims

        mul_axis = to_neg_axis(to_pos_axis(mul_axis, broadcasted_shape), broadcasted_shape)
        if new_n_axis is not None:
            new_n_axis = to_neg_axis(to_pos_axis(new_n_axis, broadcasted_shape), broadcasted_shape)

        # Checking whether MatMulDiags is possible.
        diag_candidates = []
        degenerate_diag_candidates = []

        for ax in range(-rank, 0):
            if ax == mul_axis:
                continue

            if padded_input[ax] == 1 and padded_dims[ax] == broadcasted_shape[ax]:
                if broadcasted_shape[ax] > 1:
                    diag_candidates.append(ax)
                else:
                    degenerate_diag_candidates.append(ax)

        all_candidates = [ax for ax in range(-rank, 0) if ax != mul_axis]
        mask_candidates = [ax for ax in all_candidates if broadcasted_shape[ax] > 1]

        # Normal case: use useful Diags if available.
        if (
                (new_n_axis is None and len(diag_candidates) > 0)
                or new_n_axis in diag_candidates
        ):
            candidates, impl_kind = diag_candidates, 'diags'

        # Degenerate case: no useful Masks axis exists, so use Diags even on size-1 axis.
        elif len(mask_candidates) == 0 and len(degenerate_diag_candidates) > 0:
            candidates, impl_kind = degenerate_diag_candidates, 'diags'

        else:
            candidates, impl_kind = all_candidates, 'masks'

        if new_n_axis is not None:
            if new_n_axis not in candidates:
                raise ValueError(
                    f"new_n_axis={new_n_axis} is not a valid n-axis candidate ({candidates}) "
                    f"for HomMatMul with broadcasted shape {broadcasted_shape}."
                )
            return new_n_axis, impl_kind

        # Choose the largest surviving axis
        return max(candidates, key=lambda ax: broadcasted_shape[ax]), impl_kind

    def infer_output_shape(self, input: HomValue, **kwargs) -> HomValue:
        """Infer output shape after broadcast and `mul_axis` elimination."""
        after_mul = infer_broadcast_output_shape(input, dims=self.dims)

        if input.n_axis is None:
            return infer_remove_axis_output_shape(after_mul, self.mul_axis)

        n_axis = to_pos_axis(after_mul.n_axis, after_mul.tensor_shape)
        mul_axis = to_pos_axis(self.mul_axis, after_mul.tensor_shape)
        if n_axis != mul_axis:
            chosen_axis = n_axis
        else:
            chosen_axis, _ = self._choose_matmul_new_n_axis(input.tensor_shape, self.dims, mul_axis, self.new_n_axis)
            chosen_axis = to_pos_axis(chosen_axis, after_mul.tensor_shape)
        return after_mul.make_copy(
            tensor_shape=after_mul.tensor_shape[:mul_axis] + after_mul.tensor_shape[mul_axis + 1:],
            n_axis=chosen_axis - (1 if mul_axis < chosen_axis else 0)
        )

    def infer_output_level_and_scale(
        self,
        input: HomValue,
        hom_params=None,
        **kwargs,
    ) -> HomValue:
        """Infer optional modswitch and scale-up policy for matmul."""
        return infer_optional_modswitch(
            hom_params,
            input,
            with_modswitch=self.with_modswitch,
            rows_budget=self.rows_budget,
            op_scale_up=self.pt_scale,
        )
