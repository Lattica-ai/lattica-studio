"""See `operators/shape/README.md` for usage details."""

import math
from collections.abc import Sequence
from operator import index

import torch

from lattica_build.base_classes.hom_op import HomOp
from lattica_build.base_classes.hom_value import HomValue
from lattica_build.params.level_and_scale_tracing import infer_optional_modswitch
from lattica_build.params.shape_tracing import infer_n_batch
from lattica_build.serialization.hom_op_pb2 import HomOpType


class HomBatchPermute(HomOp):
    """Moves values between the plaintexts of a batch (`HomParams.batch_size`): the batch, B plaintexts of J values,
    is the tensor [B, J], reshaped to ``view`` and permuted by ``perm``. The first permuted axis is the new batch, and
    the others, flattened, are its plaintexts:

        output = batch.reshape(view).permute(perm).reshape(view[perm[0]], -1)

    For instance, a batch of T tokens of H heads of D values each, a token's values ordered (d, h) at slot H d + h,
    becomes a batch of H heads of the D x T matrix of each, with view (T, D, H) and perm (2, 1, 0); the same op with
    view (H, D, T) and perm (2, 1, 0) takes it back.

    The values move between ciphertexts by rotations and masks, and their count depends on how far each value moves:
    the order of the values in a plaintext sets the cost. In the example, every value of a token's ciphertext moves by
    an amount that depends only on its ciphertext, its target ciphertext and a few low bits of its slot, which takes
    a few rotations per ciphertext. The order of the values comes from the op that writes them, such as the columns
    of a matrix product's weight.

    Takes one level, for the masks.

    Args:
        view: The shape the batch [B, J] is viewed as; its product is B * J.
        perm: A permutation of the axes of ``view``.
    """

    OP_TYPE = HomOpType.BatchPermute

    def __init__(self, view: Sequence[int], perm: Sequence[int]) -> None:
        super().__init__()
        self.view = tuple(index(size) for size in view)
        self.perm = tuple(index(axis) for axis in perm)
        if sorted(self.perm) != list(range(len(self.view))):
            raise ValueError(f"perm {self.perm} is not a permutation of the {len(self.view)} axes of view")

    def infer_output_shape(self, input: HomValue, internal_n: int = None, **kwargs) -> HomValue:
        if input.batch_size is None:
            raise ValueError("HomBatchPermute takes a batch of plaintexts (HomParams.batch_size)")
        if len(input.tensor_shape) != 1:
            raise ValueError(f"HomBatchPermute takes plaintexts of one axis, the slot axis; got {input.tensor_shape}")
        if math.prod(self.view) != input.batch_size * input.tensor_shape[0]:
            raise ValueError(f"view {self.view} does not hold a batch of {input.batch_size} plaintexts of "
                             f"{input.tensor_shape[0]} values")
        batch_size = self.view[self.perm[0]]
        length = math.prod(self.view) // batch_size
        n_slots = 1 << (length - 1).bit_length()
        if n_slots > internal_n:
            raise NotImplementedError(f"a plaintext of {length} values does not fit a ciphertext of {internal_n} slots")
        return input.make_copy(tensor_shape=(length,), n_slots=n_slots, batch_size=batch_size,
                               n_batch=infer_n_batch(internal_n, n_slots, batch_size))

    def infer_output_level_and_scale(self, input: HomValue, hom_params=None, **kwargs) -> HomValue:
        """The masks' plaintext multiplication and its rescale."""
        return infer_optional_modswitch(hom_params, input, with_modswitch=True, rows_budget=None, op_scale_up=None)

    def forward_clear(self, batch: torch.Tensor) -> torch.Tensor:
        """``batch`` holds the whole batch, its plaintexts on the first axis."""
        return batch.reshape(self.view).permute(self.perm).reshape(self.view[self.perm[0]], -1)
