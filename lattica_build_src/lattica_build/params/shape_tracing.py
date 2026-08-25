"""See `params/README.md` for usage details."""

import math
from typing import Sequence, Tuple

import torch

from lattica_build.base_classes.hom_value import HomValue, TensorShape


def to_pos_axis(axis: int, shape: TensorShape) -> int:
    if axis >= 0:
        return axis
    else:
        return axis + len(shape)

def resolve_n_axis(tensor_shape: TensorShape, n_axis: int | None, internal_n: int | None) -> int | None:
    rank = len(tensor_shape)

    if n_axis is not None:
        return to_pos_axis(n_axis, tensor_shape)

    if internal_n is None:  # No internal_n provided, cannot infer n-axis
        return None

    pads = [
        (internal_n - dim) % internal_n
        for dim in tensor_shape
    ]

    return int(min(range(rank), key=lambda i: pads[i]))

def to_neg_axis(axis: int, shape: TensorShape) -> int:
    if axis < 0:
        return axis
    rank = len(tuple(shape))
    return axis - rank

def infer_broadcast_output_shape(*inputs: Sequence[HomValue], dims: TensorShape | None = None) -> HomValue:
    shapes_to_broadcast = list(input.tensor_shape for input in inputs)
    if dims is not None:
        shapes_to_broadcast.append(dims)
    output_shape = torch.broadcast_shapes(*shapes_to_broadcast)

    if inputs[0].n_axis is None:
        return inputs[0].make_copy(tensor_shape=output_shape)

    # n_axis_neg is identical for the input and output shapes.
    n_axis_neg = to_neg_axis(inputs[0].n_axis, inputs[0].tensor_shape)

    # if there is more than one HomValue, ensure they all have the same n_axis
    for input in inputs:
        input_n_axis_neg = to_neg_axis(input.n_axis, input.tensor_shape)
        if not input_n_axis_neg == n_axis_neg:
            raise ValueError(f"Cannot broadcast ciphertexts with different n_axis,"
                             f"but got {input_n_axis_neg} and {n_axis_neg}.")
        if not input.tensor_shape[input_n_axis_neg] == inputs[0].tensor_shape[n_axis_neg]:
            raise ValueError(f"Cannot broadcast ciphertexts with different n_axis sizes,"
                             f"but got {input.tensor_shape[input_n_axis_neg]} and "
                             f"{inputs[0].tensor_shape[n_axis_neg]}.")

    return inputs[0].make_copy(tensor_shape=output_shape, n_axis=n_axis_neg)

def infer_insert_axis_output_shape(input: HomValue, axis: int, size: int) -> HomValue:
    output_shape = (*input.tensor_shape[:axis], size, *input.tensor_shape[axis:])

    if input.n_axis is None:
        return input.make_copy(tensor_shape=output_shape)

    return input.make_copy(
        tensor_shape=output_shape,
        n_axis=input.n_axis + (1 if axis <= input.n_axis else 0)
    )

def infer_remove_axis_output_shape(input: HomValue, axis: int) -> HomValue:
    axis = to_pos_axis(axis, input.tensor_shape)
    if axis == input.n_axis:
        raise ValueError(f"Cannot remove {axis=} because n_axis == axis.")

    output_shape = input.tensor_shape[:axis] + input.tensor_shape[axis + 1:]

    if input.n_axis is None:
        return input.make_copy(tensor_shape=output_shape)

    return input.make_copy(
        tensor_shape=output_shape,
        n_axis=input.n_axis - (1 if axis < input.n_axis else 0),
    )

def _is_full_slice(entry) -> bool:
    return (
        isinstance(entry, slice)
        and entry.start is None
        and entry.stop is None
        and (entry.step is None or entry.step == 1)
    )


def expand_ellipsis(key, ndim: int) -> tuple:
    if key is Ellipsis:
        return (slice(None),) * ndim
    if not isinstance(key, tuple):
        key = (key,)

    ellipsis_count = sum(1 for entry in key if entry is Ellipsis)
    if ellipsis_count > 1:
        raise ValueError("HomSlice indexer may contain at most one Ellipsis.")

    if ellipsis_count == 0:
        if len(key) > ndim:
            raise ValueError(
                f"HomSlice indexer specifies {len(key)} axes but tensor rank is {ndim}."
            )
        # NumPy/PyTorch: short indexers are padded with full slices on the right.
        expanded = key + (slice(None),) * (ndim - len(key))
    else:
        ellipsis_idx = key.index(Ellipsis)
        n_specified = len(key) - 1
        if n_specified > ndim:
            raise ValueError(
                f"HomSlice indexer specifies {n_specified} axes but tensor rank is {ndim}."
            )
        n_fill = ndim - n_specified
        expanded = key[:ellipsis_idx] + (slice(None),) * n_fill + key[ellipsis_idx + 1:]

    if len(expanded) != ndim:
        raise ValueError(
            f"HomSlice indexer length {len(expanded)} does not match tensor rank {ndim}."
        )
    return expanded


def resolve_single_axis_index(key, ndim: int) -> tuple[int, int | slice]:
    """Resolve a torch-style indexer to a single (dim, int|slice) HomSlice key."""
    expanded = expand_ellipsis(key, ndim)
    active = [(i, entry) for i, entry in enumerate(expanded) if not _is_full_slice(entry)]
    if len(active) == 0:
        raise ValueError("HomSlice requires exactly one active axis, but indexer is all full slices.")
    if len(active) > 1:
        dims = [i for i, _ in active]
        raise ValueError(
            f"HomSlice supports only a single active axis, but indexer touches axes {dims}."
        )
    dim, entry = active[0]
    if not isinstance(entry, (int, slice)):
        raise TypeError(
            f"HomSlice only supports int or slice indexing, got {type(entry).__name__}."
        )
    return dim, entry


def infer_reduce_axis_output_shape(input: HomValue, dim: int, keep_dim: bool) -> HomValue:
    res = infer_remove_axis_output_shape(input, dim)
    if keep_dim:
        res = infer_insert_axis_output_shape(res, dim, 1)
    return res
