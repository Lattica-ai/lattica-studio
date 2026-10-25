"""Clear execution of the slot operators on the ciphertext's slot layout.

Each case runs the traced pipeline (HomomorphicPipeline.forward_clear with hom_params) and compares it with a
slot-by-slot simulation of the ciphertext, written independently of operators.slots.ring_utils: pack the slot axis as
encryption does, apply the operator to the slots, read back the slots that hold the values, dropping the padding
and the repeats.
"""

import pytest
import torch

from lattica_build.base_classes.hom_pipeline import HomomorphicPipeline
from lattica_build.operators.composite.sequential import SequentialHomOp
from lattica_build.operators.slots.h_expand import HomExpand
from lattica_build.operators.slots.h_rotate_sum import HomRotateSum
from lattica_build.operators.slots.h_running_sum import HomRunningSum
from lattica_build.operators.slots.h_sum_slots import HomSumSlots
from lattica_build.params.params import HomParams

N = 2 ** 9  # internal_n = 256


def _params(n_slots=None):
    return HomParams(n=N, full_q_list_precision=((60,),) * 12, pt_scale=2 ** 30, num_special_primes=1,
                     n_slots=n_slots)


def _run(op, x, n_slots=None, n_axis=None):
    params = _params(n_slots)
    pipeline = HomomorphicPipeline(hom=op, input_shape=tuple(x.shape), n_axis=n_axis)
    return pipeline.forward_clear(x, hom_params=params), params


# ---------------- slot-by-slot simulation of the ciphertext

def _pack(values, n_slots, internal_n):
    """The rings (lists of internal_n slots) holding a 1-D slot axis."""
    length = len(values)
    if length <= n_slots:
        period = list(values) + [0.0] * (n_slots - length)
        return [period * (internal_n // n_slots)]
    padded = list(values) + [0.0] * (-length % internal_n)
    return [padded[start:start + internal_n] for start in range(0, len(padded), internal_n)]


def _unpack(rings, length):
    return [slot for ring in rings for slot in ring][:length]


def _per_slot_axis(x, n_axis, fn):
    """Apply fn (a list of slot-axis values -> list) to every 1-D fiber of x along n_axis."""
    moved = x.movedim(n_axis, -1)
    out = [torch.tensor(fn(fiber.tolist()), dtype=x.dtype) for fiber in moved.reshape(-1, moved.shape[-1])]
    return torch.stack(out).reshape(*moved.shape[:-1], -1).movedim(-1, n_axis)


def _rotate(values, r, n_slots, internal_n):
    rings = _pack(values, n_slots, internal_n)
    return _unpack([[ring[(s + r) % internal_n] for s in range(internal_n)] for ring in rings], len(values))


def _sum_slots(values, k, n_slots, internal_n):
    rings = _pack(values, n_slots, internal_n)
    cols = n_slots // k
    out = []
    for ring in rings:
        out.append([sum(ring[(s // n_slots) * n_slots + (s % cols) + t * cols] for t in range(k))
                    for s in range(internal_n)])
    return _unpack(out, len(values))


def _running_sum(blocks, k, internal_n):
    """blocks: rings; slot s is at row s // cols, column s % cols; rows are summed across blocks in the order row r
    of block 0, row r of block 1, ..., then row r + 1."""
    cols = internal_n // k
    out = [[0.0] * internal_n for _ in blocks]
    for c in range(cols):
        total = 0.0
        for r in range(k):
            for b, ring in enumerate(blocks):
                total += ring[r * cols + c]
                out[b][r * cols + c] = total
    return out


# ---------------- RotateSum

@pytest.mark.parametrize('length, n_slots', [(32, 32), (28, 32), (128, 256), (600, 256)])
def test_rotate_sum_rotates_in_padding(length, n_slots):
    x = torch.randn(length, dtype=torch.float64)
    result, params = _run(HomRotateSum((1, 2)), x, n_slots)
    rotated = [_rotate(x.tolist(), r, n_slots, params.internal_n) for r in (1, 2)]
    torch.testing.assert_close(result, x + sum(torch.tensor(view, dtype=torch.float64) for view in rotated))


def test_rotate_sum_on_a_slot_axis_that_is_not_last():
    x = torch.randn(28, 3, dtype=torch.float64)
    result, params = _run(HomRotateSum((1, 2)), x, 32, n_axis=0)
    expected = x + sum(_per_slot_axis(x, 0, lambda v, r=r: _rotate(v, r, 32, params.internal_n)) for r in (1, 2))
    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize('add_identity', [False, True])
def test_rotate_sum_stack_mode(add_identity):
    x = torch.randn(28, dtype=torch.float64)
    result, params = _run(HomRotateSum((1, 3), perform_sum=False, add_identity_rotation=add_identity), x, 32)
    views = [torch.tensor(_rotate(x.tolist(), r, 32, params.internal_n), dtype=torch.float64) for r in (1, 3)]
    if add_identity:
        views.insert(0, x)
    torch.testing.assert_close(result, torch.stack(views))


# ---------------- RunningSum

@pytest.mark.parametrize('length, n_slots', [(256, 256), (200, 256), (600, 256), (28, 32)])
def test_running_sum_on_the_slot_axis(length, n_slots):
    x = torch.randn(length, dtype=torch.float64)
    result, params = _run(HomRunningSum(k=8), x, n_slots)
    rings = _pack(x.tolist(), n_slots, params.internal_n)
    expected = _unpack(_running_sum(rings, 8, params.internal_n), length)
    torch.testing.assert_close(result, torch.tensor(expected, dtype=torch.float64))


def test_running_sum_over_blocks_axis_external():
    x = torch.randn(3, 256, dtype=torch.float64)
    result, params = _run(HomRunningSum(k=8, blocks_axis_external=0), x, n_axis=1)
    blocks = [_pack(row.tolist(), 256, params.internal_n)[0] for row in x]
    expected = torch.tensor(_running_sum(blocks, 8, params.internal_n), dtype=torch.float64)
    torch.testing.assert_close(result, expected)


def test_running_sum_with_the_slot_axis_first():
    x = torch.randn(256, 2, dtype=torch.float64)
    result, params = _run(HomRunningSum(k=8), x, n_axis=0)
    expected = _per_slot_axis(x, 0, lambda v: _unpack(_running_sum(_pack(v, 256, 256), 8, 256), len(v)))
    torch.testing.assert_close(result, expected)


# ---------------- SumSlots

@pytest.mark.parametrize('length, n_slots', [(28, 128), (28, None), (600, 256)])
def test_sum_slots_uses_the_traced_period(length, n_slots):
    x = torch.randn(length, dtype=torch.float64)
    result, params = _run(HomSumSlots(k=8), x, n_slots)
    period = n_slots or params.internal_n
    expected = _sum_slots(x.tolist(), 8, period, params.internal_n)
    torch.testing.assert_close(result, torch.tensor(expected, dtype=torch.float64))


def test_sum_slots_on_a_slot_axis_that_is_not_last():
    x = torch.randn(32, 3, dtype=torch.float64)
    result, params = _run(HomSumSlots(k=8), x, 32, n_axis=0)
    expected = _per_slot_axis(x, 0, lambda v: _sum_slots(v, 8, 32, params.internal_n))
    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize('length', [32, 28])
def test_rotate_sum_after_sum_slots_rotates_the_repeated_sums(length):
    # SumSlots(k=8) leaves its sums repeated every 4 slots over the ring: a rotation by 4 finds the same sums, a
    # rotation by 1 the next sum of the period
    x = torch.randn(length, dtype=torch.float64)
    result, params = _run(SequentialHomOp(HomSumSlots(k=8), HomRotateSum((1, 4))), x, 32)
    sums = torch.tensor(_sum_slots(x.tolist(), 8, 32, params.internal_n), dtype=torch.float64)
    next_sums = torch.tensor([sums[i - i % 4 + (i + 1) % 4] for i in range(length)], dtype=torch.float64)
    torch.testing.assert_close(result, 2 * sums + next_sums)


# ---------------- Expand

@pytest.mark.parametrize('shape, expand_axis, expected', [
    ((8,), 0, lambda x, n: x[:, None].expand(8, n)),          # out[i, s] = x[i]
    ((2, 8), 1, lambda x, n: x[:, :, None].expand(2, 8, n)),  # out[b, i, s] = x[b, i]
])
def test_expand_fills_the_ring_with_each_value(shape, expand_axis, expected):
    x = torch.randn(*shape, dtype=torch.float64)
    result, params = _run(HomExpand(expand_axis), x, 8, n_axis=len(shape) - 1)
    torch.testing.assert_close(result, expected(x, params.internal_n))


# ---------------- Invalid configurations

@pytest.mark.parametrize('axis', [2, -3])
def test_expand_rejects_an_axis_out_of_range(axis):
    x = torch.randn(8, dtype=torch.float64)
    with pytest.raises(ValueError, match=r'Cannot insert an axis at -?\d+ in a tensor of rank 1'):
        _run(HomExpand(axis), x, 8)
    with pytest.raises(ValueError, match=r'expected an axis in \[-2, 1\]'):
        HomExpand(axis)(x)


def test_running_sum_rejects_blocks_on_the_slot_axis():
    with pytest.raises(ValueError, match='blocks_axis_external=1 is the slot axis'):
        _run(HomRunningSum(k=8, blocks_axis_external=1), torch.randn(3, 256, dtype=torch.float64), n_axis=1)


@pytest.mark.parametrize('k', [3, 512])
def test_running_sum_rejects_a_span_that_does_not_divide_the_ring(k):
    with pytest.raises(ValueError, match=f'HomRunningSum needs k={k} to be a power of 2 dividing the 256 slots'):
        _run(HomRunningSum(k=k, stage_sizes=[k]), torch.randn(256, dtype=torch.float64))


@pytest.mark.parametrize('k', [3, 64])
def test_sum_slots_rejects_a_span_that_does_not_divide_the_period(k):
    message = f'HomSumSlots needs k={k} to be a power of 2 dividing the 32 slots of the period'
    with pytest.raises(ValueError, match=message):
        _run(HomSumSlots(k=k), torch.randn(28, dtype=torch.float64), 32)
