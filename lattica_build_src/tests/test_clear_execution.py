import itertools
import json
import pickle
import random
from pathlib import Path
from unittest import SkipTest

import torch
import pytest

from lattica_build.base_classes.hom_op_tracer import Tracer
from lattica_build.base_classes.hom_value import HomValue
from lattica_build.base_classes.hom_pipeline import HomomorphicPipeline
from lattica_build.operators.arithmetic.h_add import HomAdd
from lattica_build.operators.arithmetic.h_const_mul import HomConstMul
from lattica_build.operators.arithmetic.h_mul import HomMul
from lattica_build.operators.client_ops import Clamp, Softmax
from lattica_build.operators.composite.sequential import SequentialHomOp
from lattica_build.operators.fhe.h_bootstrap import Bootstrap
from lattica_build.operators.fhe.h_mod_switch import HomModSwitch
from lattica_build.operators.fhe.h_ring_switch import HomRingSwitch
from lattica_build.operators.ml.h_conv import HomConv
from lattica_build.operators.polynomials.h_poly_eval_base import HomPolyEvalBase
from lattica_build.operators.shape.h_reshape import HomReshape
from lattica_build.operators.shape.h_slice import HomSlice
from lattica_build.operators.shape.h_squeeze import HomSqueeze
from lattica_build.operators.shape.h_unsqueeze import HomUnsqueeze
from lattica_build.operators.slots.h_conj import HomConj
from lattica_build.operators.slots.h_expand import HomExpand
from lattica_build.operators.slots.h_rotate_sum import HomRotateSum
from lattica_build.operators.slots.h_running_sum import HomRunningSum
from lattica_build.operators.slots.h_sum_slots import HomSumSlots
from lattica_build.params.params import HomParams
from lattica_build.serialization.clear_execution import run_clear_section


def test_leaf_clear_execution_and_data():
    x = torch.arange(6, dtype=torch.float32).reshape(2, 3)

    add = HomAdd()
    assert torch.equal(add(x, x), x + x)

    mul = HomMul(axis_sum=1)
    assert torch.equal(mul(x, x), (x * x).sum(dim=1))

    const_mul = HomConstMul(dims=(3,))
    const_mul.set_data(torch.tensor([2.0, 3.0, 4.0]))
    assert torch.equal(const_mul(x), x * torch.tensor([2.0, 3.0, 4.0]))

    assert torch.equal(HomReshape((3, 2))(x), x.reshape(3, 2))
    assert torch.equal(HomSlice(1, slice(1, None))(x), x[:, 1:])
    assert torch.equal(HomUnsqueeze(0)(x), x.unsqueeze(0))
    assert torch.equal(HomSqueeze(0)(x[:1]), x[:1].squeeze(0))


def test_slot_client_and_fhe_clear_execution():
    x = torch.arange(8, dtype=torch.float32)
    assert torch.equal(HomExpand()(x), x.unsqueeze(1).expand(8, 8))
    z = torch.complex(x, -x)
    assert torch.equal(HomConj()(z), z.conj())
    assert torch.equal(HomRotateSum((1,), add_identity_rotation=True)(x), x + x.roll(-1, -1))
    assert torch.equal(HomRunningSum()(x), x.cumsum(-1))
    assert torch.equal(HomSumSlots(k=4)(x), x.reshape(4, 2).sum(0).repeat(4))
    assert torch.equal(HomModSwitch()(x), x)
    assert torch.equal(Bootstrap()(x), x)
    assert torch.equal(HomRingSwitch(10)(x), x)
    assert torch.equal(HomomorphicPipeline(hom=SequentialHomOp(Clamp(0, 8)), input_shape=x.shape).forward_clear(x), x)

    probabilities = Softmax(-1)(x)
    assert torch.allclose(probabilities.sum(), torch.tensor(1.0))
    assert torch.equal(Clamp(2, 5)(x), x.clamp(2, 5))


@pytest.mark.parametrize('axis', [0, 1, -1, -2])
def test_expand_clear_accepts_every_vector_insertion_axis(axis):
    x = torch.tensor([2., 3., 5.])
    expected = torch.tensor([[2., 3., 5.]]).repeat(3, 1)
    if axis in (0, -2):
        expected = expected.T
    torch.testing.assert_close(HomExpand(expand_axis=axis)(x), expected)


def test_expand_clear_preserves_batch_dimensions():
    x = torch.arange(6.).reshape(2, 3)
    result = HomExpand(expand_axis=1)(x)
    assert result.shape == (2, 3, 3)
    for index in range(3):
        torch.testing.assert_close(result[:, :, index], x)
    with pytest.raises(IndexError):
        HomExpand(expand_axis=3)(x)


def test_serialized_clear_execution_does_not_expand_to_ring_size():
    pipeline = HomomorphicPipeline(hom=HomExpand(expand_axis=1), input_shape=(8,))
    params = HomParams(n=256, full_q_list_precision=((60,),) * 4,
                       pt_scale=2**30, num_special_primes=1, n_slots=8)
    graph, _ = pipeline.serialize(params)
    section = json.loads(graph)['pipeline_sections']['hom']
    x = torch.arange(8.)
    result = run_clear_section(section, [x], lambda path: pytest.fail('No weights expected'))
    torch.testing.assert_close(result, x[None, :].repeat(8, 1))


def test_running_sum_clear_uses_only_the_logical_axis():
    x = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
    op = HomRunningSum(cumsum_axis=0, k=64, stage_sizes=[8, 8])
    torch.testing.assert_close(op(x), torch.tensor([[1., 2., 3.], [5., 7., 9.]]))


@pytest.mark.parametrize('include_identity', [False, True])
@pytest.mark.parametrize('perform_sum', [False, True])
def test_rotate_sum_clear_counts_each_requested_view_once(include_identity, perform_sum):
    x = torch.tensor([1., 2., 4.])
    views = [torch.tensor([2., 4., 1.])]
    if perform_sum or include_identity:
        views.insert(0, x)
    expected = sum(views) if perform_sum else torch.stack(views)
    torch.testing.assert_close(HomRotateSum([1], perform_sum, include_identity)(x), expected)


@pytest.mark.parametrize('batched', [False, True])
def test_convolution_clear_requires_packed_images(batched):
    image = torch.ones((1, 5, 5), dtype=torch.float64)
    op = HomConv(1, 1, kernel_size=3, stride=2, padding=1)
    op.set_data(torch.ones((1, 1, 3, 3), dtype=torch.float64))
    if batched:
        image = image.unsqueeze(0)
    with pytest.raises(ValueError, match='unbatched packed tensor'):
        op(image)


def test_convolution_clear_groups_dilation_and_bias():
    image = torch.arange(2 * 6 * 8., dtype=torch.float64).reshape(2, 6, 8)
    weight = torch.arange(18., dtype=torch.float64).reshape(2, 1, 3, 3)
    bias = torch.tensor([2., -3.], dtype=torch.float64)
    op = HomConv(2, 2, 3, stride=(2, 1), padding=1, groups=2, bias=True,
                 dilation=2, image_hw=(6, 8))
    op.set_data(weight, bias)
    dense = torch.nn.functional.conv2d(image, weight, bias, padding=2,
                                       dilation=2, groups=2)
    expected = torch.zeros_like(dense)
    expected[:, ::4, ::2] = dense[:, ::4, ::2]
    torch.testing.assert_close(op(image.flatten(1)), expected.flatten(1))


def test_polynomial_clear_prepends_coefficient_batch_dimensions():
    x = torch.tensor([[0., .25], [.5, 1.]], dtype=torch.float64)
    # Descending Chebyshev coefficients for x and 2*x**2 - 1.
    op = HomPolyEvalBase([[0., 1., 0.], [1., 0., 0.]])
    torch.testing.assert_close(op(x), torch.stack((x, 2 * x.square() - 1)))


def test_composites_and_pipeline_clear_execution():
    scale = HomConstMul(dims=(2,))
    scale.set_data(torch.tensor([2.0, 3.0]))
    graph = SequentialHomOp(scale, Clamp(0, 5))
    x = torch.tensor([1.0, 2.0])
    assert torch.equal(graph.forward_clear(x), torch.tensor([2.0, 5.0]))

    pipeline = HomomorphicPipeline(
        client_pre=[Clamp(0, 2)],
        hom=scale,
        client_post=[Clamp(0, 3)],
        input_shape=(2,),
    )
    expected = (x.clamp(0, 2) * torch.tensor([2.0, 3.0])).clamp(0, 3)
    assert torch.equal(pipeline.forward_clear(x), expected)


@pytest.mark.parametrize("c_in,c_out,hw,kernel,stride,padding", [
    (4, 4, (4, 4), 3, 1, 1),     # same layout
    (4, 8, (4, 4), 3, 2, 1),     # strided, re-interleaved to 8 channels
    (8, 10, (1, 1), 1, 1, 0),    # the FC head: 10 channels in 16 slots
])
def test_mux_conv_clear_matches_torch(c_in, c_out, hw, kernel, stride, padding):
    """HomMuxConv in the clear: unpack, conv2d, pack into the output layout."""
    from lattica_build.operators.ml.h_mux_conv import HomMuxConv, MuxLayout

    torch.manual_seed(0)
    x = torch.randn(c_in, *hw, dtype=torch.float64)
    weight = torch.randn(c_out, c_in, kernel, kernel, dtype=torch.float64)
    bias = torch.randn(c_out, dtype=torch.float64)
    conv = HomMuxConv((c_out, c_in, kernel, kernel), hw, stride=stride, padding=padding)
    conv.set_data(weight)
    vec = MuxLayout(c_in, *hw).pack(x)

    expected = torch.nn.functional.conv2d(x[None], weight, stride=stride, padding=padding)[0]
    out = conv.forward_clear(vec)
    assert torch.allclose(out[MuxLayout(c_out, *conv.output_hw).slot_map()], expected)
    assert int((out != 0).sum()) <= expected.numel()

    conv_bias = HomMuxConv((c_out, c_in, kernel, kernel), hw, stride=stride, padding=padding, bias=True)
    conv_bias.set_data(weight, bias)
    out = conv_bias.forward_clear(vec)
    assert torch.allclose(out[MuxLayout(c_out, *conv.output_hw).slot_map()], expected + bias[:, None, None])


def _ciphertext_slot_sums(x, n_slots, ks, internal_n):
    """Independent oracle: x zero-padded to its period n_slots and repeated over the
    ring's internal_n slots; each SumSlots(k) adds the slots rotated by every multiple of
    n_slots / k, leaving period n_slots / k. Read back at x's positions."""
    slots = torch.nn.functional.pad(x, (0, n_slots - len(x))).repeat(internal_n // n_slots)
    for k in ks:
        n_slots //= k
        slots = sum(slots.roll(-m * n_slots) for m in range(k))
    return slots[:len(x)]


def test_sum_slots_clear_matches_the_ciphertext():
    """The clear graph sums the slots the ciphertext does: 28 values zero-padded to their
    32-slot period, so padding and data mix in every sum."""
    x = torch.randn(28, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    pipeline = HomomorphicPipeline(hom=HomSumSlots(k=8), input_shape=(28,))
    params = HomParams(n=2 ** 13, full_q_list_precision=((61,), (61, 30), (61,)),
                       pt_scale=2 ** 44, num_special_primes=1, n_slots=32)
    graph, _ = pipeline.serialize(params)
    section = json.loads(graph)['pipeline_sections']['hom']
    result = run_clear_section(section, [x], lambda path: pytest.fail('No weights expected'))
    torch.testing.assert_close(result, _ciphertext_slot_sums(x, 32, (8,), params.internal_n))


def test_sum_slots_output_period():
    """The output's period is the input's (internal_n without one) over k."""
    value = HomValue(tensor_shape=(32,), n_axis=-1)
    assert HomSumSlots(k=8).infer_output_shape(value.make_copy(n_slots=4096)).n_slots == 512
    assert HomSumSlots(k=8).infer_output_shape(value, internal_n=4096).n_slots == 512


def test_clear_execution_does_not_trace():
    tensors = {}
    with Tracer(tensors) as tracer:
        result = HomAdd().forward_clear(torch.tensor([1.0]), torch.tensor([2.0]))
        composite_result = SequentialHomOp(Clamp(0, 2)).forward_clear(torch.tensor([3.0]))
    assert torch.equal(result, torch.tensor([3.0]))
    assert torch.equal(composite_result, torch.tensor([2.0]))
    assert tracer.recorded_ops == []
    assert "value" not in Tracer.serialize_hom_val(HomValue(tensor_shape=(1,)))


def test_mnist_composite_example_predicts_known_digits():
    from lattica_build.examples.advanced import mnist_fc

    examples_data = (
        Path(__file__).resolve().parents[1]
        / "lattica_build/examples/advanced/data"
    )
    batch_path = examples_data / "mnist_test_batch.pt"
    if not batch_path.exists():
        raise SkipTest("MNIST test fixture is not available")

    batch = torch.load(batch_path, weights_only=True, map_location="cpu")
    images = batch["images"]
    labels = batch["labels"]
    sample_indices = torch.randperm(images.shape[0])[:mnist_fc.BATCH]
    images = images[sample_indices]
    labels = labels[sample_indices]

    pipeline = mnist_fc.Pipeline()
    predictions = pipeline.build_pipeline().forward_clear(images).argmax(dim=-1)
    accuracy = (predictions == labels).float().mean()
    assert accuracy >= 0.95, f"clear MNIST accuracy was only {accuracy.item():.1%}"


def test_bitonic_sort_composite_example_sorts_input():
    from lattica_build.examples.advanced import bitonic_sort
    example = bitonic_sort.Pipeline()
    pipeline = example.build_pipeline()
    values = example.get_example_pt()
    result = pipeline.forward_clear(values, hom_params=example.build_params())
    expected = example.compute_expected(values)
    example.verify_results(result, expected)


def test_resnet_composite_example_matches_pretrained_model():
    from lattica_build.examples.advanced import resnet20
    from lattica_build.operators.ml.h_mux_conv import MuxLayout

    examples_data = (
        Path(__file__).resolve().parents[1]
        / "lattica_build/examples/advanced/data"
    )
    cifar_batch_path = examples_data / "cifar10_test_batch"
    if not cifar_batch_path.exists():
        raise SkipTest("local CIFAR-10 test fixture is not available")

    with cifar_batch_path.open("rb") as batch_file:
        batch = pickle.load(batch_file, encoding="bytes")

    sample_index = random.Random(0).randrange(len(batch[b"labels"]))
    normalized_image, _ = resnet20.Pipeline.load_cifar_sample(sample_index)
    reference_model = torch.hub.load(
        "chenyaofo/pytorch-cifar-models",
        "cifar10_resnet20",
        pretrained=True,
        verbose=False,
    ).eval()
    example = resnet20.Pipeline()
    pipeline = example.build_pipeline()
    params = example.build_params()
    with torch.no_grad():
        expected_class = reference_model(
            normalized_image.to(torch.float32).reshape(1, 3, 32, 32)
        ).argmax(dim=-1).item()

    # The client packs the normalized image into the mux layout; there is no client_pre.
    packed_image = MuxLayout(*resnet20.STEM_LAYOUT, resnet20.SUBRING_SLOTS).pack(
        resnet20.pad_channels(normalized_image, resnet20.STEM_CHANNELS)
    )
    result = pipeline.forward_clear(
        packed_image,
        hom_params=params,
    )
    # The head's 1x1 output layout puts logit c at slot c.
    predicted_class = result[..., :10].argmax().item()
    assert predicted_class == expected_class, (
        f"clear ResNet predicted {predicted_class} for CIFAR sample "
        f"{sample_index}, expected pretrained-model class {expected_class}"
    )


def test_mux_layout_packs_each_channel_at_its_slot():
    """Sides that are not powers of 2: the width pads to 8."""
    from lattica_build.operators.ml.h_mux_conv import MuxLayout

    x = torch.arange(1, 4 * 3 * 5 + 1, dtype=torch.float64).reshape(4, 3, 5)
    vec = MuxLayout(4, 3, 5, 128).pack(x)
    for c, i, j in itertools.product(range(4), range(3), range(5)):
        assert vec[(i * 8 + j) * 4 + c] == x[c, i, j]
    assert int((vec != 0).sum()) == x.numel()

