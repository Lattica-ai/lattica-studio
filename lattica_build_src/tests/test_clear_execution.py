import pickle
import random
from pathlib import Path
from unittest import SkipTest

import torch

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
from lattica_build.operators.shape.h_reshape import HomReshape
from lattica_build.operators.shape.h_slice import HomSlice
from lattica_build.operators.shape.h_squeeze import HomSqueeze
from lattica_build.operators.shape.h_unsqueeze import HomUnsqueeze
from lattica_build.operators.slots.h_expand import HomExpand
from lattica_build.operators.slots.h_rotate_sum import HomRotateSum
from lattica_build.operators.slots.h_running_sum import HomRunningSum
from lattica_build.operators.slots.h_sum_slots import HomSumSlots


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
    assert torch.equal(HomExpand(2)(x), x.unsqueeze(0).expand(2, 8))
    assert torch.equal(HomRotateSum((1,), add_identity_rotation=True)(x), x + x.roll(-1, -1))
    assert torch.equal(HomRunningSum()(x), x.cumsum(-1))
    assert torch.equal(HomSumSlots(k=3)(x), x[:3].sum().expand_as(x))
    assert torch.equal(HomModSwitch()(x), x)
    assert torch.equal(Bootstrap()(x), x)
    assert torch.equal(HomRingSwitch(10)(x), x)
    assert torch.equal(HomomorphicPipeline(hom=SequentialHomOp(Clamp(0, 8)), input_shape=x.shape).forward_clear(x), x)

    probabilities = Softmax(-1)(x)
    assert torch.allclose(probabilities.sum(), torch.tensor(1.0))
    assert torch.equal(Clamp(2, 5)(x), x.clamp(2, 5))


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

    predictions = mnist_fc.build_pipeline().forward_clear(images).argmax(dim=-1)
    accuracy = (predictions == labels).float().mean()
    assert accuracy >= 0.95, f"clear MNIST accuracy was only {accuracy.item():.1%}"


def test_bitonic_sort_composite_example_sorts_input():
    from lattica_build.examples.advanced import bitonic_sort

    values = torch.rand(bitonic_sort.ARRAY_LEN)
    pipeline = bitonic_sort.build_pipeline()
    result = pipeline.forward_clear(values, hom_params=bitonic_sort.build_params())
    expected = torch.sort(values).values.to(result.dtype)
    assert torch.allclose(result[:bitonic_sort.ARRAY_LEN], expected, atol=2e-2)


def test_resnet_composite_example_matches_pretrained_model():
    from lattica_build.examples.advanced import resnet20

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
    raw_image = torch.tensor(
        batch[b"data"][sample_index], dtype=torch.float32
    ).reshape(3, 32, 32)
    reference_model = torch.hub.load(
        "chenyaofo/pytorch-cifar-models",
        "cifar10_resnet20",
        pretrained=True,
        verbose=False,
    ).eval()
    pipeline = resnet20.build_pipeline()
    params = resnet20.build_params()
    with torch.no_grad():
        normalized_image = pipeline.client_pre.forward_clear(
            raw_image, internal_n=params.internal_n
        )
        expected_class = reference_model(
            normalized_image.reshape(1, 3, 32, 32)
        ).argmax(dim=-1).item()

    result = pipeline.forward_clear(
        raw_image,
        hom_params=params,
    )
    predicted_class = result[..., 0].argmax().item()
    assert predicted_class == expected_class, (
        f"clear ResNet predicted {predicted_class} for CIFAR sample "
        f"{sample_index}, expected pretrained-model class {expected_class}"
    )
