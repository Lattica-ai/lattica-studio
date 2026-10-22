import pickle
import random
from pathlib import Path
import numpy as np
import torch

from lattica_build.base_classes.hom_op import HomOp
from lattica_build.base_classes.hom_pipeline import HomomorphicPipeline
from lattica_build.base_classes.hom_value import HomValue
from lattica_build.base_classes.pipeline_wrapper import PipelineWrapper
from lattica_build.operators.fhe.h_bootstrap import Bootstrap
from lattica_build.operators.ml.h_mux_conv import HomMuxConv, HomMuxConvBnFused, MuxLayout
from lattica_build.operators.polynomials.h_poly_eval import HomPolyEval
from lattica_build.operators.slots.h_sum_slots import HomSumSlots
from lattica_build.params.params import HomParams

N = 2 ** 16
SUBRING_SLOTS = 2 ** 14                      # stem tile = 32*32*16 = 16384 < N/2
PT_SCALE = 2 ** 30
Q_LIST_PRECISION = ((60, 30),) * 4
DNUM = 2                                     # hybrid digits over the chain (9 special primes)
RELU_DEG = 119                               # 7 levels with Paterson-Stockmeyer; relu(7) + conv(1) = 8 exactly
STEM_CHANNELS = 16                           # RGB (3) zero-padded up to 16 channels

# Per-stage map (C, H, W) after the stage's downsampling.
STEM_LAYOUT = (16, 32, 32)
LAYER_IN = {1: (16, 32, 32), 2: (16, 32, 32), 3: (32, 16, 16)}

LAYER_OUT = {1: (16, 32, 32), 2: (32, 16, 16), 3: (64, 8, 8)}


# ReLU-domain scaling folded into each conv's BN affine (per block, per conv).
DELTA_INITIAL = 0.1806
DELTAS = [
    [[0.1722, 0.1310], [0.1973, 0.1273], [0.2429, 0.1275]],   # layer 1
    [[0.1906, 0.1232], [0.3359, 0.1161], [0.2869, 0.0893]],   # layer 2
    [[0.2563, 0.1762], [0.2739, 0.1279], [0.2108, 0.0348]],   # layer 3
]

# The official CIFAR-10 test batch, and the normalization the model expects.
CIFAR_BATCH = Path(__file__).parent / "data" / "cifar10_test_batch"
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
# CIFAR-10 labels in the model's output-index order.
CIFAR10_CLASSES = ('airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck')


def make_relu(delta: float) -> HomPolyEval:
    return HomPolyEval(func=lambda x: np.maximum(0, x / delta), degree=RELU_DEG)


def _conv_bn(conv, bn, layout, delta, *, weight=None):
    """A mux convolution with its batch norm and the next ReLU's 1/delta folded in."""
    _, height, width = layout
    weight = conv.weight if weight is None else weight
    op = HomMuxConvBnFused(
        kernel_shape=tuple(weight.shape), image_hw=(height, width),
        stride=conv.stride, padding=conv.padding, dilation=conv.dilation,
    )
    op.set_data(weight, None, bn.running_mean, bn.running_var, bn.weight, bn.bias, bn.eps, delta)
    return op


# A bootstrap sits right before each ReLU: the ReLU takes 7 of its 8 levels and the
# next conv the last one, so every conv runs on the bottom of the chain (2 limbs).
class _MuxStem(HomOp):
    def __init__(self, model):
        super().__init__()
        weight = torch.zeros(STEM_LAYOUT[0], STEM_CHANNELS, 3, 3, dtype=torch.float64)
        weight[:, :3] = model.conv1.weight
        self.convBN = _conv_bn(model.conv1, model.bn1, STEM_LAYOUT,
                              DELTA_INITIAL, weight=weight)
        self.bootstrap = Bootstrap()
        self.relu = make_relu(DELTA_INITIAL)

    def forward(self, x: HomValue) -> HomValue:
        x = self.convBN(x)
        x = self.bootstrap(x)
        return self.relu(x)


class _MuxBlock(HomOp):
    def __init__(self, model, layer, block):
        super().__init__()
        input_layout = LAYER_IN[layer] if block == 1 else LAYER_OUT[layer]
        output_layout = LAYER_OUT[layer]
        first_relu_scale, second_relu_scale = DELTAS[layer - 1][block - 1]
        self.is_transition = model.downsample is not None
        self.convBN1 = _conv_bn(model.conv1, model.bn1, input_layout, first_relu_scale)
        self.bootstrap1 = Bootstrap()
        self.relu1 = make_relu(first_relu_scale)
        self.convBN2 = _conv_bn(model.conv2, model.bn2, output_layout, second_relu_scale)
        if self.is_transition:
            skip_conv, skip_bn = model.downsample
            self.downsample = _conv_bn(skip_conv, skip_bn, input_layout, second_relu_scale)
        else:
            self.skip_delta = second_relu_scale
        self.bootstrap2 = Bootstrap()
        self.relu2 = make_relu(second_relu_scale)

    def forward(self, x: HomValue) -> HomValue:
        residual = self.convBN1(x)
        residual = self.bootstrap1(residual)
        residual = self.relu1(residual)
        residual = self.convBN2(residual)

        if self.is_transition:
            shortcut = self.downsample(x)
        else:
            shortcut = self.skip_delta * x

        x = residual + shortcut
        x = self.bootstrap2(x)
        return self.relu2(x)


class _MuxLayer(HomOp):
    def __init__(self, model, layer):
        super().__init__()
        self.block1 = _MuxBlock(model[0], layer, 1)
        self.block2 = _MuxBlock(model[1], layer, 2)
        self.block3 = _MuxBlock(model[2], layer, 3)

    def forward(self, x: HomValue) -> HomValue:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x


class _MuxHead(HomOp):
    def __init__(self, model):
        super().__init__()
        channels, height, width = LAYER_OUT[3]
        spatial_positions = height * width
        # the last ReLU leaves one level, for the FC
        self.avgpool = HomSumSlots(k=spatial_positions)
        self.fc = HomMuxConv(kernel_shape=(10, channels, 1, 1), image_hw=(1, 1), padding=(0, 0),
                             bias=True)
        # SumSlots supplies a sum; fold the averaging factor into the FC weights.
        fc_weights = model.fc.weight.reshape(10, channels, 1, 1)
        self.fc.set_data(weight=fc_weights / spatial_positions, bias=model.fc.bias)

    def forward(self, x: HomValue) -> HomValue:
        x = self.avgpool(x)
        return self.fc(x)


class _MuxResnetGraph(HomOp):
    def __init__(self, model):
        super().__init__()
        self.stem = _MuxStem(model)
        self.layer1 = _MuxLayer(model.layer1, 1)
        self.layer2 = _MuxLayer(model.layer2, 2)
        self.layer3 = _MuxLayer(model.layer3, 3)
        self.head = _MuxHead(model)

    def forward(self, x: HomValue) -> HomValue:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.head(x)


class Pipeline(PipelineWrapper):
    def _load_model(self):
        model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20",
                               pretrained=True, verbose=False)
        model.eval()
        return model

    def build_pipeline(self):
        self._model = self._load_model()
        return HomomorphicPipeline(
            hom=_MuxResnetGraph(self._model), input_shape=(SUBRING_SLOTS,),
        )

    def build_params(self):
        return HomParams(
            full_q_list_precision=Q_LIST_PRECISION,
            n=N,
            pt_scale=PT_SCALE,
            sk_hw=192,
            dnum=DNUM,
            n_slots=SUBRING_SLOTS,
        )

    @staticmethod
    def load_cifar_sample(index=None):
        """A test image as the model sees it, normalized float64 (3, 32, 32), and its class.
        index None picks a random image."""
        with CIFAR_BATCH.open("rb") as batch_file:
            batch = pickle.load(batch_file, encoding="bytes")
        if index is None:
            index = random.randrange(len(batch[b"labels"]))
        image = torch.tensor(batch[b"data"][index], dtype=torch.float64).reshape(3, 32, 32) / 255
        mean = torch.tensor(CIFAR10_MEAN, dtype=torch.float64)[:, None, None]
        std = torch.tensor(CIFAR10_STD, dtype=torch.float64)[:, None, None]
        return (image - mean) / std, CIFAR10_CLASSES[int(batch[b"labels"][index])]

    def _set_example_pt(self):
        self._img, self._category = self.load_cifar_sample()
        # RGB zero-padded to the stem's channels, so the stem has the layout of layer 1.
        image = torch.nn.functional.pad(self._img, (0, 0, 0, 0, 0, STEM_CHANNELS - self._img.shape[0]))
        return MuxLayout(*STEM_LAYOUT, SUBRING_SLOTS).pack(image)

    def verify_results(self, actual, expected):
        # the FC's 1x1 output layout puts logit c at slot c
        fhe_logits = actual[:10]
        with torch.no_grad():
            model_logits = self._model(self._img.to(torch.float32).unsqueeze(0))[0]
        fhe_category = CIFAR10_CLASSES[int(fhe_logits.argmax())]
        model_category = CIFAR10_CLASSES[int(model_logits.argmax())]
        print(f"[resnet20] image category: {self._category}   |   "
              f"predicted (decrypted): {fhe_category}   |   (torch model): {model_category}")
        assert fhe_category == model_category, (
            f"FHE predicted {fhe_category}, the torch model {model_category}")
