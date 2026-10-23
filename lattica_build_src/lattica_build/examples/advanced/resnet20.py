import json
import math
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
from lattica_build.operators.polynomials.h_poly_eval_base import HomPolyEvalBase
from lattica_build.operators.slots.h_sum_slots import HomSumSlots
from lattica_build.params.params import HomParams

N = 2 ** 16
SUBRING_SLOTS = 2 ** 14                      # stem tile = 32*32*16 = 16384 < N/2
PT_SCALE = 2 ** 30
# The FC's weights are encoded at 2^16, without a rescale: its output, at 2^30 x 2^16 with logits within +-12, fits the
# last ReLU's level, q0 (53 bits), so the head needs no bootstrap. At 2^16 the encrypted pipeline predicts as it did with
# the head bootstrap on 1000 of 1000 test images.
FC_PT_SCALE = 2 ** 16
# The ReLUs' |x| polynomials, one per ReLU: Chebyshev coefficients (T0 first) of |u| on [-1, 1]. Degree 30 (5 levels
# on the tree): least squares weighted by that ReLU's inputs on 5,000 CIFAR-10 training images plus 1e-3 of a uniform
# weight. Degree 14 (4 levels): trained by distillation from that fit. Both ReLUs of a block share a degree, so the
# block's level count (2 (levels + 1)) is a whole number of (60, 30) rows. PROTOTYPE: the mixed set is for timing; its
# accuracy is not validated.
RELU_COEFS = json.loads((Path(__file__).parent / "data" / "resnet20_relu_coefs_mixed.json").read_text())


def _relu_levels(name: str) -> int:
    return (len(RELU_COEFS[name]) - 1).bit_length()


def _block_rows(layer: int, block: int) -> int:
    """The (60, 30) rows a block's bootstrap gives it: conv1 (1) + ReLU1 + conv2 (1) + ReLU2."""
    levels = {_relu_levels(f"L{layer}B{block}r{r}") for r in (1, 2)}
    assert len(levels) == 1, f"L{layer}B{block}: both ReLUs of a block take the same degree"
    return levels.pop() + 1


# One bootstrap per residual block, at its input, the shortcut's scalar or downsample conv running in parallel; it
# targets the block's own rows (drop_levels leaves the chain's first others out). The head takes no level (see _MuxHead),
# so no block carries a spare one. dnum is unset: the fewest 128-bit-secure digits for the chain.
Q_LIST_PRECISION = ((60, 30),) * max(_block_rows(layer, block) for layer in (1, 2, 3) for block in (1, 2, 3))
# There is no bootstrap before the first block: the image is encrypted with the stem's levels only, the stem conv (1)
# and its ReLU.
NUM_INIT_ROWS = math.ceil((_relu_levels("stem") + 1) / 2)
STEM_CHANNELS = 16                           # RGB (3) zero-padded up to 16 channels

# Per-stage map (C, H, W) after the stage's downsampling.
STEM_LAYOUT = (16, 32, 32)
LAYER_IN = {1: (16, 32, 32), 2: (16, 32, 32), 3: (32, 16, 16)}

LAYER_OUT = {1: (16, 32, 32), 2: (32, 16, 16), 3: (64, 8, 8)}


# ReLU-domain scaling folded into each conv's BN affine (per block, per conv).
DELTA_INITIAL = 0.136372
DELTAS = [
    [[0.095998, 0.0785], [0.144863, 0.076012], [0.113038, 0.075843]],   # layer 1
    [[0.142393, 0.082133], [0.176943, 0.084878], [0.138764, 0.067013]],   # layer 2
    [[0.127263, 0.087996], [0.130662, 0.071781], [0.088909, 0.037647]],   # layer 3
]

# The official CIFAR-10 test batch, and the normalization the model expects.
CIFAR_BATCH = Path(__file__).parent / "data" / "cifar10_test_batch"
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
# CIFAR-10 labels in the model's output-index order.
CIFAR10_CLASSES = ('airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck')


class _ReLU(HomOp):
    """max(0, x / delta) = x / (2 delta) + |x| / (2 delta). |x| is even, so the poly eval
    takes it through z = T2 at half the degree; the linear term is one scalar multiply.
    delta 1 gives max(0, x) itself, in [0, 1] for a folded x in [-1, 1]: what a bootstrap takes."""

    def __init__(self, delta: float, name: str):
        super().__init__()
        self.slope = 1 / (2 * delta)
        coefs = np.asarray(RELU_COEFS[name]) / (2 * delta)
        self.abs = HomPolyEvalBase(coefs=coefs)   # lowest degree first, as the JSON holds them

    def forward(self, x: HomValue) -> HomValue:
        return self.abs(x) + self.slope * x   # the add writes into its first operand: the polynomial's output, at the lower level


def _conv_bn(conv, bn, layout, delta, *, weight=None, drop_rows=0):
    """A mux convolution with its batch norm and the next ReLU's 1/delta folded in."""
    _, height, width = layout
    weight = conv.weight if weight is None else weight
    op = HomMuxConvBnFused(
        kernel_shape=tuple(weight.shape), image_hw=(height, width),
        stride=conv.stride, padding=conv.padding, dilation=conv.dilation, drop_rows=drop_rows,
    )
    op.set_data(weight, None, bn.running_mean, bn.running_var, bn.weight, bn.bias, bn.eps, delta)
    return op


def _input_delta(layer, block):
    """The delta of the ReLU whose output is this block's input: the stem's, or the previous block's second."""
    if block > 1:
        return DELTAS[layer - 1][block - 2][1]
    if layer > 1:
        return DELTAS[layer - 2][-1][1]
    return DELTA_INITIAL


# The ReLUs that feed a bootstrap (the stem's and each block's second) output max(0, x) in their folded domain, delta
# times the activation, so the bootstrap's input stays in [0, 1]; their consumers take the 1/delta into their weights.
class _MuxStem(HomOp):
    def __init__(self, model):
        super().__init__()
        weight = torch.zeros(STEM_LAYOUT[0], STEM_CHANNELS, 3, 3, dtype=torch.float64)
        weight[:, :3] = model.conv1.weight
        self.convBN = _conv_bn(model.conv1, model.bn1, STEM_LAYOUT,
                              DELTA_INITIAL, weight=weight)
        self.relu = _ReLU(1, "stem")

    def forward(self, x: HomValue) -> HomValue:
        return self.relu(self.convBN(x))


class _MuxBlock(HomOp):
    def __init__(self, model, layer, block):
        super().__init__()
        input_layout = LAYER_IN[layer] if block == 1 else LAYER_OUT[layer]
        output_layout = LAYER_OUT[layer]
        first_relu_scale, second_relu_scale = DELTAS[layer - 1][block - 1]
        input_scale = _input_delta(layer, block)
        self.is_transition = model.downsample is not None
        self.bootstrap = Bootstrap(drop_levels=len(Q_LIST_PRECISION) - _block_rows(layer, block))
        self.convBN1 = _conv_bn(model.conv1, model.bn1, input_layout, first_relu_scale,
                                weight=model.conv1.weight / input_scale)
        self.relu1 = _ReLU(first_relu_scale, f"L{layer}B{block}r1")
        self.convBN2 = _conv_bn(model.conv2, model.bn2, output_layout, second_relu_scale)
        if self.is_transition:
            skip_conv, skip_bn = model.downsample
            # The shortcut is added after conv2, so the downsample runs near conv2's rows, not at the bootstrap's: on
            # fewer limbs, its plaintext too. It drops ReLU1's levels // 2 whole rows, fewer than conv1 and ReLU1
            # consume, so its output stays a level above the residual's: the add's scale adjustment needs one (it
            # refuses the same rows at another scale).
            self.downsample = _conv_bn(skip_conv, skip_bn, input_layout, second_relu_scale,
                                       weight=skip_conv.weight / input_scale,
                                       drop_rows=_relu_levels(f"L{layer}B{block}r1") // 2)
        else:
            self.skip_delta = second_relu_scale / input_scale
        self.relu2 = _ReLU(1, f"L{layer}B{block}r2")

    def forward(self, x: HomValue) -> HomValue:
        x = self.bootstrap(x)
        residual = self.convBN1(x)
        residual = self.relu1(residual)
        residual = self.convBN2(residual)

        if self.is_transition:
            shortcut = self.downsample(x)
        else:
            shortcut = self.skip_delta * x

        x = residual + shortcut
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
        # the last ReLU leaves only q0: the sum takes no level, the FC multiplies without a rescale
        self.avgpool = HomSumSlots(k=spatial_positions)
        self.fc = HomMuxConv(kernel_shape=(10, channels, 1, 1), image_hw=(1, 1), padding=(0, 0),
                             bias=True, with_modswitch=False, pt_scale=FC_PT_SCALE)
        # SumSlots supplies a sum, of the last ReLU's folded output: fold the averaging factor and that ReLU's
        # 1/delta into the FC weights.
        fc_weights = model.fc.weight.reshape(10, channels, 1, 1)
        self.fc.set_data(weight=fc_weights / (spatial_positions * DELTAS[-1][-1][1]), bias=model.fc.bias)

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
        # The architecture of chenyaofo's cifar10_resnet20 with osmr/pytorchcv's resnet20_cifar10 weights (MIT,
        # 94.03% on the CIFAR-10 test set; the keys renamed to chenyaofo's): same network, 1.4 points more accurate.
        model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20",
                               pretrained=False, verbose=False)
        model.load_state_dict(torch.load(Path(__file__).parent / "data" / "resnet20_cifar10_pytorchcv.pt",
                                         map_location="cpu"))
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
            n_slots=SUBRING_SLOTS,
            num_init_rows=NUM_INIT_ROWS,
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
