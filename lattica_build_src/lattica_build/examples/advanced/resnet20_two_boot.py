"""ResNet-20 with a bootstrap before every ReLU and every ReLU at degree 30: the ReLU coefficients need sample inputs
only (no training), and the pipeline keeps the model's accuracy (93.86% on the CIFAR-10 test set against the exact
model's 94.03%). resnet20 is the faster layout (one bootstrap per block, ReLUs trained to degree 14, 92.56%)."""
import json
from pathlib import Path

import torch

from lattica_build.base_classes.hom_op import HomOp
from lattica_build.base_classes.hom_pipeline import HomomorphicPipeline
from lattica_build.base_classes.hom_value import HomValue
from lattica_build.examples.advanced import resnet20
from lattica_build.examples.advanced.resnet20 import (
    DELTA_INITIAL, DELTAS, LAYER_IN, LAYER_OUT, N, PT_SCALE, STEM_CHANNELS, STEM_LAYOUT, SUBRING_SLOTS, _conv_bn, _ReLU)
from lattica_build.operators.fhe.h_bootstrap import Bootstrap
from lattica_build.operators.ml.h_mux_conv import HomMuxConv
from lattica_build.operators.slots.h_sum_slots import HomSumSlots
from lattica_build.params.params import HomParams

# 6 levels after each bootstrap: a ReLU (5) and a conv (1).
Q_LIST_PRECISION = ((60, 30),) * 3
DNUM = 2                                     # hybrid digits over the chain
# The ReLUs' |x| polynomials, degree 30 (5 levels on the tree), one per ReLU: Chebyshev coefficients (T0 first) of |u|
# on [-1, 1], least squares weighted by that ReLU's inputs on 5,000 CIFAR-10 training images plus 1e-3 of a uniform
# weight that keeps the fit bounded over the whole domain.
RELU_COEFS = json.loads((Path(__file__).parent / "data" / "resnet20_relu_coefs.json").read_text())


# A bootstrap sits right before each ReLU: the ReLU takes 5 of its 6 levels and the next conv the last one, so every
# conv runs on the bottom of the chain.
class _MuxStem(HomOp):
    """bootstrap=False: the input holds the levels of the conv and the ReLU (resnet20_n15)."""

    def __init__(self, model, coefs, bootstrap=True):
        super().__init__()
        weight = torch.zeros(STEM_LAYOUT[0], STEM_CHANNELS, 3, 3, dtype=torch.float64)
        weight[:, :3] = model.conv1.weight
        self.convBN = _conv_bn(model.conv1, model.bn1, STEM_LAYOUT, DELTA_INITIAL, weight=weight)
        self.bootstrap = Bootstrap() if bootstrap else None
        self.relu = _ReLU(DELTA_INITIAL, "stem", coefs)

    def forward(self, x: HomValue) -> HomValue:
        x = self.convBN(x)
        return self.relu(self.bootstrap(x) if self.bootstrap is not None else x)


class _MuxBlock(HomOp):
    def __init__(self, model, layer, block, coefs):
        super().__init__()
        input_layout = LAYER_IN[layer] if block == 1 else LAYER_OUT[layer]
        output_layout = LAYER_OUT[layer]
        first_relu_scale, second_relu_scale = DELTAS[layer - 1][block - 1]
        self.is_transition = model.downsample is not None
        self.convBN1 = _conv_bn(model.conv1, model.bn1, input_layout, first_relu_scale)
        self.bootstrap1 = Bootstrap()
        self.relu1 = _ReLU(first_relu_scale, f"L{layer}B{block}r1", coefs)
        self.convBN2 = _conv_bn(model.conv2, model.bn2, output_layout, second_relu_scale)
        if self.is_transition:
            skip_conv, skip_bn = model.downsample
            self.downsample = _conv_bn(skip_conv, skip_bn, input_layout, second_relu_scale)
        else:
            self.skip_delta = second_relu_scale
        self.bootstrap2 = Bootstrap()
        self.relu2 = _ReLU(second_relu_scale, f"L{layer}B{block}r2", coefs)

    def forward(self, x: HomValue) -> HomValue:
        residual = self.relu1(self.bootstrap1(self.convBN1(x)))
        residual = self.convBN2(residual)
        shortcut = self.downsample(x) if self.is_transition else self.skip_delta * x
        return self.relu2(self.bootstrap2(residual + shortcut))


class _MuxLayer(HomOp):
    def __init__(self, model, layer, coefs):
        super().__init__()
        self.block1 = _MuxBlock(model[0], layer, 1, coefs)
        self.block2 = _MuxBlock(model[1], layer, 2, coefs)
        self.block3 = _MuxBlock(model[2], layer, 3, coefs)

    def forward(self, x: HomValue) -> HomValue:
        return self.block3(self.block2(self.block1(x)))


class _MuxHead(HomOp):
    def __init__(self, model):
        super().__init__()
        channels, height, width = LAYER_OUT[3]
        spatial_positions = height * width
        # the last ReLU leaves one level, for the FC
        self.avgpool = HomSumSlots(k=spatial_positions)
        self.fc = HomMuxConv(kernel_shape=(10, channels, 1, 1), image_hw=(1, 1), padding=(0, 0), bias=True)
        # SumSlots supplies a sum; fold the averaging factor into the FC weights.
        fc_weights = model.fc.weight.reshape(10, channels, 1, 1)
        self.fc.set_data(weight=fc_weights / spatial_positions, bias=model.fc.bias)

    def forward(self, x: HomValue) -> HomValue:
        return self.fc(self.avgpool(x))


class _MuxResnetGraph(HomOp):
    """coefs: the ReLUs' |x| coefficients by name; stem_bootstrap: as _MuxStem's bootstrap."""

    def __init__(self, model, coefs=RELU_COEFS, stem_bootstrap=True):
        super().__init__()
        self.stem = _MuxStem(model, coefs, stem_bootstrap)
        self.layer1 = _MuxLayer(model.layer1, 1, coefs)
        self.layer2 = _MuxLayer(model.layer2, 2, coefs)
        self.layer3 = _MuxLayer(model.layer3, 3, coefs)
        self.head = _MuxHead(model)

    def forward(self, x: HomValue) -> HomValue:
        return self.head(self.layer3(self.layer2(self.layer1(self.stem(x)))))


class Pipeline(resnet20.Pipeline):
    """resnet20's pipeline (model, CIFAR-10 samples, verification) with this layout and chain."""

    def build_pipeline(self):
        self._model = self._load_model()
        return HomomorphicPipeline(hom=_MuxResnetGraph(self._model), input_shape=(SUBRING_SLOTS,))

    def build_params(self):
        return HomParams(full_q_list_precision=Q_LIST_PRECISION, n=N, pt_scale=PT_SCALE, sk_hw=192, dnum=DNUM,
                         n_slots=SUBRING_SLOTS)
