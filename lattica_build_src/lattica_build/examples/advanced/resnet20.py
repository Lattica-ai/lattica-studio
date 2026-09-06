import os

import numpy as np
import torch

from lattica_build.base_classes.hom_op import HomOp
from lattica_build.base_classes.hom_pipeline import HomomorphicPipeline
from lattica_build.base_classes.hom_value import HomValue
from lattica_build.operators.client_ops import Repeat
from lattica_build.operators.arithmetic.h_const_add import HomConstAdd
from lattica_build.operators.arithmetic.h_const_mul import HomConstMul
from lattica_build.operators.fhe.h_bootstrap import Bootstrap
from lattica_build.operators.ml.conv_bn_fused import HomConvBnFused
from lattica_build.operators.ml.h_linear import HomLinear
from lattica_build.operators.polynomials.h_poly_eval import HomPolyEval
from lattica_build.operators.shape.h_reshape import HomReshape
from lattica_build.operators.slots.h_rotate_sum import HomRotateSum
from lattica_build.params.bootstrapping_params import BootstrappingVariant
from lattica_build.params.params import HomParams

N = 2 ** 11         # Note: not secure for production use, run with 2**16 for secure parameters.
LOG_N_SUBRING = 11
PT_SCALE = 2 ** 30

Q_LIST_PRECISION = ((60, 30),) * 4
N_SPECIAL_PRIMES = 9

RELU_DEG = 119
IMAGE_HW = (32, 32)
HOM_INPUT_SHAPE = (3, IMAGE_HW[0] * IMAGE_HW[1])
INPUT_SHAPE = (3, *IMAGE_HW)
FINAL_PITCH = 4

# The pretrained cifar10 models expect normalized inputs
CIFAR10_MEAN = torch.tensor((0.4914, 0.4822, 0.4465))
CIFAR10_STD = torch.tensor((0.2023, 0.1994, 0.2010))

DELTA_INITIAL = 0.1806
DELTAS = [
    [[0.1722, 0.1310], [0.1973, 0.1273], [0.2429, 0.1275]],  # layer 1
    [[0.1906, 0.1232], [0.3359, 0.1161], [0.2869, 0.0893]],  # layer 2
    [[0.2563, 0.1762], [0.2739, 0.1279], [0.2108, 0.0348]],  # layer 3
]

def build_pipeline(
        log_n_subring=LOG_N_SUBRING,
        relu_deg=RELU_DEG,
        image_hw=IMAGE_HW,
        hom_input_shape=HOM_INPUT_SHAPE,
        final_pitch=FINAL_PITCH,
        delta_initial=DELTA_INITIAL,
        deltas=DELTAS,
) -> HomomorphicPipeline:
    """Construct a resnet homomorphic pipeline."""

    # Download the pretrained model from torch hub.
    # This will cache the model in the local directory.
    _model = torch.hub.load(
        "chenyaofo/pytorch-cifar-models", "cifar10_resnet20",
        pretrained=True, verbose=False)
    _model.eval()

    def make_relu(delta: float) -> HomPolyEval:
        return HomPolyEval(func=lambda x: np.maximum(0, x / delta), degree=relu_deg)

    class _InitialLayerPipeline(HomOp):
        def __init__(self, kwargs):
            super().__init__()
            self.convBN = HomConvBnFused(**kwargs['init_kwargs'])
            self.bootstrap = Bootstrap(log_n_subring=log_n_subring)
            self.relu = make_relu(kwargs['delta'])

        def forward(self, x: HomValue) -> HomValue:
            x = self.convBN(x)
            x = self.bootstrap(x)
            x = self.relu(x)
            return x


    class _BlockPipeline(HomOp):
        def __init__(self, block_kwargs):
            super().__init__()
            d1 = block_kwargs['conv1']['delta']
            d2 = block_kwargs['conv2']['delta']
            self.bootstrap = Bootstrap(log_n_subring=log_n_subring)
            self.convBN1 = HomConvBnFused(**block_kwargs['conv1']['init_kwargs'])
            self.relu1 = make_relu(d1)
            self.convBN2 = HomConvBnFused(**block_kwargs['conv2']['init_kwargs'])
            self.relu2 = make_relu(d2)

            self.has_downsample = 'downsample' in block_kwargs
            if self.has_downsample:
                self.downsample = HomConvBnFused(**block_kwargs['downsample']['init_kwargs'])
            else:
                self.skip_delta = d2

        def forward(self, x: HomValue) -> HomValue:
            res = self.convBN1(x)
            res = self.bootstrap(res)
            res = self.relu1(res)
            res = self.convBN2(res)
            skip = self.downsample(x) if self.has_downsample else self.skip_delta * x
            res = res + skip
            res = self.bootstrap(res)
            res = self.relu2(res)

            return res


    class _MainLayerPipeline(HomOp):
        def __init__(self, layer, block_kwargs):
            super().__init__()
            self.block1 = _BlockPipeline(block_kwargs[(layer, 1)])
            self.block2 = _BlockPipeline(block_kwargs[(layer, 2)])
            self.block3 = _BlockPipeline(block_kwargs[(layer, 3)])

        def forward(self, x: HomValue) -> HomValue:
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
            return x


    class _FinalLayerPipeline(HomOp):
        def __init__(self, kwargs):
            super().__init__()
            self.avgpool_sum = HomRotateSum(rotations=kwargs['avgpool_rots'], perform_sum=True)
            self.fc = HomLinear(dims=kwargs['fc']['dims'], mul_axis=-2)

        def forward(self, x: HomValue) -> HomValue:
            x = self.avgpool_sum(x)
            x = self.fc(x)
            return x

    class _ResnetPipeline(HomOp):

        def __init__(self, initial_kwargs, block_kwargs, final_kwargs):
            super().__init__()
            self.initial_layer = _InitialLayerPipeline(initial_kwargs)
            self.layer1 = _MainLayerPipeline(1, block_kwargs)
            self.layer2 = _MainLayerPipeline(2, block_kwargs)
            self.layer3 = _MainLayerPipeline(3, block_kwargs)
            self.final_layer = _FinalLayerPipeline(final_kwargs)

        def forward(self, x: HomValue) -> HomValue:
            x = self.initial_layer(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.final_layer(x)
            return x


    def _dilation(layer, block, is_transition_conv):
        if is_transition_conv and layer >= 2 and block == 1:
            return (2 ** (layer - 2),) * 2
        return (2 ** (layer - 1),) * 2

    def _convbn_kwargs(conv, bn, delta, dilation):
        data_kwargs = dict(
            weight=conv.weight.detach(),
            bias=None,
            mean=bn.running_mean.detach(),
            var=bn.running_var.detach(),
            gamma=bn.weight.detach(),
            beta=bn.bias.detach(),
            eps=float(bn.eps),
            delta=delta,
        )
        cout, cin_pg, kh, kw = conv.weight.shape
        init_kwargs = dict(
            in_channels=cin_pg * conv.groups,
            out_channels=cout,
            kernel_size=(kh, kw),
            stride=tuple(conv.stride),
            padding=(kh // 2, kw // 2),
            groups=conv.groups,
            dilation=dilation,
            image_hw=image_hw,
        )
        return {'delta': delta, 'init_kwargs': init_kwargs, 'set_data_kwargs': data_kwargs}

    def _block_kwargs(layer, block):
        stage = getattr(_model, f'layer{layer}')
        blk = stage[block - 1]
        d1, d2 = deltas[layer - 1][block - 1]
        out = {
            'conv1': _convbn_kwargs(blk.conv1, blk.bn1, d1,
                                         _dilation(layer, block, is_transition_conv=True)),
            'conv2': _convbn_kwargs(blk.conv2, blk.bn2, d2,
                                         _dilation(layer, block, is_transition_conv=False)),
        }
        if blk.downsample is not None:  # dx reuses conv2's delta (d2)
            out['downsample'] = _convbn_kwargs(blk.downsample[0], blk.downsample[1], d2,
                                                    _dilation(layer, block, is_transition_conv=True))
        return out

    def _final_layer_kwargs():
        fc = _model.fc
        cout, cin = fc.weight.shape
        pitch = final_pitch
        dim_final = image_hw[0] // pitch
        rots = [pitch * (i * image_hw[1] + j)
                for i in range(dim_final) for j in range(dim_final)
                if (i, j) != (0, 0)]

        fc_kwargs = {
            'dims': (cout, cin, 1),  # trailing 1 broadcasts over slots; cin is contracted
            'weight': (fc.weight.detach() / dim_final ** 2).reshape(cout, cin, 1),
            'bias': fc.bias.detach().reshape(cout, 1),
        }
        return {'avgpool_rots': rots, 'fc': fc_kwargs}

    initial_kwargs = _convbn_kwargs(_model.conv1, _model.bn1, delta_initial, dilation=(1, 1))
    block_kwargs = {(l, b): _block_kwargs(l, b)
                    for l in (1, 2, 3)
                    for b in (1, 2, 3)}
    final_kwargs = _final_layer_kwargs()

    hom_pipeline = HomomorphicPipeline(
        client_pre=[
            HomConstMul(dims=(3, 1, 1)).set_data(1.0 / (255.0 * CIFAR10_STD)),
            HomConstAdd(dims=(3, 1, 1)).set_data(-CIFAR10_MEAN / CIFAR10_STD),
            HomReshape(hom_input_shape),
            Repeat(dim=1),
        ],
        hom=_ResnetPipeline(initial_kwargs, block_kwargs, final_kwargs),
        input_shape=(3, *image_hw),
    )

    hom_pipeline.set_data('initial_layer.convBN', *initial_kwargs['set_data_kwargs'].values())
    for (layer, block), bk in block_kwargs.items():
        base = f'layer{layer}.block{block}'
        hom_pipeline.set_data(f'{base}.convBN1', *bk['conv1']['set_data_kwargs'].values())
        hom_pipeline.set_data(f'{base}.convBN2', *bk['conv2']['set_data_kwargs'].values())
        if 'downsample' in bk:
            hom_pipeline.set_data(f'{base}.downsample', *bk['downsample']['set_data_kwargs'].values())
    hom_pipeline.set_data('final_layer.fc',
                          final_kwargs['fc']['weight'], final_kwargs['fc']['bias'])

    return hom_pipeline

def build_params(
        q_list_precision=Q_LIST_PRECISION,
        n=N,
        pt_scale=PT_SCALE,
        num_special_primes=N_SPECIAL_PRIMES,
) -> HomParams:
    return HomParams(
        full_q_list_precision=q_list_precision,
        n=n,
        pt_scale=pt_scale,
        sk_hw=192,
        num_special_primes=num_special_primes,
        num_init_rows=1
    )
