import numpy as np
import torch

from lattica_build.base_classes.hom_op import HomOp
from lattica_build.base_classes.hom_pipeline import HomomorphicPipeline
from lattica_build.base_classes.hom_value import HomValue
from lattica_build.base_classes.pipeline_wrapper import PipelineWrapper
from lattica_build.operators.arithmetic.h_const_add import HomConstAdd
from lattica_build.operators.arithmetic.h_const_mul import HomConstMul
from lattica_build.operators.client_ops import Repeat
from lattica_build.operators.fhe.h_bootstrap import Bootstrap
from lattica_build.operators.ml.conv_bn_fused import HomConvBnFused
from lattica_build.operators.ml.h_linear import HomLinear
from lattica_build.operators.polynomials.h_poly_eval import HomPolyEval
from lattica_build.operators.shape.h_reshape import HomReshape
from lattica_build.operators.slots.h_rotate_sum import HomRotateSum
from lattica_build.params.params import HomParams

# Note: not secure for production use, run with 2**16 for secure parameters.
N = 2 ** 11
LOG_N_SUBRING = 11
PT_SCALE = 2 ** 30
Q_LIST_PRECISION = ((60, 30),) * 4
N_SPECIAL_PRIMES = 9
RELU_DEG = 119
IMAGE_HW = (32, 32)
HOM_INPUT_SHAPE = (3, IMAGE_HW[0] * IMAGE_HW[1])
FINAL_PITCH = 4
# The pretrained CIFAR-10 models expect normalized inputs.
CIFAR10_MEAN = torch.tensor((0.4914, 0.4822, 0.4465))
CIFAR10_STD = torch.tensor((0.2023, 0.1994, 0.201))
DELTA_INITIAL = 0.1806
DELTAS = [
    [[0.1722, 0.131], [0.1973, 0.1273], [0.2429, 0.1275]],  # layer 1
    [[0.1906, 0.1232], [0.3359, 0.1161], [0.2869, 0.0893]],  # layer 2
    [[0.2563, 0.1762], [0.2739, 0.1279], [0.2108, 0.0348]],  # layer 3
]


def _make_relu(delta, degree):
    return HomPolyEval(func=lambda x: np.maximum(0, x / delta), degree=degree)


def _dilation(layer, block, transition):
    if transition and layer >= 2 and block == 1:
        return (2 ** (layer - 2),) * 2
    return (2 ** (layer - 1),) * 2


def _convbn_kwargs(conv, bn, delta, dilation, image_hw):
    data = dict(
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
    init = dict(
        in_channels=cin_pg * conv.groups,
        out_channels=cout,
        kernel_size=(kh, kw),
        stride=tuple(conv.stride),
        padding=(kh // 2, kw // 2),
        groups=conv.groups,
        dilation=dilation,
        image_hw=image_hw,
    )
    return {"delta": delta, "init_kwargs": init, "set_data_kwargs": data}


def _block_kwargs(model, layer, block, deltas, image_hw):
    module = getattr(model, f"layer{layer}")[block - 1]
    d1, d2 = deltas[layer - 1][block - 1]
    result = {
        "conv1": _convbn_kwargs(
            module.conv1, module.bn1, d1, _dilation(layer, block, True), image_hw
        ),
        "conv2": _convbn_kwargs(
            module.conv2, module.bn2, d2, _dilation(layer, block, False), image_hw
        ),
    }
    if module.downsample is not None:  # dx reuses conv2's delta (d2)
        result["downsample"] = _convbn_kwargs(
            module.downsample[0],
            module.downsample[1],
            d2,
            _dilation(layer, block, True),
            image_hw,
        )
    return result


def _final_layer_kwargs(model, image_hw, final_pitch):
    fc = model.fc
    cout, cin = fc.weight.shape
    dim_final = image_hw[0] // final_pitch
    rots = [
        final_pitch * (i * image_hw[1] + j)
        for i in range(dim_final)
        for j in range(dim_final)
        if (i, j) != (0, 0)
    ]
    return {
        "avgpool_rots": rots,
        "fc": {
            "dims": (cout, cin, 1),  # trailing 1 broadcasts over slots
            "weight": (fc.weight.detach() / dim_final ** 2).reshape(cout, cin, 1),
            "bias": fc.bias.detach().reshape(cout, 1),
        },
    }


class _InitialLayerPipeline(HomOp):
    def __init__(self, kwargs, log_n_subring, relu_deg):
        super().__init__()
        self.convBN = HomConvBnFused(**kwargs["init_kwargs"])
        self.bootstrap = Bootstrap(log_n_subring=log_n_subring)
        self.relu = _make_relu(kwargs["delta"], relu_deg)

    def forward(self, x):
        return self.relu(self.bootstrap(self.convBN(x)))


class _BlockPipeline(HomOp):
    def __init__(self, kwargs, log_n_subring, relu_deg):
        super().__init__()
        d1, d2 = kwargs["conv1"]["delta"], kwargs["conv2"]["delta"]
        self.bootstrap = Bootstrap(log_n_subring=log_n_subring)
        self.convBN1 = HomConvBnFused(**kwargs["conv1"]["init_kwargs"])
        self.relu1 = _make_relu(d1, relu_deg)
        self.convBN2 = HomConvBnFused(**kwargs["conv2"]["init_kwargs"])
        self.relu2 = _make_relu(d2, relu_deg)
        self.has_downsample = "downsample" in kwargs
        if self.has_downsample:
            self.downsample = HomConvBnFused(**kwargs["downsample"]["init_kwargs"])
        else:
            self.skip_delta = d2

    def forward(self, x):
        res = self.relu1(self.bootstrap(self.convBN1(x)))
        res = self.convBN2(res)
        skip = self.downsample(x) if self.has_downsample else self.skip_delta * x
        return self.relu2(self.bootstrap(res + skip))


class _MainLayerPipeline(HomOp):
    def __init__(self, layer, kwargs, log_n_subring, relu_deg):
        super().__init__()
        self.block1 = _BlockPipeline(kwargs[layer, 1], log_n_subring, relu_deg)
        self.block2 = _BlockPipeline(kwargs[layer, 2], log_n_subring, relu_deg)
        self.block3 = _BlockPipeline(kwargs[layer, 3], log_n_subring, relu_deg)

    def forward(self, x):
        return self.block3(self.block2(self.block1(x)))


class _FinalLayerPipeline(HomOp):
    def __init__(self, kwargs):
        super().__init__()
        self.avgpool_sum = HomRotateSum(rotations=kwargs["avgpool_rots"], perform_sum=True)
        self.fc = HomLinear(dims=kwargs["fc"]["dims"], mul_axis=-2)

    def forward(self, x):
        return self.fc(self.avgpool_sum(x))


class _ResnetPipeline(HomOp):
    def __init__(self, initial, blocks, final, log_n_subring, relu_deg):
        super().__init__()
        self.initial_layer = _InitialLayerPipeline(initial, log_n_subring, relu_deg)
        self.layer1 = _MainLayerPipeline(1, blocks, log_n_subring, relu_deg)
        self.layer2 = _MainLayerPipeline(2, blocks, log_n_subring, relu_deg)
        self.layer3 = _MainLayerPipeline(3, blocks, log_n_subring, relu_deg)
        self.final_layer = _FinalLayerPipeline(final)

    def forward(self, x):
        return self.final_layer(self.layer3(self.layer2(self.layer1(self.initial_layer(x)))))


class Pipeline(PipelineWrapper):
    def build_pipeline(
        self,
        log_n_subring=LOG_N_SUBRING,
        relu_deg=RELU_DEG,
        image_hw=IMAGE_HW,
        hom_input_shape=HOM_INPUT_SHAPE,
        final_pitch=FINAL_PITCH,
        delta_initial=DELTA_INITIAL,
        deltas=DELTAS,
    ):
        # Download the pretrained model from torch hub. This is cached locally
        # after the first build.
        model = torch.hub.load(
            "chenyaofo/pytorch-cifar-models",
            "cifar10_resnet20",
            pretrained=True,
            verbose=False,
        )
        model.eval()
        initial = _convbn_kwargs(model.conv1, model.bn1, delta_initial, (1, 1), image_hw)
        blocks = {
            (layer, block): _block_kwargs(model, layer, block, deltas, image_hw)
            for layer in (1, 2, 3)
            for block in (1, 2, 3)
        }
        final = _final_layer_kwargs(model, image_hw, final_pitch)
        pipeline = HomomorphicPipeline(
            client_pre=[
                HomConstMul(dims=(3, 1, 1)).set_data(
                    1.0 / (255.0 * CIFAR10_STD)
                ),
                HomConstAdd(dims=(3, 1, 1)).set_data(
                    -CIFAR10_MEAN / CIFAR10_STD
                ),
                HomReshape(hom_input_shape),
                Repeat(dim=1),
            ],
            hom=_ResnetPipeline(initial, blocks, final, log_n_subring, relu_deg),
            input_shape=(3, *image_hw),
        )
        pipeline.set_data("initial_layer.convBN", *initial["set_data_kwargs"].values())
        for (layer, block), kwargs in blocks.items():
            base = f"layer{layer}.block{block}"
            pipeline.set_data(f"{base}.convBN1", *kwargs["conv1"]["set_data_kwargs"].values())
            pipeline.set_data(f"{base}.convBN2", *kwargs["conv2"]["set_data_kwargs"].values())
            if "downsample" in kwargs:
                pipeline.set_data(
                    f"{base}.downsample",
                    *kwargs["downsample"]["set_data_kwargs"].values(),
                )
        pipeline.set_data("final_layer.fc", final["fc"]["weight"], final["fc"]["bias"])
        return pipeline

    def build_params(
        self,
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
            num_init_rows=1,
        )
