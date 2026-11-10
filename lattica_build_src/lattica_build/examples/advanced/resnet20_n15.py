"""ResNet-20 at N = 2**15: resnet20's graph, a bootstrap before every ReLU, with osmr/pytorchcv's weights (94.03% on
the CIFAR-10 test set), a degree-30 ReLU fitted per position, and the "low" bootstrap. The stem runs without a
bootstrap, on the input's levels and the StC rows the input also holds."""
import json
from pathlib import Path

import numpy as np
import torch

from lattica_build.base_classes.hom_op import HomOp
from lattica_build.base_classes.hom_pipeline import HomomorphicPipeline
from lattica_build.base_classes.hom_value import HomValue
from lattica_build.examples.advanced import resnet20
from lattica_build.operators.polynomials.h_poly_eval_base import HomPolyEvalBase
from lattica_build.params.params import HomParams

N = 2 ** 15
PT_SCALE = 2 ** 26
# 6 levels after each bootstrap: a degree-30 ReLU (5) and a conv (1), on 26-bit primes. The sixth is the 26-bit prime
# the "low" bootstrap's base word holds, so the conv, the last level, runs on one limb.
Q_LIST_PRECISION = ((26,), (52, 26), (52, 26))
# 4 special primes of 57 bits: 3 digits, and every key within the 128-bit bound at N = 2**15
N_SPECIAL_PRIMES = 4
SPECIAL_PRIME_BITS = 57

# ReLU-domain scaling folded into each conv's BN affine (per block, per conv), for these weights
DELTA_INITIAL = 0.136372
DELTAS = [
    [[0.095998, 0.0785], [0.144863, 0.076012], [0.113038, 0.075843]],   # layer 1
    [[0.142393, 0.082133], [0.176943, 0.084878], [0.138764, 0.067013]],   # layer 2
    [[0.127263, 0.087996], [0.130662, 0.071781], [0.088909, 0.037647]],   # layer 3
]
# The ReLUs' |x| polynomials, one per position ("stem", "L{layer}B{block}r{1|2}"): Chebyshev coefficients (T0 first)
# of |u| on [-1, 1], degree 30, least squares weighted by that ReLU's inputs on 5,000 CIFAR-10 training images plus
# 1e-3 of a uniform weight that keeps the fit bounded over the whole domain.
RELU_COEFS = json.loads((Path(__file__).parent / "data" / "resnet20_relu_coefs.json").read_text())


class _ReLU(HomOp):
    """max(0, x / delta) = x / (2 delta) + |x| / (2 delta), with |x| from the position's fit."""

    def __init__(self, name: str, delta: float):
        super().__init__()
        self.slope = 1 / (2 * delta)
        self.abs = HomPolyEvalBase(coefs=np.asarray(RELU_COEFS[name]) / (2 * delta))

    def forward(self, x: HomValue) -> HomValue:
        return self.abs(x) + self.slope * x   # the add writes into the polynomial's output, at the lower level


class Pipeline(resnet20.Pipeline):
    def _load_model(self):
        # chenyaofo's cifar10_resnet20 architecture with osmr/pytorchcv's resnet20_cifar10 weights (MIT), keys renamed
        model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=False, verbose=False)
        model.load_state_dict(torch.load(Path(__file__).parent / "data" / "resnet20_cifar10_pytorchcv.pt",
                                         map_location="cpu"))
        model.eval()
        return model

    def build_pipeline(self):
        self._model = self._load_model()
        graph = resnet20._MuxResnetGraph(self._model, _ReLU, DELTA_INITIAL, DELTAS, stem_bootstrap=False)
        return HomomorphicPipeline(hom=graph, input_shape=(resnet20.SUBRING_SLOTS,))

    def build_params(self):
        return HomParams(full_q_list_precision=Q_LIST_PRECISION, n=N, pt_scale=PT_SCALE, sk_hw=1024,
                         num_special_primes=N_SPECIAL_PRIMES, special_prime_bits=SPECIAL_PRIME_BITS,
                         n_slots=resnet20.SUBRING_SLOTS, boot_precision="low", input_holds_stc_rows=True)
