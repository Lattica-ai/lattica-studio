"""resnet20_n15 with resnet20_two_boot's degree-30 ReLUs (no training, 93.86% clear): 6 levels after each bootstrap (a
degree-30 ReLU 5, a conv 1) in 26-bit primes. One level more than resnet20_n15 fits N = 2^15 at 128 bits only with a
smaller EvalMod prime: the low-precision bootstrap with q0 = 35 and log_delta 1 (actual log QP 805.9 of 809). Its
inputs are resnet20_n15's (the same deltas, a bootstrap before each ReLU), whose slot means (<= 0.09) keep the sine's
cubic error on the DC coefficient negligible at log_delta 1: the clear model with the low bootstrap's noise (2^-9.6 at
every bootstrap) and that error keeps 93.77% on the CIFAR-10 test set. The stem runs without a bootstrap, as in
resnet20_n15.
RTX 5090, CUDA graphs: 0.116 s and 6.7 GiB per CIFAR-10 image (resnet20_n15: 0.113 s, 6.7 GiB); on test images 0-999
the prediction agrees with the torch model's on 985 (resnet20_n15: 950)."""
import dataclasses
from types import MappingProxyType

from lattica_build.base_classes.hom_pipeline import HomomorphicPipeline
from lattica_build.examples.advanced import resnet20_n15, resnet20_two_boot
from lattica_build.examples.advanced.resnet20 import SUBRING_SLOTS
from lattica_build.params.bootstrapping_params import BOOT_PRECISIONS
from lattica_build.params.params import HomParams

# 6 levels after each bootstrap: a degree-30 ReLU (5) and a conv (1), in 26-bit primes.
Q_LIST_PRECISION = ((52, 26),) * 3
_LOW = BOOT_PRECISIONS["low"]
BOOT_PRECISION = dataclasses.replace(_LOW, eval_mod_scale=35, per_n_slots=MappingProxyType(
    {k: (split, cts, 1) for k, (split, cts, _) in _LOW.per_n_slots.items()}))


class Pipeline(resnet20_n15.Pipeline):
    def build_pipeline(self):
        self._model = self._load_model()
        graph = resnet20_two_boot._MuxResnetGraph(self._model, resnet20_two_boot.RELU_COEFS, stem_bootstrap=False)
        return HomomorphicPipeline(hom=graph, input_shape=(SUBRING_SLOTS,))

    def build_params(self):
        return HomParams(full_q_list_precision=Q_LIST_PRECISION, n=resnet20_n15.N, pt_scale=resnet20_n15.PT_SCALE,
                         sk_hw=192, n_slots=SUBRING_SLOTS, boot_precision=BOOT_PRECISION, input_holds_stc_rows=True)
