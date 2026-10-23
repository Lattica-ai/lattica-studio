"""ResNet-20 at N = 2^15: a bootstrap before every ReLU (resnet20_two_boot's layout) with the one-boot model's degree-14
ReLUs (resnet20_relu_coefs_mixed.json), and the low-precision bootstrap (bootstrapping_params BOOT_PRECISIONS["low"],
~10 bits; the network needs ~9), whose chain is small enough for N = 2^15 at 128 bits. 5 levels after each bootstrap
(a ReLU and a conv); the stem runs without one, on the input's levels and the StC primes it also holds
(HomParams.input_holds_stc_rows): its conv, its degree-30 ReLU and L1B1's conv. 18 bootstraps.
RTX 5090, CUDA graphs: 0.111 s and 6.4 GiB per CIFAR-10 image (resnet20: 0.143 s, 13.9 GiB); on the 10,000 test images
the prediction agrees with the torch model's on 95.65% (the polynomial network without bootstrap noise: 95.68%)."""
import json
from pathlib import Path

from lattica_build.base_classes.hom_pipeline import HomomorphicPipeline
from lattica_build.examples.advanced import resnet20_two_boot
from lattica_build.examples.advanced.resnet20 import SUBRING_SLOTS
from lattica_build.params.params import HomParams

N = 2 ** 15
PT_SCALE = 2 ** 26
# 5 levels after each bootstrap: a degree-14 ReLU (4) and a conv (1),, in 26-bit primes.
Q_LIST_PRECISION = ((52, 26), (52, 26), (26,))
RELU_COEFS = json.loads((Path(__file__).parent / "data" / "resnet20_relu_coefs_mixed.json").read_text())


class Pipeline(resnet20_two_boot.Pipeline):
    def build_pipeline(self):
        self._model = self._load_model()
        graph = resnet20_two_boot._MuxResnetGraph(self._model, RELU_COEFS, stem_bootstrap=False)
        return HomomorphicPipeline(hom=graph, input_shape=(SUBRING_SLOTS,))

    def build_params(self):
        return HomParams(full_q_list_precision=Q_LIST_PRECISION, n=N, pt_scale=PT_SCALE, sk_hw=192,
                         n_slots=SUBRING_SLOTS, boot_precision="low", input_holds_stc_rows=True)
