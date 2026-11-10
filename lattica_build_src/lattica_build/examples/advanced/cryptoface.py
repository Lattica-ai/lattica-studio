"""CryptoFace face verification (Ao and Boddeti, CVPR 2025) under encryption.

The client aligns two faces to 64x64, cuts each into four 32x32 patches (row-major) and sends the
eight patches. The server runs one small ResNet per patch position, with HerPN activations (a
per-channel quadratic in place of ReLU + BatchNorm), on both faces' patches at once (a batch of two
ciphertexts through every op), sums the four patch features of each face into its 256-d embedding
and returns the cosine of the two embeddings. The client decrypts that one score; the faces are the
same person above a threshold.

Weights: the benchmark's checkpoint, downloaded when the pipeline is built
(https://huggingface.co/halmsu/cryptoface-v1, trained by https://github.com/human-analysis/CryptoFace).
Example pairs: the FHE benchmark's 1,024 CelebA pairs (https://fhe-benchmarking.github.io/face-recognition),
aligned with InsightFace when it is installed, as the benchmark's client does, else center-cropped.

Every level is a 30-bit prime: a conv or a constant multiply takes one, a HerPN's square one. A
segment of 9 levels runs two residual blocks, so each patch bootstraps twice (the HerPN output at the
head of blocks 2 and 4, which both of that block's branches read) and each face's embedding once.
"""
import io
import math
import pickle
import random
from functools import cache
from pathlib import Path

import numpy as np
import torch
from torch import nn

from lattica_build.base_classes.hom_op import HomOp
from lattica_build.base_classes.hom_pipeline import HomomorphicPipeline
from lattica_build.base_classes.hom_value import HomValue
from lattica_build.base_classes.pipeline_wrapper import PipelineWrapper
from lattica_build.operators.arithmetic.h_const_add import HomConstAdd
from lattica_build.operators.arithmetic.h_const_mul import HomConstMul
from lattica_build.operators.arithmetic.h_mul import HomMul
from lattica_build.operators.composite.module_list import ModuleListHomOp
from lattica_build.operators.fhe.h_bootstrap import Bootstrap
from lattica_build.operators.ml.h_mux_conv import HomMuxConv, MuxLayout
from lattica_build.operators.polynomials.h_poly_eval import HomPolyEval
from lattica_build.operators.shape.h_slice import HomSlice
from lattica_build.operators.slots.h_rotate_sum import HomRotateSum
from lattica_build.operators.slots.h_sum_slots import HomSumSlots
from lattica_build.params.params import HomParams

N = 2 ** 16
SUBRING_SLOTS = 2 ** 14                      # a patch: 16 channels (RGB zero-padded) x 32 x 32
# The size of the chain's 30-bit primes (2^30.06 to 2^30.08). A HerPN's square takes the scale s to
# s^2 / q, doubling its distance in bits from the prime q: from 2^30 it fell to 2^21 by block 4 and the
# circuit lost all precision. From here it stays within 0.04 bits; the bootstraps reset it.
PT_SCALE = round(2 ** 30.07)
Q_LIST_PRECISION = ((60, 30),) * 4 + ((30,),)   # nine 30-bit primes: two blocks between bootstraps
N_SPECIAL_PRIMES = 9

CHECKPOINT_URL = "https://huggingface.co/halmsu/cryptoface-v1/resolve/main/backbone-64x64.ckpt"
PAIRS_URL = "https://huggingface.co/datasets/halmsu/celeba-1024-pairs/resolve/main/face_dataset.npy"
LABELS_URL = "https://huggingface.co/datasets/halmsu/celeba-1024-pairs/resolve/main/face_dataset_labels.txt"

FACE_SIZE = 64
PATCH_SIZE = 32
N_PATCHES = (FACE_SIZE // PATCH_SIZE) ** 2
EMBEDDING_DIM = 256
STEM_CHANNELS = 16                           # RGB (3) zero-padded up to 16 channels
# (input channels, output channels, stride, input height and width) of the five residual blocks
BLOCKS = ((16, 16, 1, 32), (16, 32, 2, 32), (32, 32, 1, 16), (32, 64, 2, 16), (64, 64, 1, 8))
BOOTSTRAPPED_BLOCKS = (1, 3)                 # bootstrap the first HerPN's output of these blocks
POOL_LAYOUT = (64, 8, 8)
POOL_WINDOW = 4                              # AdaptiveAvgPool2d((2, 2)) of the 8x8 map: 4x4 windows

# A HerPN input's range, per channel: its training mean +- this many of its training deviations (the
# statistics its bn1 kept). Every HerPN output is scaled to at most 1 in magnitude there, which the
# bootstraps need. 1.5x the largest deviation of any channel of any patch on the benchmark's 2,048
# aligned faces: the inputs are heavy-tailed (99.9% within 4-7 deviations, the largest 6-23).
HERPN_RANGE_SIGMAS = ((20, 34), (28, 20), (17, 15), (16, 14), (10, 11))   # (herpn1, herpn2) of each block
POOL_RANGE_SIGMAS = 13
BN_EPS = 1e-5
# The bootstrapped embedding is e / EMBEDDING_SCALE: e is batch-normalized (unit variance per
# coordinate), so its coordinates stay well under EMBEDDING_SCALE / 2. 1/|e| is a polynomial in
# |e|^2 / EMBEDDING_SCALE^2 over [INV_NORM_LO, INV_NORM_HI], i.e. |e|^2 from 64 to 576; the
# benchmark's faces measure 108 to 505. A domain 2 wide maps onto [-1, 1] without a level.
EMBEDDING_SCALE = 16.0
INV_NORM_LO, INV_NORM_HI = 0.25, 2.25
INV_NORM_DEGREE = 15


# ---------- the plaintext model (eval mode, the original repo's module names) ----------

class _TorchHerPN(nn.Module):
    """g * (bn0(1) / sqrt(2 pi) + bn1(x) / 2 + bn2((x^2 - 1) / sqrt 2) / sqrt(4 pi)) + b"""

    def __init__(self, planes):
        super().__init__()
        self.bn0, self.bn1, self.bn2 = (nn.BatchNorm2d(planes, affine=False) for _ in range(3))
        self.weight = nn.Parameter(torch.ones(planes, 1, 1))
        self.bias = nn.Parameter(torch.zeros(planes, 1, 1))

    def forward(self, x):
        hermite = (self.bn0(torch.ones_like(x)) / math.sqrt(2 * math.pi) + self.bn1(x) / 2
                   + self.bn2((x * x - 1) / math.sqrt(2)) / math.sqrt(4 * math.pi))
        return self.weight * hermite + self.bias


class _TorchBlock(nn.Module):
    def __init__(self, in_planes, planes, stride):
        super().__init__()
        self.herpn1 = _TorchHerPN(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.herpn2 = _TorchHerPN(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                                          nn.BatchNorm2d(planes))

    def forward(self, x):
        x = self.herpn1(x)
        return self.conv2(self.herpn2(self.conv1(x))) + self.shortcut(x)


class _TorchHerPNPool(nn.Module):
    def __init__(self, planes):
        super().__init__()
        self.herpn = _TorchHerPN(planes)
        self.pool = nn.AdaptiveAvgPool2d((2, 2))

    def forward(self, x):
        return self.pool(self.herpn(x))


class _TorchBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, STEM_CHANNELS, 3, 1, 1, bias=False)
        self.layers = nn.Sequential(*(_TorchBlock(c_in, c_out, stride) for c_in, c_out, stride, _ in BLOCKS))
        self.herpnpool = _TorchHerPNPool(POOL_LAYOUT[0])
        self.bn = nn.BatchNorm1d(EMBEDDING_DIM)

    def forward(self, x):
        return self.bn(self.herpnpool(self.layers(self.conv(x))).flatten(1))


class _TorchCryptoFace(nn.Module):
    def __init__(self):
        super().__init__()
        self.nets = nn.ModuleList(_TorchBackbone() for _ in range(N_PATCHES))
        self.linear = nn.Linear(N_PATCHES * EMBEDDING_DIM, EMBEDDING_DIM)
        self.bn = nn.BatchNorm1d(EMBEDDING_DIM, affine=False)
        self.jigsaw = nn.Linear(EMBEDDING_DIM, N_PATCHES)   # a training head, unused here

    def forward(self, faces):
        """Faces (B, 3, 64, 64) in [-1, 1] -> embeddings (B, 256)."""
        features = [net(patch) for net, patch in zip(self.nets, _patches(faces))]
        return self.bn(self.linear(torch.cat(features, dim=1)))


def _patches(faces):
    """The 32x32 patches of (..., C, 64, 64) faces, row-major."""
    grid = FACE_SIZE // PATCH_SIZE
    return [faces[..., PATCH_SIZE * row:PATCH_SIZE * (row + 1), PATCH_SIZE * col:PATCH_SIZE * (col + 1)]
            for row in range(grid) for col in range(grid)]


@cache
def _load_checkpoint() -> dict:
    """The checkpoint's state dict, downloaded once into torch's hub cache, in float64."""
    checkpoint = torch.hub.load_state_dict_from_url(CHECKPOINT_URL, map_location="cpu", weights_only=True,
                                                    file_name="cryptoface-backbone-64x64.ckpt")
    return {name: tensor.double() if tensor.is_floating_point() else tensor
            for name, tensor in checkpoint["backbone"].items()}


@cache
def load_torch_model() -> nn.Module:
    model = _TorchCryptoFace().double()
    model.load_state_dict(_load_checkpoint())
    return model.eval()


# ---------- the weights folded for the encrypted circuit ----------

class _FoldedHerPN:
    """A HerPN, w2 x^2 + w1 x + w0 per channel, as w2 a^2 (((x + h) / a)^2 + c / a^2): its producer emits
    (x + h) / a (input_scale = 1 / a, shift = h / a, both folded into the producer's weights and bias), the
    circuit squares that and adds offset = c / a^2, and gain = w2 a^2 goes into the consumers' weights.
    a scales the output to at most 1 in magnitude over the input's range."""

    def __init__(self, state, prefix, range_sigmas):
        g, b = state[prefix + "weight"].reshape(-1), state[prefix + "bias"].reshape(-1)
        (m0, v0), (m1, v1), (m2, v2) = ((state[f"{prefix}bn{i}.running_mean"], state[f"{prefix}bn{i}.running_var"] + BN_EPS)
                                        for i in range(3))
        # CryptoFace's HerPN_Fuse
        w2 = g / torch.sqrt(8 * math.pi * v2)
        w1 = g / (2 * torch.sqrt(v1))
        w0 = b + g * ((1 - m0) / torch.sqrt(2 * math.pi * v0) - m1 / (2 * torch.sqrt(v1))
                      - (1 + math.sqrt(2) * m2) / torch.sqrt(8 * math.pi * v2))
        h = w1 / (2 * w2)
        c = w0 / w2 - h * h
        low, high = m1 - range_sigmas * torch.sqrt(v1), m1 + range_sigmas * torch.sqrt(v1)
        # (x + h)^2 + c peaks in magnitude at an end of the range, or at its vertex x = -h
        vertex_inside = (low <= -h) & (-h <= high)
        peak = torch.maximum(((low + h) ** 2 + c).abs(), ((high + h) ** 2 + c).abs())
        peak = torch.where(vertex_inside, torch.maximum(peak, c.abs()), peak)
        a = torch.sqrt(peak)
        self.input_scale = 1 / a
        self.shift = h / a
        self.offset = c / a ** 2
        self.gain = w2 * a ** 2


def _channels_at(layout: MuxLayout, values: torch.Tensor) -> torch.Tensor:
    """A per-channel vector at every pixel of its channel, as slots."""
    return layout.pack(values[:, None, None].expand(layout.channels, layout.height, layout.width))


def _layout(channels, size) -> MuxLayout:
    return MuxLayout(channels, size, size)


def _conv(weight, bias, image_size, *, stride=1, padding=1, dilation=1) -> HomMuxConv:
    op = HomMuxConv(kernel_shape=tuple(weight.shape), image_hw=(image_size, image_size), stride=stride,
                    padding=padding, dilation=dilation, bias=bias is not None)
    op.set_data(weight, bias)
    return op


# ---------- the encrypted circuit ----------

class _HerPN(HomOp):
    """Square the shifted, scaled input and add the per-channel offset: the HerPN, but for its gain."""

    def __init__(self, folded: _FoldedHerPN, layout: MuxLayout):
        super().__init__()
        self.square = HomMul()
        self.offset = HomConstAdd(dims=(layout.n_slots,)).set_data(_channels_at(layout, folded.offset))

    def forward(self, x: HomValue) -> HomValue:
        return self.offset(self.square(x, x))


class _Block(HomOp):
    """HerPN, conv, HerPN, conv, plus the shortcut from the first HerPN: an identity scaled per channel
    (a constant multiply) or a 1x1 conv with its BatchNorm. conv2 emits the next HerPN's input."""

    def __init__(self, state, prefix, block, herpn1: _FoldedHerPN, herpn2: _FoldedHerPN, next_herpn: _FoldedHerPN):
        super().__init__()
        c_in, c_out, stride, size = BLOCKS[block]
        in_layout, out_layout = _layout(c_in, size), _layout(c_out, size // stride)
        self.herpn1 = _HerPN(herpn1, in_layout)
        self.bootstrap = Bootstrap(target_output_scale=PT_SCALE) if block in BOOTSTRAPPED_BLOCKS else None
        weight1 = state[prefix + "conv1.weight"] * herpn1.gain[None, :, None, None]
        self.conv1 = _conv(weight1 * herpn2.input_scale[:, None, None, None], herpn2.shift, size, stride=stride)
        self.herpn2 = _HerPN(herpn2, out_layout)

        shortcut_bias = torch.zeros(c_out, dtype=torch.float64)
        if stride != 1 or c_in != c_out:
            bn = {name: state[f"{prefix}shortcut.1.{name}"] for name in ("weight", "bias", "running_mean", "running_var")}
            bn_scale = bn["weight"] / torch.sqrt(bn["running_var"] + BN_EPS)
            shortcut_bias = bn["bias"] - bn["running_mean"] * bn_scale
            weight = (state[prefix + "shortcut.0.weight"] * bn_scale[:, None, None, None]
                      * herpn1.gain[None, :, None, None] * next_herpn.input_scale[:, None, None, None])
            self.shortcut = _conv(weight, None, size, stride=stride, padding=0)
        else:
            gain = herpn1.gain * next_herpn.input_scale
            self.shortcut = HomConstMul(dims=(in_layout.n_slots,)).set_data(_channels_at(in_layout, gain))

        weight2 = state[prefix + "conv2.weight"] * herpn2.gain[None, :, None, None]
        bias2 = next_herpn.shift + shortcut_bias * next_herpn.input_scale
        self.conv2 = _conv(weight2 * next_herpn.input_scale[:, None, None, None], bias2, size // stride)

    def forward(self, x: HomValue) -> HomValue:
        z = self.herpn1(x)
        if self.bootstrap is not None:
            z = self.bootstrap(z)
        residual = self.conv2(self.herpn2(self.conv1(z)))
        return residual + self.shortcut(z)   # the add writes into its first operand, at the lower level


class _Backbone(HomOp):
    """One patch position's network, from the patch's slots to its share of the face's (scaled) embedding:
    stem conv, five blocks, the pooling HerPN, 4x4 window sums, and one conv for the average pool, flatten,
    BatchNorm and this patch's columns of the linear layer and the final BatchNorm."""

    def __init__(self, state, patch):
        super().__init__()
        prefix = f"nets.{patch}."
        block_prefixes = [f"{prefix}layers.{block}." for block in range(len(BLOCKS))]
        herpns = [[_FoldedHerPN(state, f"{block_prefix}herpn{k + 1}.", sigmas[k]) for k in range(2)]
                  for block_prefix, sigmas in zip(block_prefixes, HERPN_RANGE_SIGMAS)]
        pool = _FoldedHerPN(state, prefix + "herpnpool.herpn.", POOL_RANGE_SIGMAS)
        next_herpns = [herpn1 for herpn1, _ in herpns[1:]] + [pool]

        stem = torch.zeros(STEM_CHANNELS, STEM_CHANNELS, 3, 3, dtype=torch.float64)
        stem[:, :3] = state[prefix + "conv.weight"]
        self.stem = _conv(stem * herpns[0][0].input_scale[:, None, None, None], herpns[0][0].shift, PATCH_SIZE)
        self.blocks = ModuleListHomOp(_Block(state, block_prefix, block, *herpns[block], next_herpns[block])
                                      for block, block_prefix in enumerate(block_prefixes))

        channels, size, _ = POOL_LAYOUT
        pool_layout = MuxLayout(*POOL_LAYOUT)
        self.pool_herpn = _HerPN(pool, pool_layout)
        # each window's sum at its top-left pixel, in halving stages: a pixel is `channels` slots, a row `size` pixels
        halvings = [2 ** stage for stage in range(POOL_WINDOW.bit_length() - 1)]
        self.window_sums = ModuleListHomOp(HomRotateSum((step * pitch,))
                                           for pitch in (channels, size * channels) for step in halvings)
        self.head = _conv(*self._head(state, patch, pool), size, stride=POOL_WINDOW, padding=0, dilation=POOL_WINDOW)

    @staticmethod
    def _head(state, patch, pool: _FoldedHerPN):
        """The 2x2 conv (dilation 4) over the window sums that gives this patch's term of e / EMBEDDING_SCALE.

        e = bn((sum_p W_p (s_p f_p + t_p)) + b): W_p this patch's linear columns, s_p, t_p its own BatchNorm1d as
        an affine map, f_p its pooled features, bn the final (affine-free) BatchNorm. Each patch takes a
        quarter of the shared bias."""
        prefix = f"nets.{patch}.bn."
        s = state[prefix + "weight"] / torch.sqrt(state[prefix + "running_var"] + BN_EPS)
        t = state[prefix + "bias"] - state[prefix + "running_mean"] * s
        linear = state["linear.weight"][:, patch * EMBEDDING_DIM:(patch + 1) * EMBEDDING_DIM]
        bn_std = torch.sqrt(state["bn.running_var"] + BN_EPS)
        bias = (linear @ t + (state["linear.bias"] - state["bn.running_mean"]) / N_PATCHES) / bn_std
        matrix = linear * s[None, :] / bn_std[:, None]
        # flatten order: channel, then the 2x2 window grid; the window sums still lack the HerPN's gain and the
        # average's 1 / 16
        weight = matrix.reshape(EMBEDDING_DIM, POOL_LAYOUT[0], 2, 2) * pool.gain[None, :, None, None] / POOL_WINDOW ** 2
        return weight / EMBEDDING_SCALE, bias / EMBEDDING_SCALE

    def forward(self, x: HomValue) -> HomValue:
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = self.pool_herpn(x)
        for window_sum in self.window_sums:
            x = window_sum(x)
        return self.head(x)


class _CosineScore(HomOp):
    """<e0, e1> / (|e0| |e1|) of the two scaled embeddings (a batch of two): products summed over the slots,
    and 1/|e| as a Chebyshev polynomial in |e|^2, both faces' at once."""

    def __init__(self):
        super().__init__()
        self.product = HomMul()
        # the sum of the embedding's 256-slot period, in every slot: the polynomial's constants need the slot axis
        self.sum = HomSumSlots(reduce_output=False)
        self.inverse_norm = HomPolyEval(lambda t: 1 / np.sqrt(t), degree=INV_NORM_DEGREE,
                                        left=INV_NORM_LO, right=INV_NORM_HI)
        self.scale = HomMul()

    def forward(self, embeddings: HomValue) -> HomValue:
        inverse = self.inverse_norm(self.sum(self.product(embeddings, embeddings)))
        dot = self.sum(self.product(embeddings[0], embeddings[1]))
        return self.scale(self.scale(dot, inverse[0]), inverse[1])


class _CryptoFaceGraph(HomOp):
    """Eight patches (each position's on both faces) to the cosine score, in every slot."""

    def __init__(self, state):
        super().__init__()
        # one network per patch position, run on both faces at once
        self.backbones = ModuleListHomOp(_Backbone(state, patch) for patch in range(N_PATCHES))
        self.bootstrap = Bootstrap(target_output_scale=PT_SCALE)
        self.score = _CosineScore()

    def forward(self, patches: HomValue) -> HomValue:
        terms = [backbone(patches[patch]) for patch, backbone in enumerate(self.backbones)]
        return self.score(self.bootstrap((terms[0] + terms[1]) + (terms[2] + terms[3])))


# ---------- example pairs ----------

def _stem_layout() -> MuxLayout:
    return MuxLayout(STEM_CHANNELS, PATCH_SIZE, PATCH_SIZE, SUBRING_SLOTS)


def _normalization() -> tuple[torch.Tensor, torch.Tensor]:
    """Per-slot scale and shift that turn the client's raw 0-255 pixels into x / 127.5 - 1."""
    rgb = torch.zeros(STEM_CHANNELS, dtype=torch.float64)
    rgb[:3] = 1
    return _channels_at(_stem_layout(), rgb / 127.5), _channels_at(_stem_layout(), -rgb)


class _NumpyOnly(pickle.Unpickler):
    """The benchmark's pairs are a pickled object array of JPEG bytes: allow numpy's own globals only."""

    _ALLOWED = {("numpy", "ndarray"), ("numpy", "dtype"), ("numpy._core.multiarray", "_reconstruct"),
                ("numpy.core.multiarray", "_reconstruct")}

    def find_class(self, module, name):
        if (module, name) not in self._ALLOWED:
            raise pickle.UnpicklingError(f"refusing {module}.{name} in the face pairs file")
        return super().find_class(module, name)


def _download(url: str) -> Path:
    path = Path(torch.hub.get_dir()) / "cryptoface" / url.rsplit("/", 1)[-1]
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(url, str(path))
    return path


@cache
def _benchmark_pairs() -> tuple[np.ndarray, np.ndarray]:
    """The benchmark's 1,024 pairs of JPEG bytes and their labels (1: same person)."""
    with _download(PAIRS_URL).open("rb") as file:
        version = np.lib.format.read_magic(file)
        np.lib.format.read_array_header_1_0(file) if version == (1, 0) else np.lib.format.read_array_header_2_0(file)
        pairs = _NumpyOnly(file).load()
    labels = np.array([int(line) for line in _download(LABELS_URL).read_text().split()], dtype=np.int8)
    return pairs, labels


@cache
def _face_detector():
    """InsightFace's detector as the benchmark's client runs it, or None without InsightFace."""
    try:
        from insightface.app import FaceAnalysis
    except ImportError:
        print("[cryptoface] InsightFace is not installed: center-cropping the example faces, not aligning them")
        return None
    detector = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"], allowed_modules=["detection"])
    detector.prepare(ctx_id=-1, det_size=(640, 640))
    return detector


def align_face(jpeg: bytes) -> np.ndarray:
    """A photo as the model's 64x64 RGB input (uint8, HWC): InsightFace's 5-point alignment to 112x112,
    resized to 64 (the benchmark's preprocessing), or a center crop without InsightFace or a face."""
    from PIL import Image

    image = Image.open(io.BytesIO(jpeg)).convert("RGB")
    detector = _face_detector()
    if detector is not None:
        import cv2
        from insightface.utils.face_align import norm_crop

        bgr = np.asarray(image)[:, :, ::-1].copy()
        faces = detector.get(bgr)
        if faces:
            aligned = norm_crop(bgr, faces[0].kps, image_size=112)
            return cv2.resize(aligned, (FACE_SIZE, FACE_SIZE), interpolation=cv2.INTER_LINEAR)[:, :, ::-1].copy()
    side = min(image.size)
    left, top = (image.width - side) // 2, (image.height - side) // 2
    crop = image.crop((left, top, left + side, top + side)).resize((FACE_SIZE, FACE_SIZE), Image.BILINEAR)
    return np.asarray(crop)


class Pipeline(PipelineWrapper):
    def __init__(self, pair_index: int | None = None):
        """pair_index: the benchmark pair the example query sends; None picks one at random."""
        self.pair_index = pair_index

    def build_pipeline(self):
        scale, shift = _normalization()
        return HomomorphicPipeline(
            # the client normalizes the raw pixels it packed
            client_pre=[
                HomConstMul(dims=(SUBRING_SLOTS,)).set_data(scale),
                HomConstAdd(dims=(SUBRING_SLOTS,)).set_data(shift),
            ],
            hom=_CryptoFaceGraph(_load_checkpoint()),
            # the score fills every slot
            client_post=[HomSlice(dim=0, key=slice(0, 1))],
            input_shape=(N_PATCHES, 2, SUBRING_SLOTS),
        )

    def build_params(self):
        return HomParams(
            full_q_list_precision=Q_LIST_PRECISION,
            n=N,
            pt_scale=PT_SCALE,
            sk_hw=192,
            num_special_primes=N_SPECIAL_PRIMES,
            n_slots=SUBRING_SLOTS,
        )

    @staticmethod
    def load_pair(index: int | None = None) -> tuple[np.ndarray, np.ndarray, bool]:
        """A benchmark pair as two 64x64 RGB faces (uint8, HWC), and whether they are the same person."""
        pairs, labels = _benchmark_pairs()
        if index is None:
            index = random.randrange(len(labels))
        face0, face1 = (align_face(bytes(jpeg)) for jpeg in pairs[index])
        return face0, face1, bool(labels[index])

    @staticmethod
    def pack_pair(face0, face1) -> torch.Tensor:
        """Two 64x64 RGB faces (uint8, HWC), raw 0-255, as the (4 patch positions, 2 faces, 2^14) slots the
        client encrypts."""
        packed = []
        for face in (face0, face1):
            image = torch.tensor(np.asarray(face), dtype=torch.float64).permute(2, 0, 1)
            # RGB zero-padded to the stem's channels, so the stem has the layout of block 1
            patches = [torch.nn.functional.pad(patch, (0, 0, 0, 0, 0, STEM_CHANNELS - 3)) for patch in _patches(image)]
            packed.append(torch.stack([_stem_layout().pack(patch) for patch in patches]))
        return torch.stack(packed, dim=1)

    @staticmethod
    def reference_score(face0, face1) -> float:
        """The plaintext model's cosine of the two faces' embeddings."""
        faces = torch.as_tensor(np.stack([face0, face1]), dtype=torch.float64).permute(0, 3, 1, 2) / 127.5 - 1
        with torch.no_grad():
            embeddings = load_torch_model()(faces)
        return torch.nn.functional.cosine_similarity(embeddings[0], embeddings[1], dim=0).item()

    def _set_example_pt(self):
        self._face0, self._face1, self._same = self.load_pair(self.pair_index)
        return self.pack_pair(self._face0, self._face1)

    def verify_results(self, actual, expected):
        # actual: the decrypted score; expected: the runtime's clear run of this circuit
        score = actual.reshape(-1)[0].item()
        reference = self.reference_score(self._face0, self._face1)
        print(f"[cryptoface] same person: {self._same}   |   score (decrypted): {score:.5f}   |   "
              f"circuit in the clear: {expected.reshape(-1)[0].item():.5f}   |   torch model: {reference:.5f}")
        torch.testing.assert_close(actual.reshape(-1)[:1], expected.reshape(-1)[:1], rtol=0, atol=2 ** -10)
        assert abs(score - reference) < 2 ** -7, f"decrypted score {score} vs the torch model's {reference}"
