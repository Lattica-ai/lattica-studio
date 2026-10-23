"""RESEARCH (NeuJeans-style ResNet-20, not for merge): the weights side of convolutions fused into the bootstrap's
SlotsToCoefs (Ju et al., NeuJeans, CCS'24).

Layout. Every feature map is a set of slices, one per (channel, phase): a slice is a polynomial of length SLICE = 128
holding an 8x8 sub-image, pixel (i, j) at coefficient i * ROW + j (ROW = 9: one zero column, then a zero tail). A map
at resolution 8 p is kept in p x p polyphase form, sub-image (a, b)[i, j] = X[p i + a, p j + b]: L1 (32x32) 16
channels x 16 phases, L2 (16x16) 32 x 4, L3 (8x8) 64 x 1. Slice s = (channel * p + a) * p + b.

Values. After CtS and EvalMod a bootstrap holds the values (the coefficients of its input) in slots, in the order
that SlotsToCoefs (StC) takes them back: CtS(StC(v)) = v. Slice s fills the SLICE contiguous slots of block s, its
coefficient c at slot BITREV[c]. The values form [x, y] (2 n_slots reals) and StC returns StC_plain(x + i y):
  repacked (a complex bootstrap on a proper subring, h_dft._repacked_factors): one vector [x, y] of 2 n_slots slots;
  full (n_slots = N / 2, no repack): x and y are two vectors of n_slots slots (EvalMod's real and imaginary parts).
x holds slices [0, n_slots / SLICE), y the others.

StC split. The library's StC factors in application order are F_0 .. F_{log2 n_slots - 1}; F_s pairs slots at
distance 2^s. S1 = F_{LOG_SLICE - 1} .. F_0 acts inside each block, the same map in every block and at every size
(a row-scaled evaluation map: CinS), so a convolution of slices is an elementwise product after S1, with the
kernel's spectrum. S2 = the remaining factors. A fused tail is S2 (+ the repack's join I + R) times the conv in S1's
basis, one matrix applied after S1:
  repacked input: tails = [T]       output = T S1 D [x, y]          (D = diag(1, i), folded into S1's first factor)
  full input:     tails = [Tx, Ty]  output = Tx S1 x + Ty S1 y
The output is the coefficient-encoded conv output: StC of its values, repeated with the output's period (a stride-2
transition's output has fewer slots than its input). Each tail is square on the input's StC working dimension
(2 n_slots repacked, n_slots full), stored as the library's DiagonalMatrix diagonals (input-aligned:
(M v)[r] = sum_rot v[r + rot] diag_rot[r + rot]); every rotation is a multiple of SLICE.
"""
import math
from dataclasses import dataclass, field

import numpy as np
import torch
from numpy.polynomial import chebyshev

SLICE = 128
LOG_SLICE = 7
SUB = 8
ROW = SUB + 1
STAGE_CHANNELS = {1: 16, 2: 32, 3: 64}
STAGE_PHASES = {1: 4, 2: 2, 3: 1}
LOG_DELTA = 4


def _bitrev(n):
    k = int(math.log2(n))
    return np.array([int(format(i, f"0{k}b")[::-1], 2) for i in range(n)])


BITREV = _bitrev(SLICE)
IMAGE_MASK = np.zeros(SLICE)                 # coefficient order: 1 at the image positions
for _i in range(SUB):
    IMAGE_MASK[_i * ROW:_i * ROW + SUB] = 1
SLOT_MASK = IMAGE_MASK[BITREV]               # slot order inside a block


# ---- packing ----
def pack(x, p):
    """x [..., C, 8p, 8p] -> slices [..., C * p * p, SLICE] (coefficient order)."""
    lead = x.shape[:-3]
    c = x.shape[-3]
    out = np.zeros(lead + (c, p, p, SLICE))
    for a in range(p):
        for b in range(p):
            sub = x[..., a::p, b::p]
            for i in range(SUB):
                out[..., a, b, i * ROW:i * ROW + SUB] = sub[..., i, :]
    return out.reshape(lead + (c * p * p, SLICE))


def unpack(s, c, p):
    """slices [..., C * p * p, SLICE] -> [..., C, 8p, 8p]"""
    lead = s.shape[:-2]
    s = s.reshape(lead + (c, p, p, SLICE))
    x = np.zeros(lead + (c, p * SUB, p * SUB), dtype=s.dtype)
    for a in range(p):
        for b in range(p):
            for i in range(SUB):
                x[..., a + p * i, b::p] = s[..., a, b, i * ROW:i * ROW + SUB]
    return x


def slices_to_slots(s):
    """slices [..., S, SLICE] (coefficient order) -> values [..., S * SLICE] (slot order)"""
    return s[..., BITREV].reshape(s.shape[:-2] + (-1,))


def slots_to_slices(v):
    return v.reshape(v.shape[:-1] + (-1, SLICE))[..., BITREV]


# ---- the library's StC (ckks_cyclic_packing.generate_transform_factors, SlotsToCoefs) as butterflies ----
def _twiddles(m):
    """factor with groups of m: w_j = omega^((5^j mod 4m) (n_slots / m)), omega = exp(i pi / (2 n_slots))"""
    return np.exp(1j * np.pi * np.array([pow(5, j, 4 * m) for j in range(m // 2)]) / (2 * m))


def stc_stages(v, first, last):
    """StC factors first .. last - 1 on the last axis: out[u] = v[u] + w v[u + m/2], out[u + m/2] = v[u] - w v[u + m/2]"""
    v = np.asarray(v, dtype=complex)
    shape = v.shape
    for s in range(first, last):
        m = 2 ** (s + 1)
        g = v.reshape(shape[:-1] + (-1, m))
        lo, hi = g[..., :m // 2], _twiddles(m) * g[..., m // 2:]
        v = np.concatenate((lo + hi, lo - hi), axis=-1).reshape(shape)
    return v


def stc_plain(w):
    """the library's plain StC on n_slots (the last axis) complex slots"""
    return stc_stages(w, 0, int(math.log2(w.shape[-1])))


def s1(v):
    return stc_stages(v, 0, LOG_SLICE)


def _s1_block():
    """S1 on one block, columns in coefficient order: B[:, c] = S1(coefficient c)"""
    return stc_stages(np.eye(SLICE), 0, LOG_SLICE).T[:, BITREV]


S1_BLOCK = _s1_block()
S1_ROW_SCALE = S1_BLOCK[:, 0]
S1_POINTS = S1_BLOCK[:, 1] / S1_ROW_SCALE    # S1_BLOCK = diag(row scale) Vandermonde(points)


def s2_blocks(n_slots):
    """S2 at n_slots as blocks: S2[a, b, t] = the entry from slot (b, t) to slot (a, t) (S2 is diagonal per block)."""
    nb = n_slots // SLICE
    x = np.zeros((nb, nb, SLICE))
    x[np.arange(nb), np.arange(nb)] = 1
    y = stc_stages(x.reshape(nb, n_slots), LOG_SLICE, int(math.log2(n_slots)))
    return y.reshape(nb, nb, SLICE).transpose(1, 0, 2)


# ---- convolutions in S1's basis ----
def _shift_spectrum(t):
    """the spectrum of Y^-t: the product with it reads x[n + t] (wrapped terms only meet the zero padding)"""
    return S1_POINTS ** (-t)


def conv_spectra(weight, stride, p_in):
    """weight [Co, Ci, kh, kw] (padding kh // 2) -> K [Co p_out^2, Ci p_in^2, SLICE] in S1's basis:
    S1(out slice o) = sum_c K[o, c] * S1(in slice c)."""
    co, ci, kh, kw = weight.shape
    p_out = p_in // stride
    K = np.zeros((co, p_out, p_out, ci, p_in, p_in, SLICE), dtype=complex)
    spectra = {}
    for c1 in range(p_out):
        for d1 in range(p_out):
            for di in range(-(kh // 2), kh // 2 + 1):
                for dj in range(-(kw // 2), kw // 2 + 1):
                    r, s = stride * c1 + di, stride * d1 + dj
                    a, si = r % p_in, r // p_in
                    b, sj = s % p_in, s // p_in
                    t = si * ROW + sj
                    if t not in spectra:
                        spectra[t] = _shift_spectrum(t)
                    K[:, c1, d1, :, a, b, :] += weight[:, :, di + kh // 2, dj + kw // 2, None] * spectra[t]
    return K.reshape(co * p_out * p_out, ci * p_in * p_in, SLICE)


def identity_spectra(n_slices, scale=1.0):
    K = np.zeros((n_slices, n_slices, SLICE), dtype=complex)
    K[np.arange(n_slices), np.arange(n_slices)] = scale
    return K


# ---- fused tails ----
@dataclass
class StcForm:
    """A stage's StC: n_slots of the bootstrap, repacked or full (x and y in two vectors)."""
    n_slots: int
    repacked: bool

    @property
    def dimension(self):
        return 2 * self.n_slots if self.repacked else self.n_slots

    @property
    def n_parts(self):
        return 1 if self.repacked else 2


def stage_form(stage, n):
    """the StC form of a stage's bootstrap in a ring of dimension n (2 n_slots real values)"""
    values = STAGE_CHANNELS[stage] * STAGE_PHASES[stage] ** 2 * SLICE
    n_slots = values // 2
    if n_slots > n // 2:
        raise ValueError(f"stage {stage} needs {values} real slots, the ring has {n // 2} complex")
    return StcForm(n_slots, repacked=n_slots < n // 2)


@dataclass
class FusedTail:
    """matrices: one per input part (the library's DiagonalMatrix diagonals {rotation: complex [dimension]});
    bias: the slot-domain vector to add after the tails (StC of the bias values, repeated), or None."""
    name: str
    dimension: int
    output_slots: int                  # the output's period (its bootstrap's n_slots)
    matrices: list
    bias: np.ndarray = None
    blocks: list = field(default=None, repr=False)   # [rows, cols, SLICE] per part (rows: the output period's blocks)

    def apply(self, parts):
        """parts: S1 outputs [..., dimension] per part -> the output [..., dimension] (block form)"""
        out = 0
        for blk, x in zip(self.blocks, parts):
            lead = x.shape[:-1]
            xb = x.reshape((-1, x.shape[-1] // SLICE, SLICE)).transpose(2, 1, 0)       # [t, c, batch]
            y = np.matmul(blk.transpose(2, 0, 1), xb).transpose(2, 1, 0)              # [batch, a, t]
            reps = xb.shape[1] // y.shape[1]
            out = out + np.concatenate([y] * reps, axis=1).reshape(lead + (x.shape[-1],))
        if self.bias is not None:
            out = out + self.bias
        return out


def _to_diagonals(blk, dimension):
    """blocks [nb_o, nb, SLICE] (rows repeated with period nb_o) -> {rotation: diagonal} (input-aligned)"""
    nb = dimension // SLICE
    rows = np.arange(nb)
    diags = {}
    for e in range(nb):
        d = blk[(rows + e) % blk.shape[0], rows]          # entry from block b to block b + e
        if np.any(d):
            diags[(-e * SLICE) % dimension] = d.reshape(dimension)
    return diags


def fused_tail(name, K, form_in, form_out, bias_values=None):
    """K [2 nb_out, n_in_slices, SLICE]: S1(out slice o) = sum_c K[o, c] S1(in slice c) -> FusedTail."""
    nb_o = form_out.n_slots // SLICE
    assert K.shape[0] == 2 * nb_o, (K.shape, form_out)
    # output slice o is block o mod nb_o of x' + i y'
    Kc = K[:nb_o] + 1j * K[nb_o:]
    T = np.matmul(s2_blocks(form_out.n_slots).transpose(2, 0, 1), Kc.transpose(2, 0, 1)).transpose(1, 2, 0)
    if form_in.repacked:
        nb_half = form_in.n_slots // SLICE
        scale_in = np.where(np.arange(K.shape[1]) < nb_half, 1, 1j)
        parts = [T / scale_in[None, :, None]]
    else:
        nb_i = form_in.n_slots // SLICE
        parts = [T[:, :nb_i], T[:, nb_i:]]
    dim = form_in.dimension
    assert all(p.shape[1] * SLICE == dim for p in parts)
    bias = None
    if bias_values is not None:
        # per output slice constant at every position (the padding is masked after the next ReLU)
        v = np.repeat(bias_values, SLICE).reshape(2, -1)
        t = stc_plain(v[0] + 1j * v[1])
        bias = np.tile(t, dim // t.size)
    return FusedTail(name, dim, form_out.n_slots, [_to_diagonals(p, dim) for p in parts], bias, parts)


def input_mask(form):
    """the 0/1 value mask (slot order) per input part, to fold into S1's first factor (re-zeroes the padding)"""
    return np.tile(SLOT_MASK, form.dimension // SLICE)


# ---- ReLU folded with EvalMod's sine ----
def relu_poly(abs_coefs):
    """max(0, u) ~ (u + p(u)) / 2, p the Chebyshev series of |u| (T0 first): the Chebyshev series of the ReLU"""
    c = np.asarray(abs_coefs, dtype=np.float64) / 2
    c[1] += 0.5
    return c


def fold_inverse_sine(coefs, samples=None, degree=None, log_delta=LOG_DELTA, lam=1e-2, u_max=1.0):
    """Chebyshev coefs P with P(y) ~ f(m(y)), f the series coefs, y = D sin(2 pi m / D) / (2 pi) what EvalMod returns
    for the message m (D = 2^log_delta), m(y) its inverse: least squares over m in [-u_max, u_max], weighted by the
    samples of |m| (total weight 1, both signs) plus lam of a uniform weight."""
    D = 2.0 ** log_delta
    degree = len(coefs) - 1 if degree is None else degree
    uni = np.linspace(-u_max, u_max, 20001)
    m, w = uni, np.full(uni.size, lam / uni.size)
    if samples is not None:
        s = np.minimum(np.abs(samples), u_max)
        m = np.concatenate((m, s, -s))
        w = np.concatenate((w, np.full(2 * s.size, 0.5 / s.size)))
    y = D * np.sin(2 * np.pi * m / D) / (2 * np.pi)
    return chebyshev.chebfit(y, chebyshev.chebval(m, coefs), degree, w=np.sqrt(w))


def eval_mod_sine(u, log_delta=LOG_DELTA):
    D = 2.0 ** log_delta
    return D * np.sin(2 * np.pi * u / D) / (2 * np.pi)


# ---- the network ----
def bn_fold(conv, bn):
    """conv + BN -> (weight [Co, Ci, kh, kw], bias [Co]) float64"""
    g = (bn.weight / torch.sqrt(bn.running_var + bn.eps)).detach().double()
    w = conv.weight.detach().double() * g[:, None, None, None]
    beta = (bn.bias.detach().double() - bn.running_mean.detach().double() * g)
    return w.numpy(), beta.numpy()


@dataclass
class NJBoot:
    """One bootstrap: ReLU (folded Chebyshev coefs on EvalMod's output) -> mask -> S1 -> tails.
    tails: FusedTail outputs (each one coefficient-encoded ciphertext), or head (the last bootstrap)."""
    name: str
    form: StcForm
    relu_coefs: np.ndarray
    mask: np.ndarray
    tails: list
    head: "NJHead" = None


@dataclass
class NJHead:
    """logits = rotate-sum(matrix @ relu output) + bias: the matrix pools each block's masked values (avgpool, the
    ReLU's 1/delta) and applies the FC, writing class k at slot k of every block; the rotations (multiples of SLICE)
    sum the blocks; logit k is then at slot k (of every block)."""
    dimension: int
    matrix: dict
    rotations: list
    bias: np.ndarray
    blocks: np.ndarray = field(default=None, repr=False)   # [nb, 10, SLICE]: per block, class x slot

    def apply(self, r):
        rb = r.reshape(r.shape[:-1] + (-1, SLICE))
        return np.einsum("bkt,...bt->...k", self.blocks, rb) + self.bias[:10]


def head(model, delta, form, n_classes=10):
    """avgpool + FC on the last ReLU's values (L3 layout, channel c = block c)"""
    w = model.fc.weight.detach().double().numpy() / (SUB * SUB * delta)    # [10, 64]
    b = model.fc.bias.detach().double().numpy()
    dim = form.dimension
    nb = dim // SLICE
    blocks = w.T[:, :, None] * SLOT_MASK[None, None, :]                  # [nb, 10, SLICE]
    # matrix entry from slot (b, t) to slot (b, k): rotation t - k, diagonal at the input position (b, t)
    diags = {}
    for k in range(n_classes):
        for t in np.nonzero(SLOT_MASK)[0]:
            rot = (int(t) - k) % dim
            d = diags.setdefault(rot, np.zeros(dim, complex))
            d[np.arange(nb) * SLICE + t] += blocks[:, k, t]
    bias = np.zeros(dim)
    bias[(np.arange(nb)[:, None] * SLICE + np.arange(n_classes)[None]).reshape(-1)] = np.tile(b, nb)
    rotations = [SLICE * 2 ** j for j in range(int(math.log2(nb)))]
    return NJHead(dim, diags, rotations, bias, blocks)


def _names():
    return ["stem"] + [f"L{s}B{b}r{r}" for s in (1, 2, 3) for b in (1, 2, 3) for r in (1, 2)]


def build_network(model, relu_table, delta_initial, deltas, n=2 ** 15, samples=None, fold=True,
                  log_delta=LOG_DELTA, folded_table=None):
    """The NeuJeans ResNet-20: the stem tail (on the client's S1(image)) and the 19 bootstraps.

    relu_table[name]: Chebyshev coefs of |u| on [-1, 1] per ReLU (u = delta * pre-activation, the bootstrap's input);
    deltas: resnet20.DELTAS. Each bootstrap outputs max(0, u) on the values (delta times the activation); the next
    linear maps take the 1/delta, the ReLU after them its own delta (and the bias, BN folded).
    samples[name]: |u| samples for the data-weighted fold (optional); folded_table[name]: coefs already folded (e.g.
    data/resnet20_relu_coefs_nj_folded_ld4.json), used as they are. Returns (stem_tail, boots)."""
    delta = {"stem": delta_initial}
    for s in (1, 2, 3):
        for b in (1, 2, 3):
            delta[f"L{s}B{b}r1"], delta[f"L{s}B{b}r2"] = deltas[s - 1][b - 1]
    forms = {s: stage_form(s, n) for s in (1, 2, 3)}

    def relu_of(name):
        if folded_table is not None:
            return np.asarray(folded_table[name], dtype=np.float64)
        rc = relu_poly(relu_table[name])
        if not fold:
            return rc
        return fold_inverse_sine(rc, None if samples is None else samples.get(name), log_delta=log_delta)

    # stem: the client encrypts S1(x) of the RGB image in the L1 layout (x part; y = 0)
    w, beta = bn_fold(model.conv1, model.bn1)
    w16 = np.zeros((w.shape[0], STAGE_CHANNELS[1]) + w.shape[2:])
    w16[:, :w.shape[1]] = w
    d0 = delta["stem"]
    stem = fused_tail("stem", conv_spectra(w16 * d0, 1, STAGE_PHASES[1]), forms[1], forms[1],
                      np.repeat(beta * d0, STAGE_PHASES[1] ** 2))
    boots = []
    prev_name, stage = "stem", 1
    for s, layer in zip((1, 2, 3), (model.layer1, model.layer2, model.layer3)):
        for b, blk in enumerate(layer, start=1):
            r1, r2 = f"L{s}B{b}r1", f"L{s}B{b}r2"
            stride = blk.conv1.stride[0]
            f_in, f_out = forms[stage], forms[s]
            p_in, p_out = STAGE_PHASES[stage], STAGE_PHASES[s]
            d_in = delta[prev_name]
            # the input's bootstrap (prev_name): conv1 and the shortcut
            w1, b1 = bn_fold(blk.conv1, blk.bn1)
            tails = [fused_tail(f"{r1}.conv1", conv_spectra(w1 * delta[r1] / d_in, stride, p_in), f_in, f_out,
                                np.repeat(b1 * delta[r1], p_out ** 2))]
            if blk.downsample is not None:
                wd, bd = bn_fold(blk.downsample[0], blk.downsample[1])
                tails.append(fused_tail(f"{r2}.shortcut", conv_spectra(wd * delta[r2] / d_in, stride, p_in), f_in,
                                        f_out, np.repeat(bd * delta[r2], p_out ** 2)))
            else:
                n_sl = STAGE_CHANNELS[s] * p_out ** 2
                tails.append(fused_tail(f"{r2}.shortcut", identity_spectra(n_sl, delta[r2] / d_in), f_in, f_out))
            boots.append(NJBoot(prev_name, f_in, relu_of(prev_name), input_mask(f_in), tails))
            # the block's first ReLU: conv2
            w2, b2 = bn_fold(blk.conv2, blk.bn2)
            boots.append(NJBoot(r1, f_out, relu_of(r1), input_mask(f_out),
                                [fused_tail(f"{r2}.conv2", conv_spectra(w2 * delta[r2] / delta[r1], 1, p_out), f_out,
                                            f_out, np.repeat(b2 * delta[r2], p_out ** 2))]))
            prev_name, stage = r2, s
    boots.append(NJBoot(prev_name, forms[3], relu_of(prev_name), input_mask(forms[3]), [],
                        head(model, delta[prev_name], forms[3])))
    return stem, boots
