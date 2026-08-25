"""
minimax_v3.py — minimax polynomials for CKKS: composite sign + EvalMod (mod reduction).

Two classes only:
    Remez       — fits one minimax polynomial to a function over a union of intervals.
    Polynomial  — a composite P = p_k o ... o p_0 (each stage a Chebyshev series
                  on [-1, 1]); evaluates, reports depth, and plots.

Plus module-level helper functions and two high-level entry points:
    sign_minimax       — brute-force searches degree schedules for the SMALLEST evaluation
                         depth that hits a requested accuracy both in y (closeness of the
                         output to +/-1) and in x (how close to 0 inputs can be).
    eval_mod_minimax   — builds the homomorphic modular reduction normod(x) = x - round(x)
                         as the composition h3~ . h2^ell . h1~ of an optimal-minimax scaled
                         COSINE (multi-interval) and INVERSE SINE (single interval), the
                         method of Lee et al., "High-Precision Bootstrapping of RNS-CKKS ...
                         Optimal Minimax Polynomial Approximation and Inverse Sine Function"
                         (eprint 2020/552).

------------------------------------------------------------------------------------------
Two Remez modes
------------------------------------------------------------------------------------------
The default mode (general=False) is tuned for the sign function: because sign is
piecewise-constant, the error extrema coincide with the extrema of the polynomial p, which
are cheap to find as the roots of p'. The improved mode (general=True) implements the
multi-interval Remez of Lee et al. for a GENERAL (non-constant) target. It differs in two
ways that matter for the cosine and inverse sine:
  * error extrema are extrema of r = p - f (found by a fine scan + binary/ternary search),
    not extrema of p;
  * the n+1 references are chosen from all the extrema by the paper's three criteria
    (Algorithm 3): the local-extreme-value condition (mu*r >= |E|, applied with an adaptive
    threshold for numerical slack), the alternating condition, and the maximum-absolute-sum
    condition. A precision-aware noise floor discards spurious extrema that appear when the
    solve overfits toward the arithmetic floor.

------------------------------------------------------------------------------------------
EvalMod composition
------------------------------------------------------------------------------------------
    normod(x) = x - round(x)  ~  h3~ . h2(.) applied ell times . h1~ (x)
        h1(x) = cos((2*pi/2^ell)(x - 1/4))    scaled cosine, minimax over U_i [i-eps, i+eps]
        h2(y) = 2*y^2 - 1  ( = T_2 )          double-angle: ell steps send cos(theta/2^ell)
                                              to cos(theta) = cos(2pi(x-1/4)) = sin(2pi x)
        h3(y) = (1/2pi) arcsin(y)             inverse sine, minimax over [-s, s], s=sin(2pi eps)
h1~ and h3~ are fit by the improved multi-interval Remez; h2 is exact.

------------------------------------------------------------------------------------------
Working envelope (verified by running; see benchmark_paper.py)
------------------------------------------------------------------------------------------
  * Inverse sine (single interval): converges to the true minimax across the paper's degree
    range; float64 equioscillates to about 2^-33 (deg 9) and the reported minimax error is
    accurate to about 2^-47 (deg 15), improving ~2.5 bits/degree. Coefficients come out
    exactly odd. Pass precision=<bits> to equioscillate deeper.
  * Scaled cosine (multi-interval): cosine_minimax uses DEGREE CONTINUATION (fit a low
    degree, then raise it by 2 a step at a time, re-seeding the references each step) in two
    phases. Phase 1 runs in float64 and tracks the paper's Table 2 curve up to its
    conditioning floor (~2^-31, degree ~58 at K=25, eps=2^-12). Phase 2, enabled by
    precision=<bits> or precision=True, carries the continuation on IN MPMATH from the last
    converged degree, finding the sub-float64 error extrema in full precision; in high
    precision the global reference selection is stable (an ill-conditioned solve no longer
    blows up). This reproduces the paper's Table 2 EXACTLY through degree 68:
        deg 60 -> 2^-35.72   deg 64 -> 2^-44.89   deg 68 -> 2^-53.75
        deg 62 -> 2^-40.79   deg 66 -> 2^-49.19   (paper rows match to <0.1 bit)
  * Full composite normod validated end to end (e.g. ell=2, K=8, eps=2^-6, cos deg 24,
    arcsin deg 9 -> normod error ~2^-27).

Performance: the float64 phase is vectorised (the whole error-extrema scan AND the batched
ternary refinement are single numpy calls). The mpmath phase uses a hand-rolled Gaussian
elimination on flat mpf lists (far faster than mpmath's matrix LU here) and golden-section
extremum search; a degree-68 fit (the paper's deepest) runs in well under two minutes.

Plotting (matplotlib): Remez.plot() draws a single fit and its error, exposing the
equioscillation (target vs approximant for a single interval; the error stitched across all
subintervals for the multi-interval cosine). plot_eval_mod(em) draws the composite: the true
sawtooth normod with the approximation and the shaded approximation intervals, plus the error
across D -- the analogue of the sign composite plot.

Background (sign): a single polynomial approximating sign(x) on [-1,-g] U [g,1] needs huge
degree when g is tiny. Instead we COMPOSE several moderate-degree minimax polynomials; each
one widens the usable gap, so the overall cost grows with depth, not with 1/g.

See `operators/polynomials/README.md` for usage details."""
from __future__ import annotations
import math
from itertools import combinations_with_replacement
from types import SimpleNamespace
import numpy as np
from numpy.polynomial.chebyshev import poly2cheb, cheb2poly, chebval, chebder, chebroots

try:
    import mpmath as mp                                       # optional: only for >float64 precision
except Exception:
    mp = None


# ======================================================================
# Degree / depth model
# ----------------------------------------------------------------------
# A degree-d polynomial is evaluated with a Paterson-Stockmeyer / BSGS scheme
# whose multiplicative depth is ceil(log2(d)) once d is rounded up to one of the
# supported "bucket" degrees below. POLY_EVAL_DEG are the only degrees we ever
# instantiate; each costs DEPTH_OF_DEGREE[d] levels.
# ======================================================================
POLY_EVAL_DEG = [5, 13, 27, 59, 119, 247]                    # supported per-stage degrees
DEPTH_OF_DEGREE = {d: int(math.ceil(math.log2(d))) for d in POLY_EVAL_DEG}
#   {5:3, 13:4, 27:5, 59:6, 119:7, 247:8}
DEPTH_TO_DEGREE = {depth: d for d, depth in DEPTH_OF_DEGREE.items()}


def _bucket_depth(degree):
    """Depth charged for an arbitrary polynomial `degree`: round up to the next
    supported bucket and return its evaluation depth (levels)."""
    for d in POLY_EVAL_DEG:
        if degree <= d:
            return DEPTH_OF_DEGREE[d]
    raise ValueError(f"degree {degree} exceeds largest supported degree {POLY_EVAL_DEG[-1]}")


# ======================================================================
# Small Chebyshev helpers (float64 via numpy; an mpmath path lives in Remez)
# ======================================================================
def _design_matrix(nodes, basis_degrees):
    """Build M[i, j] = T_{basis_degrees[j]}(nodes[i]).

    nodes         : reference points (in the hull-normalized coordinate u in [-1,1]).
    basis_degrees : which Chebyshev degrees are free parameters (all degrees, or only
                    odd ones when the target is odd).
    Returns a (len(nodes) x len(basis_degrees)) numpy matrix.
    """
    pts = np.asarray(nodes, dtype=float)
    columns = []
    for k in basis_degrees:
        unit = np.zeros(k + 1); unit[k] = 1.0                # coeffs of the single basis fn T_k
        columns.append(chebval(pts, unit))
    return np.vstack(columns).T


def _error_extrema(coeffs, subdomains_u):
    """Locations (in u) of all local extrema of p(u)=sum_k coeffs[k] T_k(u) that lie
    inside the given subdomains, plus each subdomain's endpoints.

    Extrema are the real roots of p'(u), found exactly via the Chebyshev colleague
    matrix (chebroots) — no scanning, nothing missed. Because the target sign(.) is
    constant on each subdomain, extrema of the approximation error coincide with
    extrema of p.
    """
    derivative = chebder(np.asarray(coeffs, dtype=float))
    critical = chebroots(derivative) if len(derivative) > 1 else np.array([])
    critical = critical[np.abs(critical.imag) < 1e-9].real   # keep real roots only
    locations = []
    for a, b in subdomains_u:
        lo, hi = (a, b) if a < b else (b, a)
        locations.append(lo); locations.append(hi)           # endpoints are always candidates
        locations.extend(u for u in critical if lo < u < hi)
    return sorted(set(locations))


def _pick_alternating(candidates, num_nodes):
    """Reduce a list of (position, signed_error) extrema to exactly `num_nodes`
    reference points with strictly alternating error sign (the Remez exchange).

    Step 1: collapse maximal runs of same-signed error to their largest-|error| member.
    Step 2: drop whichever END of the resulting alternating sequence has the smaller
            |error| until exactly num_nodes remain (this always keeps the global
            maximum and preserves alternation).
    Returns the surviving (position, signed_error) pairs.
    """
    runs = []
    for u, err in candidates:
        if err == 0:
            continue
        if runs and (err > 0) == (runs[-1][1] > 0):          # same sign as current run
            if abs(err) > abs(runs[-1][1]):
                runs[-1] = (u, err)
        else:
            runs.append((u, err))
    while len(runs) > num_nodes:
        if abs(runs[0][1]) <= abs(runs[-1][1]):
            runs.pop(0)
        else:
            runs.pop()
    return runs


# ======================================================================
# Improved multi-interval Remez: error extrema of (p - f) and reference selection
# (Lee, Lee, Lee, Kim, No, "High-Precision Bootstrapping ...", eprint 2020/552)
# ======================================================================
def _general_extrema(coeffs, target_vec, target_at, subdomains_u, scan_per_sub, refine_iters):
    """All local extrema of the error r(u) = p(u) - f(u) inside `subdomains_u`,
    classified by type, found WITHOUT assuming f is constant.

    coeffs     : Chebyshev coefficients of p (on the hull u in [-1, 1]).
    target_vec : vectorized f-on-u, target_vec(grid_array) -> array.
    target_at  : scalar f-on-u (kept for API compatibility; unused in the fast path).

    Each subdomain is scanned on a fine grid (the paper's scan step); interior peaks and
    valleys are located by a sign change of the discrete slope. Every interior bracket from
    every subdomain is then sharpened TOGETHER with a single vectorized ternary search
    (the paper's binary-search refinement, batched), so the cost is `refine_iters`
    vectorized chebval calls regardless of how many extrema there are -- not one scalar
    search per extremum. Interval endpoints are always included as one-sided extrema.

    Returns a position-sorted list of (u, r(u), mu); mu = +1 peak, mu = -1 valley.
    """
    cf = np.asarray(coeffs, dtype=float)
    out = []
    blo, bhi, bmu = [], [], []
    for a, b in subdomains_u:
        lo, hi = (a, b) if a <= b else (b, a)
        grid = np.linspace(lo, hi, scan_per_sub)
        vals = chebval(grid, cf) - np.asarray(target_vec(grid), dtype=float)
        out.append((grid[0], float(vals[0]), 1 if vals[0] >= vals[1] else -1))
        inner, ln, rn = vals[1:-1], vals[:-2], vals[2:]
        peak = (inner >= ln) & (inner >= rn) & ((inner > ln) | (inner > rn))
        valley = (inner <= ln) & (inner <= rn) & ((inner < ln) | (inner < rn))
        for i in (np.nonzero(peak)[0] + 1):
            blo.append(grid[i - 1]); bhi.append(grid[i + 1]); bmu.append(1.0)
        for i in (np.nonzero(valley)[0] + 1):
            blo.append(grid[i - 1]); bhi.append(grid[i + 1]); bmu.append(-1.0)
        out.append((grid[-1], float(vals[-1]), 1 if vals[-1] >= vals[-2] else -1))

    if blo:                                                   # batched ternary refinement
        lo = np.array(blo); hi = np.array(bhi); mu = np.array(bmu)
        for _ in range(refine_iters):
            m1 = lo + (hi - lo) / 3.0
            m2 = hi - (hi - lo) / 3.0
            g1 = mu * (chebval(m1, cf) - np.asarray(target_vec(m1), dtype=float))
            g2 = mu * (chebval(m2, cf) - np.asarray(target_vec(m2), dtype=float))
            left = g1 < g2
            lo = np.where(left, m1, lo)
            hi = np.where(left, hi, m2)
        u = 0.5 * (lo + hi)
        r = chebval(u, cf) - np.asarray(target_vec(u), dtype=float)
        out.extend((float(u[k]), float(r[k]), int(mu[k])) for k in range(len(u)))
    out.sort(key=lambda t: t[0])
    return out


def _ternary(err_at, lo, hi, mu, iters):
    """Refine an extremum of r in [lo, hi]: maximize mu*r (mu=+1 -> a peak of r,
    mu=-1 -> a valley). Derivative-free, so it works for any target f."""
    g = (lambda u: err_at(u)) if mu > 0 else (lambda u: -err_at(u))
    for _ in range(iters):
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        if g(m1) < g(m2):
            lo = m1
        else:
            hi = m2
    u = 0.5 * (lo + hi)
    return u, err_at(u)


def _mp_solve(rows, rhs):
    """Solve the dense system (rows) x = rhs in mpmath via Gaussian elimination with
    partial pivoting, operating on plain Python lists of mpf. This is several times faster
    than mpmath's matrix LU here, whose per-element __getitem__/__setitem__ overhead (millions
    of calls for an n~70 system) dominates the high-degree exchange. `rows` is a list of n
    lists of mpf, `rhs` a list of n mpf; returns the solution list."""
    n = len(rhs)
    M = [rows[i] + [rhs[i]] for i in range(n)]               # augmented rows
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        if piv != col:
            M[col], M[piv] = M[piv], M[col]
        Mc = M[col]
        inv = 1 / Mc[col]
        for r in range(col + 1, n):
            Mr = M[r]
            f = Mr[col] * inv
            if f != 0:
                for k in range(col, n + 1):
                    Mr[k] -= f * Mc[k]
    x = [None] * n
    for i in range(n - 1, -1, -1):
        Mi = M[i]
        s = Mi[n]
        for k in range(i + 1, n):
            s -= Mi[k] * x[k]
        x[i] = s / Mi[i]
    return x


def _select_references(extrema, num_nodes, level, noise=0.0):
    """Choose `num_nodes` reference points from `extrema` obeying the paper's three
    criteria (Algorithm 3): local-extreme-value, alternating, and maximum-absolute-sum.

    `extrema` is the position-sorted (u, r, mu) list of every peak/valley of r = p - f
    (mu = +1 peak, -1 valley); `level` is the current |E|; `noise` is the arithmetic
    floor below which an extremum is pure rounding (so it is discarded -- when the solve
    overfits toward the float/precision floor the scan otherwise reports spurious wiggles).

    The local-extreme-value condition keeps only points with mu*r >= |E|; Theorem 3.1
    guarantees at least num_nodes alternating points survive in exact arithmetic, but a
    real separating valley can sit a hair below |E| numerically, so the threshold is
    relaxed adaptively (only as far as needed). Then trim to num_nodes by repeatedly
    discarding the least-valuable removal that preserves alternation and never the global
    peak -- the maximum-absolute-sum condition.
    """
    def collapse(points):                                    # alternating skeleton
        alt = []
        for pt in points:
            if alt and alt[-1][2] == pt[2]:
                if abs(pt[1]) > abs(alt[-1][1]):
                    alt[-1] = pt
            else:
                alt.append(pt)
        return alt

    if noise > 0:
        extrema = [(u, r, mu) for (u, r, mu) in extrema if abs(r) >= noise]

    alt = None
    for frac in (1.0, 0.5, 0.25, 0.1, 0.0):                  # local-extreme-value threshold
        kept = [(u, r, mu) for (u, r, mu) in extrema if mu * r >= level * frac]
        cand = collapse(kept)
        if len(cand) >= num_nodes:
            alt = cand
            break
    if alt is None:
        alt = collapse(extrema)

    # Pad a degenerate (too-short) alternating set with the tallest leftover extrema so the
    # exchange always gets a full reference set and can self-correct, rather than freezing.
    if len(alt) < num_nodes:
        chosen_u = set(u for u, _, _ in alt)
        extra = sorted((p for p in extrema if p[0] not in chosen_u), key=lambda t: -abs(t[1]))
        alt = sorted(alt + extra[:num_nodes - len(alt)], key=lambda t: t[0])

    if len(alt) <= num_nodes:
        return alt

    protect = alt[max(range(len(alt)), key=lambda k: abs(alt[k][1]))][0]
    while len(alt) > num_nodes:
        m = len(alt)
        absr = [abs(r) for _, r, _ in alt]
        if m == num_nodes + 1:                               # remove one (an endpoint)
            if alt[0][0] == protect:
                alt.pop()
            elif alt[-1][0] == protect:
                alt.pop(0)
            else:
                alt.pop(0 if absr[0] <= absr[-1] else -1)
            continue
        pairs = [(absr[i] + absr[i + 1], i) for i in range(m - 1)]
        pairs.append((absr[0] + absr[-1], -1))               # wrap-around (both ends)
        pairs.sort()
        for _, i in pairs:
            if i == -1:
                if alt[0][0] != protect and alt[-1][0] != protect:
                    alt.pop(); alt.pop(0); break
            elif i == 0:                                      # pair touches left end
                if alt[0][0] != protect:
                    alt.pop(0); break
            elif i == m - 2:                                  # pair touches right end
                if alt[-1][0] != protect:
                    alt.pop(); break
            else:
                if alt[i][0] != protect and alt[i + 1][0] != protect:
                    alt.pop(i + 1); alt.pop(i); break
        else:                                                # global peak boxed in; drop an end
            alt.pop(0 if absr[0] <= absr[-1] else -1)
    return alt


# ======================================================================
# Class 1: Remez — minimax fit of one polynomial
# ======================================================================
class Remez:
    """Fit the minimax (equioscillating) polynomial approximation of a function
    over a union of real intervals, using the multi-interval Remez exchange.

    Constructor parameters
    ----------------------
    func        : the target function (called on real x).
    degree      : polynomial degree to fit.
    intervals   : list of (a, b) real intervals the approximation must be good on.
    odd         : if True, assume `func` is odd on a symmetric domain and fit only
                  odd Chebyshev coefficients on the positive branch (half the
                  unknowns, exactly odd result). Use this for sign().
    precision   : None -> work in float64 (fast). An int -> after the float64 fit,
                  polish to that many bits with mpmath (needed only when the target
                  error is below ~2^-52).

    After calling .fit(), the instance carries the results as attributes:
        coeffs        : Chebyshev coefficients on [-1, 1] (even ones are 0 if odd).
        leveled_error : |E|, the equioscillation error magnitude (the minimax error).
        max_error     : most-positive value of (p - func) over the domain.
        min_error     : magnitude of the most-negative value of (p - func).
                        (For an odd fit both equal leveled_error.)
        iters         : number of exchange iterations performed.
        backend       : "float64" or "mpmath(<bits>)".
        spread        : final relative spread of |error| across reference points
                        (a convergence diagnostic; ~0 means well equioscillated).
    """

    def __init__(self, func, degree, intervals, *, odd=False, precision=None,
                 general=False):
        self.func = func
        self.degree = int(degree)
        self.odd = bool(odd)
        self.precision = precision
        # `general=True` selects the improved multi-interval Remez of Lee et al.
        # (correct error extrema of p - f for a NON-constant target, plus the
        # local-extreme-value / alternating / maximum-absolute-sum reference
        # selection). The default (False) keeps the fast sign-tuned path, which
        # assumes f is piecewise-constant so error extrema coincide with extrema of p.
        self.general = bool(general)

        intervals = [(float(a), float(b)) for a, b in intervals]
        if self.odd and not self.general:
            # Symmetric hull [-R, R]; work on the positive branch only, in u = x / R.
            radius = max(max(abs(a), abs(b)) for a, b in intervals)
            self.hull_lo, self.hull_hi = -radius, radius
            self.subdomains = [(min(abs(a), abs(b)) / radius, max(abs(a), abs(b)) / radius)
                               for a, b in intervals]
            self.basis_degrees = list(range(1, self.degree + 1, 2))   # odd Chebyshev terms only
        else:
            self.hull_lo = min(a for a, _ in intervals)
            self.hull_hi = max(b for _, b in intervals)
            half = (self.hull_hi - self.hull_lo) / 2
            mid = (self.hull_hi + self.hull_lo) / 2
            self.subdomains = [((a - mid) / half, (b - mid) / half) for a, b in intervals]
            self.basis_degrees = list(range(0, self.degree + 1))

        # Remez needs (#free coefficients + 1) reference points (the +1 is the error E).
        self.num_nodes = len(self.basis_degrees) + 1

        # filled in by fit()
        self.coeffs = None
        self.leveled_error = None
        self.max_error = None
        self.min_error = None
        self.iters = 0
        self.backend = None
        self.spread = None

    # ---- map hull-normalized u back to real x, and evaluate the target there ----
    def _to_x(self, u):
        return (self.hull_hi - self.hull_lo) / 2 * u + (self.hull_hi + self.hull_lo) / 2

    def _target(self, u):
        return float(self.func(self._to_x(u)))

    def _target_vec(self, u):
        """Vectorized target on a u-array. Uses self.func on the whole array when it is
        numpy-aware (cos/arcsin are), else falls back to a scalar loop."""
        x = self._to_x(np.asarray(u, dtype=float))
        try:
            y = np.asarray(self.func(x), dtype=float)
            if y.shape == x.shape:
                return y
        except Exception:
            pass
        return np.array([float(self.func(xx)) for xx in np.atleast_1d(x)])

    # ---- float64 exchange phase ----
    def _fit_float64(self, max_iter, tol):
        # Initialize reference points as Chebyshev-Lobatto nodes spread across the
        # subdomains (exactly num_nodes of them).
        nodes = []
        num_sub = len(self.subdomains)
        base, remainder = divmod(self.num_nodes, num_sub)
        for idx, (a, b) in enumerate(self.subdomains):
            count = max(base + (1 if idx < remainder else 0), 2)
            for j in range(count):
                t = math.cos(math.pi * j / (count - 1))
                nodes.append((a + b) / 2 + (b - a) / 2 * t)
        nodes = sorted(set(nodes))
        if len(nodes) > self.num_nodes:                      # trim evenly if we overshot
            nodes = [nodes[round(i * (len(nodes) - 1) / (self.num_nodes - 1))]
                     for i in range(self.num_nodes)]
        while len(nodes) < self.num_nodes:                   # pad by bisecting the widest gap
            gaps = [(nodes[i + 1] - nodes[i], i) for i in range(len(nodes) - 1)]
            _, i = max(gaps)
            nodes.insert(i + 1, (nodes[i] + nodes[i + 1]) / 2)

        coeffs_full = np.zeros(self.degree + 1)
        spread = 1.0
        best_spread = float("inf")
        stall = 0
        for it in range(max_iter):
            # Solve the linear system: at each node, p(node) - target(node) = (-1)^i * E,
            # with unknowns = the free Chebyshev coefficients and the error E.
            A = np.zeros((self.num_nodes, self.num_nodes))
            A[:, :-1] = _design_matrix(nodes, self.basis_degrees)
            A[:, -1] = [(-1) ** i for i in range(self.num_nodes)]
            rhs = np.array([self._target(u) for u in nodes])
            solution = np.linalg.solve(A, rhs)
            free_coeffs, E = solution[:-1], solution[-1]

            coeffs_full = np.zeros(self.degree + 1)
            for k, c in zip(self.basis_degrees, free_coeffs):
                coeffs_full[k] = c

            # Bail out cheaply on an ill-conditioned (blown-up) fit, BEFORE the costly
            # root-finding below — this is what keeps the brute-force search fast.
            if np.max(np.abs(coeffs_full)) > COEFF_LIMIT:
                return coeffs_full, nodes, abs(E), 1.0, it + 1

            # Exchange: move reference points onto the current error extrema.
            candidates = [(u, chebval(u, coeffs_full) - self._target(u))
                          for u in _error_extrema(coeffs_full, self.subdomains)]
            selected = _pick_alternating(candidates, self.num_nodes)
            errs = [abs(e) for _, e in selected]
            spread = (max(errs) - min(errs)) / max(errs) if errs else 0.0
            if len(selected) == self.num_nodes:
                nodes = [u for u, _ in selected]
            elif len(selected) > self.num_nodes:
                nodes = [u for u, _ in selected[:self.num_nodes]]
            # (if fewer, keep current nodes and try again)
            if spread < tol:
                break
            # Stop if the spread stops improving (e.g. the error has hit the float64
            # floor, so further iterations are just noise).
            if spread < best_spread * (1 - 1e-3):
                best_spread = spread
                stall = 0
            else:
                stall += 1
                if stall >= 4:
                    break
        return coeffs_full, nodes, abs(E), spread, it + 1

    # ---- optional mpmath polish (only when sub-float64 accuracy is requested) ----
    def _fit_mpmath(self, init_nodes, prec, max_iter, tol):
        mp.mp.prec = prec

        def cheb(coeffs, u):                                  # Clenshaw evaluation in mpf
            n = len(coeffs) - 1
            if n < 0: return mp.mpf(0)
            if n == 0: return mp.mpf(coeffs[0])
            b1 = mp.mpf(0); b2 = mp.mpf(0)
            for k in range(n, 0, -1):
                b1, b2 = 2 * u * b1 - b2 + coeffs[k], b1
            return coeffs[0] + u * b1 - b2

        def cheb_deriv(coeffs):                               # Chebyshev derivative coeffs (mpf)
            c = [mp.mpf(x) for x in coeffs]; n = len(c) - 1
            if n <= 0: return [mp.mpf(0)]
            d = [mp.mpf(0)] * n; d[n - 1] = 2 * n * c[n]
            if n >= 2: d[n - 2] = 2 * (n - 1) * c[n - 1]
            for k in range(n - 3, -1, -1): d[k] = d[k + 2] + 2 * (k + 1) * c[k + 1]
            d[0] /= 2; return d

        nodes = [mp.mpf(u) for u in init_nodes]
        coeffs_full = [mp.mpf(0)] * (self.degree + 1)
        spread = mp.mpf(1)
        for it in range(max_iter):
            n = self.num_nodes
            A = mp.matrix(n, n); rhs = mp.matrix(n, 1)
            for i, u in enumerate(sorted(nodes)):
                for j, k in enumerate(self.basis_degrees):
                    unit = [mp.mpf(0)] * (k + 1); unit[k] = mp.mpf(1)
                    A[i, j] = cheb(unit, u)
                A[i, n - 1] = mp.mpf((-1) ** i)
                rhs[i] = mp.mpf(self.func(self._to_x(u)))
            solution = mp.lu_solve(A, rhs)
            coeffs_full = [mp.mpf(0)] * (self.degree + 1)
            for j, k in enumerate(self.basis_degrees):
                coeffs_full[k] = solution[j]
            E = solution[n - 1]

            d1 = cheb_deriv(coeffs_full); d2 = cheb_deriv(d1)
            coeffs_f64 = np.array([float(c) for c in coeffs_full])
            candidates = []
            for u0 in _error_extrema(coeffs_f64, self.subdomains):   # float64 warm-starts
                u = mp.mpf(u0)
                for _ in range(60):                                    # Newton-polish to full precision
                    g = cheb(d1, u); gp = cheb(d2, u)
                    if gp == 0: break
                    step = g / gp; u -= step
                    if abs(step) < abs(u) * mp.power(2, -prec + 8) + mp.power(2, -prec + 8):
                        break
                candidates.append((u, cheb(coeffs_full, u) - mp.mpf(self.func(self._to_x(u)))))
            selected = _pick_alternating(candidates, self.num_nodes)
            errs = [abs(e) for _, e in selected]
            spread = (max(errs) - min(errs)) / max(errs) if errs else mp.mpf(0)
            if len(selected) == self.num_nodes:
                nodes = [u for u, _ in selected]
            if spread < tol:
                break
        return coeffs_full, abs(E), spread, it + 1

    # ---- shared initial reference points (Chebyshev-Lobatto, spread across subdomains) ----
    def _initial_nodes(self):
        nodes = []
        num_sub = len(self.subdomains)
        base, remainder = divmod(self.num_nodes, num_sub)
        for idx, (a, b) in enumerate(self.subdomains):
            count = max(base + (1 if idx < remainder else 0), 2)
            for j in range(count):
                t = math.cos(math.pi * j / (count - 1))
                nodes.append((a + b) / 2 + (b - a) / 2 * t)
        nodes = sorted(set(nodes))
        if len(nodes) > self.num_nodes:
            nodes = [nodes[round(i * (len(nodes) - 1) / (self.num_nodes - 1))]
                     for i in range(self.num_nodes)]
        while len(nodes) < self.num_nodes:                   # pad by bisecting the widest gap
            gaps = [(nodes[i + 1] - nodes[i], i) for i in range(len(nodes) - 1)]
            _, i = max(gaps)
            nodes.insert(i + 1, (nodes[i] + nodes[i + 1]) / 2)
        return nodes

    # ---- improved multi-interval exchange (float64), for a general target f ----
    def _fit_general_float64(self, max_iter, tol, scan_per_sub=48, refine_iters=24,
                             warm_nodes=None):
        nodes = list(warm_nodes) if warm_nodes is not None else self._initial_nodes()
        coeffs_full = np.zeros(self.degree + 1)
        spread = 1.0
        E = 0.0
        # Below the float64 floor the exchange cannot equioscillate and starts to wander,
        # so we remember the best polynomial seen (smallest true peak error) and return it.
        best = (float("inf"), coeffs_full, list(nodes), 0.0, 1.0)
        for it in range(max_iter):
            A = np.zeros((self.num_nodes, self.num_nodes))
            A[:, :-1] = _design_matrix(sorted(nodes), self.basis_degrees)
            A[:, -1] = [(-1) ** i for i in range(self.num_nodes)]
            rhs = np.array([self._target(u) for u in sorted(nodes)])
            solution = np.linalg.solve(A, rhs)
            free_coeffs, E = solution[:-1], solution[-1]
            coeffs_full = np.zeros(self.degree + 1)
            for k, c in zip(self.basis_degrees, free_coeffs):
                coeffs_full[k] = c

            err_at = lambda u: float(chebval(u, coeffs_full) - self._target(u))
            extrema = _general_extrema(coeffs_full, self._target_vec, self._target,
                                       self.subdomains, scan_per_sub, refine_iters)
            peak = max((abs(r) for _, r, _ in extrema), default=float("inf"))
            selected = _select_references(extrema, self.num_nodes, abs(E),
                                          noise=(1e-13 if len(self.subdomains) > 1 else 0.0))
            mags = [abs(r) for _, r, _ in selected]
            spread = (max(mags) - min(mags)) / max(mags) if mags and max(mags) > 0 else 1.0
            if peak < best[0]:
                refs = [u for u, _, _ in selected] if len(selected) == self.num_nodes \
                    else sorted(nodes)
                best = (peak, coeffs_full, refs, abs(E), spread)
            if len(selected) == self.num_nodes:
                nodes = [u for u, _, _ in selected]
            if spread < tol:
                break
        # If we equioscillated, the last iterate is optimal; otherwise fall back to best.
        if spread < tol:
            return coeffs_full, [u for u, _, _ in selected] if len(selected) == self.num_nodes \
                else sorted(nodes), abs(E), spread, it + 1
        return best[1], best[2], best[3], best[4], it + 1

    # ---- improved multi-interval exchange, full precision (mpmath) ----
    def _fit_general_mpmath(self, init_coeffs, prec, max_iter, tol, init_nodes=None,
                            scan_per_sub=14, refine_iters=44):
        """Run the improved multi-interval Remez entirely in mpmath, so the error extrema
        (which are below the float64 floor at high degree) are found accurately. Seeded from
        `init_nodes` -- the converged references of the previous (lower) degree -- this is
        the engine behind the mpmath degree continuation in cosine_minimax. In full precision
        the global reference selection is stable: an ill-conditioned solve no longer blows up,
        so an interval briefly losing its references simply regains them next iteration.

        Returns (coeffs[mpf list], E, spread, refs[float list], iters)."""
        mp.mp.prec = prec
        maxd = self.degree
        n = self.num_nodes
        bd = self.basis_degrees

        def basis(u):                                         # [T_0(u), ..., T_maxd(u)]
            T = [mp.mpf(1), u]
            for k in range(2, maxd + 1):
                T.append(2 * u * T[-1] - T[-2])
            return T

        def cheb(c, u):
            tu = 2 * u                                        # precompute, used every term
            b1 = mp.mpf(0); b2 = mp.mpf(0)
            for k in range(maxd, 0, -1):
                b1, b2 = tu * b1 - b2 + c[k], b1
            return c[0] + u * b1 - b2

        def err(c, u):
            return cheb(c, u) - self.func(self._to_x(u))

        GR = (mp.sqrt(5) - 1) / 2                              # golden ratio, set once at this prec

        def scan_extrema(c):
            """Locate + refine all error extrema in full precision: scan each subdomain, then
            sharpen every bracket with a golden-section search (one new evaluation per step,
            so ~3x fewer polynomial evaluations than a ternary search of equal accuracy)."""
            out = []; brackets = []
            for a, b in self.subdomains:
                g = [mp.mpf(a) + (mp.mpf(b) - mp.mpf(a)) * t / (scan_per_sub - 1)
                     for t in range(scan_per_sub)]
                v = [err(c, u) for u in g]
                out.append((g[0], v[0], 1 if v[0] >= v[1] else -1))
                for i in range(1, scan_per_sub - 1):
                    if v[i] >= v[i - 1] and v[i] >= v[i + 1] and (v[i] > v[i - 1] or v[i] > v[i + 1]):
                        brackets.append((g[i - 1], g[i + 1], 1))
                    elif v[i] <= v[i - 1] and v[i] <= v[i + 1] and (v[i] < v[i - 1] or v[i] < v[i + 1]):
                        brackets.append((g[i - 1], g[i + 1], -1))
                out.append((g[-1], v[-1], 1 if v[-1] >= v[-2] else -1))
            for lo, hi, mu in brackets:
                s = 1 if mu > 0 else -1
                x1 = hi - GR * (hi - lo); x2 = lo + GR * (hi - lo)
                f1 = s * err(c, x1); f2 = s * err(c, x2)
                for _ in range(refine_iters):
                    if f1 < f2:
                        lo, x1, f1 = x1, x2, f2
                        x2 = lo + GR * (hi - lo); f2 = s * err(c, x2)
                    else:
                        hi, x2, f2 = x2, x1, f1
                        x1 = hi - GR * (hi - lo); f1 = s * err(c, x1)
                u = (lo + hi) / 2
                out.append((u, err(c, u), mu))
            out.sort(key=lambda t: t[0])
            return out

        # Seed: prefer the supplied (converged lower-degree) references; else the error
        # extrema of the warm float64 polynomial.
        if init_nodes is not None and len(init_nodes) == n:
            nodes = [mp.mpf(u) for u in sorted(init_nodes)]
        else:
            cw = np.asarray(init_coeffs, dtype=float)
            seed = _general_extrema(cw, self._target_vec, self._target,
                                    self.subdomains, 48, 20)
            seed = _select_references(seed, n,
                                      np.median([abs(r) for _, r, _ in seed]) if seed else 0.0,
                                      noise=(1e-13 if len(self.subdomains) > 1 else 0.0))
            nodes = [mp.mpf(u) for u, _, _ in seed]
            if len(nodes) != n:
                nodes = [mp.mpf(u) for u in self._initial_nodes()]

        noise = float(mp.power(2, -(prec - 20)))
        best = (mp.inf, [mp.mpf(0)] * (maxd + 1), mp.mpf(1), nodes, 0)
        for it in range(max_iter):
            srt = sorted(nodes)
            rows = []; rhs = []
            for i, u in enumerate(srt):
                T = basis(u)
                row = [T[k] for k in bd]
                row.append(mp.mpf((-1) ** i))
                rows.append(row)
                rhs.append(self.func(self._to_x(u)))
            sol = _mp_solve(rows, rhs)
            c = [mp.mpf(0)] * (maxd + 1)
            for j, k in enumerate(bd):
                c[k] = sol[j]
            E = sol[n - 1]

            ext = scan_extrema(c)
            sel = _select_references(ext, n, abs(E), noise=noise)
            mags = [abs(r) for _, r, _ in sel]
            peak = max((abs(r) for _, r, _ in ext), default=mp.inf)
            spread = (max(mags) - min(mags)) / max(mags) if mags and max(mags) > 0 else mp.mpf(1)
            refs = [u for u, _, _ in sel] if len(sel) == n else srt
            if peak < best[0]:
                best = (peak, c, spread, refs, it + 1)
            if spread < tol:
                return c, abs(E), spread, [float(u) for u in refs], it + 1
            if len(sel) == n:
                nodes = [u for u, _, _ in sel]
        return best[1], best[0], best[2], [float(u) for u in best[3]], best[4]

    def fit(self, max_iter=60, tol=None, warm_nodes=None):
        """Run the exchange to convergence and store the results on this instance.
        `warm_nodes` optionally seeds the general exchange (used by degree continuation).
        Returns `self` so you can write `r = Remez(...).fit()`."""
        float_tol = 1e-13 if tol is None else tol
        if self.general:
            coeffs, nodes, E, spread, iters = self._fit_general_float64(
                max_iter, float_tol, warm_nodes=warm_nodes)
        else:
            coeffs, nodes, E, spread, iters = self._fit_float64(max_iter, float_tol)
        backend = "float64"
        general_peak = None                                   # set when the mpmath path measures it

        if self.precision is not None and self.precision > 52:
            if mp is None:
                raise RuntimeError("mpmath is required for precision > float64")
            polish_tol = mp.power(2, -self.precision // 2)
            if self.general:
                mc, mpeak, mspread, mrefs, extra = self._fit_general_mpmath(
                    coeffs, self.precision, max_iter, polish_tol, init_nodes=nodes)
                # Keep the full-precision result only if it actually improves the true peak.
                # For a single high-degree fit (non-continuation warm start) the exchange can
                # wander; this guarantees precision=<bits> never returns a worse polynomial.
                def _peak64(cs):
                    c = np.array([float(x) for x in cs], dtype=float)
                    ext = _general_extrema(c, self._target_vec, self._target,
                                           self.subdomains, 64, 30)
                    return max((abs(r) for _, r, _ in ext), default=float("inf"))
                if float(mpeak) < _peak64(coeffs):
                    coeffs, spread, nodes = mc, mspread, mrefs
                    general_peak = float(mpeak)
                    backend = f"mpmath({self.precision})"
            else:
                coeffs, E, spread, extra = self._fit_mpmath(
                    nodes, self.precision, max_iter, polish_tol)
                iters += extra
                backend = f"mpmath({self.precision})"

        self.coeffs = list(coeffs)
        if self.odd and not self.general:                     # odd & symmetric => symmetric error band
            self.max_error = self.min_error = float(E)
        elif self.general and general_peak is not None:       # measured in full precision
            self.max_error = self.min_error = general_peak
        elif self.general:                                    # extrema of (p - f), measured in float64
            c_f64 = np.array([float(c) for c in coeffs])
            band = [r for _, r, _ in _general_extrema(c_f64, self._target_vec, self._target,
                                                      self.subdomains, 64, 30)]
            self.max_error = max(band)
            self.min_error = -min(band)
        else:
            band = [chebval(u, np.array([float(c) for c in coeffs])) - self._target(u)
                    for u in _error_extrema(np.array([float(c) for c in coeffs]), self.subdomains)]
            self.max_error = max(band)
            self.min_error = -min(band)
        self.leveled_error = general_peak if general_peak is not None else float(E)
        self.iters = iters
        self.backend = backend
        self.spread = float(spread)
        self.ref_nodes = [float(u) for u in nodes]            # converged references (for continuation)
        return self

    def plot(self, num=3000, save=None, show=True, title=None):
        """Plot this fitted minimax approximation and its error, exposing the
        equioscillation. For a single interval (e.g. the inverse sine) the top panel shows
        the target f and the approximant p, and the bottom panel the error p - f with the
        +/- max-error band. For a multi-interval fit (the scaled cosine) the subintervals
        are stitched left to right and the error is drawn across all of them, so the
        equioscillation between +E and -E is visible at a glance."""
        import matplotlib.pyplot as plt
        if not hasattr(self, "coeffs"):
            raise RuntimeError("call .fit() before .plot()")
        c = np.array([float(x) for x in self.coeffs])
        nsub = len(self.subdomains)
        band = self.max_error
        bl = math.log2(band) if band > 0 else float("-inf")

        # For a high-precision (mpmath) fit the error is below the float64 floor, so a float64
        # p - f would show rounding noise instead of the true equioscillation. Evaluate the
        # error in mpmath in that case (the tiny float results are themselves representable).
        mp_mode = mp is not None and len(self.coeffs) and isinstance(self.coeffs[0], mp.mpf)
        if mp_mode:
            cf = self.coeffs
            def err_vals(u_arr):
                out = []
                for uu in u_arr:
                    um = mp.mpf(float(uu)); tu = 2 * um
                    b1 = mp.mpf(0); b2 = mp.mpf(0)
                    for k in range(len(cf) - 1, 0, -1):
                        b1, b2 = tu * b1 - b2 + cf[k], b1
                    out.append(float(cf[0] + um * b1 - b2 - self.func(self._to_x(um))))
                return np.array(out)
        else:
            def err_vals(u_arr):
                return chebval(u_arr, c) - np.asarray(self._target_vec(u_arr), dtype=float)

        segs = []
        per = max(40, num // nsub)
        for au, bu in self.subdomains:
            u = np.linspace(au, bu, per)
            segs.append((self._to_x(u), u, err_vals(u)))

        if nsub == 1:
            fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True,
                                         gridspec_kw={"height_ratios": [2, 1]})
            x, u, err = segs[0]
            a1.plot(x, self._target_vec(u), color="#bcbcbc", lw=3, label="target f")
            a1.plot(x, chebval(u, c), color="#1b4965", lw=1.6, label=f"minimax p (deg {self.degree})")
            a1.legend(loc="best", fontsize=9); a1.grid(alpha=0.25); a1.set_ylabel("value")
            a2.plot(x, err, color="#bb3e03", lw=1.2)
            a2.axhline(band, ls="--", color="#888"); a2.axhline(-band, ls="--", color="#888")
            a2.fill_between(x, -band, band, color="#ffd6a5", alpha=0.35)
            a2.grid(alpha=0.25); a2.set_ylabel("error  p - f"); a2.set_xlabel("x")
            fig.suptitle(title or f"minimax fit  (deg {self.degree}, err 2^{bl:.1f}, "
                                  f"spread {self.spread:.0e})", fontsize=12)
        else:
            fig, ax = plt.subplots(figsize=(11, 4.6))
            pos = 0
            for x, u, err in segs:
                ax.plot(np.arange(pos, pos + len(err)), err, color="#1b4965", lw=0.8)
                pos += len(err)
                ax.axvline(pos - 0.5, color="#e8e8e8", lw=0.5)
            ax.axhline(band, ls="--", color="#888", label=f"+/- max error 2^{bl:.1f}")
            ax.axhline(-band, ls="--", color="#888")
            ax.fill_between([0, pos], -band, band, color="#ffd6a5", alpha=0.35)
            ax.set_xlim(0, pos); ax.set_ylabel("error  p - f")
            ax.set_xlabel(f"sample index  ({nsub} subintervals stitched left to right)")
            ax.grid(alpha=0.25); ax.legend(loc="upper right", fontsize=9)
            fig.suptitle(title or f"scaled-cosine minimax error across {nsub} intervals  "
                                  f"(deg {self.degree}, err 2^{bl:.1f}, spread {self.spread:.0e})",
                         fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        if save: fig.savefig(save, dpi=150, bbox_inches="tight")
        if show: plt.show()
        return fig


# ======================================================================
# Class 2: Polynomial — the composite, and how to use it
# ======================================================================
class Polynomial:
    """A composite P(x) = p_k( ... p_1( p_0(x) ) ... ).

    Each stage is a list of Chebyshev coefficients on [-1, 1]; the stages are
    composed by feeding one stage's output straight into the next (the coefficients
    are pre-normalized so no rescaling is needed between stages).

    Attributes
    ----------
    stages : list of per-stage Chebyshev coefficient lists.
    gap    : the don't-care radius g; inputs with |x| < g are intentionally
             unconstrained. (None if unknown.) Used to auto-shade plots.
    bits   : achieved y-accuracy in bits (-log2 of the worst-case |P(x) - sign(x)|
             outside the gap). (None if unknown.)
    """

    def __init__(self, stages, *, gap=None, bits=None):
        self.stages = [list(s) for s in stages]
        self.gap = gap
        self.bits = bits

    # ---- evaluation ----
    def evaluate(self, x, precision=None):
        """Evaluate P at x.
        x         : a scalar or numpy array.
        precision : None -> fast vectorized float64. An int -> mpmath at that many
                    bits (use this to actually see accuracy beyond ~15 digits).
        """
        if precision is None:
            y = np.asarray(x, dtype=float)
            for stage in self.stages:
                y = chebval(y, np.asarray(stage, dtype=float))
            return y
        mp.mp.prec = precision

        def cheb(coeffs, u):
            n = len(coeffs) - 1; b1 = mp.mpf(0); b2 = mp.mpf(0)
            for k in range(n, 0, -1):
                b1, b2 = 2 * u * b1 - b2 + coeffs[k], b1
            return coeffs[0] + u * b1 - b2

        scalar = np.ndim(x) == 0
        out = []
        for v in np.atleast_1d(x):
            y = mp.mpf(v)
            for stage in self.stages:
                y = cheb([mp.mpf(c) for c in stage], y)
            out.append(y)
        return out[0] if scalar else out

    def partials(self, x):
        """Return [P_0(x), P_1(x), ..., P_k(x)] — the output after each successive
        stage (float64). Useful for visualizing how the composition sharpens."""
        y = np.asarray(x, dtype=float)
        out = []
        for stage in self.stages:
            y = chebval(y, np.asarray(stage, dtype=float))
            out.append(y.copy())
        return out

    # ---- structure ----
    def degrees(self):
        """Per-stage polynomial degrees."""
        return [len(s) - 1 for s in self.stages]

    def depth(self):
        """Total multiplicative depth = sum over stages of the (bucketed) evaluation
        depth ceil(log2(bucket_degree)). This is the quantity the search minimizes."""
        return sum(_bucket_depth(len(s) - 1) for s in self.stages)

    def coefs(self):
        """List of ndarrays of coefficients of each stage, in reverse order"""
        return [np.array(f[::-1], dtype=np.float64) for f in self.stages]

    # ---- refinement ----
    def refine(self, times=1, order=3):
        """Append `times` copies of the order-`order` refinement polynomial (a
        closed-form polynomial that is maximally flat at +/-1, so it sharpens a
        value already near +/-1 toward exactly +/-1). order 1 = X2 (x2 the correct
        digits), order 3 = X4 (x4). Returns a new Polynomial."""
        rc = list(refinement_poly(order))
        return Polynomial(self.stages + [rc] * times, gap=self.gap, bits=None)

    # ---- plotting (auto-shades the don't-care band from self.gap) ----
    def plot(self, mode="composite", interval=(-1.0, 1.0), num=4000,
             gap="auto", show_sign=True, ax=None, save=None, show=True, title=None):
        """Plot the result.
        mode : "composite" (the full P vs sign), "partial" (output after each stage),
               "stages" (each stage polynomial on its own), or "both".
        gap  : "auto" uses self.gap; pass a number to override, or None to disable shading.
        """
        import matplotlib.pyplot as plt
        g = self.gap if gap == "auto" else gap
        a, b = float(interval[0]), float(interval[1])
        x = np.linspace(a, b, num)
        coeffs = [np.asarray(s, dtype=float) for s in self.stages]
        k = len(coeffs)

        def shade(axp):
            if g:
                axp.axvspan(-g, g, color="#ffd6a5", alpha=0.7, label=f"don't-care |x|<{g:.2g}")

        def draw_composite(axp):
            shade(axp)
            if show_sign: axp.plot(x, np.sign(x), color="#bcbcbc", lw=3, label="sign(x)")
            axp.plot(x, self.partials(x)[-1], color="#1b4965", lw=1.9,
                     label=f"P (deg {sum(self.degrees())}, depth {self.depth()})")
            axp.set_title("full composite P"); axp.set_ylim(-1.25, 1.25)

        def draw_partial(axp):
            shade(axp)
            if show_sign: axp.plot(x, np.sign(x), color="#bcbcbc", lw=3, label="sign(x)")
            for j, y in enumerate(self.partials(x)):
                axp.plot(x, y, color=plt.cm.viridis(0.12 + 0.78 * j / max(k - 1, 1)), lw=1.6,
                         label=f"P{j} (deg {self.degrees()[j]})")
            axp.set_title("partial compositions"); axp.set_ylim(-1.25, 1.25)

        def draw_stages(axp):
            t = np.linspace(-1, 1, num)
            if show_sign: axp.plot(t, np.sign(t), color="#bcbcbc", lw=3, label="sign(t)")
            for i, c in enumerate(coeffs):
                axp.plot(t, chebval(t, c), color=plt.cm.plasma(0.08 + 0.74 * i / max(k - 1, 1)),
                         lw=1.6, label=f"p{i} (deg {self.degrees()[i]})")
            axp.set_title("individual stage polynomials")

        drawers = {"composite": draw_composite, "partial": draw_partial, "stages": draw_stages}
        if mode == "both":
            fig, axes = plt.subplots(1, 2, figsize=(14, 5.2)); draw_stages(axes[0]); draw_partial(axes[1])
        elif mode in drawers:
            if ax is None: fig, ax = plt.subplots(figsize=(8, 5.2))
            else: fig = ax.figure
            drawers[mode](ax); axes = ax
        else:
            raise ValueError("mode must be 'composite', 'partial', 'stages', or 'both'")
        for axx in (axes if isinstance(axes, np.ndarray) else [axes]):
            axx.set_xlabel("x"); axx.grid(alpha=0.25); axx.legend(loc="lower right", fontsize=8)
        if title: fig.suptitle(title, fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.95] if title else None)
        if save: fig.savefig(save, dpi=150, bbox_inches="tight")
        if show: plt.show()
        return fig, axes


# ======================================================================
# Closed-form refinement-polynomial family (eprint 2019/1234)
# ----------------------------------------------------------------------
# f_order(x) = (integral_0^x (1-t^2)^order dt) / (integral_0^1 ...).
# Its derivative is proportional to (1-x^2)^order, so it is maximally flat at +/-1:
# applied to a value within e of +/-1 it returns a value within ~e^(order+1), i.e.
# it multiplies the correct digits by (order+1).  order 1 -> X2 (deg 3),
# order 3 -> X4 (deg 7).  (NB: the input is the `order`, not the polynomial degree.)
# ======================================================================
def refinement_poly(order=3):
    """Chebyshev coefficients on [-1, 1] of the order-`order` refinement polynomial."""
    from fractions import Fraction as Fr
    from math import comb
    monomial = [Fr(0)] * (2 * order + 2)
    norm = sum(Fr((-1) ** j * comb(order, j), 2 * j + 1) for j in range(order + 1))
    for j in range(order + 1):
        monomial[2 * j + 1] = Fr((-1) ** j * comb(order, j), 2 * j + 1) / norm
    return list(poly2cheb(np.array([float(c) for c in monomial])))


def _refinement_depth(order):
    """Evaluation depth charged for one order-`order` refinement polynomial
    (its degree is 2*order+1, bucketed)."""
    return _bucket_depth(2 * order + 1)


# ======================================================================
# Degree-schedule enumeration (the brute-force search space)
# ======================================================================
def enumerate_degree_lists(total_depth, min_depth_part=3):
    """Yield every NON-DECREASING list of degrees drawn from POLY_EVAL_DEG whose
    per-stage depths sum to exactly `total_depth`.

    Implemented as integer partitions of `total_depth` into parts taken from the
    available per-stage depths {3,4,5,6,7,8}, parts kept non-decreasing (which makes
    the degrees non-decreasing, since higher degree = higher depth). Each part maps
    to its bucket degree via DEPTH_TO_DEGREE.

    Example: enumerate_degree_lists(10) yields (as degree lists)
             [5,119], [5,5,13], [13,59], [27,27]
             with depths (3,7), (3,3,4), (4,6), (5,5).
    """
    available = sorted(DEPTH_TO_DEGREE)                       # [3,4,5,6,7,8]
    if total_depth == 0:
        yield []
        return
    for part in available:
        if part < min_depth_part:                            # enforce non-decreasing parts
            continue
        if part > total_depth:
            break
        for tail in enumerate_degree_lists(total_depth - part, part):
            yield [DEPTH_TO_DEGREE[part]] + tail


def _refinement_combos(orders, max_count):
    """All non-decreasing tuples of refinement orders (including the empty tuple),
    of length 0..max_count, drawn from `orders`."""
    orders = sorted(set(orders))
    combos = [()]
    for n in range(1, max_count + 1):
        combos.extend(combinations_with_replacement(orders, n))
    return combos


# ======================================================================
# Composite construction over the sign() don't-care domain
# ======================================================================
def _sign(z):
    return -1.0 if z < 0 else (1.0 if z > 0 else 0.0)


def _initial_interval(gap, err_pad, odd):
    """Stage 0's domain: [gap, 1+err_pad] (positive branch only, for odd), padded on
    the outer end by the scheme-error allowance so a noisy ciphertext near +/-1 stays
    inside the approximation interval."""
    return [(gap, 1.0 + err_pad)] if odd else [(-1.0 - err_pad, -gap), (gap, 1.0 + err_pad)]


def _next_interval(stage_record, err_pad, odd):
    """The domain of the next stage: the image of the previous stage's output, i.e.
    [1 - min_error - err_pad, 1 + max_error + err_pad]. Each stage WIDENS this gap."""
    outer = 1.0 + stage_record["max_error"] + err_pad
    inner = 1.0 - stage_record["min_error"] - err_pad
    return [(inner, outer)] if odd else [(-outer, -inner), (inner, outer)]


def _fit_stage(degree, interval, precision, odd):
    """Fit one Remez stage and return a compact record of what the composite needs."""
    r = Remez(_sign, degree, interval, odd=odd, precision=precision).fit()
    return {"coeffs": r.coeffs, "max_error": r.max_error,
            "min_error": r.min_error, "error": r.leveled_error}


def _assemble(stage_records, refinement_orders, gap, err_pad, bits):
    """Turn per-stage records + a refinement spec into a runnable Polynomial.

    Every Remez stage except the last is renormalized (divided by its output radius
    1+max_error+err_pad) so its output lands in [-1,1] for the next stage. The last
    Remez stage is left raw; refinement polynomials, if any, are appended after it
    (their flatness at +/-1 absorbs the last stage's slight overshoot).
    """
    coeffs = []
    for i, rec in enumerate(stage_records):
        if i < len(stage_records) - 1:
            outer = 1.0 + rec["max_error"] + err_pad
            coeffs.append([c / outer for c in rec["coeffs"]])
        else:
            coeffs.append(list(rec["coeffs"]))               # last Remez stage: raw
    for order in refinement_orders:
        coeffs.append(refinement_poly(order))
    return Polynomial(coeffs, gap=gap, bits=bits)


# ----------------------------------------------------------------------
# Numerical guards / measurement
# ----------------------------------------------------------------------
# float64 Remez is only well-conditioned when the degree is modest relative to
# the gap; otherwise the linear solve returns garbage coefficients (~1e15) while
# still reporting a tiny "leveled error". Genuine fits have coefficients of order 1,
# so we reject any stage whose coefficients blow past this limit.
COEFF_LIMIT = 1e3
# Worst-case error we can trust from a float64 composite evaluation (below this the
# measurement is just rounding noise). Targets beyond it need the refinement path.
FLOAT64_BIT_CAP = 44.0


def _measure_bits(poly, gap, samples=3000):
    """Achieved y-accuracy of `poly`, measured by EVALUATING the composite (float64)
    on a dense grid of the positive domain [gap, 1] and taking the worst |P(x) - 1|.

    Returns -log2(worst_error), capped at FLOAT64_BIT_CAP. Returns -inf if the
    composite is numerically broken (overflowed to inf/nan, or so wrong it fails to
    classify) — this is what catches unstable high-degree schedules.

    We measure by evaluation rather than trusting the last stage's leveled error
    because a mis-fit stage can report a small error it does not actually deliver.
    """
    grid = np.concatenate([gap * np.power(gap, -np.linspace(0.0, 1.0, samples)),
                           np.linspace(gap, 1.0, samples)])
    with np.errstate(all="ignore"):
        values = poly.evaluate(grid)                         # P(x) on the positive branch ~ +1
    worst = np.max(np.abs(values - 1.0))
    if not np.isfinite(worst) or worst > 0.5:
        return -np.inf
    return min(-math.log2(worst), FLOAT64_BIT_CAP)


def build_composite(degrees, x_accuracy, *, refinement_orders=(),
                    logerr=None, precision=None, odd=True):
    """Build the composite sign approximation for an EXPLICIT degree list.

    degrees           : per-stage Remez degrees (non-decreasing recommended).
    x_accuracy        : the don't-care gap is 2^-x_accuracy.
    refinement_orders : optional refinement-polynomial orders appended at the end.
    logerr            : outer interval padding 2^-logerr (default x_accuracy+5).
    precision         : None -> float64; int -> mpmath bits for the Remez stages.
    odd               : exploit oddness of sign (recommended True).

    Returns a Polynomial with .gap set and .bits = measured base accuracy times the
    refinement factor prod(order+1). (The base is measured by evaluation; refinement
    multiplies the correct digits by order+1 per polynomial.)
    """
    gap = 2.0 ** -x_accuracy
    err_pad = 2.0 ** -(logerr if logerr is not None else x_accuracy + 5)
    records, interval = [], _initial_interval(gap, err_pad, odd)
    for d in degrees:
        rec = _fit_stage(d, interval, precision, odd)
        records.append(rec)
        interval = _next_interval(rec, err_pad, odd)
    base_bits = _measure_bits(_assemble(records, (), gap, err_pad, None), gap)
    bits = base_bits
    for order in refinement_orders:
        bits *= (order + 1)
    return _assemble(records, refinement_orders, gap, err_pad, bits)


# ======================================================================
# High-level entry point: brute-force minimal-depth search
# ======================================================================
def sign_minimax(x_accuracy, y_accuracy, *, use_refinement=False,
                 refinement_orders=(2, 3), max_refinements=3,
                 accuracy_epsilon=0.1, max_total_depth=40,
                 logerr=None, precision=None, odd=True, verbose=False):
    """Find the SMALLEST-DEPTH composite sign approximation meeting both accuracies.

    Parameters
    ----------
    x_accuracy        : x-accuracy in bits. Don't-care gap is 2^-x_accuracy; inputs
                        with |x| >= 2^-x_accuracy must be classified correctly.
    y_accuracy        : y-accuracy in bits. Requires |P(x) - sign(x)| <= 2^-y_accuracy
                        outside the gap. Without refinement this is limited to
                        ~FLOAT64_BIT_CAP bits (float64 can't measure finer); use
                        use_refinement=True for more.
    use_refinement    : if True, the search may append closed-form refinement
                        polynomials (always last) to reach y_accuracy cheaply. Each
                        refinement of order o multiplies the correct digits by (o+1).
    refinement_orders : refinement orders the search may use (default 2 and 3:
                        order 2 = degree 5 / depth 3, order 3 = degree 7 / depth 4).
    max_refinements   : cap on number of appended refinement polynomials.
    accuracy_epsilon  : a candidate passes if its y-accuracy >= y_accuracy - this
                        (absorbs measurement noise / near-ties).
    max_total_depth   : give up if nothing reaches the target by this depth.
    logerr, precision, odd : as in build_composite.

    Search (your rules)
    -------------------
    For total_depth = 3, 4, 5, ...: enumerate every non-decreasing degree list (from
    POLY_EVAL_DEG) of that depth, optionally reserving some depth for trailing
    refinement polynomials; build the composite and MEASURE its accuracy by evaluating
    it (so unstable high-degree fits, which overflow, are discarded). Stop at the first
    depth where something passes; among passers there, pick the FEWEST polynomials
    (ties broken by higher accuracy). Degree-list prefixes are cached and unstable
    prefixes are remembered, so each distinct stage is fit at most once.

    Returns the chosen Polynomial (with .gap and .bits set); raises if unreachable.
    """
    gap = 2.0 ** -x_accuracy
    err_pad = 2.0 ** -(logerr if logerr is not None else x_accuracy + 5)

    # ---- prefix-cached stage builder; returns None for unstable/garbage prefixes ----
    stage_cache = {(): []}

    def records_for(degree_tuple):
        if degree_tuple in stage_cache:
            return stage_cache[degree_tuple]
        prefix = records_for(degree_tuple[:-1])
        if prefix is None:                                   # an earlier stage already failed
            stage_cache[degree_tuple] = None
            return None
        interval = (_next_interval(prefix[-1], err_pad, odd) if prefix
                    else _initial_interval(gap, err_pad, odd))
        try:
            rec = _fit_stage(degree_tuple[-1], interval, precision, odd)
        except Exception:
            stage_cache[degree_tuple] = None                 # singular solve etc.
            return None
        if max(abs(c) for c in rec["coeffs"]) > COEFF_LIMIT:  # blown-up (unstable) fit
            stage_cache[degree_tuple] = None
            return None
        stage_cache[degree_tuple] = prefix + [rec]
        return stage_cache[degree_tuple]

    # ---- base accuracy of a Remez-only composite, measured by evaluation ----
    base_bits_cache = {}

    def base_bits_for(degree_tuple):
        if degree_tuple in base_bits_cache:
            return base_bits_cache[degree_tuple]
        records = records_for(degree_tuple)
        if records is None:
            base_bits_cache[degree_tuple] = -np.inf
        else:
            poly = _assemble(records, (), gap, err_pad, None)
            base_bits_cache[degree_tuple] = _measure_bits(poly, gap)
        return base_bits_cache[degree_tuple]

    refine_options = (_refinement_combos(refinement_orders, max_refinements)
                      if use_refinement else [()])

    for total_depth in range(3, max_total_depth + 1):
        passing = []   # (num_polynomials, -bits, degrees, refine_combo, bits)
        for refine_combo in refine_options:
            refine_depth = sum(_refinement_depth(o) for o in refine_combo)
            remez_depth = total_depth - refine_depth
            if remez_depth < 3:                              # need >= one Remez stage
                continue
            refine_factor = 1.0
            for order in refine_combo:
                refine_factor *= (order + 1)                 # digits multiplier
            for degrees in enumerate_degree_lists(remez_depth):
                base_bits = base_bits_for(tuple(degrees))
                if not np.isfinite(base_bits):
                    continue                                 # unstable / broken schedule
                bits = base_bits * refine_factor
                if bits >= y_accuracy - accuracy_epsilon:
                    num_polys = len(degrees) + len(refine_combo)
                    passing.append((num_polys, -bits, degrees, refine_combo, bits))
        if passing:
            passing.sort(key=lambda c: (c[0], c[1]))         # fewest polynomials, then most bits
            num_polys, _, degrees, refine_combo, bits = passing[0]
            poly = _assemble(records_for(tuple(degrees)), refine_combo, gap, err_pad, bits)
            if verbose:
                extra = f" + refine {list(refine_combo)}" if refine_combo else ""
                print(f"min total depth = {total_depth}: degrees {degrees}{extra}"
                      f"  ->  {bits:.2f} bits, {num_polys} polynomials, depth {poly.depth()}")
            return poly
    raise RuntimeError(
        f"could not reach {y_accuracy} bits within total depth {max_total_depth}"
        + ("" if use_refinement else "; try use_refinement=True for high y-accuracy"))

# ======================================================================
# Improved multi-interval Remez applied to CKKS bootstrapping (EvalMod)
# ----------------------------------------------------------------------
# Lee, Lee, Lee, Kim, No, "High-Precision Bootstrapping of RNS-CKKS ...",
# eprint 2020/552. The homomorphic modular reduction is approximated as a
# composition of three stable polynomials:
#
#     normod(x) = x - round(x)  =  h3 . (h2 applied ell times) . h1 (x)
#
#       h1(x) = cos( (2*pi / 2^ell) * (x - 1/4) )   the SCALED COSINE, optimal-minimax
#                                                   over the union of intervals near
#                                                   each integer (multi-interval).
#       h2(y) = 2*y^2 - 1  ( = T_2 )                the double-angle map: applying it ell
#                                                   times turns cos(theta/2^ell) into
#                                                   cos(theta), here cos(2*pi*(x-1/4)) =
#                                                   sin(2*pi*x).
#       h3(y) = (1/2pi) * arcsin(y)                 the INVERSE SINE, optimal-minimax over
#                                                   the single interval [-s, s].
#
# h1 and h3 are produced by the improved multi-interval Remez above
# (Remez(..., general=True)); h2 is exact. This module composes them and
# measures the resulting approximation of normod.
# ======================================================================

def mod_reduction_intervals(K, eps):
    """The approximation region D = U_{i=-(K-1)}^{K-1} [i-eps, i+eps] near each integer,
    where K sets how many integers are covered and eps = 2^-delta_diff the half-width."""
    return [(i - eps, i + eps) for i in range(-(K - 1), K)]


def scaled_cosine_target(ell):
    """h1(x) = cos((2*pi/2^ell)(x - 1/4)) as a numpy- and mpmath-aware callable."""
    w = 2 * math.pi / (2 ** ell)
    def h1(x):
        if mp is not None and isinstance(x, mp.mpf):
            return mp.cos(mp.mpf(w) * (x - mp.mpf(1) / 4))
        return np.cos(w * (np.asarray(x, dtype=float) - 0.25))
    return h1


def inverse_sine_target():
    """h3(y) = (1/2pi) arcsin(y) as a numpy- and mpmath-aware callable."""
    def h3(y):
        if mp is not None and isinstance(y, mp.mpf):
            return mp.asin(y) / (2 * mp.pi)
        return np.arcsin(np.asarray(y, dtype=float)) / (2 * math.pi)
    return h3


def cosine_minimax(degree, *, ell=2, K=25, eps=2.0 ** -12, precision=None, max_iter=80,
                   continuation=True, start_degree=30, step=2, mp_max_iter=40):
    """Optimal minimax polynomial h1~ of the scaled cosine over the modular-reduction
    region, by the improved multi-interval Remez. Returns the fitted Remez instance.

    The high-degree multi-interval exchange must be warm-started or it overfits and the
    reference selection abandons intervals. With `continuation=True` (default) we fit a low
    degree, then raise it by `step` at a time, re-seeding the references from the previous
    converged fit -- degree continuation that keeps every step inside the convergence basin.

    Two phases:
      * float64 continuation tracks the paper's curve up to its conditioning floor
        (~2^-31 / degree ~58 for K=25, eps=2^-12), fast.
      * if precision=<bits> is given and float64 stops converging before `degree`, the
        continuation carries on IN MPMATH from the last converged degree, finding the
        (sub-float64) extrema in full precision. This reproduces the paper's deepest rows
        (degree 64-68, errors 2^-45...2^-54) exactly. Coefficients are then stored as mpf
        and max_error is measured in full precision.
    Without precision, the float64 result at `degree` is returned (check .spread)."""
    target = scaled_cosine_target(ell)
    ints = mod_reduction_intervals(K, eps)
    if precision is True:                                     # auto: enough bits for this degree
        precision = degree + 50
    s0 = max(6, min(start_degree, len(ints) - 2))
    if not continuation or degree <= s0:
        return Remez(target, degree, ints, general=True, precision=precision).fit(max_iter=max_iter)

    def pad(nodes, count):                                    # add references by bisecting widest gaps
        nodes = sorted(nodes)
        while len(nodes) < count:
            i = max(range(len(nodes) - 1), key=lambda j: nodes[j + 1] - nodes[j])
            nodes.insert(i + 1, (nodes[i] + nodes[i + 1]) / 2)
        return nodes

    # ---- phase 1: float64 degree continuation ----
    r = Remez(target, s0, ints, general=True).fit(max_iter=max_iter)
    last_good = r; prev_refs = r.ref_nodes
    while r.degree < degree:
        d = r.degree + step
        r = Remez(target, d, ints, general=True)
        r.fit(max_iter=max_iter, warm_nodes=pad(prev_refs, r.num_nodes))
        prev_refs = r.ref_nodes
        if r.spread < 1e-2:
            last_good = r
        elif precision is not None and mp is not None:
            break                                            # hand over to mpmath
    if precision is None or mp is None or last_good.degree >= degree:
        return last_good if last_good.degree >= degree else r

    # ---- phase 2: mpmath degree continuation from the last converged float64 fit ----
    prec = int(precision)
    nodes = last_good.ref_nodes
    result = last_good
    d = last_good.degree
    while d < degree:
        d += step
        rr = Remez(target, d, ints, general=True, precision=prec)
        c, peak, spread, refs, iters = rr._fit_general_mpmath(
            None, prec, mp_max_iter, mp.power(2, -45), init_nodes=pad(nodes, rr.num_nodes))
        rr.coeffs = list(c)
        rr.max_error = rr.min_error = rr.leveled_error = float(peak)
        rr.spread = float(spread); rr.iters = iters
        rr.backend = f"mpmath({prec})"; rr.ref_nodes = refs
        nodes = refs; result = rr
    return result


def arcsin_minimax(degree, *, eps=2.0 ** -12, s=None, precision=None, max_iter=80):
    """Optimal minimax polynomial h3~ of (1/2pi)arcsin over a single interval [-s, s],
    where s = sin(2*pi*eps) is the range of sin(2*pi*normod) for |normod| <= eps. arcsin
    is odd, so the even Chebyshev coefficients come out ~0. Returns the fitted Remez."""
    if s is None:
        s = math.sin(2 * math.pi * eps)
    return Remez(inverse_sine_target(), degree, [(-s, s)],
                 general=True, precision=precision).fit(max_iter=max_iter)


def arcsin_minimax_coefs(degree, eps, delta=1, tol=1e-7):
    asin = arcsin_minimax(degree=degree, eps=eps)
    c_as  = np.array([np.float64(c) for c in asin.coeffs])
    c_as *= delta
    d_asin = poly2cheb(cheb2poly(c_as) / (asin.hull_hi ** np.arange(len(c_as))))
    f_arcsin = d_asin[::-1].astype(np.float64)
    f_arcsin[f_arcsin < tol] = 0.0
    return f_arcsin

def eval_mod(cos_fit, asin_fit, ell):
    """Compose the EvalMod approximation normod(x) ~ h3~ . h2^ell . h1~(x) into a single
    callable. The inter-stage scalings are explicit (they differ per stage): the input is
    normalized over the cosine hull, h2 keeps values in [-1, 1], and the value is rescaled
    by 1/s before the inverse-sine stage. Accepts a scalar or a numpy array."""
    chull = cos_fit.hull_hi                                  # cosine hull is symmetric, mid = 0
    s = asin_fit.hull_hi                                     # inverse-sine half-width
    ccos = np.array([float(c) for c in cos_fit.coeffs])
    casin = np.array([float(c) for c in asin_fit.coeffs])

    def approx(x):
        y = chebval(np.asarray(x, dtype=float) / chull, ccos)   # ~ cos((2pi/2^ell)(x-1/4))
        for _ in range(ell):
            y = 2.0 * y * y - 1.0                                # double-angle -> ~ sin(2pi x)
        return chebval(y / s, casin)                            # ~ normod(x)
    return approx


def eval_mod_error(cos_fit, asin_fit, ell, K, eps, samples=400):
    """Worst-case |eval_mod(x) - normod(x)| over the modular-reduction region D."""
    approx = eval_mod(cos_fit, asin_fit, ell)
    worst = 0.0
    for a, b in mod_reduction_intervals(K, eps):
        xs = np.linspace(a, b, samples)
        worst = max(worst, float(np.max(np.abs(approx(xs) - (xs - np.round(xs))))))
    return worst


def eval_mod_minimax(*, ell=2, K=13, eps=2.0 ** -8, cos_degree=36, asin_degree=11,
                     precision=None, max_iter=80, verbose=False):
    """Build the full EvalMod approximation of normod: fit the scaled cosine h1~ and the
    inverse sine h3~ with the improved multi-interval Remez, then compose them with ell
    double-angle steps. Returns a SimpleNamespace with fields:
        cos   : fitted Remez for h1~ (the scaled cosine)
        asin  : fitted Remez for h3~ (the inverse sine)
        eval  : the composed callable approximating normod
        error : worst-case |approx - normod| over D
        s     : the inverse-sine half-width sin(2*pi*eps)
        ell, K, eps : the parameters used
    Note: h3~ is fit on [-s, s] with s = sin(2*pi*eps), matching the range of sin(2*pi*x)
    on D, so the cosine and inverse-sine domains are consistent by construction."""
    s = math.sin(2 * math.pi * eps)
    cos_fit = cosine_minimax(cos_degree, ell=ell, K=K, eps=eps,
                             precision=precision, max_iter=max_iter)
    asin_fit = arcsin_minimax(asin_degree, s=s, precision=precision, max_iter=max_iter)
    err = eval_mod_error(cos_fit, asin_fit, ell, K, eps)
    if verbose:
        ce = cos_fit.max_error if cos_fit.max_error is not None else float("nan")
        ae = asin_fit.max_error if asin_fit.max_error is not None else float("nan")
        print(f"cos  deg {cos_degree}: minimax err 2^{math.log2(max(ce,1e-300)):.2f} "
              f"(spread {cos_fit.spread:.1e})")
        print(f"asin deg {asin_degree}: minimax err 2^{math.log2(max(ae,1e-300)):.2f} "
              f"(spread {asin_fit.spread:.1e})")
        print(f"composite normod error over D: 2^{math.log2(max(err,1e-300)):.2f}")
    return SimpleNamespace(cos=cos_fit, asin=asin_fit, eval=eval_mod(cos_fit, asin_fit, ell),
                           error=err, s=s, ell=ell, K=K, eps=eps)


def plot_eval_mod(em, window=3, num=4000, save=None, show=True, title=None):
    """Plot the composite EvalMod approximation, analogous to the sign composite plot.
    Top: the true modular reduction normod(x)=x-round(x) (a sawtooth) and the composed
    approximation eval(x) over a few integers, with the approximation intervals [i-eps,i+eps]
    shaded (the approximation is only meant to match inside those bands; between them it is
    free). Bottom: the error |eval(x) - normod(x)| across the whole region D (subintervals
    stitched left to right, log scale), with the worst-case error marked.

    `em` is the namespace returned by eval_mod_minimax. `window` is how many integers on each
    side of 0 to draw in the top panel."""
    import matplotlib.pyplot as plt
    approx, eps, K = em.eval, em.eps, em.K
    bl = math.log2(em.error) if em.error > 0 else float("-inf")

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7),
                                 gridspec_kw={"height_ratios": [3, 2]})

    win = min(window, K - 1)
    xs = np.linspace(-win - 0.5, win + 0.5, num)
    a1.plot(xs, xs - np.round(xs), color="#bcbcbc", lw=2.5, label="normod(x) = x - round(x)")
    a1.plot(xs, approx(xs), color="#1b4965", lw=1.4,
            label=f"eval ~ h3 . h2^{em.ell} . h1  (cos deg {em.cos.degree}, asin deg {em.asin.degree})")
    for i in range(-win, win + 1):
        a1.axvspan(i - eps, i + eps, color="#ffd6a5", alpha=0.8,
                   label=f"approx region |x-i|<=2^{math.log2(eps):.0f}" if i == -win else None)
    a1.set_ylim(-0.75, 0.75); a1.set_xlabel("x"); a1.set_ylabel("value")
    a1.grid(alpha=0.25); a1.legend(loc="upper center", fontsize=9, ncol=1)
    a1.set_title("composite modular reduction: approximation vs truth")

    # bottom: error across all of D, subintervals stitched
    pos = 0
    for a, b in mod_reduction_intervals(K, eps):
        xx = np.linspace(a, b, 60)
        err = np.abs(approx(xx) - (xx - np.round(xx)))
        a2.semilogy(np.arange(pos, pos + len(xx)), np.maximum(err, 1e-300),
                    color="#1b4965", lw=0.7)
        pos += len(xx)
    a2.axhline(em.error, ls="--", color="#bb3e03", label=f"worst error 2^{bl:.1f}")
    a2.set_xlim(0, pos); a2.set_xlabel(f"sample index  ({2 * K - 1} intervals stitched)")
    a2.set_ylabel("|eval - normod|"); a2.grid(alpha=0.25, which="both")
    a2.legend(loc="upper right", fontsize=9)

    fig.suptitle(title or f"EvalMod over D (K={K}, eps=2^{math.log2(eps):.0f}, "
                          f"ell={em.ell}): normod error 2^{bl:.1f}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if save: fig.savefig(save, dpi=150, bbox_inches="tight")
    if show: plt.show()
    return fig


# ======================================================================
# Demo
# ======================================================================
if __name__ == "__main__":
    # "Give me the cheapest (smallest-depth) sign polynomial that resolves inputs
    #  down to 2^-12 from zero, with 15 bits of output accuracy."
    P = sign_minimax(x_accuracy=12, y_accuracy=15, verbose=True)
    print("chosen degrees:", P.degrees(), " depth:", P.depth(), " bits:", round(P.bits, 2))

    # EvalMod: approximate the modular reduction normod(x) = x - round(x) as the
    # composition of an optimal-minimax scaled cosine (multi-interval) and inverse sine.
    print("\n--- EvalMod: normod(x) ~ h3~ . h2^ell . h1~(x) ---")
    em = eval_mod_minimax(ell=2, K=25, eps=2.0 ** -12 ,
                          cos_degree=60, asin_degree=5, precision=True, verbose=True)
    for x in (0.013, 0.97, 2.004, -2.98):
        print(f"  x={x:+.3f}: approx={em.eval(x):+.6f}  normod={x - round(x):+.6f}")