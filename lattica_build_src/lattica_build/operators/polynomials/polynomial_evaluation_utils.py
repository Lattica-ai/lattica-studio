"""See `operators/polynomials/README.md` for usage details."""

import numpy as np
from numpy.polynomial.chebyshev import chebfit, chebval
from numpy.polynomial import Chebyshev
from scipy import signal


def _indicator_cheb_coeffs(max_val, j, deg, peak_width, left_tol=0.01, num=10_000, tol=None, plot=False):
    """
    We want to nullify the nearest nodes as the next ones will be exponentially suppressed by the gaussian.
    If the distribution around the nodes is very narrow, we don't waste polynomial energy on most of the gaussian,
    while forcing the fit to achieve zero amplitude at the adjacent nodes.

    peak_width: width of region around the peak that the will be fitted, while the rest of the peak is ignored
    left_tol: as the input to this polynomial is not exact and some values can be mapped outside the [-1,1] range,
              we force the fit to be accurate for another small interval left to -1.
              (at the other boundary of 1, we didn't observe wild behaviour of the polynomial)
    """

    nodes_distance = 2 / max_val

    # a heuristic: the next node should be at 5*sigma (to be very close to 0: exp(-25/2) =~ 2^-18)
    sigma = nodes_distance / 5

    mu = 2 * j / max_val - 1

    # force the polynomial to be zero outside this interval
    center_boundaries = [max(-1, mu - nodes_distance + peak_width), min(mu + nodes_distance - peak_width, 1)]

    def _gaussian_indicator(x):
        return np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

    # densely sample the tails, while sparsely sample the center because the zeros accumulate in the sums afterwards
    left_margin = max(center_boundaries[0] + 1, 0)
    right_margin = max(1 - center_boundaries[1], 0)
    ratio_left = left_margin / (left_margin + right_margin)
    num_pts_left = int(num * ratio_left)
    num_pts_right = num - num_pts_left
    num_pts_center = 30  # fixed number of points in the center (need to manually tune if sigma is drastically changed)

    xl = np.linspace(-1 - left_tol, center_boundaries[0], num_pts_left)
    xc = np.linspace(mu - peak_width, mu + peak_width, num_pts_center)
    xr = np.linspace(center_boundaries[1], 1, num_pts_right)
    x_fit = np.concatenate([xl, xc, xr])

    y_fit = _gaussian_indicator(x_fit)

    w = 100
    weight = np.concat([w * np.ones_like(xl), np.ones_like(xc), w * np.ones_like(xr)])

    coefs = chebfit(x_fit, y_fit, deg=deg, w=weight)

    if tol is not None:
        coefs[abs(coefs) < tol] = 0

    if plot:
        import matplotlib.pyplot as plt
        xx = np.linspace(-1 - left_tol, 1, 10_000)
        center_mask = (xx > mu - peak_width) & (xx < mu + peak_width)
        tail_mask = (xx < center_boundaries[0]) | (xx > center_boundaries[1])
        target = _gaussian_indicator(xx)
        approx = chebval(xx, coefs)
        err = np.abs(approx - target)
        err_center = err[center_mask]
        # the error of the tail is compared to ZERO, so the error is just the value of the approx
        err_tail = np.abs(approx[tail_mask])

        plt.figure()
        plt.plot(xx, target, label='Original Function')
        plt.plot(xx, approx, label=f'Chebyshev Approximation {deg=}')
        plt.axvspan(center_boundaries[0], mu - peak_width, color='gray', alpha=0.2)
        plt.axvspan(mu + peak_width, center_boundaries[1], color='gray', alpha=0.2)
        plt.title(f'max absolute error tail: {np.log2(err_tail.max())}\nmax absolute error center: {np.log2(err_center.max())}')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

    return coefs[::-1].copy()


def indicator_cheb_coeffs_packed(max, deg, peak_width=0.01, num=10_000, tol=None, plot=False):
    packed_indicators = []
    for j in range(1, max + 1):
        cheb_coefs = _indicator_cheb_coeffs(max, j, deg=deg, peak_width=peak_width, num=num, tol=tol, plot=plot)
        packed_indicators.append(cheb_coefs)

    return np.stack(packed_indicators)


def get_cheb_coefs(
        deg,
        func,
        x_fit=None,
        y_fit=None,
        left=-1,
        right=1,
        num=10001,
        plot=False,
        weight=None,
        tol=None,
):
    """Fit Chebyshev coeffs mapped to the domain [left, right].

    Uses the Chebyshev.fit class method which automatically maps
    the input domain [left, right] to the canonical window [-1, 1]
    before evaluating the series.

    If ``x_fit`` / ``y_fit`` are provided explicitly, they should now
    be passed in their true original domain [left, right], rather
    than manually pre-mapped to [-1, 1].
    """
    # 1. Generate or use data in the original [left, right] domain
    if x_fit is None:
        x_fit = np.linspace(left, right, num)

    if y_fit is None:
        y_fit = func(x_fit)

    # 2. Fit using the class method (handles the [-1, 1] mapping internally)
    fitted_series = Chebyshev.fit(x_fit, y_fit, deg=deg, domain=[left, right], w=weight)

    # 3. Extract the canonical coefficients (mapped to the [-1, 1] window)
    coefs = fitted_series.coef

    # 4. Apply tolerance threshold
    if tol is not None:
        coefs[abs(coefs) < tol] = 0
        # Update the series object so the plot reflects the zeroed coefficients
        fitted_series.coef = coefs

    if plot:
        import matplotlib.pyplot as plt
        plt.figure()
        xx = np.linspace(left, right, 1000)

        if func is not None:
            plt.plot(xx, func(xx), label='Original Function')
        if y_fit is not None and x_fit is not None:
            plt.plot(x_fit, y_fit, label='Fitted Data')
        plt.plot(xx, fitted_series(xx), label=f'Chebyshev Approximation\n(degree={deg})')
        plt.legend()
        plt.show()

    return coefs[::-1].copy()  # copy ndarray is equivalent to contiguous


def heaviside_sigmoid_cheb_coeffs(step, deg=100, sharpness=79, out_val=1, num=10001, tol=None, plot=False):

    def _sigmoid_heaviside(x):
        return out_val / (1 + np.exp(-(x - step) * sharpness))

    return get_cheb_coefs(deg, _sigmoid_heaviside, num=num, tol=tol, plot=plot)


def heaviside_peacewise_linear_cheb_coeffs(deg, margin, out_val=1, tol=None, plot=False):
    def _piecewise_linear(x):
        a, b = margin
        y = np.zeros_like(x, dtype=x.dtype)

        # Linear region mask
        mask = (x >= a) & (x <= b)

        # Scale linearly between a and b
        y[mask] = (x[mask] - a) / (b - a) * out_val

        # Values above b are clipped to out_val
        y[x > b] = out_val

        return y

    xl = np.linspace(-1, margin[0], 1000)
    xr = np.linspace(margin[1], 1, 1000)
    x_fit = np.concatenate([xl, xr])

    return get_cheb_coefs(deg, _piecewise_linear, x_fit=x_fit, tol=tol, plot=plot)


def threshold_cheb_coeffs_minimax(deg, margin, out_val=1, domain_start=-1, tol=None, plot=False):
    """
    Calculates the Minimax Chebyshev coefficients for a threshold function ignoring the margin using the Remez algorithm.

    margin: [left, right] of the ignored transition region
    domain_start: ignored region on the left of the domain
    """
    assert domain_start < margin[0], "domain must start left to the margin"

    # 1. Number of taps must be odd to create a symmetric Type I filter
    # A filter of length 2*deg + 1 yields a polynomial of degree 'deg'
    numtaps = 2 * deg + 1

    # 1. Map coordinates to frequencies (descending order)
    f_right_end = np.arccos(margin[1]) / np.pi
    f_left_start = np.arccos(margin[0]) / np.pi
    if domain_start > -1:
        f_domain_end = np.arccos(domain_start) / np.pi

        # We need a tiny frequency gap before the leash starts to prevent band overlap
        epsilon = 0.01
        f_leash_start = min(f_domain_end + epsilon, 1.0)

    # 2. Define the three bands
    # Band 1: Target = 1 (The region > margin[1])
    # Band 2: Target = 0 (The active region we care about < margin[0])
    # Band 3: Target = 0 (The empty region we want to leash)
    if domain_start == -1:
        bands = [0, f_right_end, f_left_start, 1]
    else:
        bands = [
            0, f_right_end,
            f_left_start, f_domain_end,
            f_leash_start, 1.0
        ]

    # Desired amplitudes for each band
    desired = [out_val, 0] if domain_start == -1 else [out_val, 0, 0]

    # 3. Apply the Weights
    # We weight the regions we care about heavily
    # We weight the empty region lightly just to keep it bounded
    weights = None if domain_start == -1 else [100, 100, 1]

    # 4. Run the Parks-McClellan (Remez) algorithm
    h = signal.remez(numtaps, bands, desired, weight=weights, fs=2)

    # 5. Extract coefficients
    M = deg
    cheb_coeffs = np.zeros(M + 1)

    cheb_coeffs[0] = h[M]
    for k in range(1, M + 1):
        cheb_coeffs[k] = 2 * h[M - k]

    if tol is not None:
        cheb_coeffs[abs(cheb_coeffs) < tol] = 0

    if plot:
        import matplotlib.pyplot as plt
        x_plot = np.linspace(domain_start, 1, 10_000)
        y_approx = chebval(x_plot, cheb_coeffs)

        y_target = (np.sign(x_plot - np.mean(margin)) + 1) / 2 * out_val
        mask = (x_plot < margin[0]) | (x_plot > margin[1])
        err_th = np.abs(y_approx[mask] - y_target[mask])
        print(f'THRESH: log2 max err: {np.log2(err_th.max())}, log2 mean err: {np.log2(err_th.mean())}')

        plt.figure(figsize=(10, 6))
        plt.plot(x_plot, y_approx, label=f'Remez Approx ({deg=})', color='blue')
        plt.plot(x_plot, y_target, alpha=0.3, label='Target', color='black', linewidth=2)
        plt.axvspan(margin[0], margin[1], color='gray', alpha=0.2, label='Don\'t Care Gap')
        plt.title(
            f'Remez Split-Domain Approximation\nMax Absolute Error in Care Bands: {np.log2(err_th.max()):.2f} (log2)')
        plt.legend()
        plt.grid(True)

        plt.show()

    return cheb_coeffs[::-1].copy()


def find_optimal_k_m(deg):
    """ Compute positive integers k,m such that n < k(2^m-1), k is close to sqrt(n/2)
    and the depth = ceil(log2(k))+m is minimized. Moreover, for that depth the
    number of homomorphic multiplications = k+2m+2^(m-1)-4 is minimized.
    """

    m_list_intervals = [0, 2, 11, 13, 17, 55, 59, 76, 239, 247, 284, 991, 1007, 1083, 2015, 2031, 2204]
    m_list_values = [1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5, 6, 5, 6, 7, 6]

    if deg < 1 or deg > 2204:
        raise ValueError("n must be in the range [1, 2204]")
    for i in range(len(m_list_intervals)):
        if deg > m_list_intervals[i] and deg <= m_list_intervals[i + 1]:
            m = m_list_values[i]
            k = int(np.floor(deg / (2 ** m - 1) + 1))
            return m, k
