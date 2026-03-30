"""
This module provides helper and plotting functions built on top of the
VIPurPCA library for uncertainty-aware principal component analysis (PCA).

It enables the computation of eigenvectors under uncertainty and the
visualization of distribution trajectories in principal component space.

For details on the VIPurPCA method, see the corresponding paper:
https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10368349

"""


import jax
import jax.numpy as np
from jax import vmap
import numpy
import warnings
from scipy.linalg import block_diag
from uadapy import Distribution
from scipy.linalg import null_space
from vipurpca import PCA
from vipurpca.helper_functions import equipotential_standard_normal


def _prepare_pca_inputs(dists):
    """
    Prepare stacked means, block-diagonal covariance, and labels from distributions.

    Parameters
    ----------
    dists : list of Distribution
        List of distribution objects. Each distribution must provide
        `.mean()` and `.cov()`

    Returns
    -------
    Y : ndarray of shape (n, p)
        Stacked mean vectors.
    C : ndarray of shape (n*p, n*p)
        Block-diagonal covariance matrix.
    """
    means = []
    cov_blocks = []

    for dist in dists:
        mu = numpy.atleast_2d(dist.mean())
        cov = numpy.asarray(dist.cov())
        means.append(mu)
        cov_blocks.append(cov)

    Y = numpy.vstack(means).astype(numpy.float32)
    C = block_diag(*cov_blocks).astype(numpy.float32)

    return np.asarray(Y), np.asarray(C)


def _effective_rank_from_X(X, tol=None):
    """
    Compute the effective numerical rank of a matrix using SVD.

    Parameters
    ----------
    X : ndarray of shape (n, p)
        Input data matrix.
    tol : float, optional
        Numerical tolerance. If None, a default based on matrix size
        and precision is used.

    Returns
    -------
    rank : int
        Effective rank of the input matrix.
    """
    Xc = X - np.mean(X, axis=0, keepdims=True)
    s = np.linalg.svd(Xc, compute_uv=False)
    if tol is None:
        tol = max(Xc.shape) * np.finfo(s.dtype).eps * s[0]
    return int(np.sum(s > tol))


def _fit_pca_with_uncertainty(Y, cov_Y, n_components=2):
    """
    Fit PCA model with uncertainty propagation.

    Parameters
    ----------
    Y : ndarray of shape (n, p)
        Stacked mean vectors.
    cov_Y : ndarray of shape (n*p, n*p)
        Covariance matrix (full).
    n_components : int, default=2
        Number of principal components to retain.

    Returns
    -------
    model : PCA
        PCA model containing eigenvalues, eigenvectors, and
        covariance of eigenvectors.
    """
    r = _effective_rank_from_X(Y)
    if n_components > r:
        warnings.warn(
            f"Reducing n_components from {n_components} to {r} due to rank deficiency.",
            category=UserWarning,
            stacklevel=2,
        )
        n_components = r

    model = PCA(matrix=Y,
                sample_cov=None,
                feature_cov=None,
                full_cov=cov_Y,
                n_components=n_components,
                axis=0)

    model.pca_value()
    model.compute_cov_eigenvectors()
    return model


def fit_distribution_pca(dists, n_components=2):
    """
    Fit an uncertainty-aware PCA model on a set of distributions and return the full PCA model.

    Parameters
    ----------
    dists : list of Distribution
        List of distribution objects with `.mean()` and `.cov()`.
    n_components : int, default=2
        Number of principal components to retain.

    Returns
    -------
    model : vipurpca.PCA
        A fitted VIPurPCA PCA model. The returned object exposes both standard PCA outputs and
        uncertainty-related quantities needed for distribution plots.
    """
    Y, cov_Y = _prepare_pca_inputs(dists)
    model = _fit_pca_with_uncertainty(Y, cov_Y, n_components)
    return model


def compute_distribution_trajectories(distributions, n_components=2, n_frames=10, seed=55):
    """
    Compute trajectories of distributions under PCA with uncertainty.

    For each distribution, PCA directions are sampled according to the
    propagated covariance of eigenvectors. Trajectories in the principal
    component space are then computed for each sample.

    Parameters
    ----------
    distributions : list of Distribution
        List of distribution objects with `.mean()` and `.cov()`
    n_components : int, default=2
        Number of principal components to retain.
    n_frames : int, default=10
        Number of trajectory samples to draw.
    seed : int, default=55
        Random seed for reproducibility.

    Returns
    -------
    trajectories : ndarray of shape (n_frames + 1, n_samples, n_components)
        Trajectory coordinates in the selected PC plane for each sample.
        The first dimension corresponds to sampled eigenvector frames.

    """
    numpy.random.seed(seed)

    model = fit_distribution_pca(distributions, n_components)

    if model.cov_eigenvectors is None:
        raise RuntimeError("Uncertainty of eigenvectors has not been computed.")

    # Sample eigenvectors under uncertainty
    S = equipotential_standard_normal(model.size[1] * model.n_components, n_frames + 1)
    L, _ = jax.scipy.linalg.cho_factor(
        model.cov_eigenvectors + 1e-5 * np.eye(model.cov_eigenvectors.shape[0]),
        lower=True,
    )
    eigv_samples = np.transpose(np.dot(L, S)) + np.ravel(model.eigenvectors, "F")

    # Reshape sampled eigenvectors into matrices and project data
    samples_reshaped = vmap(
        lambda s: np.transpose(
            np.reshape(s, (min(model.size[0], model.n_components), model.size[1]), "C")
        )
    )(eigv_samples)

    trajectories = np.array([model.X_unflattener(model.X_flat) @ i for i in samples_reshaped])

    return trajectories


def estimate_pc_direction_uncertainty(mean_w_matrix, cov_eigenvectors, n_samples=1000, seed=55):
    """
    Estimate principal-direction mean and angular uncertainty for N-dimensional PCs.

    For each PC, samples directions from the Gaussian uncertainty, resolves sign
    ambiguity, computes the Frechet mean direction, and estimates angular spread
    via signed tangent-plane projection.

    Parameters
    ----------
    mean_w_matrix : array-like of shape (p, n_components)
        Eigenvector matrix from model.eigenvectors.
        Each column is one principal component direction.
    cov_eigenvectors : array-like of shape (n_components*p, n_components*p)
        Joint covariance of all eigenvector weights from model.cov_eigenvectors.
    n_samples : int, default=1000
        Number of Monte Carlo samples per PC.
    seed : int, default=55
        Random seed for reproducibility.

    Returns
    -------
    results : list of dict, one per PC
        Each dict contains:
            'pc'    : int          - PC index (1-based)
            'mu'    : ndarray (p,) - unit mean direction vector
            'sigma' : float        - angular standard deviation in radians
    """
    rng = numpy.random.default_rng(seed)

    mean_w_matrix    = numpy.asarray(mean_w_matrix, float)
    cov_eigenvectors = numpy.asarray(cov_eigenvectors, float)

    p = mean_w_matrix.shape[0]
    n_components = mean_w_matrix.shape[1]

    results = []

    for k in range(n_components):

        # Extract mean vector and covariance block for this PC
        mean_w = mean_w_matrix[:, k]
        cov_w  = cov_eigenvectors[k*p:(k+1)*p, k*p:(k+1)*p]

        # Sample possible directions from Gaussian uncertainty
        W = rng.multivariate_normal(mean=mean_w, cov=cov_w, size=n_samples)

        # Normalize onto unit hypersphere
        norms = numpy.linalg.norm(W, axis=1)
        valid = norms > 1e-12
        W = W[valid] / norms[valid, None]

        # Resolve sign ambiguity
        mw   = mean_w / (numpy.linalg.norm(mean_w) + 1e-12)
        dots = W @ mw
        W[dots < 0] *= -1

        # Average unit vectors and re-normalize
        mean_dir = numpy.mean(W, axis=0)
        mean_dir /= numpy.linalg.norm(mean_dir)

        # Project samples onto tangent plane at mean_dir
        # v_tangent = v - (v · mean_dir) * mean_dir
        dots_mean = W @ mean_dir
        W_tangent = W - dots_mean[:, None] * mean_dir

        # Signed deviations via tangent reference axis
        # null_space gives orthonormal basis of the hyperplane perpendicular to mean_dir
        tangent_ref       = null_space(mean_dir.reshape(1, -1))[:, 0]
        signed_deviations = W_tangent @ tangent_ref

        sigma = float(numpy.std(signed_deviations))

        results.append({
            'pc'   : k + 1,
            'mu'   : mean_dir,
            'sigma': sigma,
        })

    return results


def compute_uncertain_projections(model, pcx=1, pcy=2, n_samples=1000, seed=55):
    """
    Computes projected distributions for VIPurPCA model eigenvector samples.

    Parameters
    ----------
    model : vipurpca.PCA
        Fitted VIPurPCA model with computed eigenvectors and eigenvector covariance.
    pcx, pcy : int
        1-based indices of principal components to return.
    n_samples : int, default=1000
        Number of eigenvector samples to draw.
    seed : int, default=55
        Random seed for reproducibility.

    Returns
    -------
    projected_distributions : list
        list of Distribution objects (one per data point)
    """
    rng = numpy.random.default_rng(seed)

    X_np = numpy.asarray(model.X_unflattener(model.X_flat))
    n, p = model.size
    n_components = model.n_components
    ix, iy = pcx - 1, pcy - 1

    mean_flat = numpy.asarray(model.eigenvectors).T.flatten()
    cov_flat  = numpy.asarray(model.cov_eigenvectors)

    samples = rng.multivariate_normal(
        mean=mean_flat,
        cov=cov_flat + 1e-5 * numpy.eye(cov_flat.shape[0]),
        size=n_samples
    )

    W_samples = samples.reshape(n_samples, n_components, p).transpose(0, 2, 1)

    # Project mean-centered data: (n_samples, n_data, n_components)
    projections = numpy.einsum('ip, spk -> sik', X_np, W_samples)

    projected_distributions = []
    for i in range(n):
        pts = projections[:, i, :][:, [ix, iy]]
        projected_distributions.append(Distribution(pts, name="samples"))

    return projected_distributions


def draw_pc_wedge(res, ax, line_len, label, fill_alpha, line_alpha, n_std):
    """
    Draws a single PC directional uncertainty wedge (lines + filled region) onto existing axes.

    Parameters
    ----------
    res : dict
        Single entry from estimate_pc_direction_uncertainty output,
        with keys 'pc', 'mu' (unit vector), 'sigma' (float, radians).
    ax : matplotlib.axes.Axes
        Axes to draw onto.
    line_len : float
        Half-length of the direction lines (typically from ax xlim/ylim).
    label : str
        Legend label for this PC line.
    fill_alpha : float
        Opacity of the uncertainty wedge fill.
    line_alpha : float
        Opacity of the direction lines.
    n_std : float
        Number of std deviations for the wedge width.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The input axes with the PC direction line, dashed uncertainty
        boundary lines, and filled wedge added in-place.
    """
    mu    = float(numpy.arctan2(res['mu'][1], res['mu'][0]))
    sigma = res['sigma']

    # main direction line
    d = numpy.array([numpy.cos(mu), numpy.sin(mu)])
    ax.plot(
        [-line_len * d[0], line_len * d[0]],
        [-line_len * d[1], line_len * d[1]],
        linewidth=3,
        alpha=line_alpha,
        label=label,
    )

    # dashed +/- n_std lines
    for sgn in (-1, +1):
        a  = mu + sgn * n_std * sigma
        dd = numpy.array([numpy.cos(a), numpy.sin(a)])
        ax.plot(
            [-line_len * dd[0], line_len * dd[0]],
            [-line_len * dd[1], line_len * dd[1]],
            linestyle="--",
            linewidth=2,
            alpha=line_alpha,
        )

    # filled wedge between -/+ n_std
    a1 = mu - n_std * sigma
    a2 = mu + n_std * sigma
    A  = numpy.array([numpy.cos(a1), numpy.sin(a1)]) * line_len
    B  = numpy.array([numpy.cos(a2), numpy.sin(a2)]) * line_len

    ax.fill([0,  A[0],  B[0]], [0,  A[1],  B[1]], alpha=fill_alpha)
    ax.fill([0, -A[0], -B[0]], [0, -A[1], -B[1]], alpha=fill_alpha)

    return ax

def _estimate_feature_angular_uncertainty(W, cov_eigenvectors, ix, iy, p):
    """
    Computes angular uncertainty (std dev in radians) for each feature's
    2D loading vector in the (ix, iy) PC plane.

    Parameters
    ----------
    W : numpy.ndarray, shape (p, n_components)
        Eigenvector matrix from the fitted model.
    cov_eigenvectors : numpy.ndarray
        Joint covariance of eigenvectors from model.cov_eigenvectors.
    ix : int
        0-based index of the x-axis PC.
    iy : int
        0-based index of the y-axis PC.
    p : int
        Number of features.

    Returns
    -------
    feature_sigmas : list of float
        Angular std dev in radians for each of the p features.
    """
    cov_w         = numpy.asarray(cov_eigenvectors)
    feature_sigmas = []
    rng            = numpy.random.default_rng(55)

    for j in range(p):
        row_ix = ix * p + j
        row_iy = iy * p + j
        cov_2d = numpy.array([
            [cov_w[row_ix, row_ix], cov_w[row_ix, row_iy]],
            [cov_w[row_iy, row_ix], cov_w[row_iy, row_iy]],
        ])
        mean2d  = numpy.array([W[j, ix], W[j, iy]])
        samples = rng.multivariate_normal(
            mean = mean2d,
            cov  = cov_2d + 1e-9 * numpy.eye(2),
            size = 1000,
        )
        angles = numpy.arctan2(samples[:, 1], samples[:, 0])
        ref    = numpy.arctan2(mean2d[1], mean2d[0])
        angles = numpy.where(angles - ref >  numpy.pi, angles - 2 * numpy.pi, angles)
        angles = numpy.where(angles - ref < -numpy.pi, angles + 2 * numpy.pi, angles)
        feature_sigmas.append(float(numpy.std(angles)))

    return feature_sigmas

def _draw_loading_wedge(ax, dx, dy, sigma, n_std, color, fill_alpha, line_alpha):
    """
    Draws a single one-directional angular uncertainty wedge (filled region
    + dashed boundary lines) for a loading arrow onto existing axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw onto.
    dx : float
        x-component of the scaled loading arrow (plot units).
    dy : float
        y-component of the scaled loading arrow (plot units).
    sigma : float
        Angular standard deviation in radians for this feature.
    n_std : float
        Number of std deviations for the wedge width.
    color : str
        Color for the wedge fill and dashed boundary lines.
    fill_alpha : float
        Opacity of the wedge fill.
    line_alpha : float
        Opacity of the dashed boundary lines.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The input axes with wedge drawn in-place.
    """
    mu       = numpy.arctan2(dy, dx)
    line_len = numpy.sqrt(dx**2 + dy**2)

    a1 = mu - n_std * sigma
    a2 = mu + n_std * sigma
    A  = numpy.array([numpy.cos(a1), numpy.sin(a1)]) * line_len
    B  = numpy.array([numpy.cos(a2), numpy.sin(a2)]) * line_len

    ax.fill(
        [0,  A[0],  B[0]],
        [0,  A[1],  B[1]],
        color  = color,
        alpha  = fill_alpha,
        zorder = 3,
    )

    for sgn in (-1, +1):
        a  = mu + sgn * n_std * sigma
        dd = numpy.array([numpy.cos(a), numpy.sin(a)]) * line_len
        ax.plot(
            [0, dd[0]], [0, dd[1]],
            linestyle = "--",
            linewidth = 1.0,
            color     = color,
            alpha     = line_alpha,
            zorder    = 3,
        )

    return ax

def draw_loading_arrows(
    model,
    ax,
    pcx=1,
    pcy=2,
    feature_names=None,
    point_names=None,
    arrow_scale=1.0,
    arrow_color="red",
    arrow_lw=2.0,
    arrow_head_width=0.2,
    arrow_head_length=0.4,
    point_label_fontsize=9,
    feature_label_fontsize=10,
    show_uncertainty=False,
    n_std=1.0,
    wedge_fill_alpha=0.15,
    wedge_line_alpha=0.5
):
    """
    Draws PCA loading arrows for each feature and labels each data point
    at its mean score position onto existing axes. Optionally draws an
    angular uncertainty wedge around each loading arrow.

    Parameters
    ----------
    model : vipurpca.PCA
        Fitted VIPurPCA model with computed eigenvectors and eigenvector covariance.
    ax : matplotlib.axes.Axes
        Existing axes to draw onto.
    pcx : int, optional
        1-based index of the PC to use as the x-axis. Defaults to 1.
    pcy : int, optional
        1-based index of the PC to use as the y-axis. Defaults to 2.
    feature_names : list of str, optional
        Names for each feature used to label loading arrows.
        Defaults to ['x1', 'x2', ...] if not provided.
    point_names : list of str, optional
        Names for each data point used to label score positions.
        Defaults to ['P1', 'P2', ...] if not provided.
    arrow_scale : float, optional
        Multiplicative scaling factor applied to all loading arrows
        relative to the plot range. Defaults to 1.0.
    arrow_color : str, optional
        Color of loading arrows, feature labels, and uncertainty wedges.
        Defaults to 'red'.
    arrow_lw : float, optional
        Line width of the loading arrows. Defaults to 2.0.
    arrow_head_width : float, optional
        Width of the arrow head. Defaults to 0.2.
    arrow_head_length : float, optional
        Length of the arrow head. Defaults to 0.4.
    point_label_fontsize : int, optional
        Font size for data point labels drawn at score positions.
        Defaults to 9.
    feature_label_fontsize : int, optional
        Font size for feature name labels drawn at arrow tips.
        Defaults to 10.
    show_uncertainty : bool, optional
        If True, draws an angular uncertainty wedge around each loading
        arrow using the covariance of eigenvectors from the model.
        Defaults to False.
    n_std : float, optional
        Number of standard deviations used for the wedge half-width.
        Defaults to 1.0.
    wedge_fill_alpha : float, optional
        Opacity of the uncertainty wedge filled region. Defaults to 0.15.
    wedge_line_alpha : float, optional
        Opacity of the dashed wedge boundary lines. Defaults to 0.5.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The input axes with loading arrows, feature labels, point labels,
        and optional uncertainty wedges drawn in-place.
    """
    n, p = model.size
    ix   = pcx - 1
    iy   = pcy - 1
    W    = numpy.asarray(model.eigenvectors)
    cov_eigenvectors = numpy.asarray(model.cov_eigenvectors)

    if feature_names is None:
        feature_names = [f"x{j+1}" for j in range(p)]
    if point_names is None:
        point_names = [f"P{i+1}" for i in range(n)]

    # Label each blob at its mean score position
    X_np   = numpy.asarray(model.X_unflattener(model.X_flat))
    scores = X_np @ W
    for i, name in enumerate(point_names):
        ax.text(
            scores[i, ix],
            scores[i, iy],
            f"  {name}",
            fontsize = point_label_fontsize,
            color    = "black",
            va       = "center",
        )

    # Compute per-feature angular uncertainty if requested
    feature_sigmas = None
    if show_uncertainty and cov_eigenvectors is not None:
        feature_sigmas = _estimate_feature_angular_uncertainty(W, cov_eigenvectors, ix, iy, p)

    # Scale arrows to fit plot range
    xlim   = ax.get_xlim()
    ylim   = ax.get_ylim()
    x_span = xlim[1] - xlim[0]
    y_span = ylim[1] - ylim[0]
    scale  = 0.4 * min(x_span, y_span) * arrow_scale

    # Draw loading arrows (+ optional wedges)
    for j, fname in enumerate(feature_names):
        dx = W[j, ix] * scale
        dy = W[j, iy] * scale

        # uncertainty wedge behind the arrow
        if show_uncertainty and feature_sigmas is not None:
            _draw_loading_wedge(
                ax, dx, dy,
                sigma      = feature_sigmas[j],
                n_std      = n_std,
                color      = arrow_color,
                fill_alpha = wedge_fill_alpha,
                line_alpha = wedge_line_alpha,
            )

        # main arrow
        ax.annotate(
            text       = "",
            xy         = (dx, dy),
            xytext     = (0, 0),
            arrowprops = dict(
                arrowstyle = f"->, head_width={arrow_head_width}, head_length={arrow_head_length}",
                color      = arrow_color,
                lw         = arrow_lw,
                shrinkA    = 0,
                shrinkB    = 8,
            ),
            zorder = 5,
        )

        # label with white background to avoid collision
        ax.text(
            dx * 1.2,
            dy * 1.2,
            fname,
            fontsize   = feature_label_fontsize,
            color      = arrow_color,
            ha         = "center",
            va         = "center",
            fontweight = "bold",
            bbox       = dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8),
        )

    return ax