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
        Each COLUMN is one principal component direction.
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
