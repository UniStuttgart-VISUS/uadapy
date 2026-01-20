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
from scipy.linalg import block_diag
from vipurpca import PCA
from vipurpca.helper_functions import equipotential_standard_normal
import warnings


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


def compute_distribution_eigenvectors(dists, n_components=3):
    """
    Compute eigenvectors of PCA fitted on distributions with uncertainty.

    Parameters
    ----------
    dists : list of Distribution
        List of distribution objects with `.mean()` and `.cov()`.
    n_components : int, default=3
        Number of principal components to retain.

    Returns
    -------
    eigenvectors : ndarray of shape (p, n_components)
        Principal component eigenvectors.
    """
    Y, cov_Y = _prepare_pca_inputs(dists)
    model = _fit_pca_with_uncertainty(Y, cov_Y, n_components)
    return model.eigenvectors

def compute_distribution_trajectories(dists, n_components=2, n_frames=10, seed=55):
    """
    Compute trajectories of distributions under PCA with uncertainty.

    For each distribution, PCA directions are sampled according to the
    propagated covariance of eigenvectors. Trajectories in the principal
    component space are then computed for each sample.

    Parameters
    ----------
    dists : list of Distribution
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

    Y, cov_Y = _prepare_pca_inputs(dists)
    model = _fit_pca_with_uncertainty(Y, cov_Y, n_components)

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
