import numpy as np
from uadapy import Distribution
from scipy.stats import multivariate_normal

def uapca(distributions, n_dims: int = 2, weights: np.ndarray = None):
    """
    Applies UAPCA algorithm to the distribution and returns the distribution
    in lower-dimensional space. It assumes a normal distribution. If you apply
    other distributions that provide mean and covariance, these values would be used
    to approximate a normal distribution.

    Parameters
    ----------
    distributions : list
        List of input distributions
    n_dims : int
        Target dimension. Default is 2.
    weights : np.ndarray, optional
        Array of weights for each distribution. If None, uniform weights are used.

    Returns
    -------
    list
        List of distributions in low-dimensional space.
    """

    try:
        means = np.array([d.mean() for d in distributions])
        covs = np.array([d.cov() for d in distributions])
        means_pca, covs_pca = transform_uapca(means, covs, n_dims, weights)
        dist_pca = []
        for (m, c) in zip(means_pca, covs_pca):
            dist_pca.append(Distribution(multivariate_normal(m, c)))
        return dist_pca
    except Exception as e:
        raise Exception(f'Something went wrong. Did you input normal distributions? Exception:{e}')



# Computing methods

def compute_ua_cov(means: np.ndarray, covs: np.ndarray, weights: np.ndarray = None) -> np.ndarray:
    """
    Computes the weighted uncertainty-aware covariance matrix. If weights is None,
    uniform weights are assumed.

    Parameters
    ----------
    means : np.ndarray
        Array of mean vectors.
    covs : np.ndarray
        Array of covariance matrices.
    weights : np.ndarray, optional
        Array of weights of shape (n,).

    Returns
    -------
    np.ndarray
        Weighted uncertainty-aware covariance matrix.
    """

    n = means.shape[0]
    d = means.shape[1]
    # Set uniform weights if not provided
    if weights is None:
        weights = np.ones(n) / n
    else:
        weights = np.array(weights)
        weights = weights / weights.sum()
    # empirical mean
    mu = np.sum(weights[:, None] * means, axis=0)
    # centering matrix
    centering = np.outer(mu, mu)
    # average covariance matrix
    avg_cov = np.sum(weights[:, None, None] * covs, axis=0)
    # sample covariance
    sample_cov = np.sum(
        weights[:, None, None] * np.array([np.outer(means[i], means[i]) for i in range(n)]),
        axis=0
    )
    # final uncertainty aware covariance matrix
    return sample_cov + avg_cov - centering


def compute_uapca(means: np.ndarray, covs: np.ndarray, weights: np.ndarray = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the principal components for uncertainty-aware PCA.

    Parameters
    ----------
    means : np.ndarray
        Array of mean vectors.
    covs : np.ndarray
        Array of covariance matrices.
    weights : np.ndarray, optional
        Array of weights for each distribution.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Eigenvectors and eigenvalues.
    """

    cov = compute_ua_cov(means, covs, weights)
    u,s,vh = np.linalg.svd(cov, full_matrices=True)
    return u, s


def transform_uapca(means, covs, dims: int=2, weights: np.ndarray = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Projects mean and covariance matrices into a lower-dimensional space.

    Parameters
    ----------
    means : np.ndarray
        Array of mean vectors.
    covs : np.ndarray
        Array of covariance matrices.
    dims : int
        Target dimension for projection.
    weights : np.ndarray, optional
        Array of weights for each distribution.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Projected mean vectors and covariance matrices.
    """

    n = means.shape[0]
    d = means.shape[1]
    eigvecs, eigvals = compute_uapca(means, covs, weights)
    projmat = eigvecs[:, :dims]
    projected_means = means @ projmat
    projected_covs = np.array([projmat.T @ covs[i] @ projmat for i in range(n)])
    return projected_means, projected_covs