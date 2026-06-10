import numpy as np


def uapca_revisited(distributions: list, n_dims: int = 2, n_samples: int = 10000, seed: int = None) -> list:
    """
    Applies UAPCA Revisited algorithm to the distributions and returns the projected samples
    for each distribution. It assumes normal distributions. If you apply other distributions
    that provide mean and covariance, these values would be used to approximate a normal distribution.

    Parameters
    ----------
    distributions : list
        List of input distributions
    n_dims : int
        Target dimension. Default is 2.
    n_samples : int
        Number of Monte Carlo samples. Default is 10000.
    seed : int, optional
        Random seed for reproducibility. Default is None.

    Returns
    -------
    list
        List of arrays containing projected samples for each distribution.
        Each array has shape (n_samples, n_dims).
    """

    try:
        if seed is not None:
            np.random.seed(seed)
        means = np.array([d.mean() for d in distributions])
        covs = np.array([d.cov() for d in distributions])
        projected_samples = _sample_and_project(means, covs, n_dims, n_samples)
        return projected_samples
    except Exception as e:
        raise Exception(f'Something went wrong. Did you input normal distributions? Exception:{e}')


# Computing methods

def _sample_and_project(means: np.ndarray, covs: np.ndarray, n_dims: int, n_samples: int) -> list:
    """
    Performs sampling-based projection using multiple PCA realizations.

    Parameters
    ----------
    means : np.ndarray
        Array of mean vectors.
    covs : np.ndarray
        Array of covariance matrices.
    n_dims : int
        Target dimension for projection.
    n_samples : int
        Number of Monte Carlo samples.

    Returns
    -------
    list
        List of arrays containing projected samples for each distribution.
    """
    
    n = means.shape[0]
    d = means.shape[1]
    projected_samples = [np.zeros((n_samples, n_dims)) for _ in range(n)]
    reference_eigvec = None
    
    for k in range(n_samples):
        # Sample one realization from each distribution
        realization = np.array([np.random.multivariate_normal(means[i], covs[i]) for i in range(n)])
        # Compute mean and covariance for this realization
        mean_k = np.mean(realization, axis=0)
        cov_k = _compute_realization_cov(realization, mean_k)
        # Compute PCA projection for this realization
        projmat_k = _compute_projection_matrix(cov_k, n_dims)
        # Align eigenvectors for orientation consistency
        if reference_eigvec is None:
            reference_eigvec = projmat_k.copy()
        else:
            projmat_k = _align_eigenvectors(projmat_k, reference_eigvec)
            reference_eigvec = _update_reference(reference_eigvec, projmat_k, k)
        # Project each point in the realization
        for i in range(n):
            projected_samples[i][k] = (realization[i] - mean_k) @ projmat_k
    
    return projected_samples


def _compute_realization_cov(points: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """
    Computes covariance matrix for a realization.

    Parameters
    ----------
    points : np.ndarray
        Array of sampled points.
    mean : np.ndarray
        Mean of the points.

    Returns
    -------
    np.ndarray
        Covariance matrix.
    """
    
    n = points.shape[0]
    centered = points - mean
    return (centered.T @ centered) / n


def _compute_projection_matrix(cov: np.ndarray, n_dims: int) -> np.ndarray:
    """
    Computes PCA projection matrix from covariance matrix.

    Parameters
    ----------
    cov : np.ndarray
        Covariance matrix.
    n_dims : int
        Target dimension.

    Returns
    -------
    np.ndarray
        Projection matrix with shape (n_features, n_dims).
    """
    
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    projmat = eigenvectors[:, :n_dims]
    return projmat


def _align_eigenvectors(current: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Aligns eigenvectors with reference to ensure consistent orientation.

    Parameters
    ----------
    current : np.ndarray
        Current eigenvectors.
    reference : np.ndarray
        Reference eigenvectors.

    Returns
    -------
    np.ndarray
        Aligned eigenvectors.
    """
    
    aligned = current.copy()
    for i in range(current.shape[1]):
        if np.dot(current[:, i], reference[:, i]) < 0:
            aligned[:, i] = -aligned[:, i]
    return aligned


def _update_reference(reference: np.ndarray, current: np.ndarray, iteration: int) -> np.ndarray:
    """
    Updates reference eigenvectors as running average.

    Parameters
    ----------
    reference : np.ndarray
        Current reference eigenvectors.
    current : np.ndarray
        Newly aligned eigenvectors.
    iteration : int
        Current iteration number.

    Returns
    -------
    np.ndarray
        Updated reference eigenvectors.
    """
    
    weight = iteration / (iteration + 1)
    updated = reference * weight + current * (1 - weight)
    return updated