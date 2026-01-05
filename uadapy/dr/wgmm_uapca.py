import numpy as np
from sklearn.mixture import GaussianMixture

from uadapy import Distribution
from uadapy.distributions import MultivariateGMM
from uadapy.dr.uapca import compute_uapca


def wgmm_uapca(distributions: list, weights: np.ndarray = None, n_dims: int = 2) -> list:
    """
    Applies weighted GMM UAPCA to Gaussian Mixture Model distributions
    and returns the projected distributions in lower-dimensional space.

    Parameters
    ----------
    distributions : list
        List of input Distribution objects wrapping MultivariateGMM models.
    weights : np.ndarray, optional
        Array of weights for each distribution. If None, uniform weights are used.
    n_dims : int
        Target dimension. Default is 2.

    Returns
    -------
    list
        List of Distribution objects wrapping projected MultivariateGMM models.
    """
    try:
        # Validate inputs
        for d in distributions:
            if not isinstance(d.model, MultivariateGMM):
                raise ValueError("All distributions must be MultivariateGMM models")
            if d.n_dims < n_dims:
                raise ValueError("All distributions must have dimensionality greater than or equal to n_dims")
        if not (weights is None or len(weights) == len(distributions)):
            raise ValueError("Weights array must be None or have the same length as distributions list")
        
        # Extract overall means and covariances from each GMM
        means = np.array([d.mean() for d in distributions])
        covs = np.array([d.cov() for d in distributions])
        
        # Project distributions
        projected_gmms = transform_wgmm_uapca(distributions, means, covs, weights, n_dims)
        
        # Turn GaussianMixture objects back into Distribution objects wrapping MultivariateGMM
        dist_pca = []
        for projected_gmm in projected_gmms:
            dist_pca.append(Distribution(MultivariateGMM(projected_gmm)))
        
        return dist_pca
        
    except Exception as e:
        raise Exception(f'Something went wrong. Exception: {e}')


# Computing methods

def transform_wgmm_uapca(distributions: list, means: np.ndarray, covs: np.ndarray, 
                          weights: np.ndarray = None, n_dims: int = 2) -> list[GaussianMixture]:
    """
    Projects GMM distributions into a lower-dimensional space.

    Parameters
    ----------
    distributions : list
        List of Distribution objects wrapping MultivariateGMM models.
    means : np.ndarray
        Array of overall mean vectors from each GMM.
    covs : np.ndarray
        Array of overall covariance matrices from each GMM.
    weights : np.ndarray, optional
        Array of weights for each distribution. If None, uniform weights are used.
    n_dims : int
        Target dimension for projection.

    Returns
    -------
    list[GaussianMixture]
        List of projected GaussianMixture objects.
    """
    # Compute projection matrix
    eigvecs, eigvals = compute_uapca(means, covs, weights)
    projmat = eigvecs[:, :n_dims]
    
    # Project each GMM's components
    projected_gmms = []
    for d in distributions:
        gmm = d.model
        
        # Project all components
        projected_means = gmm.means_ @ projmat
        projected_covs = np.array([projmat.T @ cov @ projmat for cov in gmm.covariances_])
        
        # Create new GaussianMixture with projected parameters
        projected_gmm = create_gmm_from_components(projected_means, projected_covs, gmm.weights_)
        projected_gmms.append(projected_gmm)
    
    return projected_gmms


def create_gmm_from_components(means: np.ndarray, covariances: np.ndarray, weights: np.ndarray) -> GaussianMixture:
    """
    Creates a fitted GaussianMixture object from component parameters.

    Parameters
    ----------
    means : np.ndarray
        Array of component means.
    covariances : np.ndarray
        Array of component covariances.
    weights : np.ndarray
        Array of component weights.

    Returns
    -------
    GaussianMixture
        A fitted GaussianMixture object with the given parameters.
    """
    n_components = len(weights)
    n_dims = means.shape[1]
    
    # Create a GaussianMixture object
    gmm = GaussianMixture(n_components=n_components, covariance_type='full')
    
    # Manually set the parameters
    gmm.means_ = means
    gmm.covariances_ = covariances
    gmm.weights_ = weights
    gmm.n_features_in_ = n_dims
    
    # Set other required attributes for a "fitted" GMM
    gmm.precisions_cholesky_ = np.array([
        np.linalg.cholesky(np.linalg.inv(cov)) 
        for cov in gmm.covariances_
    ])
    gmm.converged_ = True
    gmm.n_iter_ = 0
    gmm.lower_bound_ = np.nan
    
    return gmm