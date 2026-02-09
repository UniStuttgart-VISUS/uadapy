import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import gaussian_kde


class MultivariateGMM:
    """
    Wrapper around sklearn's GaussianMixture providing a consistent interface
    for use with the Distribution class.

    This class supports all sklearn covariance types of sklearn's GaussianMixture.
    Covariances are always converted to full format if necessary for consistent handling.

    Parameters
    ----------
    gmm : GaussianMixture
        A fitted sklearn GaussianMixture model.
        
    Attributes
    ----------
    gmm : GaussianMixture
        The underlying fitted GaussianMixture model.
    n_components : int
        Number of mixture components.
    n_dims : int
        Dimensionality of the distribution.
    covariance_type : str
        Type of covariance parameters: "full", "tied", "diag", or "spherical".
    means_ : np.ndarray
        Means of each mixture component, shape (n_components, n_dims).
    covariances_ : np.ndarray
        Covariances of each mixture component in "full" format, shape (n_components, n_dims, n_dims).
    weights_ : np.ndarray
        Weights of each mixture component, shape (n_components,).
    """
    
    def __init__(self, gmm: GaussianMixture):
        """
        Initialize the MultivariateGMM wrapper.
        
        Parameters
        ----------
        gmm : GaussianMixture
            A fitted sklearn GaussianMixture model.
            
        Raises
        ------
        ValueError
            If the GaussianMixture model has not been fitted yet.
        """
        if not hasattr(gmm, 'means_'):
            raise ValueError("GaussianMixture model must be fitted before wrapping")
        
        self.gmm = gmm
        self.n_components = gmm.n_components
        self.n_dims = gmm.means_.shape[1]
        self.covariance_type = gmm.covariance_type
        
        # Expose key attributes for direct access
        self.means_ = gmm.means_
        self.covariances_ = self._convert_to_full_covariances(gmm)
        self.weights_ = gmm.weights_
    
    def sample(self, n: int, seed: int = None) -> np.ndarray:
        """
        Creates samples from the Gaussian Mixture Model.

        Parameters
        ----------
        n : int
            Number of samples.
        seed : int, optional
            Seed for the random number generator for reproducibility, default is None.

        Returns
        -------
        np.ndarray
            Samples of the Gaussian Mixture Model.
        """
        if seed is not None:
            # Create temporary random state without modifying the model
            rng = np.random.RandomState(seed)
            
            # Sample component indices according to mixture weights
            component_indices = rng.choice(
                self.n_components,
                size=n,
                p=self.weights_
            )
            
            # Sample from each chosen component
            samples = np.zeros((n, self.n_dims))
            for i in range(n):
                comp_idx = component_indices[i]
                samples[i] = rng.multivariate_normal(
                    self.means_[comp_idx],
                    self.covariances_[comp_idx]
                )
            return samples
        else:
            # Use the model's built-in sampling
            return self.gmm.sample(n)[0]
    
    def pdf(self, x: np.ndarray) -> np.ndarray | float:
        """
        Computes the probability density function.

        Parameters
        ----------
        x : np.ndarray or float
            The position(s) where the pdf should be evaluated.

        Returns
        -------
        np.ndarray or float
            Probability values of the distribution at the given sample point(s).
        """
        # Ensure x is 2D for score_samples
        x = np.atleast_2d(x)

        # Compute log-likelihood and exponentiate to get pdf
        log_likelihood = self.gmm.score_samples(x)
        result = np.exp(log_likelihood)
        
        # Return scalar if input was a single point
        return result[0] if result.shape[0] == 1 else result
    
    def mean(self) -> np.ndarray:
        """
        Compute the overall mean of the Gaussian Mixture Model.
        
        Returns
        -------
        np.ndarray
            Overall mean vector.
        """
        # Weighted mean of all components
        return np.sum(self.weights_[:, None] * self.means_, axis=0)
    
    def cov(self) -> np.ndarray:
        """
        Compute the overall covariance of the Gaussian Mixture Model.
        
        Returns
        -------
        np.ndarray
            Overall covariance matrix.
        """
        overall_mean = self.mean()
        
        # Within-component covariance (weighted average)
        weighted_cov = np.sum(
            self.weights_[:, None, None] * self.covariances_,
            axis=0
        )
        
        # Between-component covariance
        mean_diff = self.means_ - overall_mean
        weighted_outer = np.sum(
            self.weights_[:, None, None] * (mean_diff[:, :, None] @ mean_diff[:, None, :]),
            axis=0
        )
        
        return weighted_cov + weighted_outer
    
    def _convert_to_full_covariances(self, gmm: GaussianMixture) -> np.ndarray:
        """
        Convert sklearn's covariances to full format regardless of original type.
        
        Parameters
        ----------
        gmm : GaussianMixture
            Fitted GaussianMixture model.
            
        Returns
        -------
        np.ndarray
            Covariances in full format, shape (n_components, n_dims, n_dims).
        """
        cov_type = gmm.covariance_type
        
        if cov_type == "full":
            return gmm.covariances_
        elif cov_type == "tied":
            return np.array([gmm.covariances_] * self.n_components)
        elif cov_type == "diag":
            full_covs = np.zeros((self.n_components, self.n_dims, self.n_dims))
            for i in range(self.n_components):
                full_covs[i] = np.diag(gmm.covariances_[i])
            return full_covs
        elif cov_type == "spherical":
            full_covs = np.zeros((self.n_components, self.n_dims, self.n_dims))
            for i in range(self.n_components):
                full_covs[i] = np.eye(self.n_dims) * gmm.covariances_[i]
            return full_covs
        else:
            raise ValueError(f"Unknown covariance type: {cov_type}")


def gmm_from_kde(kde: gaussian_kde):
    """
    Converts a scipy gaussian_kde object to MultivariateGMM.
    KDE consists of gaussians at every point of the dataset it was estimated from.
    Each gaussian shares the same covariance matrix which makes it easy to convert
    to GMM.
    
    Returns
    -------
    MultivariateGMM
        The mGMM equivalent to specified KDE.
    """
    centers = kde.dataset.T
    n_components = len(centers)

    # Create a GaussianMixture object
    gmm = GaussianMixture(n_components=n_components, covariance_type="full")

    # Manually set the parameters
    gmm.weights_ = kde.weights
    gmm.means_ = centers
    gmm.covariances_ = np.array([kde.covariance for _ in range(n_components)])

    return MultivariateGMM(gmm)
