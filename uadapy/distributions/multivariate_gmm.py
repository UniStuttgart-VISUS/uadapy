import numpy as np
from sklearn.mixture import GaussianMixture


class MultivariateGMM:
    """
    Wrapper around sklearn's GaussianMixture providing a consistent interface
    for use with the Distribution class.
    
    This class delegates to the underlying GaussianMixture model and implements
    additional functionality like pdf evaluation and reproducible sampling.
    
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
    means_ : np.ndarray
        Means of each mixture component, shape (n_components, n_dims).
    covariances_ : np.ndarray
        Covariances of each mixture component, shape (n_components, n_dims, n_dims).
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
        
        # Expose key attributes for direct access
        self.means_ = gmm.means_
        self.covariances_ = gmm.covariances_
        self.weights_ = gmm.weights_
    
    def sample(self, n: int, seed: int = None) -> np.ndarray:
        """
        Generate random samples from the Gaussian Mixture Model.
        
        Parameters
        ----------
        n : int
            Number of samples to generate.
        seed : int, optional
            Random seed for reproducibility. If None, uses the model's random state.
            
        Returns
        -------
        np.ndarray
            Generated samples, shape (n, n_dims).
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
        Evaluate the probability density function at given points.
        
        Parameters
        ----------
        x : np.ndarray
            Points at which to evaluate the pdf.
            Shape should be (n_samples, n_dims) or (n_dims,) for a single point.
            
        Returns
        -------
        np.ndarray or float
            PDF values at the given points.
            Returns a scalar if input was 1D, array otherwise.
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
            Overall mean vector, shape (n_dims,).
        """
        # Weighted mean of all components
        return np.sum(self.weights_[:, None] * self.means_, axis=0)
    
    def cov(self) -> np.ndarray:
        """
        Compute the overall covariance of the Gaussian Mixture Model.
        
        Uses the law of total covariance:
        Cov(X) = E[Cov(X|Z)] + Cov(E[X|Z])
        where Z is the latent component indicator.
        
        Returns
        -------
        np.ndarray
            Overall covariance matrix, shape (n_dims, n_dims).
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