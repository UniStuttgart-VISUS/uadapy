from uadapy import distribution
import numpy as np

class TimeSeries:
    """
    The TimeSeries class provides a consistent interface to model an uncertain, univariate time series.
    It strongly builds on the Distribution class.
    It provides wrapper functions to some of the important functions of the Distribution class
    and adds additional convenience functions that are commonly used in time series.
    
    Attributes
    ----------
    distribution: Distribution
        The underlying distribution of the time series
    timesteps: np.ndarray
        The time steps of the time series
    """

    def __init__(self, model, timesteps=None, name="", n_dims=1):
        """
        Creates a time series object. 
        The underlying distribution is created in agreement with the Distribution class.
        
        Parameters
        ----------
        model: 
            A scipy.stats distribution or samples
        timesteps: np.ndarray, optional
            The time steps of the time series. If none is provided, the time steps are assumed to be [0, 1, 2, ...].
        name: str, optional
            The name of the distribution
        n_dims: int, optional
            The dimensionality of the distribution (default is 1)
        """
        self.distribution = distribution.Distribution(model, name, n_dims)
        self.timesteps = timesteps

    def sample(self, n: int, seed: int = None) -> np.ndarray:
        """
        Creates samples from the time series (one specific instance).

        Parameters
        ----------
        n : int
            Number of samples (time series).
        seed : int, optional
            Seed for the random number generator for reproducibility, default is None.

        Returns
        -------
        np.ndarray
            Time series instances that represent samples of the distribution.
        """
        return self.distribution.sample(n, seed)
    
    def pdf(self, x: np.ndarray | float) -> np.ndarray | float:
        """
        Computes the probability density function based on the whole
        probability distribution of the time series.

        Parameters
        ----------
        x : np.ndarray or float
            The position where the pdf should be evaluated.

        Returns
        -------
        np.ndarray or float
            Probability values of the distribution at the given sample points.
        """
        return self.distribution.pdf(x)
    
    def mean(self) -> np.ndarray | float:
        """
        Expected value of the time series.

        Returns
        -------
        np.ndarray or float
            Expected value of the time series.
        """
        return self.distribution.mean()

    def cov(self) -> np.ndarray | float:
        """
        Covariance of the time series.

        Returns
        -------
        np.ndarray or float
            Covariance of all time series points.
        """
        return self.distribution.cov()
    
    def variance(self) -> np.ndarray | float:
        """
        Variance of the time series.

        Returns
        -------
        np.ndarray or float
            Variance of the time series.
        """
        return self.distribution.cov().diag()

class CorrelatedDistributions:
    """
    Class for managing and analyzing correlated distributions or time series.

    Attributes
    ----------
    distributions : list[Distribution]
        List of individual distributions.
    covariance_matrix : np.ndarray
        Pairwise covariance matrix of the distributions.
    """

    def __init__(self, distributions: list[distribution.Distribution]):
        """
        Initializes the CorrelatedDistributions object.

        Parameters
        ----------
        distributions : list[Distribution]
            A list of Distribution objects.
        """
        self.distributions = distributions
        self.n_distributions = len(distributions)
        self.covariance_matrix = self._compute_covariance_matrix()

    def _compute_covariance_matrix(self) -> np.ndarray:
        """
        Computes the pairwise covariance matrix for the distributions.

        Returns
        -------
        np.ndarray
            Pairwise covariance matrix.
        """
        n = self.n_distributions
        cov_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    cov_matrix[i, j] = self.distributions[i].cov()
                else:
                    cov_matrix[i, j] = cov_matrix[j, i] = self._cross_covariance(i, j)
        
        return cov_matrix

    def _cross_covariance(self, dim_i: int, dim_j: int) -> float:
        """
        Computes the cross-covariance between two distributions.
        
        Parameters
        ----------
        dim_i : int
            Index of the first distribution.
        dim_j : int
            Index of the second distribution.
        
        Returns
        -------
        float
            Cross-covariance between the two distributions.
        """

        cov_i = self.distributions[dim_i].cov()
        cov_j = self.distributions[dim_j].cov()
        cross_cov = np.sqrt(cov_i * cov_j)

        return cross_cov

    def mean(self, dim_i: int) -> float:
        """
        Returns the mean of the i-th distribution.

        Parameters
        ----------
        dim_i : int
            Index of the distribution.

        Returns
        -------
        float
            Mean of the i-th distribution.
        """
        return self.distributions[dim_i].mean()

    def cov(self, dim_i: int, dim_j: int) -> np.ndarray | float:
        """
        Returns the covariance or cross-covariance between two distributions.

        Parameters
        ----------
        dim_i : int
            Index of the first distribution.
        dim_j : int
            Index of the second distribution.

        Returns
        -------
        np.ndarray | float
            Covariance matrix if i = j, otherwise cross-covariance value.
        """
        return self.covariance_matrix[dim_i, dim_j]

    def sample(self, n_samples: int, seed: int = None) -> np.ndarray:
        """
        Samples from the joint distribution of all correlated distributions.

        Parameters
        ----------
        n_samples : int
            Number of samples to draw.
        seed : int, optional
            Seed for random number generation.

        Returns
        -------
        np.ndarray
            Samples from the joint distribution.
        """
        np.random.seed(seed)
        means = [dist.mean() for dist in self.distributions]
        joint_samples = np.random.multivariate_normal(means, self.covariance_matrix, size=n_samples)
        return joint_samples