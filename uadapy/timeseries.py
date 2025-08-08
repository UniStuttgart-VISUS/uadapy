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
    timesteps: int
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
            The time steps of the time series.
        name: str, optional
            The name of the distribution
        n_dims: int, optional
            The dimensionality of the distribution (default is 1)
        """
        self.distribution = distribution.Distribution(model, name, n_dims)
        if np.any(timesteps):
            self.timesteps = timesteps
        else:
            self.timesteps = np.arange(0, len(self.distribution.mean()))

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
    distributions : list[Distribution] or list[Timeseries]
        List of individual distributions or timeseries.
    n_distributions : int
        Number of distributions or timeseries.
    covariance_matrix : np.ndarray
        Pairwise covariance matrix of the distributions.
    """

    def __init__(self, distributions: list[distribution.Distribution], covariance_matrix=None):
        """
        Initializes the CorrelatedDistributions object.

        Parameters
        ----------
        distributions : list[Distribution]
            A list of Distribution objects.
        covariance_matrix : np.ndarray
            Pairwise covariance matrix of the distributions.
        """
        self.distributions = distributions
        self.n_distributions = len(distributions)
        self._check_covariance_matrix_consistency(covariance_matrix)
        self.covariance_matrix = covariance_matrix

    def _check_covariance_matrix_consistency(self, covariance_matrix) -> bool:
        """
        Checks if the provided covariance matrix matches the covariances computed from the distributions.

        Returns
        -------
        bool
            True if the covariance matrix matches, False otherwise.
        """

        if covariance_matrix is None:
            raise ValueError("Covariance matrix must be provided to validate the distributions.")

        for i in range(self.n_distributions):
            computed_variance = self.distributions[i].cov()
            if not all(np.allclose(arr1, arr2, atol=1e-8) for arr1, arr2 in zip(computed_variance, covariance_matrix[i][i])):
                raise ValueError(
                    f"Variance of distribution {i} does not match the provided covariance matrix."
                )

        return True

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
        Returns the covariance matrix between two distributions.

        Parameters
        ----------
        dim_i : int
            Index of the first distribution.
        dim_j : int
            Index of the second distribution.

        Returns
        -------
        np.ndarray
            Covariance matrix between i-th and j-th distribution.
        """
        return self.covariance_matrix[dim_i][dim_j]

    def sample(self, n_samples: int, seed: int = None) -> np.ndarray:
        """
        Samples from the joint distribution of all correlated distributions in block structure.

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
        means = np.concatenate([dist.mean() for dist in self.distributions])
        block_rows = [np.concatenate(row, axis=1) for row in self.covariance_matrix]
        covariance_matrix = np.concatenate(block_rows, axis=0)
        joint_samples = np.random.multivariate_normal(means, covariance_matrix, size=n_samples)

        return joint_samples