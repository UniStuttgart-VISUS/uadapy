import distribution

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