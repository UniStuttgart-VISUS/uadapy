import numpy as np
import scipy as sp
from scipy import stats
from scipy.stats import _multivariate as mv


class Distribution:
    """
    The Distribution class provides a consistent interface to a variety of distributions.
    
    Attributes
    ----------
    model
        The underlying concrete distribution model, a `scipy.stats` distribution object or an array of samples
    name : str 
        Name of the distribution type, e.g. 'Normal'
    n_dims : int
        Dimensionality of the distribution
    """

    def __init__(self, model, name="", n_dims=1):
        """
        Creates a distribution, if samples are passed as the first parameter,
        no assumptions about the distribution are made. For the pdf and the sampling,
        a KDE is used. If the name is "Normal", the samples
        are treated as samples of a normal distribution.

        Parameters
        ----------
        model: 
            A scipy.stats distribution or samples
        name: str, optional
            The name of the distribution
        n_dims: int, optional
            The dimensionality of the distribution (default is 1)
        """
        if name:
            self.name = name
        else:
            self.name = model.__class__.__name__
        if isinstance(model, np.ndarray) and name == "Normal":
            mean = np.mean(model, axis=0)
            cov = np.cov(model, rowvar=False)
            self.model = stats.multivariate_normal(mean, cov)
        else:
            self.model = model
        mean = self.mean()
        if isinstance(mean, np.ndarray):
            self.n_dims = len(self.mean())
        else:
            self.n_dims = 1
        self.kde = None
        if isinstance(self.model, np.ndarray):
            self.kde = stats.gaussian_kde(self.model.T)

    def sample(self, n: int, seed: int = None) -> np.ndarray:
        """
        Creates samples from the distribution.

        Parameters
        ----------
        n : int
            Number of samples.
        seed : int, optional
            Seed for the random number generator for reproducibility, default is None.

        Returns
        -------
        np.ndarray
            Samples of the distribution.
        """
        if isinstance(self.model, np.ndarray):
            return self.kde.resample(n, seed).T
        if hasattr(self.model, 'sample') and callable(self.model.sample):
            return self.model.sample(n, seed=seed)
        if hasattr(self.model, 'rvs') and callable(self.model.rvs):
            return self.model.rvs(size=n, random_state=seed)
        if hasattr(self.model, 'resample') and callable(self.model.resample):
            return self.model.resample(size=n, seed=seed)

    def pdf(self, x: np.ndarray | float) -> np.ndarray | float:
        """
        Computes the probability density function.

        Parameters
        ----------
        x : np.ndarray or float
            The position where the pdf should be evaluated.

        Returns
        -------
        np.ndarray or float
            Probability values of the distribution at the given sample points.
        """
        if isinstance(self.model, np.ndarray):
            return self.kde.pdf(x.T)
        if not hasattr(self.model, 'pdf'):
            raise AttributeError(f"The model has no pdf. {self.model.__class__.__name__}")
        else:
            return self.model.pdf(x)
    
    def cdf(self, x: np.ndarray | float) -> np.ndarray | float:
        """
        Computes the cumulative density function.

        Parameters
        ----------
        x : np.ndarray or float
            The position where the cdf should be evaluated.

        Returns
        -------
        np.ndarray or float
            Cumulative probability values of the distribution at the given sample points.
        """
        if isinstance(self.model, np.ndarray):
            raise AttributeError("CDF not implemented for sample-based distributions.")
        if not hasattr(self.model, 'cdf'):
            raise AttributeError(f"The model has no cdf. {self.model.__class__.__name__}")
        else:
            return self.model.cdf(x)

    def mean(self) -> np.ndarray | float:
        """
        Expected value of the distribution.

        Returns
        -------
        np.ndarray or float
            Expected value of the distribution.
        """
        if isinstance(self.model, np.ndarray):
            return np.mean(self.model, axis=0)
        if hasattr(self.model, 'mean'):
            if callable(self.model.mean):
                return self.model.mean()
            return self.model.mean
        if hasattr(self.model, 'loc'):
            return self.model.loc
        if hasattr(self.model, 'mu'):
            return self.model.mu
        else:
           raise AttributeError(f"Mean not implemented yet! {self.model.__class__.__name__}")

    def cov(self) -> np.ndarray | float:
        """
        Covariance of the distribution.

        Returns
        -------
        np.ndarray or float
            Covariance of the distribution.
        """
        if isinstance(self.model, np.ndarray):
            return np.cov(self.model.T)
        if hasattr(self.model, 'cov'):
            if callable(self.model.cov):
                return self.model.cov()
            return self.model.cov
        if hasattr(self.model, 'covariance'):
            if callable(self.model.covariance):
                return self.model.covariance
            return self.model.covariance
        if hasattr(self.model, 'var'):
            if callable(self.model.var):
                return self.model.var()
            return self.model.var
        if isinstance(self.model, mv.multivariate_t_frozen):
            return self.model.shape * (self.model.df / (self.model.df - 2))
        raise AttributeError(f"Covariance not implemented yet! {self.model.__class__.__name__}")


    def skew(self) -> np.ndarray | float:
        """
        Skewness of the distribution.

        Returns
        -------
        np.ndarray or float
            Skewness of the distribution.
        """
        if isinstance(self.model, np.ndarray):
            return stats.skew(self.model)
        if hasattr(self.model, 'stats') and callable(self.model.stats):
            return self.model.stats(moments='s')
        if isinstance(self.model, mv.multivariate_t_frozen):
            return 0

    def kurt(self) -> np.ndarray | float:
        """
        Kurtosis of the distribution.

        Returns
        -------
        np.ndarray or float
            Kurtosis of the distribution.
        """
        if isinstance(self.model, np.ndarray):
            return stats.kurtosis(self.model)
        if hasattr(self.model, 'stats') and callable(self.model.stats):
            return self.model.stats(moments='k')
        if isinstance(self.model, stats.multivariate_normal):
            return 0
