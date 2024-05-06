import numpy as np
import scipy as sp
from scipy import stats
from scipy.stats import _multivariate as mv


class distribution:

    def __init__(self, model, name="", dim = 1):
        """
        Creates a distribution, if samples are passed as the first parameter,
        no assumptions about the distribution are made. For the pdf and the sampling,
        a KDE is used. If the name is "Normal", the samples
        are treated as samples of a normal distribution.
        :param model: A scipy.stats distribution or samples
        :param name: The name of the distribution
        :param dim: The dimensionality of the distribution
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
            self.dim = len(self.mean())
        else:
            self.dim = 1
        self.kde = None
        if isinstance(self.model, np.ndarray):
            self.kde = stats.gaussian_kde(self.model.T)

    def sample(self, n: int, random_state: int = None) -> np.ndarray:
        if isinstance(self.model, np.ndarray):
            return self.kde.resample(n, random_state).T
        if hasattr(self.model, 'rvs') and callable(self.model.rvs):
            return self.model.rvs(size=n, random_state=random_state)
        if hasattr(self.model, 'resample') and callable(self.model.resample):
            return self.model.resample(size=n, seed=random_state)

    def pdf(self, x: np.ndarray | float) -> np.ndarray | float:
        if isinstance(self.model, np.ndarray):
            return self.kde.pdf(x.T)
        if not hasattr(self.model, 'pdf'):
            raise AttributeError(f"The model has no pdf.{self.model.__class__.__name__}")
        else:
            return self.model.pdf(x)

    def mean(self) -> np.ndarray | float:
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
        if isinstance(self.model, np.ndarray):
            return stats.skew(self.model)
        if hasattr(self.model, 'stats') and callable(self.model.stats):
            return self.model.stats(moments='s')
        if isinstance(self.model, multivariate_t_frozen):
            return 0

    def kurt(self) -> np.ndarray | float:
        if isinstance(self.model, np.ndarray):
            return stats.kurtosis(self.model)
        if hasattr(self.model, 'stats') and callable(self.model.stats):
            return self.model.stats(moments='k')
        if isinstance(self.model, stats.multivariate_normal):
            return 0
