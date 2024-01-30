import numpy as np
import scipy as sp


class distribution:

    def __init__(self, model, name="", dim = 1):
        """
        Creates a distribution, if samples are passed as the first parameter,
        no assumptions about the distribution are made. If the name is "Normal", the samples
        are treated as samples of a normal distribution
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
            self.model = sp.stats.multivariate_normal(mean, cov)
        elif isinstance(model, np.ndarray):
            ...
        else:
            self.model = model
        mean = self.mean()
        if isinstance(mean, np.ndarray):
            self.dim = len(self.mean())
        else:
            self.dim = 1

    def sample(self, n: int, random_state : int = None) -> np.ndarray:
        if hasattr(self.model, 'rvs') and callable(self.model.rvs):
            return self.model.rvs(size=n, random_state=random_state)
        if hasattr(self.model, 'resample') and callable(self.model.resample):
            return self.model.resample(size=n, seed=random_state)

    def pdf(self, x: np.ndarray | float) -> np.ndarray | float:
        if not hasattr(self.model, 'pdf'):
            print("The model has no pdf.")
        else:
            return self.model.pdf(x)


    def mean(self) -> np.ndarray | float:
        if hasattr(self.model, 'mean'):
            if callable(self.model.mean):
                return self.model.mean()
            return self.model.mean
        else:
            print("Mean not implemented yet!")

    def cov(self) -> np.ndarray | float:
        if hasattr(self.model, 'cov'):
            return self.model.cov
        if hasattr(self.model, 'covariance'):
            return self.model.covariance
        if hasattr(self.model, 'var') and callable(self.model.var):
            return self.model.var()
        print("Covariance not implemented yet!")


    def skew(self) -> np.ndarray | float:
        if hasattr(self.model, 'stats') and callable(self.model.stats):
            return self.model.stats(moments='s')

    def kurt(self) -> np.ndarray | float:
        if hasattr(self.model, 'stats') and callable(self.model.stats):
            return self.model.stats(moments='k')