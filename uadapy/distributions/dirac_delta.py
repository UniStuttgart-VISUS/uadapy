import scipy.stats as stats
import numpy as np

class DiracDelta:
    """
    Dirac Delta distribution class.
    To actually be able to work with this distribution, a very small tolerance can be specified and the distribution will
    then mimick a tiny uniform distribution.
    This class is intended to be used when a variable has no uncertainty but the input needs to be specified in terms of a distribution.

    Attributes
    ----------
    mean : float
        The mean of the distribution (the location of the peak).
    tol : float
        The tolerance for the distribution. If tol is 0, the distribution is a true Dirac Delta. 
        If tol > 0, the distribution is a uniform distribution centered at mean with width tol.
    """

    def __init__(self, mean, tol=0.0):
        self.mean = mean
        self.tol = tol
    
    def sample(self, n, seed=None, tol=None):
        if tol is None:
            tol = self.tol
        if tol == 0.0:
            return np.ones(n) * self.mean
        else:
            return stats.uniform.rvs(loc=self.mean - self.tol/2, scale=self.tol, size=n, random_state=seed)
    
    def var(self):
        return (self.tol*self.tol)/12.0 # variance of uniform distribution
    
    def pdf(self, x, tol=None):
        if tol is None:
            tol = self.tol
        if tol == 0.0:
            values = np.where(x==self.mean, np.inf, 0.0)
            return np.asarray(values).item() if np.isscalar(values) or len(values) == 1  else values
        else:
            return stats.uniform.pdf(x, loc=self.mean-self.tol/2, scale=self.tol)

    def cdf(self, x, tol=None):
        if tol is None:
            tol = self.tol
        if tol == 0.0:
            values = np.where(x < self.mean, 0.0, 1.0)
            return np.asarray(values).item() if np.isscalar(values) or len(values) == 1  else values
        else:
            return stats.uniform.cdf(x, loc=self.mean-self.tol/2, scale=self.tol)

