import scipy.stats
import numpy as np

class DiracDelta:

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
        return (self.tol*self.tol)/12 # variance of uniform distribution
    
    def pdf(self, x, tol=None):
        if tol is None:
            tol = self.tol
        if tol == 0.0:
            values = np.where(x==self.mean, np.inf, 0.0)
            return np.asscalar(values) if np.isscalar(values) else values
        else:
            return stats.uniform.pdf(x, loc=self.mean-self.tol/2, scale=self.tol)

    def cdf(self, x, tol=None):
        if tol is None:
            tol = self.tol
        if tol == 0.0:
            values = np.where(x < self.mean, 0.0, 1.0)
            return np.asscalar(values) if np.isscalar(values) else values
        else:
            return stats.uniform.cdf(x, loc=self.mean-self.tol/2, scale=self.tol)

