import numpy as np
import scipy.stats as stats
from uadapy import Distribution

class IndependentJoint:
    """
    Joint Distributiuon of independent continuous distributions. 
    This class allows to combine multiple independent distributions into a single multivariate joint distribution.
    Univariate as well as multivariate distributions can be joined. 
    The resulting joint distribution will have a dimensionality equal to the sum of the dimensionalities of the individual distributions.

    Example usage::

        import scipy.stats as stats
        from uadapy.distributions import IndependentJoint
        a = stats.Normal()
        b = stats.t(5)
        j = IndependentJoint([a, b])

    """

    def __init__(self, distributions):
        if not isinstance(distributions, list) or len(distributions) < 2:
            raise ValueError("distributions must be a list of at least 2 Distribution objects or Distribution wrappable objects")
        self.distributions = [d if isinstance(d, Distribution) else Distribution(d) for d in distributions]
        self.dim = sum(d.n_dims for d in self.distributions)
    
    def sample(self, n, seed=None):
        samples = []
        for d in self.distributions:
            s = d.sample(n, seed=seed)
            samples.append(s if len(s.shape) > 1 else s[:,None])
        return np.hstack(samples)
    
    def cov(self):
        covs = [d.cov() for d in self.distributions]
        cov = np.zeros((self.dim, self.dim))
        idx = 0
        for c in covs:
            dim = 1 if np.isscalar(c) else c.shape[0]
            cov[idx:idx+dim, idx:idx+dim] = c
            idx += dim
        return cov
    
    def pdf(self, x):
        idx = 0
        pdfs = np.ones(x.shape[0])
        for d in self.distributions:
            dim = d.n_dims
            x_ = x[:, idx:idx+dim] if len(x.shape) > 1 else x[idx:idx+dim]
            if dim == 1:
                x_ = x_.ravel() # flatten to 1D array for univariate distributions
            if len(x_) == 1:
                x_ = x_.item() # convert to scalar if univariate and single sample
            p = d.pdf(x_)
            pdfs *= p
            idx += dim
        return pdfs
    
    def mean(self):
        m = np.zeros(self.dim)
        idx = 0
        for d in self.distributions:
            dim = d.n_dims
            m[idx:idx+dim] = d.mean()
            idx += dim
        return m
    