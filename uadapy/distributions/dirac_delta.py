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
        """
        Creates a Dirac Delta distribution with the specified mean and tolerance.

        Parameters
        ----------
        mean : float
            The mean of the distribution (the location of the peak).
        tol : float, optional
            The tolerance for the distribution. If tol is 0, the distribution is a true Dirac Delta. 
            If tol > 0, the distribution is a uniform distribution centered at mean with width tol. Default is 0.0.
        """
        self.mean = mean
        self.tol = tol
    
    def sample(self, n, seed=None, tol=None):
        """
        Draws n samples from the Dirac Delta distribution.
        If tol is None, the tolerance specified in the constructor is used. If tol is 0, all samples will be equal to the mean.

        Parameters
        ----------
        n : int
            Number of samples to draw.
        seed : int or np.random.Generator, optional
            Random seed or random number generator for reproducibility. Default is None.
        tol : float, optional
            The tolerance for the distribution. If tol is 0, all samples will be equal to the mean. Default is None.

        Returns
        -------
        samples : ndarray, shape (n,)
            The drawn samples from the distribution.
        """
        if tol is None:
            tol = self.tol
        if tol == 0.0:
            return np.ones(n) * self.mean
        else:
            return stats.uniform.rvs(loc=self.mean - self.tol/2, scale=self.tol, size=n, random_state=seed)
    
    def var(self):
        """
        Returns the variance of the distribution.

        Returns
        -------
        variance : float
            The variance of the distribution. For a true Dirac Delta (tol=0), the variance is 0. For a uniform distribution (tol>0), the variance is (tol^2)/12.
        """
        return (self.tol*self.tol)/12.0 # variance of uniform distribution
    
    def pdf(self, x, tol=None):
        """
        Computes the probability density function of the Dirac Delta distribution at the given points x.

        Parameters
        ----------
        x : array-like, shape (n_samples,) or scalar
            Points at which to evaluate the PDF.
        tol : float, optional
            The tolerance for the distribution. If tol is None, the tolerance specified in the constructor is used. Default is None.
            When tol is 0, the PDF is infinite at the mean and 0 elsewhere. When tol > 0, the PDF is uniform.
        
        Returns
        -------
        pdfs : ndarray or scalar
            Probability density values at the given points x. 
        """
        if tol is None:
            tol = self.tol
        if tol == 0.0:
            values = np.where(x==self.mean, np.inf, 0.0)
            return np.asarray(values).item() if np.isscalar(values) or len(values) == 1  else values
        else:
            return stats.uniform.pdf(x, loc=self.mean-self.tol/2, scale=self.tol)

    def cdf(self, x, tol=None):
        """
        Computes the cumulative distribution function of the distribution at the given points x.
        
        Parameters
        ----------
        x : array-like, shape (n_samples,) or scalar
            Points at which to evaluate the CDF.
        tol : float, optional
            The tolerance for the distribution. If tol is None, the tolerance specified in the constructor is used. Default is None.
            When tol is 0, the CDF is 0 for x < mean and 1 for x >= mean. When tol > 0, the CDF is that of the corresponding uniform distribution.
        """
        if tol is None:
            tol = self.tol
        if tol == 0.0:
            values = np.where(x < self.mean, 0.0, 1.0)
            return np.asarray(values).item() if np.isscalar(values) or len(values) == 1  else values
        else:
            return stats.uniform.cdf(x, loc=self.mean-self.tol/2, scale=self.tol)

