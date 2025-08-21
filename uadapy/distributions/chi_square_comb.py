from uadapy import Distribution
import numpy as np

class ChiSquareComb:
    """
    The ChiSquareComb class provides a consistent interface to a combination of chi-square distribution.
    
    Currently, we only support the distribution created by summing up two squares
    of complex normal distributions.

    Attributes
    ----------
    mu_complex: np.ndarray
        The complex mean of the distribution before squaring
    covariance: np.ndarray
        The complex covariance matrix of the distribution before squaring
    pseudo_covariance: np.ndarray
        The complex pseudo-covariance matrix of the distribution before squaring
    mu_real: np.ndarray
        The mean of the distribution where the first half corresponds to the real part
        and the second half to the imaginary part
    self.cov_real: np.ndarray
        The covariance matrix of the distribution where the first half corresponds to the real part
        and the second half to the imaginary part, it also contains the cross-correlation terms

    """
    def __init__(self, mean: np.ndarray, cov: np.ndarray, pseudo_cov: np.ndarray):
        """
        Creates a distribution based on the descriptors in the complex space.

        Parameters
        ----------
        mean: np.ndarray
            The complex mean of the distribution
        cov: np.ndarray
            The complex covariance matrix of the distribution
        pseudo_cov: np.ndarray
            The complex pseudo-covariance matrix of the distribution
        """
        self.mu_complex = mean
        self.covariance = cov
        self.pseudo_covariance = pseudo_cov
        self.mu_real, self.cov_real = self._complex_to_real()

    def resample(self, size: int, seed: int = 0) -> np.ndarray:
        """
        Resamples the distribution.

        Parameters
        ----------
        size: int
            The number of samples to be drawn
        seed: int, optional
            The seed for the random number generator, default is 0

        Returns
        -------
        np.ndarray
            The samples
        """
        # Sample from (real) multivariate normal and square
        samples = np.random.multivariate_normal(self.mu_real, self.cov_real, size)
        N = int(len(self.mu_complex)/2)
        print(N)
        return samples[:,:N]**2 + samples[:,N:]**2
    
    def mean(self) -> np.ndarray:
        """
        Returns the mean of the distribution.

        Returns
        -------
        np.ndarray
            The mean of the distribution
        """
        N = len(self.mu_complex)
        return np.diag(self.cov_real)[:int(N/2)]+np.diag(self.cov_real)[int(N/2):]+self.mu_real[:int(N/2)]+self.mu_real[int(N/2):]

    def cov(self) -> np.ndarray:
        """
        Returns the covariance matrix of the distribution.

        Returns
        -------
        np.ndarray
            The covariance matrix of the distribution
        """
        N = len(self.mu_complex)
        cov1 = 2*self.cov_real[:int(N/2),:int(N/2)]**2+4*np.outer(self.mu_real[:int(N/2)],self.mu_real[:int(N/2)])*self.cov_real[:int(N/2),:int(N/2)]
        cov2 = 2*self.cov_real[:int(N/2),int(N/2):]**2+4*np.outer(self.mu_real[:int(N/2)],self.mu_real[int(N/2):])*self.cov_real[:int(N/2),int(N/2):]
        cov3 = 2*self.cov_real[int(N/2):,:int(N/2)]**2+4*np.outer(self.mu_real[int(N/2):],self.mu_real[:int(N/2)])*self.cov_real[int(N/2):,:int(N/2)]
        cov4 = 2*self.cov_real[int(N/2):,int(N/2):]**2+4*np.outer(self.mu_real[int(N/2):],self.mu_real[int(N/2):])*self.cov_real[int(N/2):,int(N/2):]
        return cov1 + cov2 + cov3 + cov4

    def _complex_to_real(self):
        """
        Create a real normal distribution where the first part represents the real part
        and the second part the imaginary part of the complex distribution.

        Returns
        -------
        np.ndarray
            The mean of the real normal distribution
        np.ndarray
            The covariance matrix of the real normal distribution
        """
        N = len(self.mu_complex)
        mu_X = np.real(self.mu_complex)
        mu_X[int(N/2):] = np.imag(self.mu_complex)[:int(N/2)]
        Sigma = 0.5*np.real(self.covariance+self.pseudo_covariance)
        Sigma[int(N/2):, int(N/2):] = 0.5*np.real(self.covariance-self.pseudo_covariance)[:int(N/2), :int(N/2)]
        Sigma[int(N/2):, :int(N/2)] = 0.5*np.imag(self.covariance+self.pseudo_covariance)[:int(N/2), :int(N/2)]
        Sigma[:int(N/2), int(N/2):] = 0.5*np.imag(self.pseudo_covariance-self.covariance)[:int(N/2), :int(N/2)]
        return mu_X, Sigma