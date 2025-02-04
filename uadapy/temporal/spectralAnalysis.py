from uadapy import TimeSeries
import numpy as np
import math
import uadapy.distributions.chi_square_comb as chi_square_comb

def _fourier(i, j, N):
    """
    Returns the Fourier basis function for the i-th and j-th frequencies.
    
    Parameters
    ----------
    i : int
        Index of the first frequency.
    j : int
        Index of the second frequency.
    N : int
        Number of samples.

    Returns
    -------
    complex
        Fourier basis function.
    """
    return np.exp(-2j*math.pi*i*j/N)

def _ua_fourier_transform(timeseries: TimeSeries) -> np.ndarray:
    """
    Computes the Fourier basis functions for the given time series.

    Parameters
    ----------
    timeseries : TimeSeries
        Time series data.

    Returns
    -------
    np.ndarray
        Fourier-transformend (complex) mean
    np.ndarray
        Fourier-transformend (complex) covariance matrix
    np.ndarray
        Fourier-transformend (complex) pseudo-covariance matrix
    """
    N = len(timeseries.mean())
    W = 1/np.sqrt(N)*np.array([[_fourier(i, j, N) for j in range(N)] for i in range(N)], dtype=complex)
    fftMu = np.dot(W, timeseries.mean())
    fftGamma = np.dot(W, np.dot(timeseries.cov(), W.conj().T))
    fftC = np.dot(W, np.dot(timeseries.cov(), W.T))
    return fftMu, fftGamma, fftC

def ua_fourier_spectrum(timeseries: TimeSeries) -> np.ndarray:
    """
    Computes the Fourier spectrum for the given time series.

    
    This function assumes that the time steps are numerical values 
    and the distances between the time steps are equidistant.

    Parameters
    ----------
    timeseries : TimeSeries
        Time series data.

    Returns
    -------
    TimeSeries
        Fourier spectrum (energy spectral density)
    """
    fftMu, fftGamma, fftC = _ua_fourier_transform(timeseries)
    dt = timeseries.timesteps[1]-timeseries.timesteps[0]
    frequencies = (np.arange(len(timeseries.timesteps)) + 2) / len(timeseries.timesteps) / (2*dt)
    return TimeSeries(chi_square_comb.ChiSquareComb(fftMu, fftGamma, fftC), frequencies)