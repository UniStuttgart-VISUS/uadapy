from uadapy import TimeSeries
import numpy as np
import math
import uadapy.distributions.chi_square_comb as chi_square_comb
import os
import scipy.optimize as optimize
from array import array
import platform

from cffi import FFI
ffibuilder = FFI()

# Get the directory where this module is located
current_dir = os.path.dirname(os.path.abspath(__file__))
header_path = os.path.join(current_dir, "chi2comb", "interface.h")

# Platform-specific library loading
if platform.system() == "Windows":
    lib_path = os.path.join(current_dir, "chi2comb", "chi2comb.dll")
elif platform.system() == "Linux":
    lib_path = os.path.join(current_dir, "chi2comb", "libchi2comb.so")
elif platform.system() == "Darwin":  # macOS
    lib_path = os.path.join(current_dir, "chi2comb", "libchi2comb.dylib")
else:
    raise OSError(f"chi2comb library is not available for {platform.system()} systems.")

# Check if library file exists
if not os.path.exists(lib_path):
    raise OSError(f"chi2comb library not found at {lib_path}")

with open(header_path, "r") as f:
    ffibuilder.cdef(f.read())

lib = ffibuilder.dlopen(lib_path)

# Copied over from chi2comb python bindings
class ChiSquared(object):
    """
    Noncentral χ² distribution.
    """

    def __init__(self, coef, ncent, dof):
        self.coef = float(coef)
        self.ncent = float(ncent)
        self.dof = int(dof)

    def __repr__(self):
        msg = "(coef={}, ncent={}, dof={})".format(self.coef, self.ncent, self.dof)
        return "ChiSquared" + msg


class Info(object):
    """
    Algorithm information.
    """

    def __init__(self):
        self.emag = 0.0
        self.niterms = 0
        self.nints = 0
        self.intv = 0.0
        self.truc = 0.0
        self.sd = 0.0
        self.ncycles = 0

    def __repr__(self):
        msg = "(emag={}, niterms={}, ".format(self.emag, self.niterms)
        msg += "nints={}, ".format(self.nints)
        msg += "intv={}, truc={}, sd={}".format(self.intv, self.truc, self.sd)
        msg += ", ncycles={})".format(self.ncycles)
        return "Info" + msg


def chi2comb_cdf(q, chi2s, gcoef, lim=1000, atol=1e-4):
    """
    Function distribution of combination of noncentral chi-squared distributions.

    Parameters
    ----------
    q : float
        Value point at which distribution function is to be evaluated.
    chi2s : list
        List of ChiSquared objects defining noncentral χ² distributions.
    gcoef : float
        Coefficient of the standard Normal distribution.
    lim : int, optional
        Maximum number of integration terms. It defaults to ``1000``.
    atol : float, optional
        Absolute error tolerance. It defaults to ``1e-4``.

    Returns
    -------
    result : float
        Estimated c.d.f. evaluated at ``q``.
    error : int
        0: completed successfully
        1: required accuracy not achieved
        2: round-off error possibly significant
        3: invalid parameters
        4: unable to locate integration parameters
        5: out of memory
    info : Info
        Algorithm information.
    """

    int_type = "i"
    if array(int_type, [0]).itemsize != ffibuilder.sizeof("int"):
        int_type = "l"
        if array(int_type, [0]).itemsize != ffibuilder.sizeof("int"):
            raise RuntimeError("Could not infer a proper integer representation.")

    if array("d", [0.0]).itemsize != ffibuilder.sizeof("double"):
        raise RuntimeError("Could not infer a proper double representation.")

    q = float(q)
    c_chi2s = ffibuilder.new("struct chi2comb_chisquareds *")
    c_info = ffibuilder.new("struct chi2comb_info *")

    ncents = array("d", [float(i.ncent) for i in chi2s])
    coefs = array("d", [float(i.coef) for i in chi2s])
    dofs = array(int_type, [int(i.dof) for i in chi2s])

    c_chi2s.ncents = ffibuilder.cast("double *", ncents.buffer_info()[0])
    c_chi2s.coefs = ffibuilder.cast("double *", coefs.buffer_info()[0])
    c_chi2s.dofs = ffibuilder.cast("int *", dofs.buffer_info()[0])
    c_chi2s.n = len(chi2s)

    result = ffibuilder.new("double *")
    errno = lib.chi2comb_cdf(q, c_chi2s, gcoef, lim, atol, c_info, result)

    info = Info()
    methods = ["emag", "niterms", "nints", "intv", "truc", "sd", "ncycles"]
    for i in methods:
        setattr(info, i, getattr(c_info, i))

    return (result[0], errno, info)

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
    # Normalize the time series if required
    # Compute the Fourier transform 
    fftMu, fftGamma, fftC = _ua_fourier_transform(timeseries)
    dt = timeseries.timesteps[1]-timeseries.timesteps[0]
    frequencies = (np.arange(len(timeseries.timesteps)) + 2) / len(timeseries.timesteps) / (2*dt)
    frequencies = frequencies[:len(frequencies)//2]
    return TimeSeries(chi_square_comb.ChiSquareComb(fftMu, fftGamma, fftC), frequencies)

def compute_percentiles_complex(timeseries, p):
    """
    Computes the percentiles of the spectrum which is defined by computing the absolute square of the complex
    normal distribution defined by mu, gamma and c.

    Parameters
    ----------
    timeseries : TimeSeries
        Time series data that results out of the Fourier transformation of the original time series.
    p : list
        List of percentiles to compute.

    Returns
    -------
    np.ndarray
        Percentiles of the spectrum.
    """
    mu = timeseries.distribution.model.mu_complex
    gamma = timeseries.distribution.model.covariance
    c = timeseries.distribution.model.pseudo_covariance
    mu_re = np.real(mu)
    mu_im = np.imag(mu)
    a = np.diagonal(0.5 * np.real(gamma + c))
    d = np.diagonal(0.5 * np.real(gamma - c))
    b = np.diagonal(0.5 * np.imag(gamma + c))
    return _compute_percentiles_diagonals(mu_re, mu_im, a, b, d, p)

def _compute_percentiles_diagonals(mu1, mu2, a, b, d, p=None):
    EPS = 10e-8
    if p is None:
        p = [0.025, 0.25, 0.5, 0.75, 0.975]
    l1 = 0.5 * ((a + d) + np.sqrt((a - d) ** 2 + 4 * b ** 2))
    l2 = 0.5 * ((a + d) - np.sqrt((a - d) ** 2 + 4 * b ** 2))
    if np.any(l2 < EPS):
        d = np.copy(d)
        d[d<0] = 0
        a = np.copy(a)
        a[a<0] = 0
        b1 = np.zeros(l1.shape)
        b2 = np.zeros(l1.shape)
        b1[l2<EPS] = mu1[l2 < EPS]*np.sqrt(a[l2 < EPS])+mu2[l2 < EPS]*np.sqrt(d[l2 < EPS])
        if np.any(l2 >= EPS):
            p11 = 1 / np.sqrt((b[l2 >= EPS] ** 2 + (l1[l2 >= EPS] - a[l2 >= EPS]) ** 2)) * b[l2 >= EPS]
            p21 = 1 / np.sqrt((b[l2 >= EPS] ** 2 + (l2[l2 >= EPS] - a[l2 >= EPS]) ** 2)) * b[l2 >= EPS]
            p12 = 1 / np.sqrt((b[l2 >= EPS] ** 2 + (l1[l2 >= EPS] - a[l2 >= EPS]) ** 2)) * (l1[l2 >= EPS] - a[l2 >= EPS])
            p22 = 1 / np.sqrt((b[l2 >= EPS] ** 2 + (l2[l2 >= EPS] - a[l2 >= EPS]) ** 2)) * (l2[l2 >= EPS] - a[l2 >= EPS])
            b2[l2 >= EPS] = (mu1[l2 >= EPS] * p21 + mu2[l2 >= EPS] * p22) / np.sqrt(l2[l2 >= EPS])
            b1[l2 >= EPS] = (mu1[l2 >= EPS] * p11 + mu2[l2 >= EPS] * p12) / np.sqrt(l1[l2 >= EPS])
        if np.any(a + d < EPS):
            l1[a + d < EPS] = 0
        mask = l2<EPS#np.logical_and(l2<EPS, a+d>=EPS)
        #b2[mask] = mu1[mask]**2+mu2[mask]**2-(b1[mask])**2/(a[mask]+d[mask])
        b2[mask] = mu1[mask]**2+mu2[mask]**2-b1[mask]**2/l1[mask]
    else:
        p11 = 1 / np.sqrt((b ** 2 + (l1 - a) ** 2)) * b
        p21 = 1 / np.sqrt((b ** 2 + (l2 - a) ** 2)) * b
        p12 = 1 / np.sqrt((b ** 2 + (l1 - a) ** 2)) * (l1 - a)
        p22 = 1 / np.sqrt((b ** 2 + (l2 - a) ** 2)) * (l2 - a)
        if np.any(np.abs(b) < EPS):
            p11[b < EPS] = 1
            p12[b < EPS] = 0
            p21[b < EPS] = 0
            p22[b < EPS] = 1
        b1 = (mu1 * p11 + mu2 * p12) / np.sqrt(l1)
        b2 = (mu1 * p21 + mu2 * p22) / np.sqrt(l2)
    return _percentiles_over_time(l1, l2, b1, b2, min=0, max=max(1000, np.max((mu1**2+mu2**2)*5)), p=p)

def _percentiles_over_time(l1, l2, b1, b2, min=0.01, max=100, p = None):
    THRESHOLD = 0.01
    percentiles = np.zeros((len(p), len(l1)))

    # l1Max = np.max(l1)
    # l2Max = np.max(l2)
    l1 = np.abs(l1)
    l2 = np.abs(l2)
    # Iterate over each set of parameters and compute the percentiles
    for i, (l1, l2, b1, b2) in enumerate(zip(l1, l2, b1, b2)):
        percentiles[:,i] = _get_percentiles(l1, l2, b1, b2, min=min, max=max, p=p)
    # Assume symmetry of real time series and remove the second half of the spectrum
    percentiles = percentiles[:, :len(percentiles[0])//2]
    return percentiles

def cdfSingular(x, l, b, c):
    gcoef = 0
    chi2s = [ChiSquared(l, (b/l)**2, 1)]
    r, _, _ = chi2comb_cdf(x-c, chi2s, gcoef)
    return r

def cdf(x, l1, l2, b1, b2):
  CHI2COMBTOL = 1e-3
  #return integrate.quad(pdf, 0, x, args=(l1, l2, b1, b2))[0]
  gcoef = 0
  ncents = [b1**2, b2**2]
  dofs = [1, 1]
  coefs = [l1, l2]
  chi2s = [ChiSquared(coefs[i], ncents[i], dofs[i]) for i in range(2)]
  r, _, _ = chi2comb_cdf(x, chi2s, gcoef, atol=CHI2COMBTOL)
  if r < 0:
        if np.abs(r) < CHI2COMBTOL:
            r = 0
        else:
            print("wrong " + str(r))
            print(ncents)
            print(coefs)
  return r

def _get_percentiles(l1, l2, b1, b2, min=0.00001, max=100, p=None):
  EPS = 10e-8
  if l2 < EPS:
      if l1 < EPS:
          cdfVal = lambda x: 0
      else:
          cdfVal = lambda x: cdfSingular(x, l=l1, b=b1, c=b2)
  else:
      cdfVal = lambda x: cdf(x, l1=l1, l2=l2, b1=b1, b2=b2)
  res = np.zeros((len(p)))
  for i, percentile in enumerate(p):
      try:
        res[i] = optimize.bisect(lambda x: cdfVal(x) - percentile, min, max)#, rtol=0.000001)
      except:
        print("Did not work for " + str(percentile))
        print(str(cdfVal(min) - percentile) + " " + str(cdfVal(max) - percentile))
        print(str(min) + " " + str(max))
  return res