============
Distribution
============

The `distribution` class serves as a core component for handling probability distributions, both parametric and non-parametric. It allows you to create a distribution from either a statistical model (such as one from `scipy.stats`) or directly from a dataset (samples). The class also supports handling multivariate distributions and automatically distinguishes between univariate and multivariate cases. This class abstracts away the complexity of working with different types of distributions while providing a uniform interface for statistical operations.

Creating a distribution
-----------------------
.. code-block:: python

   def __init__(self, model, name="", dim=1)

:Parameters:
   - **model**: A `scipy.stats` distribution object or an array of samples.
   - **name** *(str)*: The name of the distribution (optional; default is inferred from the model).
   - **dim** *(int)*: Dimensionality of the distribution (optional; default is `1`).

If a set of samples is passed instead of a statistical model, a Kernel Density Estimate (KDE) is used for estimating the probability density function (PDF). If the distribution is named "Normal", the class assumes the samples are from a normal distribution and fits a multivariate normal model to the data.

Working with distributions
--------------------------
**Distribution Properties**: Provides methods for calculating key statistical properties such as:

- **mean() -> np.ndarray | float**:
  
  Returns the mean of the distribution.

- **cov() -> np.ndarray | float**:
  
  Returns the covariance matrix of the distribution.

- **skew() -> np.ndarray | float**:
  
  Returns the skewness of the distribution.

- **kurt() -> np.ndarray | float**:

  Returns the kurtosis of the distribution.
  
**Sampling and PDF Evaluation**: 

- **sample(n: int, random_state: int = None) -> np.ndarray**:
  
  Generates `n` random samples from the distribution.

- **pdf(x: np.ndarray | float) -> np.ndarray | float**:
  
  Evaluates the probability density function (PDF) at the given point `x`.