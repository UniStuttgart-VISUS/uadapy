=================
TimeSeries
=================

The `TimeSeries` class provides a structured interface for modeling uncertain, univariate time series.
It is built on top of the `Distribution` class and provides convenient wrapper functions to key distribution methods,
as well as additional functions relevant for time series analysis.

Creating a time series
----------------------
.. code-block:: python

   def __init__(self, model, timesteps, name="", n_dims=1)

:Parameters:
   - **model**: A `scipy.stats` distribution object or an array of samples.
   - **timesteps** *(int)*: The number of time steps in the time series.
   - **name** *(str, optional)*: The name of the distribution (default is inferred from the model).
   - **n_dims** *(int, optional)*: Dimensionality of the distribution (default is `1`).

The class constructs an internal `Distribution` object using the provided model and stores the associated time steps.

Working with time series
--------------------------
**Time Series Properties**: Provides methods for computing key statistical properties such as:

- **mean() -> np.ndarray | float**:

  Returns the expected value of the time series.

- **cov() -> np.ndarray | float**:

  Returns the covariance of all time series points.

- **variance() -> np.ndarray | float**:

  Returns the variance of the time series (diagonal of the covariance matrix).

**Sampling and PDF Evaluation**:

- **sample(n: int, seed: int = None) -> np.ndarray**:

  Generates `n` random samples from the time series.

- **pdf(x: np.ndarray | float) -> np.ndarray | float**:

  Evaluates the probability density function (PDF) at the given points.

=========================
CorrelatedDistributions
=========================

The `CorrelatedDistributions` class provides a way to manage and analyze correlated distributions or time series.
It enables sampling from a joint distribution, computing covariance matrices, and accessing means of individual distributions.

Creating correlated distributions
---------------------------------
.. code-block:: python

   def __init__(self, distributions: list[Distribution], covariance_matrix=None)

:Parameters:
   - **distributions** *(list[Distribution])*: A list of individual distributions or time series.
   - **covariance_matrix** *(np.ndarray, optional)*: The pairwise covariance matrix of the distributions.

The class validates whether the provided covariance matrix aligns with the variances of the individual distributions.

Working with correlated distributions
--------------------------------------
**Statistical Properties**:

- **mean(dim_i: int) -> float**:

  Returns the mean of the `i`-th distribution.

- **cov(dim_i: int, dim_j: int) -> np.ndarray | float**:

  Returns the covariance between the `i`-th and `j`-th distributions.

**Sampling**:

- **sample(n_samples: int, seed: int = None) -> np.ndarray**:

  Draws `n_samples` from the joint distribution of all correlated distributions.

  Sampling follows a block structure, where means are concatenated and a covariance matrix is assembled to generate
  multivariate normal samples.
