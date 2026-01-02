from sklearn import datasets
from sklearn.mixture import GaussianMixture
import numpy as np
from uadapy import Distribution, TimeSeries
from uadapy.distributions import MultivariateGMM
from scipy import stats

def load_iris_normal():
    """
    Uses the iris dataset and fits a normal distribution
    :return:
    """
    iris = datasets.load_iris()
    dist = []
    for c in np.unique(iris.target):
        dist.append(Distribution(np.array(iris.data[iris.target == c]), "Normal"))
    return dist

def load_iris():
    """
    Uses the iris dataset and fits a normal distribution
    :return:
    """
    iris = datasets.load_iris()
    dist = []
    for c in np.unique(iris.target):
        dist.append(Distribution(np.array(iris.data[iris.target == c])))
    return dist

def generate_synthetic_timeseries(timesteps=200, trend = 0.1):
    """
    Generates synthetic time series data by modeling a combination of trend,
    periodic patterns, and noise using a multivariate normal distribution
    with an exponential quadratic kernel for covariance.

    Parameters
    ----------
    timesteps : int
        The time steps of the time series.
        Default value is 200.

    Returns
    -------
    timeseries : Timeseries object
        An instance of the TimeSeries class, which represents a univariate time series.
    """
    np.random.seed(0)
    t = np.arange(1, timesteps + 1)
    trend = t * trend
    periodic = 10 * np.sin(2 * np.pi * t / 100)
    noise = 2 * (np.random.rand(timesteps) - 0.5)
    mu = trend + periodic + noise
    sigma2 = 20 * np.ones(timesteps)
    sigma_sq = np.sqrt(sigma2)
    sigma = np.zeros((timesteps, timesteps))

    def ex_qu_kernel(x, y, sigma_i, sigma_j, l):
        return sigma_i * sigma_j * np.exp(-0.5 * np.linalg.norm(x - y)**2 / l**2)
    
    for i in range(timesteps):
        for j in range(timesteps):
            sigma[i, j] = ex_qu_kernel(t[i], t[j], sigma_sq[i], sigma_sq[j], 5)

    # Ensure symmetry
    sigma = (sigma + sigma.T) / 2
    
    # Ensure positive definiteness
    epsilon = 1e-6
    sigma += np.eye(sigma.shape[0]) * epsilon
    model = stats.multivariate_normal(mu, sigma)
    timeseries = TimeSeries(model, None)

    return timeseries

def generate_synthetic_gmm(n_classes=3, n_dims=4, random_state=0):
    """
    Generates synthetic Gaussian Mixture Model distributions for testing and examples.
    
    Creates multiple classes, each represented as a GMM with varying numbers of components,
    means, and covariances. Useful for demonstrating dimensionality reduction techniques
    like wGMM-UAPCA.
    
    Parameters
    ----------
    n_classes : int, optional
        Number of classes to generate.
        Default value is 3.
    n_dims : int, optional
        Dimensionality of the original data space.
        Default value is 4.
    random_state : int, optional
        Random seed for reproducibility.
        Default value is 42.
    
    Returns
    -------
    list
        List of Distribution objects, each wrapping a MultivariateGMM model.
    """
    np.random.seed(random_state)

    distributions = []
    
    for class_idx in range(n_classes):
        # Vary the number of components per class (1 to 3)
        n_components = np.random.randint(1, 4)
        
        # Generate synthetic data for this class
        samples_per_component = 100
        class_data = []
        
        for comp_idx in range(n_components):
            # Random mean in hypercube
            mean = np.random.randn(n_dims) * 3 + class_idx * 5
            
            # Random covariance matrix (positive definite)
            A = np.random.randn(n_dims, n_dims)
            cov = A @ A.T + np.eye(n_dims) * 0.5
            
            # Generate samples
            samples = np.random.multivariate_normal(mean, cov, samples_per_component)
            class_data.append(samples)
        
        # Combine all component samples
        class_data = np.vstack(class_data)
        
        # Fit GMM
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type='full',
            random_state=random_state + class_idx
        )
        gmm.fit(class_data)
        
        # Wrap and store
        distributions.append(Distribution(MultivariateGMM(gmm)))
    
    return distributions
