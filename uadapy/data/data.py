from sklearn import datasets
import numpy as np
from uadapy import Distribution
from scipy import stats

def load_iris_normal():
    """
    Uses the iris dataset and fits a normal distribution
    :return:
    """
    iris = datasets.load_iris()
    print(type(iris))
    print(iris)
    print(type(iris.target))
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

def generate_synthetic_data(n=200):
    """
    Generates synthetic time series data by modeling a combination of trend,
    periodic patterns, and noise using a multivariate normal distribution
    with an exponential quadratic kernel for covariance.
    """
    np.random.seed(0)
    t = np.arange(1, n + 1)
    trend = t / 10
    periodic = 10 * np.sin(2 * np.pi * t / 100)
    noise = 2 * (np.random.rand(n) - 0.5)
    mu = trend + periodic + noise
    sigma2 = 20 * np.ones(n)
    sigma_sq = np.sqrt(sigma2)
    sigma = np.zeros((n, n))

    def ex_qu_kernel(x, y, sigma_i, sigma_j, l):
        return sigma_i * sigma_j * np.exp(-0.5 * np.linalg.norm(x - y)**2 / l**2)
    
    for i in range(n):
        for j in range(n):
            sigma[i, j] = ex_qu_kernel(t[i], t[j], sigma_sq[i], sigma_sq[j], 5)

    # Ensure symmetry
    sigma = (sigma + sigma.T) / 2
    
    # Ensure positive definiteness
    epsilon = 1e-6
    sigma += np.eye(sigma.shape[0]) * epsilon
    model = stats.multivariate_normal(mu, sigma)

    return model
