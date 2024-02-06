import numpy as np
import uadapy as ua
from scipy.stats import multivariate_normal

def uapca(distributions, dims: int):
    """
    Applies UAPCA algorithm to the distribution and returns the distribution
    in lower-dimensional space. It assumes a normal distributions. If you apply
    other distributions that provide mean and covariance, these values would be used
    to approximate a normal distribution
    :param distributions: List of input distributions
    :param dims: Target dimension
    :return: List of distributions in low-dimensional space
    """
    try:
        means = np.array([d.mean() for d in distributions])
        covs = np.array([d.cov() for d in distributions])
        means_pca, covs_pca = transform_uapca(means, covs, dims)
        dist_pca = []
        for (m, c) in zip(means_pca, covs_pca):
            dist_pca.append(ua.distribution.distribution(multivariate_normal(m, c)))
        return dist_pca
    except Exception as e:
        raise Exception(f'Something went wrong. Did you input normal distributions? Exception:{e}')



# Computing methods

def compute_ua_cov(means: np.ndarray, covs: np.ndarray) -> np.ndarray:
    n = means.shape[0]
    d = means.shape[1]
    # empirical mean
    mu = means.mean(axis=0)
    # centering matrix
    centering = np.outer(mu, mu)
    # average covariance matrix
    avg_cov = covs.reshape((n,d,d)).mean(axis=0)
    # sample covariance
    sample_cov = np.array([np.outer(means[i,:], means[i,:]) for i in range(n)]).mean(axis=0)
    # final uncertainty aware covariance matrix
    return sample_cov + avg_cov - centering


def compute_uapca(means: np.ndarray, covs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cov = compute_ua_cov(means, covs)
    u,s,vh = np.linalg.svd(cov, full_matrices=True)
    return u, s


def transform_uapca(means, covs, dims: int=2) -> tuple[np.ndarray, np.ndarray]:
    n = means.shape[0]
    d = means.shape[1]
    eigvecs, eigvals = compute_uapca(means, covs)
    projmat = eigvecs[:, :dims]
    projected_means = means @ projmat
    projected_covs = np.vstack(
        [projmat.T @ covs[i*d:(i+1)*d, :] @ projmat for i in range(n)]
    )
    return projected_means, projected_covs