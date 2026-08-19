import numpy as np
import scipy.stats as stats
from uadapy.distributions import IndependentJoint
import pytest

def test_independent_joint():
    a,b,c = stats.norm(), stats.multivariate_normal(mean=np.array([3,4,5]), cov=np.array([[1,.5,.1],[.5,1,-.5],[.1,-.5,1]])), stats.t(5)
    j = IndependentJoint([a,b,c])
    expected_mean = np.array([0, 3, 4, 5, 0])
    assert np.allclose(j.mean(), expected_mean), f"Expected mean: {expected_mean}, but got: {j.mean()}"
    expected_cov = np.eye(5)
    expected_cov[4,4] = c.var()
    expected_cov[1:4, 1:4] = b.cov
    assert np.allclose(j.cov(), expected_cov), f"Expected covariance: {expected_cov}, but got: {j.cov()}"
    sampled_mean = np.mean(j.sample(10_000), axis=0)
    assert np.allclose(sampled_mean, expected_mean, atol=0.1), f"Expected sampled mean: {expected_mean}, but got: {sampled_mean}"

    samples = j.sample(10_000)
    densities = j.pdf(samples)
    assert np.all(densities > 0), "PDF values should be positive for all samples"
    # check that highest density is close to mean
    assert np.allclose(samples[np.argmax(densities)], expected_mean, atol=0.3), f"Sample with highest density should be close to mean: {expected_mean}, but got: {samples[np.argmax(densities)]}"

    # test permutation
    permutation=np.array([2,3,1,4,0])
    jp = IndependentJoint([a,b,c], permutation=permutation)
    assert np.allclose(jp.mean(), expected_mean[permutation]), f"Expected permuted mean: {expected_mean[permutation]}, but got: {jp.mean()}"
    assert np.allclose(jp.cov(), expected_cov[permutation,:][:,permutation]), f"Expected permuted covariance: {expected_cov[permutation,:][:,permutation]}, but got: {jp.cov()}"
    samples_perm = jp.sample(10_000)
    assert np.allclose(samples_perm.mean(axis=0), expected_mean[permutation], atol=0.1), f"Expected permuted sampled mean: {expected_mean[permutation]}, but got: {samples_perm.mean(axis=0)}"
    densities_perm = jp.pdf(samples_perm)
    # check that highest density is close to mean
    assert np.allclose(samples_perm[np.argmax(densities_perm)], expected_mean[permutation], atol=0.3), f"Sample with highest density should be close to permuted mean: {expected_mean[permutation]}, but got: {samples_perm[np.argmax(densities_perm)]}"


    # test marginal
    dims = [2,1,0,3]
    j_ = j.marginal(dims)
    assert np.allclose(j_.mean(), expected_mean[dims]), f"Expected marginal mean: {expected_mean[dims]}, but got: {j_.mean()}"
    assert np.allclose(j_.cov(), expected_cov[dims,:][:,dims]), f"Expected marginal covariance: {expected_cov[dims,:][:,dims]}, but got: {j_.cov()}"
    samples_marginal = j_.sample(10_000)
    assert np.allclose(samples_marginal.mean(axis=0), expected_mean[dims], atol=0.1), f"Expected marginal sampled mean: {expected_mean[dims]}, but got: {samples_marginal.mean(axis=0)}"
    densities_marginal = j_.pdf(samples_marginal)
    # check that highest density is close to mean
    assert np.allclose(samples_marginal[np.argmax(densities_marginal)], expected_mean[dims], atol=0.3), f"Sample with highest density should be close to marginal mean: {expected_mean[dims]}, but got: {samples_marginal[np.argmax(densities_marginal)]}"

    
