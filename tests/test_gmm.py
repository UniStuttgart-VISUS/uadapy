# make script aware of parent directory where uadapy is located
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import uadapy
from uadapy import data, dr, distributions
from sklearn.mixture import GaussianMixture
from sklearn.datasets import load_iris
import numpy as np


def test_gmm_pipeline():
    # check all types of covariance supported by sklearns GaussianMixture
    for cov_type in ["full", "tied", "diag", "spherical"]:
        gmm = GaussianMixture(4, covariance_type="tied")
        dists = uadapy.data.load_iris()
        data = np.vstack([d.sample(100) for d in dists])
        gmm.fit(data)
        mgmm = uadapy.distributions.MultivariateGMM(gmm)
        dist = uadapy.Distribution(model=mgmm)
        projected = uadapy.dr.wgmm_uapca([dist])

def test_kde2gmm():
    randdat = np.random.rand(5,3)
    distr_kde = uadapy.Distribution(randdat)
    gmm = uadapy.distributions.gmm_from_kde(distr_kde.kde)
    distr_gmm = uadapy.Distribution(gmm)
    projected = uadapy.dr.wgmm_uapca([distr_gmm])


test_gmm_pipeline()
test_kde2gmm()


