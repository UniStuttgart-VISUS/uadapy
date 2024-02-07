import traceback

import numpy as np
import scipy as sp
import scipy.stats as st
import distribution as ds



def test_distrib_class():

    model_1D = [
        st.alpha(1.0),
        st.anglit(),
        st.arcsine(),
        st.argus(1.0),
        st.beta(1.0, 1.0),
        st.betaprime(1.0, 1.0),
        st.bradford(1.0),
        st.burr(1.0, 1.0),
        st.burr12(1.0, 1.0),
        st.cauchy(),
        st.chi(1.0),
        st.chi2(1.0),
        st.cosine(),
        st.crystalball(1.0, 1.0),
        st.dgamma(1.0),
        st.dweibull(1.0),
        st.erlang(1),
        st.expon(),
        st.exponnorm(1.0),
        st.exponweib(1.0, 1.0),
        st.exponpow(1.0),
        st.f(1.0, 1.0),
        st.fatiguelife(1.0),
        st.fisk(1.0),
        st.foldcauchy(1.0),
        st.foldnorm(1.0),
        # many skipped, need to be added later
        st.norm(),
        st.t(1.0),
        st.uniform(1.0, 2.0)
    ]

    n=2
    vec = np.ones(n)
    mat = np.eye(n)
    model_nD = [
        st.multivariate_normal(mean=vec, cov=mat),
        st.dirichlet(alpha=vec),
        st.dirichlet_multinomial(alpha=vec, n=4),
        st.wishart(df=n, scale=mat),
        st.multinomial(8, vec),
        st.multivariate_t(loc=vec, shape=mat, df=3),
        st.multivariate_hypergeom(m=[1 for i in range(n)], n=4),
        st.vonmises_fisher(mu=vec/np.linalg.norm(vec), kappa=1.0),
        np.random.rand(1000,n)
    ]

    model_nxnD = [
        st.invwishart(df=n, scale=mat),
        st.matrix_normal(mean=mat),
        st.special_ortho_group(dim=n),
        st.ortho_group(dim=n),
        st.unitary_group(dim=n),
        st.random_correlation(eigs=vec)
    ]

    # initialize distribution object for each of the scipy distribs (univariate)
    for scipi_distrib in model_1D:
        distrib = ds.distribution(scipi_distrib)
    # initialize distribution object for each of the scipy distribs (multivariate)
    for scipi_distrib in model_nD:
        try:
            distrib = ds.distribution(scipi_distrib)
            cov = distrib.cov()
            if cov.shape[0] != n or cov.shape[1] != n:
                raise RuntimeError(f"shape expected to be {n} x {n}, but was {cov.shape}")
            samples = distrib.sample(1000)
            densities = distrib.pdf(samples)
        except Exception as e:
            print(f"Exception occured: {e} (error encountered with {scipi_distrib.__class__.__name__})")
            traceback.print_exception(e)





if __name__ == '__main__':
    test_distrib_class()


    