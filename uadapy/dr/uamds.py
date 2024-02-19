"""
This module contains functions that implement the uncertainty aware multidimensional scaling (UAMDS) algorithm.
This is a dimensionality reduction algorithm for sets of normally distributed random vectors (i.e. multivariate
normal distributions). See the corresponding paper at https://doi.org/10.1109/TVCG.2022.3209420.

Copyright: (c) 2024 David Haegele, Patrick Paetzold, Ruben Bauer, Marina Evers

License: MIT
"""

import numba
import numpy as np
from scipy.spatial import distance_matrix
from scipy.optimize import minimize
import uadapy as ua
from scipy.stats import multivariate_normal


def precalculate_constants(normal_distr_spec: np.ndarray) -> tuple:
    """
    Computes constant expressions used in the stress and gradient calculations.
    These constants are specific properties of the individual distributions, e.g. the SVDs of the covariance matrices,
    or relationships beetween the distributions, such as the pairwise squared distances between the distribution means
    (similar to the dissimilarity matrix in regular MDS).

    Parameters
    ----------
    normal_distr_spec : np.ndarray
        normal distributions specification (block of means followed by block of covariance matrices)

    Returns
    -------
    tuple
        a tuple containing the computed constant expressions
    """
    d_hi = normal_distr_spec.shape[1]
    n = normal_distr_spec.shape[0] // (d_hi+1)  # array of (d_hi x d_hi) cov matrices and (1 x d_hi) means

    # extract means and covs
    mu = [normal_distr_spec[i, :] for i in range(n)]
    cov = [normal_distr_spec[n+d_hi*i:n+d_hi*(i+1), :] for i in range(n)]

    # compute singular value decomps of covs
    svds = [np.linalg.svd(cov[i], full_matrices=True) for i in range(n)]
    U = [svds[i].U for i in range(n)]
    S = [np.diag(svds[i].S) for i in range(n)]
    Ssqrt = [np.diag(np.sqrt(svds[i].S)) for i in range(n)]

    # combinations used in stress terms
    norm2_mui_sub_muj = [[np.dot(mu[i]-mu[j], mu[i]-mu[j]) for j in range(n)] for i in range(n)]
    Ssqrti_UiTUj_Ssqrtj = [[Ssqrt[i] @ U[i].T @ U[j] @ Ssqrt[j] for j in range(n)] for i in range(n)]
    mui_sub_muj_TUi = [[(mu[i]-mu[j]) @ U[i] for j in range(n)] for i in range(n)]
    mui_sub_muj_TUj = [[(mu[i]-mu[j]) @ U[j] for j in range(n)] for i in range(n)]
    Zij = [[U[i].T @ U[j] for j in range(n)] for i in range(n)]

    # constants = {
    #     'mu': mu,
    #     'cov': cov,
    #     'U': U,
    #     'S': S,
    #     'Ssqrt': Ssqrt,
    #     'norm2_mui_sub_muj': norm2_mui_sub_muj,
    #     'Ssqrti_UiTUj_Ssqrtj': Ssqrti_UiTUj_Ssqrtj,
    #     'mui_sub_muj_TUi': mui_sub_muj_TUi,
    #     'mui_sub_muj_TUj': mui_sub_muj_TUj,
    #     'Zij': Zij
    # }
    constants = (
        np.stack(mu),
        np.stack(cov),
        np.stack(U),
        np.stack(S),
        np.stack(Ssqrt),
        np.stack(norm2_mui_sub_muj),
        np.stack(Ssqrti_UiTUj_Ssqrtj),
        np.stack(mui_sub_muj_TUi),
        np.stack(mui_sub_muj_TUj),
        np.stack(Zij)
    )
    return constants


@numba.njit(cache=True)
def _stress_ij(i: int, j: int, normal_distr_spec: np.ndarray, uamds_transforms: np.ndarray,
               mu,
               cov,
               U,
               S,
               Ssqrt,
               norm2_mui_sub_muj,
               Ssqrti_UiTUj_Ssqrtj,
               mui_sub_muj_TUi,
               mui_sub_muj_TUj,
               Zij
               ) -> float:
    d_hi = normal_distr_spec.shape[1]
    d_lo = uamds_transforms.shape[1]
    n = normal_distr_spec.shape[0] // (d_hi + 1)
    # get constants
    # (
    #     mu,
    #     cov,
    #     U,
    #     S,
    #     Ssqrt,
    #     norm2_mui_sub_muj,
    #     Ssqrti_UiTUj_Ssqrtj,
    #     mui_sub_muj_TUi,
    #     mui_sub_muj_TUj,
    #     Zij
    # ) = pre

    # get some objects for i
    Si = S[i]
    Ssqrti = Ssqrt[i]
    ci = uamds_transforms[i, :]
    Bi = uamds_transforms[n+i*d_hi : n+(i+1)*d_hi, :]

    # get some objects for j
    Sj = S[j]
    Ssqrtj = Ssqrt[j]
    cj = uamds_transforms[j, :]
    Bj = uamds_transforms[n+j*d_hi : n+(j+1)*d_hi, :]

    ci_sub_cj = ci-cj

    # compute term 1 : part 1 : ||Si - Si^(1/2)Bi^T BiSi^(1/2)||_F^2
    temp = Ssqrti @ Bi
    temp = Si - (temp @ temp.T)
    part1 = (temp*temp).sum()  # sum of squared elements = squared frobenius norm
    # compute term 1 : part 2 : same as part 1 but with j
    temp = Ssqrtj @ Bj
    temp = Sj - (temp @ temp.T)
    part2 = (temp*temp).sum()  # sum of squared elements = squared frobenius norm
    # compute term 1 : part 3
    temp = (Ssqrti @ Bi) @ (Bj.T @ Ssqrtj)  # outer product of transformed Bs
    temp = Ssqrti_UiTUj_Ssqrtj[i][j] - temp
    part3 = (temp*temp).sum()  # sum of squared elements = squared frobenius norm
    term1 = 2*(part1+part2)+4*part3

    # compute term 2 : part 1 : sum_k^n [ Si_k * ( <Ui_k, mui-muj> - <Bi_k, ci-cj> )^2 ]
    temp = ci_sub_cj @ Bi.T
    temp = mui_sub_muj_TUi[i][j] - temp
    temp = temp*temp  # squared
    part1 = (temp @ Si).sum()
    # compute term 2 : part 2 : same as part 1 but with j
    temp = ci_sub_cj @ Bj.T
    temp = mui_sub_muj_TUj[i][j] - temp
    temp = temp*temp  # squared
    part2 = (temp @ Sj).sum()
    term2 = part1+part2

    # compute term 3 : part 1
    norm1 = norm2_mui_sub_muj[i][j]
    norm2 = np.dot(ci_sub_cj,ci_sub_cj)  # squared norm
    part1 = norm1-norm2
    # compute term 3 : part 2
    part2 = 0
    part3 = 0
    for k in range(d_hi):
        sigma_i = Si[k, k]
        sigma_j = Sj[k, k]
        bik = Bi[k, :]
        bjk = Bj[k, :]
        part2 += (1 - np.dot(bik,bik))*sigma_i
        part3 += (1 - np.dot(bjk,bjk))*sigma_j
    term3 = (part1 + part2 + part3)**2

    return term1+term2+term3


@numba.njit(cache=True)
def _gradient_ij_optimized(i: int, j: int, normal_distr_spec: np.ndarray, uamds_transforms: np.ndarray,
                           S, norm2_mui_sub_muj, mui_sub_muj_TUi, mui_sub_muj_TUj, Z, BiSi, Bi, Si, BiT, part1i) -> tuple:
    d_hi = normal_distr_spec.shape[1]
    # d_lo = uamds_transforms.shape[1]
    n = normal_distr_spec.shape[0] // (d_hi + 1)
    # get some objects for i
    #Si = S[i].copy()
    # mui = mu[i]
    ci = uamds_transforms[i, :]
    #Bi = uamds_transforms[n + i * d_hi: n + (i + 1) * d_hi, :].T.copy()
    #BiSi = Bi @ Si

    # get some objects for j
    Sj = S[j].copy()
    # muj = mu[j]
    cj = uamds_transforms[j, :]
    Bj = uamds_transforms[n + j * d_hi:n + (j + 1) * d_hi, :].T.copy()
    BjSj = Bj @ Sj

    # mui_sub_muj = mui - muj
    ci_sub_cj = ci - cj

    # compute term 1 :
    Zij = Z[i][j].copy()
    #BiT = Bi.T.copy()
    BjT = Bj.T.copy()
    #part1i = (BiSi @ BiT @ BiSi) - (BiSi @ Si)
    part1j = (BjSj @ BjT @ BjSj) - (BjSj @ Sj)
    part2i = (BjSj @ BjT @ BiSi) - (BjSj @ Zij.T @ Si)
    part2j = (BiSi @ BiT @ BjSj) - (BiSi @ Zij @ Sj)
    dBi = (part1i + part2i) * 8
    dBj = (part1j + part2j) * 8

    # compute term 2 :
    dci = np.zeros(ci.shape)
    dcj = np.zeros(cj.shape)
    if i != j:
        # gradient part for B matrices
        part3i = (np.outer(ci_sub_cj, (ci_sub_cj @ Bi)) - np.outer(ci_sub_cj, mui_sub_muj_TUi[i][j])) @ Si
        part3j = (np.outer(ci_sub_cj, (ci_sub_cj @ Bj)) - np.outer(ci_sub_cj, mui_sub_muj_TUj[i][j])) @ Sj
        dBi += 2 * part3i
        dBj += 2 * part3j
        # gradient part for c vectors
        part4i = (mui_sub_muj_TUi[i][j] - (ci_sub_cj @ Bi)) @ BiSi.T
        part4j = (mui_sub_muj_TUj[i][j] - (ci_sub_cj @ Bj)) @ BjSj.T
        part4 = -2 * (part4i + part4j)
        dci += part4
        dcj -= part4

    # compute term 3 :
    norm1 = norm2_mui_sub_muj[i][j]
    norm2 = np.dot(ci_sub_cj, ci_sub_cj)
    part1 = norm1 - norm2
    part2 = part3 = 0.0
    for k in range(d_hi):
        sigma_i = Si[k, k]
        sigma_j = Sj[k, k]
        bik = Bi[:, k].copy()
        bjk = Bj[:, k].copy()
        part2 += (1 - np.dot(bik, bik)) * sigma_i
        part3 += (1 - np.dot(bjk, bjk)) * sigma_j
    term3 = -4 * (part1 + part2 + part3)
    dBi += BiSi * term3
    dBj += BjSj * term3

    if i != j:
        dci += ci_sub_cj * term3
        dcj -= ci_sub_cj * term3

    return dBi.T, dBj.T, dci, dcj


def stress(normal_distr_spec: np.ndarray, uamds_transforms: np.ndarray, precalc_constants: tuple=None) -> float:
    if precalc_constants is None:
        precalc_constants = precalculate_constants(normal_distr_spec)
    return _stress_numba(normal_distr_spec, uamds_transforms, precalc_constants)


@numba.njit(parallel=True, cache=True)
def _stress_numba(normal_distr_spec: np.ndarray, uamds_transforms: np.ndarray, precalc_constants: tuple) -> float:
    d_hi = normal_distr_spec.shape[1]
    d_lo = uamds_transforms.shape[1]
    n = normal_distr_spec.shape[0] // (d_hi + 1)  # array of (d_hi x d_hi) cov matrices and (1 x d_hi) means

    sum = 0
    for i in numba.prange(n):
        for j in numba.prange(i, n):
            sum += _stress_ij(i, j, normal_distr_spec, uamds_transforms, *precalc_constants)
    return sum


@numba.njit(parallel=False, cache=True)
def _gradient_numba_optimized(normal_distr_spec: np.ndarray, uamds_transforms: np.ndarray, S, norm2_mui_sub_muj,
                              mui_sub_muj_TUi, mui_sub_muj_TUj, Z, n, d_hi):
    # compute the gradients of all affine transforms
    grad = np.zeros(uamds_transforms.shape)
    for i in numba.prange(n):

        Si = S[i].copy()
        # mui = mu[i]
        Bi = uamds_transforms[n + i * d_hi: n + (i + 1) * d_hi, :].T.copy()
        BiSi = Bi @ Si
        BiT = Bi.T.copy()
        part1i = (BiSi @ BiT @ BiSi) - (BiSi @ Si)

        for j in numba.prange(i, n):
            dBi, dBj, dci, dcj = _gradient_ij_optimized(i, j, normal_distr_spec, uamds_transforms, S, norm2_mui_sub_muj,
                                                        mui_sub_muj_TUi, mui_sub_muj_TUj, Z, BiSi, Bi, Si, BiT, part1i)
            # c gradients on top part of matrix
            grad[i, :] += dci
            grad[j, :] += dcj
            # B gradients below c part of matrix
            grad[n + i * d_hi:n + (i + 1) * d_hi, :] += dBi
            grad[n + j * d_hi:n + (j + 1) * d_hi, :] += dBj
    return grad


def gradient(normal_distr_spec: np.ndarray, uamds_transforms: np.ndarray, precalc_constants: tuple) -> np.ndarray:
    # print("grad")
    d_hi = normal_distr_spec.shape[1]
    # d_lo = uamds_transforms.shape[1]
    n = normal_distr_spec.shape[0] // (d_hi + 1)

    S = precalc_constants[3]
    norm2_mui_sub_muj = precalc_constants[5]
    mui_sub_muj_TUi = precalc_constants[7]
    mui_sub_muj_TUj = precalc_constants[8]
    Z = precalc_constants[9]

    return _gradient_numba_optimized(normal_distr_spec, uamds_transforms, S, norm2_mui_sub_muj,
                                     mui_sub_muj_TUi, mui_sub_muj_TUj, Z, n, d_hi)


def iterate_simple_gradient_descent(
        normal_distr_spec: np.ndarray,
        uamds_transforms_init: np.ndarray,
        precalc_constants: tuple = None,
        num_iter: int = 100,
        a: float = 0.0001,
        optimizer="plain",
        b1: float = 0.9,
        b2: float = 0.999,
        e: float = 10e-8,
        mass=0.8
) -> np.ndarray:
    """
    Performs gradient descent on the UAMDS stress to find an optimal projection.
    This uses a fixed number of iterations after which the method returns.
    There are 3 different gradient descent schemes to choose from.
    Alternatively, the method minimize_scipy(...) can be used to minimize the stress, which runs until convergence is
    reached.

    Parameters
    ----------
    normal_distr_spec : np.ndarray
        Normal distributions specification. Matrix starting with n row vectors (means) followed by
        n square matrices (covariances).
    uamds_transforms_init : np.ndarray
        uamds transformations for each distribution (low-dim means followed by local projection matrices B_i)
    precalc_constants : tuple
        a tuple containing the pre-computed constant expressions of the stress and gradient.
        Can be None and will be computed by precalculate_constants(normal_distr_spec)
    num_iter : int
        number of iterations to perform. The required number of iterations varies with the used descent scheme.
    a : float
        step size (learning rate). Depends on the size of the optimization problem and used descent scheme.
        Adam and momentum can usually employ larger learning rates than plain gradient descent.
    optimizer : str
        one of 'adam', 'momentum', 'plain'.
    b1 : float
        only used with 'adam', exponential decay rate for the 1st moment estimates.
    b2 : float
        only used with 'adam', exponential decay rate for the 2nd moment estimates
    mass : flaot
        only used with 'momentum', mass parameter in ]0, 1[. When heavy, the descent direction changes only slightly by
        the current gradient in each iteration.

    Returns
    -------
    np.ndarray
        the optimized uamds transforms. The method convert_xform_uamds_to_affine(normal_distr_spec, uamds_transforms)
        can be used to obtain the corresponding affine transformations.
    """
    if precalc_constants is None:
        precalc_constants = precalculate_constants(normal_distr_spec)
    uamds_transforms = uamds_transforms_init
    match optimizer:
        case "adam":
            m = np.zeros_like(uamds_transforms_init)
            v = np.zeros_like(uamds_transforms_init)
            for i in range(num_iter):
                grad = gradient(normal_distr_spec, uamds_transforms, precalc_constants)
                m = (1 - b1) * grad + b1 * m  # first  moment estimate.
                v = (1 - b2) * (grad ** 2) + b2 * v  # second moment estimate.
                mhat = m / (1 - b1 ** (i + 1))  # bias correction.
                vhat = v / (1 - b2 ** (i + 1))
                uamds_transforms = uamds_transforms - a * mhat / (np.sqrt(vhat) + e)

        case "momentum":
            velocity = np.zeros_like(uamds_transforms_init)
            for i in range(num_iter):
                grad = gradient(normal_distr_spec, uamds_transforms, precalc_constants)
                velocity = mass * velocity + (1.0 - mass) * grad
                uamds_transforms = uamds_transforms - a * velocity

        case _:
            for i in range(num_iter):
                grad = gradient(normal_distr_spec, uamds_transforms, precalc_constants)
                uamds_transforms = uamds_transforms - a * grad

    return uamds_transforms


def minimize_scipy(
        normal_distr_spec: np.ndarray,
        uamds_transforms_init: np.ndarray,
        precalc_constants: tuple = None,
        method: str = "BFGS"
) -> np.ndarray:
    """
    Minimizes the UAMDS stress using scipy.optimize.
    This will run until scipy's optimization routine is returning, i.e., until convergence is reached.
    Alternatively, the method iterate_simple_gradient_descent(...) can be used to perform a fixed number of
    gradient descent iterations.

    Parameters
    ----------
    normal_distr_spec : np.ndarray
        Normal distributions specification. Matrix starting with n row vectors (means) followed by
        n square matrices (covariances).
    uamds_transforms_init : np.ndarray
        uamds transformations for each distribution (low-dim means followed by local projection matrices B_i)
    precalc_constants : tuple
        a tuple containing the pre-computed constant expressions of the stress and gradient.
        Can be None and will be computed by precalculate_constants(normal_distr_spec)
    method : str
        an unconstrained scipy optimization method, 'BFGS' by default.

    Returns
    -------
    np.ndarray
        the optimal uamds transforms. The method convert_xform_uamds_to_affine(normal_distr_spec, uamds_transforms) can
        be used to obtain the corresponding affine transformations.
    """
    if precalc_constants is None:
        precalc_constants = precalculate_constants(normal_distr_spec)
    pre = precalc_constants

    # minimization problem
    x_shape = uamds_transforms_init.shape

    def fx(x: np.ndarray):
        return stress(normal_distr_spec, x.reshape(x_shape), pre)

    def dfx(x: np.ndarray):
        grad = gradient(normal_distr_spec, x.reshape(x_shape), pre)
        return grad.flatten()

    # minimization
    solution = minimize(fx, uamds_transforms_init.flatten(), method=method, jac=dfx)
    return solution.x.reshape(x_shape)


def perform_projection(normal_distr_spec: np.ndarray, uamds_transforms: np.ndarray) -> np.ndarray:
    """
    Projects the distributions specified in normal_distr_spec using the provided uamds_transforms.

    Parameters
    ----------
    normal_distr_spec : np.ndarray
        Normal distributions specification. Matrix starting with n row vectors (means) followed by
        n square matrices (covariances).
    uamds_transforms : np.ndarray
        uamds transformations for each distribution (low-dim means followed by local projection matrices B_i)

    Returns
    -------
    np.ndarray
        projected normal distributions in the normal distribution specification format (block of means followed by
        block of covariance matrices).
    """
    d_hi = normal_distr_spec.shape[1]
    # d_lo = uamds_transforms.shape[1]
    n = normal_distr_spec.shape[0] // (d_hi + 1)

    mus = []
    covs = []
    for i in range(n):
        mu_lo = uamds_transforms[i, :]
        cov_hi = normal_distr_spec[n+i*d_hi : n+(i+1)*d_hi, :]
        B = uamds_transforms[n+i*d_hi : n+(i+1)*d_hi, :]
        S = np.diag(np.linalg.svd(cov_hi, full_matrices=True).S)
        cov_lo = B.T @ S @ B
        mus.append(mu_lo)
        covs.append(cov_lo)
    return mk_normal_distr_spec(mus, covs)


def apply_uamds(means: list[np.ndarray], covs: list[np.ndarray], target_dim=2) -> dict[str, list[np.ndarray] | float]:
    """
    Applies UAMDS to the specified normal distributions (given as means and covariance matrices).

    Parameters
    ----------
    means : list
        list of vectors that resemble the means of the normal distributions
    covs : list
        list of matrices that resemble the covariances of the normal distributions
    target_dim : int
        the dimensionality of the projection space, 2 by default

    Returns
    -------
    dict
        dictionary containing the results:
        ::
            ['means']: list of projected means
            ['covs']: list of projected covariances
            ['translations']: list of low-dimensional translation vectors for affine transform of high-dimensional means
            ['projection']: list of projection matrices for affine transform of high-dimensional means and covs
            ['stress']: remaining stress of the projection
    """
    normal_distr_spec = mk_normal_distr_spec(means, covs)
    d_hi = normal_distr_spec.shape[1]
    n = normal_distr_spec.shape[0] // (d_hi + 1)
    # initialization
    uamds_transforms = np.random.rand(normal_distr_spec.shape[0], target_dim)
    avg_dist_hi = distance_matrix(normal_distr_spec[:n,:], normal_distr_spec[:n,:]).mean()
    avg_dist_lo = distance_matrix(uamds_transforms[:n,:], uamds_transforms[:n,:]).mean()
    uamds_transforms[:n,:] *= (avg_dist_hi/avg_dist_lo)
    # compute UAMDS
    pre = precalculate_constants(normal_distr_spec)
    uamds_transforms = minimize_scipy(normal_distr_spec, uamds_transforms, pre)
    s = stress(normal_distr_spec, uamds_transforms, pre)
    # perform projection
    normal_distribs_lo = perform_projection(normal_distr_spec, uamds_transforms)
    means_lo, covs_lo = get_means_covs(normal_distribs_lo)
    affine_transforms = convert_xform_uamds_to_affine(normal_distr_spec, uamds_transforms)
    translations = affine_transforms[:n,:]
    translations = [translations[i, :] for i in range(n)]
    projection_matrices = affine_transforms[n:,:]
    projection_matrices = [projection_matrices[i*d_hi:(i+1)*d_hi, :] for i in range(n)]
    return {
        'means': means_lo,
        'covs': covs_lo,
        'translations': translations,
        'projections': projection_matrices,
        'stress': s
    }


def uamds(distributions: list, dims: int=2):
    """
    Applies the UAMDS algorithm to the provided distributions and returns the projected distributions
    in lower-dimensional space. It assumes multivariate normal distributions.
    If you supply other distributions that provide mean and covariance, these values would be used
    to approximate a normal distribution.

    Parameters
    ----------
    distributions : list
        list of input distributions (distribution objects offering mean() and cov() methods)
    dims : int
        target dimensionality, 2 by default.

    Returns
    -------
    list
        List of distributions living in projection space (i.e. of provided dimensionality)
    """
    try:
        means = [d.mean() for d in distributions]
        covs = [d.cov() for d in distributions]
        result = apply_uamds(means, covs, dims)
        distribs_lo = []
        for (m, c) in zip(result['means'], result['covs']):
            distribs_lo.append(ua.distribution.distribution(multivariate_normal(m, c)))
        return distribs_lo
    except Exception as e:
        raise Exception(f'Something went wrong. Did you input normal distributions? Exception:{e}')


####################################
# utility methods ##################
####################################


def get_means_covs(normal_distr_spec: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Separates the mean vectors and covariance matrices from the stacked normal distributions specification format.

    Parameters
    ----------
    normal_distr_spec : np.ndarray
        Normal distributions specification. Matrix starting with n row vectors (means) followed by
        n square matrices (covariances).

    Returns
    -------
    tuple[list[np.ndarray], list[np.ndarray]]
        a list of mean vectors and a list of covariance matrices
    """
    d_hi = normal_distr_spec.shape[1]
    n = normal_distr_spec.shape[0] // (d_hi + 1)
    means = []
    covs = []
    for i in range(n):
        means.append(normal_distr_spec[i, :])
        covs.append(normal_distr_spec[n+i*d_hi : n+(i+1)*d_hi, :])
    return means, covs


def mk_normal_distr_spec(means: list[np.ndarray], covs: list[np.ndarray]) -> np.ndarray:
    """
    Creates a normal distributions specification matrix for the provided means and covariance matrices.
    This is a matrix starting with n row vectors (means) followed by n square matrices (covariances).

    Parameters
    ----------
    means : list[np.ndarray]
        list of mean vectors
    covs : list[list[np.ndarray]]
        list of covariance matrices

    Returns
    -------
    np.ndarray
        normal distributions specification, n means followed by n covariance matrices
    """
    mean_block = np.vstack(means)
    cov_block = np.vstack(covs)
    return np.vstack([mean_block, cov_block])


def convert_xform_uamds_to_affine(normal_distr_spec: np.ndarray, uamds_transforms: np.ndarray) -> np.ndarray:
    """
    Converts the internally used and optimized transformations into generally applicable affine transformations.
    UAMDS optimizes an affine transform per distribution, each consisting of a projection matrix and translation vector.
    However, the transforms are reformulated with respect to their distribution to allow for more efficient computations
    of the stress and gradient. These are called the uamds_transforms and each of them live in their own coordinate
    system. What this method returns are these transformations, but all with respect to the same coordinate system.

    Parameters
    ----------
    normal_distr_spec : np.ndarray
        normal distributions specification (means followed by covariance matrices)
    uamds_transforms : np.ndarray
        uamds transformations for each distribution (low-dim means followed by local projection matrices B_i)

    Returns
    -------
    np.ndarray
        affine transforms for each distribution. This is a matrix starting with n translation vectors followed by
        n projection matrices. An affine transform is a function of the form f(x)=x*P+t where x is a high-dimensional
        point, P is a projection matrix into low-dimensional space, and t is a low-dimensional translation.
        ::
            affine_transforms[:n,:] is the block of translation vectors
            affine_transforms[n:,:] is the block of projection matrices
    """
    d_hi = normal_distr_spec.shape[1]
    # d_lo = uamds_transforms.shape[1]
    n = normal_distr_spec.shape[0] // (d_hi + 1)

    translations = []
    projections = []
    for i in range(n):
        mu_lo = uamds_transforms[i, :]
        mu_hi = normal_distr_spec[i, :]
        B = uamds_transforms[n+i*d_hi : n+(i+1)*d_hi, :]
        cov_hi = normal_distr_spec[n+i*d_hi : n+(i+1)*d_hi, :]
        U = np.linalg.svd(cov_hi, full_matrices=True).U
        P = U @ B
        t = mu_lo - (mu_hi @ P)
        translations.append(t)
        projections.append(P)
    return np.vstack([np.vstack(translations), np.vstack(projections)])
        

def convert_xform_affine_to_uamds(normal_distr_spec: np.ndarray, affine_transforms: np.ndarray) -> np.ndarray:
    """
    Does the opposite of convert_xform_uamds_to_affine.

    Parameters
    ----------
    normal_distr_spec : np.ndarray
        normal distributions specification (means followed by covariance matrices)
    affine_transforms : np.ndarray
        affine transformations for each distribution (low-dim translations followed by projection matrices)

    Returns
    -------
    np.ndarray
        uamds transforms for each distribution. This is a matrix starting with n low-dimensional mean vectors followed by
        n distribution specific projection matrices B_i. The distribution specific projection matrix is the result
        of transforming it by the basis of the covariance matrix (cov=U*S*U' -> B = U'*P).
        ::
            uamds_transforms[:n,:] is the block of mean vectors
            uamds_transforms[n:,:] is the block of local projection matrices
    """
    d_hi = normal_distr_spec.shape[1]
    # d_lo = affine_transforms.shape[1]
    n = normal_distr_spec.shape[0] // (d_hi + 1)

    mus_lo = []
    Bs = []
    for i in range(n):
        t = affine_transforms[i, :]
        mu_hi = normal_distr_spec[i, :]
        P = affine_transforms[n+i*d_hi : n+(i+1)*d_hi, :]
        cov_hi = normal_distr_spec[n+i*d_hi : n+(i+1)*d_hi, :]
        U = np.linalg.svd(cov_hi, full_matrices=True).U
        B = U.T @ P
        mu_lo = (mu_hi @ P) + t
        mus_lo.append(mu_lo)
        Bs.append(B)
    return np.vstack([np.vstack(mus_lo), np.vstack(Bs)])


