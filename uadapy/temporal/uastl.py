import numpy as np
from uadapy import TimeSeries, CorrelatedDistributions
from scipy.stats import multivariate_normal

def _convmtx(h, n):
    h = np.array(h)
    m = len(h)
    H = np.zeros((m + n - 1, n))

    # Fill the transposed convolution matrix
    for i in range(m + n - 1):
        for j in range(n):
            if i - j >= 0 and i - j < m:
                H[i, j] = h[i - j].item()
    
    return H

def _loessmtx(a, s, d, omega=None):
    # Initialization and checks
    if isinstance(a, int):
        n = a
        a = np.arange(1, n + 1)
    elif isinstance(a, (list, np.ndarray)) and len(np.shape(a)) == 1:
        a = np.array(a)
        n = len(a)
    else:
        raise ValueError("loessmtx: related positions a must be a vector or an integer.")
    
    a = a.reshape(-1, 1)
    
    if omega is None:
        omega = np.ones(n)
    else:
        omega = np.array(omega).reshape(-1, 1)
    
    if s < 4:
        raise ValueError("loessmtx: span s is too small, it should be at least 4.")
    
    s = int(min(s, n))
    
    B = np.zeros((n, n))
    
    for i in range(n):
        # Find the s-nearest neighbors of a_i
        distances = np.abs(a - a[i])
        idx = np.argsort(distances, axis=0)[:s].flatten()
        
        # Center positions
        a_tild = a[idx] - a[i]
        
        # Compute scaling
        a_tild_abs = np.abs(a_tild)
        scaling = (1 - (a_tild_abs / np.max(a_tild_abs))**3)**3
        
        # Define Vandermonde matrix
        V = np.vander(a_tild.flatten(), N=d+1, increasing=True)
        
        # Compute diagonal matrix
        D = np.diag((scaling.flatten() * omega[idx].flatten()))
        
        # Weighted linear least squares solution
        try:
            pinv_VDV = np.linalg.pinv(V.T @ D @ V)
        except np.linalg.LinAlgError:
            pinv_VDV = np.linalg.pinv(V.T @ D @ V + np.eye(V.shape[1]) * 1e-10)
        a_i = np.zeros(n)
        a_i[idx] = ((np.arange(1, d+2) == 1).astype(int)) @ (pinv_VDV @ V.T @ D)
        
        # Insert row to loess matrix B
        B[i, :] = a_i
    
    return B

def _decompose_distribution(result, timesteps, num_periods):
    """
    Decomposes the result dictionary into a list of time series and a block-structured covariance matrix.

    Parameters
    ----------
    result : dict
        Dictionary containing 'mu_new' (means) and 'sigma_new' (covariance matrix).
    timesteps : int
        Number of timesteps in each component.
    num_periods : int
        Number of independent periods (or components).

    Returns
    -------
    tuple
        A list of time series (TimeSeries objects) and a block-structured covariance matrix.
    """
    means = result['means']
    cov = result['covs']
    num_components = num_periods + 3

    # Create a list to hold time series objects
    ts_list = []
    for i in range(num_components):
        start = i * timesteps
        end = (i + 1) * timesteps
        mean_vec = means[start:end].flatten()
        comp_cov = cov[start:end, start:end]
        ts_list.append(TimeSeries(multivariate_normal(mean_vec, comp_cov, allow_singular=True), timesteps))

    # Create a block-structured covariance matrix
    block_cov = []
    for i in range(num_components):
        row = []
        for j in range(num_components):
            start_i = i * timesteps
            end_i = (i + 1) * timesteps
            start_j = j * timesteps
            end_j = (j + 1) * timesteps
            row.append(cov[start_i:end_i, start_j:end_j])
        block_cov.append(row)

    return ts_list, block_cov

def apply_uastl(mean, cov, periods, timesteps):
    """
    Applies UAMDS to the specified normal distributions (given as means and covariance matrices).

    Parameters
    ----------
    mean : list
        Vectors that resemble the mean of the normal distributions
    cov : list
        Matrix that resemble the covariance of the normal distributions
    periods : list
        Periods for STL
    timesteps : int
        The time steps of the time series

    Returns
    -------
    dict
        dictionary containing the results:
        ::
            ['means']: list of projected means
            ['covs']: list of projected covariances
    """

    n = timesteps
    l = len(periods)
    robust = False
    n_o = 1
    n_i = 2
    n_t = 5
    n_s = [np.maximum(2 * (np.ceil(n / p).astype(int) // 2) + 1, 5) for p in periods]
    n_l = [np.maximum(2 * (p // 2) + 1, 5) for p in periods]
    n_t = [2 * (int((1.5 * p) / (1 - 1.5 / max(np.ceil(n / p), 5))) // 2) + 1 for p in periods]
    post_smoothing_seasonal = True
    post_smoothing_trend = True
    post_smoothing_trend_n = 5
    post_smoothing_seasonal_n = [np.maximum(2 * (p // 2 // 2) + 1, 5) for p in periods]
    post_smoothing_trend_n = np.maximum(2 * (n // 2 // 2) + 1, 5)

    n_hat = (3 + l) * n
    mu_new = np.zeros((n_hat, 1))
    mu_new[:n] = np.array(mean).reshape(-1, 1)
    sigma_new = np.zeros((n_hat, n_hat))
    sigma_new[:n, :n] = cov

    a_hat_global = np.eye(n_hat)
    weights = np.ones(n)

    for outer_loop in range(n_o):
        a_hat = np.eye(n_hat)
        for inner_loop in range(n_i):

            for k in range(l):
                a_delta_t = np.zeros((3+l, 3+l))
                a_delta_t[2+k, 0] = 1
                a_delta_t[2+k, 1:2+k] = -1
                a_delta_t[2+k, 2+k] = 0
                
                e_ext = np.zeros((n+2*periods[k], n))
                e_ext[periods[k]:periods[k]+n, :n] = np.eye(n)
                e_ext[:periods[k], :periods[k]] = np.eye(periods[k])
                e_ext[-periods[k]:, -periods[k]:] = np.eye(periods[k])

                e_split = np.zeros((n+2*periods[k], n+2*periods[k]))
                b_ns = np.zeros((n+2*periods[k], n+2*periods[k]))
                indx = 0
                for i in range(0, periods[k]):
                    cycle_subseries = np.arange(i-periods[k], i+np.floor((n-i -1)/periods[k])*periods[k]+periods[k] + 1, periods[k]) + periods[k]
                    len_cs = len(cycle_subseries)
                    cycle_subseries = cycle_subseries.astype(int)
                    cycle_subseries_weights = weights[i::periods[k]]
                    f_e = np.array([cycle_subseries_weights[0]])
                    l_e = np.array([cycle_subseries_weights[-1]])
                    b_ns[indx:indx+len_cs, indx:indx+len_cs] = _loessmtx(len_cs, n_s[k], 2, np.concatenate((f_e, cycle_subseries_weights, l_e)))
                    e_split[indx:indx+len_cs, cycle_subseries] = np.eye(len_cs)
                    indx += len_cs

                h = 3 * np.arange(periods[k]+1).reshape(-1, 1)
                h[[0, -1]] += np.array([1, -2]).reshape(-1, 1)
                h = np.concatenate((h, h[-2::-1])) / (periods[k]**2 * 3)

                a_l = _convmtx(h, n)
                b_nl = _loessmtx(n, n_l[k], 1)

                p_1n = np.zeros((n, n+2*periods[k]))
                p_1n[:, periods[k]:periods[k]+n] = np.eye(n)

                a_p = (p_1n - b_nl @ a_l.T) @ e_split.T @ b_ns @ e_split @ e_ext

                a_s_id = np.eye(3+l)
                a_s_id[2+k, 2+k] = 0

                a_hat = (np.kron(a_s_id, np.eye(n)) + np.kron(a_delta_t, a_p)) @ a_hat

            a_t_dash = np.zeros((3+l, 3+l))
            a_t_dash[1, 0] = 1
            a_t_dash[1, 2:2+l] = -1
            a_t_dash[1, 1] = 0
            a_t_id = np.eye(3+l)
            a_t_id[1, 1] = 0
            a_hat = (np.kron(a_t_id, np.eye(n)) + np.kron(a_t_dash, _loessmtx(n, n_t[0], 1))) @ a_hat

        if post_smoothing_seasonal:
            for k in range(l):
                a_s_post = np.zeros((l+3, l+3))
                a_s_post[2+k, 2+k] = 1
                a_hat = (np.kron(np.eye(l+3) - a_s_post, np.eye(n)) + np.kron(a_s_post, _loessmtx(n, post_smoothing_seasonal_n[k], 2))) @ a_hat

        if post_smoothing_trend:
            a_hat = (np.kron(a_t_id, np.eye(n)) + np.kron(a_t_dash, _loessmtx(n, post_smoothing_trend_n, 2))) @ a_hat

        tmpmtx = np.eye(l+3)
        tmpmtx[l+3-1, 0] = 1
        tmpmtx[l+3-1, 1:l+2] = -1
        tmpmtx[l+3-1, l+3-1] = 0
        a_hat = np.kron(tmpmtx, np.eye(n)) @ a_hat
        mu_new = a_hat @ mu_new
        sigma_new = a_hat @ sigma_new @ a_hat.T
        a_hat_global = a_hat @ a_hat_global

        if robust and outer_loop < n_o - 1:
            r = np.random.multivariate_normal(mu_new.flatten(), sigma_new, n)
            r = r[:, -n:]
            h = 6 * np.median(np.abs(r), axis=0)
            u = np.abs(r) / h
            u2 = (1 - u**2)**2
            u2[u > 1] = 0
            weights = np.mean(u2, axis=0)

    return {'means': mu_new, 'covs': sigma_new}


def uastl(timeseries : TimeSeries, periods : list, seed: int = 0):
    """
    Applies the Uncertainty-Aware Seasonal-Trend Decomposition based on Loess for Gaussian distributed data.

    Parameters
    ----------
    timeseries : Timeseries object
        An instance of the TimeSeries class, which represents a univariate time series.
    periods : list
        Periods of STL
    seed : int
        Set the random seed for the initialization, 0 by default

    Returns
    -------
    CorrelatedDistributions object
        List of time series or distributions with covariance matrix
    """
    try:
        np.random.seed(seed)
        mean = timeseries.mean()
        cov = timeseries.cov()
        result = apply_uastl(mean, cov, periods, len(timeseries.timesteps))
        ts_list, block_cov = _decompose_distribution(result, len(timeseries.timesteps), len(periods))
        corr_timeseries = CorrelatedDistributions(ts_list, block_cov)
        return corr_timeseries
    except Exception as e:
        raise Exception(f'Something went wrong. Did you input normal distributions? Exception:{e}')
