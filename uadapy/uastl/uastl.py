import numpy as np
from scipy.linalg import kron
from dataclasses import dataclass

@dataclass
class uncertain_data:
    mu: np.ndarray
    sigma: np.ndarray
    samples: np.ndarray
    cor_mat: np.ndarray
    a_hat: np.ndarray

@dataclass
class options:
    robust: bool = False
    n_o: int = 1
    n_i: int = 2
    n_s: int = None #np.ndarray = None 
    n_l: int = None #np.ndarray = None
    n_t: int = None
    post_smoothing_seasonal: bool = True
    post_smoothing_seasonal_n: int = None #np.ndarray = None
    post_smoothing_trend: bool = True
    post_smoothing_trend_n: int = None

def convmtx(h, n):
    """
    Create a transposed convolution matrix.

    Parameters:
        h (array-like): The impulse response vector.
        n (int): The number of columns of the transposed convolution matrix.

    Returns:
        numpy.ndarray: The transposed convolution matrix.
    """
    h = np.array(h)
    m = len(h)
    H = np.zeros((m + n - 1, n))

    # Fill the transposed convolution matrix
    for i in range(m + n - 1):
        for j in range(n):
            if i - j >= 0 and i - j < m:
                H[i, j] = h[i - j]
    
    return H

def loessmtx(a, s, d, omega=None):
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

def uastl(X: uncertain_data, p, opts: options):
    n = len(X.mu)
    L = 1 #len(p)

    opts.n_s = np.maximum(2 * (np.ceil(n / p).astype(int) // 2) + 1, 5)
    opts.n_l = np.maximum(2 * (p // 2) + 1, 5)
    opts.n_t = 2 * (int((1.5 * p) / (1 - 1.5 / max(np.ceil(n / p), 5))) // 2) + 1 #p[-1]
    opts.post_smoothing_seasonal_n = np.maximum(2 * (p // 2 // 2) + 1, 5)
    opts.post_smoothing_trend_n = np.maximum(2 * (n // 2 // 2) + 1, 5)

    print("#################################################################################")
    print("##### Uncertainty-Aware Seasonal-Trend Decomposition Based on Loess (UASTL) #####")
    print("#################################################################################")
    q = []
    q.append(p)
    x_data = uncertain_data(mu=X.mu.copy(), sigma=X.sigma.copy(), samples=X.samples.copy(), cor_mat=X.cor_mat.copy(), a_hat=X.a_hat.copy())
    x_hat = uncertain_data(mu=X.mu.copy(), sigma=X.sigma.copy(), samples=X.samples.copy(), cor_mat=X.cor_mat.copy(), a_hat=X.a_hat.copy())


    n_hat = (3 + L) * n
    x_hat.mu = np.zeros((n_hat, 1))
    x_hat.mu[:n] = x_data.mu.reshape(-1, 1)
    x_hat.sigma = np.zeros((n_hat, n_hat))
    x_hat.sigma[:n, :n] = x_data.sigma

    a_hat_global = np.eye(n_hat)
    weights = np.ones(n)

    for outer_loop in range(opts.n_o):
        a_hat = np.eye(n_hat)
        for inner_loop in range(opts.n_i):

            for k in range(L):
                # Line 7: Update a_hat regarding seasonal trends --- Steps 1-4
                
                # ------- Step 1: detrending()
                a_delta_t = np.zeros((3+L, 3+L))
                a_delta_t[2+k, 0] = 1
                a_delta_t[2+k, 1] = -1 # 1+k+1
                a_delta_t[2+k, 2+k] = 0
                
                # ------- Step 2: cycle_subseries_smoothing(p_k, n_s, omega)
                e_ext = np.zeros((n+2*q[k], n))
                e_ext[q[k]:q[k]+n, :n] = np.eye(n)
                e_ext[:q[k], :q[k]] = np.eye(q[k])
                e_ext[-q[k]:, -q[k]:] = np.eye(q[k])


                e_split = np.zeros((n+2*q[k], n+2*q[k]))
                b_ns = np.zeros((n+2*q[k], n+2*q[k]))
                indx = 0
                for i in range(0, q[k]):
                    cycle_subseries = np.arange(i-q[k], i+np.floor((n-i -1)/q[k])*q[k]+q[k] + 1, q[k]) + q[k]
                    len_cs = len(cycle_subseries)
                    cycle_subseries = cycle_subseries.astype(int)  # Ensure integer indices
                    cycle_subseries_weights = weights[i::q[k]]
                    f_e = np.array([cycle_subseries_weights[0]])
                    l_e = np.array([cycle_subseries_weights[-1]])
                    b_ns[indx:indx+len_cs, indx:indx+len_cs] = loessmtx(len_cs, opts.n_s, 2, np.concatenate((f_e, cycle_subseries_weights, l_e)))
                    e_split[indx:indx+len_cs, cycle_subseries] = np.eye(len_cs)
                    indx += len_cs

                # ------- Step 3: cycle_subseries_low_pass_filtering(p_k, n_l)
                h = 3 * np.arange(q[k]+1).reshape(-1, 1)
                h[[0, -1]] += np.array([1, -2]).reshape(-1, 1)
                h = np.concatenate((h, h[-2::-1])) / (q[k]**2 * 3)

                a_l = convmtx(h, n)
                b_nl = loessmtx(n, opts.n_l, 1)
                

                # ------- Step 4: cycle_subseries_detrending()
                p_1n = np.zeros((n, n+2*q[k]))
                p_1n[:, q[k]:q[k]+n] = np.eye(n)

                a_p = (p_1n - b_nl @ a_l.T) @ e_split.T @ b_ns @ e_split @ e_ext

                # update STL matrix a_hat regarding seasonal trends
                a_s_id = np.eye(3+L)
                a_s_id[2+k, 2+k] = 0

                a_hat = (np.kron(a_s_id, np.eye(n)) + np.kron(a_delta_t, a_p)) @ a_hat

            a_t_dash = np.zeros((3+L, 3+L))
            a_t_dash[1, 0] = 1
            a_t_dash[1, 2] = -1 # 2:2 + L
            a_t_dash[1, 1] = 0
            a_t_id = np.eye(3+L)
            a_t_id[1, 1] = 0
            a_hat = (np.kron(a_t_id, np.eye(n)) + np.kron(a_t_dash, loessmtx(n, opts.n_t, 1))) @ a_hat

        if opts.post_smoothing_seasonal:
            for k in range(L):
                a_s_post = np.zeros((L+3, L+3))
                a_s_post[2+k, 2+k] = 1
                a_hat = (kron(np.eye(L+3) - a_s_post, np.eye(n)) + kron(a_s_post, loessmtx(n, opts.post_smoothing_seasonal_n, 2))) @ a_hat

        if opts.post_smoothing_trend:
            a_hat = (kron(a_t_id, np.eye(n)) + kron(a_t_dash, loessmtx(n, opts.post_smoothing_trend_n, 2))) @ a_hat

        tmpmtx = np.eye(L+3)
        tmpmtx[L+3-1, 0] = 1
        tmpmtx[L+3-1, 1:L+2] = -1
        tmpmtx[L+3-1, L+3-1] = 0
        a_hat = kron(tmpmtx, np.eye(n)) @ a_hat
        x_hat.mu = a_hat @ x_hat.mu
        x_hat.sigma = a_hat @ x_hat.sigma @ a_hat.T
        a_hat_global = a_hat @ a_hat_global

        if opts.robust and outer_loop < opts.n_o - 1:
            r = np.random.multivariate_normal(x_hat.mu, x_hat.sigma, n)
            r = r[:, -n:]
            h = 6 * np.median(np.abs(r), axis=0)
            u = np.abs(r) / h
            u2 = (1 - u**2)**2
            u2[u > 1] = 0
            weights = np.mean(u2, axis=0)

    print("#################################################################################")
    print("##### UASTL DONE #####")
    print("#################################################################################")
    
    return x_hat, a_hat_global
