import numpy as np
from scipy.linalg import kron
from dataclasses import dataclass

@dataclass
class UncertainData:
    mu: np.ndarray
    Sigma: np.ndarray
    samples: np.ndarray
    corMat: np.ndarray
    AHat: np.ndarray

@dataclass
class Options:
    robust: bool = False
    n_o: int = 1
    n_i: int = 2
    n_s: int = None #np.ndarray = None 
    n_l: int = None #np.ndarray = None
    n_t: int = None
    postSmoothingSeasonal: bool = True
    postSmoothingSeasonal_n: int = None #np.ndarray = None
    postSmoothingTrend: bool = True
    postSmoothingTrend_n: int = None

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
        aTild = a[idx] - a[i]
        
        # Compute scaling
        aTildAbs = np.abs(aTild)
        scaling = (1 - (aTildAbs / np.max(aTildAbs))**3)**3
        
        # Define Vandermonde matrix
        V = np.vander(aTild.flatten(), N=d+1, increasing=True)
        
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

def uastl(X: UncertainData, p, opts: Options):
    n = len(X.mu)
    L = 1 #len(p)

    opts.n_s = np.maximum(2 * (np.ceil(n / p).astype(int) // 2) + 1, 5)
    opts.n_l = np.maximum(2 * (p // 2) + 1, 5)
    opts.n_t = 2 * (int((1.5 * p) / (1 - 1.5 / max(np.ceil(n / p), 5))) // 2) + 1 #p[-1]
    opts.postSmoothingSeasonal_n = np.maximum(2 * (p // 2 // 2) + 1, 5)
    opts.postSmoothingTrend_n = np.maximum(2 * (n // 2 // 2) + 1, 5)

    print("#################################################################################")
    print("##### Uncertainty-Aware Seasonal-Trend Decomposition Based on Loess (UASTL) #####")
    print("#################################################################################")
    q = []
    q.append(p)
    X_data = UncertainData(mu=X.mu.copy(), Sigma=X.Sigma.copy(), samples=X.samples.copy(), corMat=X.corMat.copy(), AHat=X.AHat.copy())
    XHat = UncertainData(mu=X.mu.copy(), Sigma=X.Sigma.copy(), samples=X.samples.copy(), corMat=X.corMat.copy(), AHat=X.AHat.copy())


    nHat = (3 + L) * n
    XHat.mu = np.zeros((nHat, 1))
    XHat.mu[:n] = X_data.mu.reshape(-1, 1)
    XHat.Sigma = np.zeros((nHat, nHat))
    XHat.Sigma[:n, :n] = X_data.Sigma

    AHatGlobal = np.eye(nHat)
    weights = np.ones(n)

    for outer_loop in range(opts.n_o):
        AHat = np.eye(nHat)
        for inner_loop in range(opts.n_i):

            for k in range(L):
                # Line 7: Update AHat regarding seasonal trends --- Steps 1-4
                
                # ------- Step 1: detrending()
                A_Deltat = np.zeros((3+L, 3+L))
                A_Deltat[2+k, 0] = 1
                A_Deltat[2+k, 1] = -1 # 1+k+1
                A_Deltat[2+k, 2+k] = 0
                
                # ------- Step 2: cycle_subseries_smoothing(p_k, n_s, omega)
                E_ext = np.zeros((n+2*q[k], n))
                E_ext[q[k]:q[k]+n, :n] = np.eye(n)
                E_ext[:q[k], :q[k]] = np.eye(q[k])
                E_ext[-q[k]:, -q[k]:] = np.eye(q[k])


                E_split = np.zeros((n+2*q[k], n+2*q[k]))
                B_ns = np.zeros((n+2*q[k], n+2*q[k]))
                indx = 0
                for i in range(0, q[k]):
                    cycleSubseries = np.arange(i-q[k], i+np.floor((n-i -1)/q[k])*q[k]+q[k] + 1, q[k]) + q[k]
                    lenCS = len(cycleSubseries)
                    cycleSubseries = cycleSubseries.astype(int)  # Ensure integer indices
                    cycleSubseriesWeights = weights[i::q[k]]
                    f_e = np.array([cycleSubseriesWeights[0]])
                    l_e = np.array([cycleSubseriesWeights[-1]])
                    B_ns[indx:indx+lenCS, indx:indx+lenCS] = loessmtx(lenCS, opts.n_s, 2, np.concatenate((f_e, cycleSubseriesWeights, l_e)))
                    E_split[indx:indx+lenCS, cycleSubseries] = np.eye(lenCS)
                    indx += lenCS

                # ------- Step 3: cycle_subseries_low_pass_filtering(p_k, n_l)
                h = 3 * np.arange(q[k]+1).reshape(-1, 1)
                h[[0, -1]] += np.array([1, -2]).reshape(-1, 1)
                h = np.concatenate((h, h[-2::-1])) / (q[k]**2 * 3)

                A_L = convmtx(h, n)
                B_nl = loessmtx(n, opts.n_l, 1)
                

                # ------- Step 4: cycle_subseries_detrending()
                P_1n = np.zeros((n, n+2*q[k]))
                P_1n[:, q[k]:q[k]+n] = np.eye(n)

                A_p = (P_1n - B_nl @ A_L.T) @ E_split.T @ B_ns @ E_split @ E_ext

                # update STL matrix AHat regarding seasonal trends
                A_S_id = np.eye(3+L)
                A_S_id[2+k, 2+k] = 0

                AHat = (np.kron(A_S_id, np.eye(n)) + np.kron(A_Deltat, A_p)) @ AHat

            A_tDash = np.zeros((3+L, 3+L))
            A_tDash[1, 0] = 1
            A_tDash[1, 2] = -1 # 2:2 + L
            A_tDash[1, 1] = 0
            A_T_id = np.eye(3+L)
            A_T_id[1, 1] = 0
            AHat = (np.kron(A_T_id, np.eye(n)) + np.kron(A_tDash, loessmtx(n, opts.n_t, 1))) @ AHat

        if opts.postSmoothingSeasonal:
            for k in range(L):
                A_S_post = np.zeros((L+3, L+3))
                A_S_post[2+k, 2+k] = 1
                AHat = (kron(np.eye(L+3) - A_S_post, np.eye(n)) + kron(A_S_post, loessmtx(n, opts.postSmoothingSeasonal_n, 2))) @ AHat

        if opts.postSmoothingTrend:
            AHat = (kron(A_T_id, np.eye(n)) + kron(A_tDash, loessmtx(n, opts.postSmoothingTrend_n, 2))) @ AHat

        tmpmtx = np.eye(L+3)
        tmpmtx[L+3-1, 0] = 1
        tmpmtx[L+3-1, 1:L+2] = -1
        tmpmtx[L+3-1, L+3-1] = 0
        AHat = kron(tmpmtx, np.eye(n)) @ AHat
        XHat.mu = AHat @ XHat.mu
        XHat.Sigma = AHat @ XHat.Sigma @ AHat.T
        AHatGlobal = AHat @ AHatGlobal

        if opts.robust and outer_loop < opts.n_o - 1:
            r = np.random.multivariate_normal(XHat.mu, XHat.Sigma, n)
            r = r[:, -n:]
            h = 6 * np.median(np.abs(r), axis=0)
            u = np.abs(r) / h
            u2 = (1 - u**2)**2
            u2[u > 1] = 0
            weights = np.mean(u2, axis=0)

    print("#################################################################################")
    print("##### UASTL DONE #####")
    print("#################################################################################")
    
    return XHat, AHatGlobal
