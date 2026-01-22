import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
from uadapy import Distribution
from uadapy.plotting import utils

# -----------------------------
# Attraction precompute with parallelization
# -----------------------------
@njit(cache=True, parallel=True)
def precompute_attraction_field(u, k=1.0):
    H, W = u.shape
    F_field = np.zeros((H, W, 2), dtype=np.float64)

    for y0 in prange(H):  # parallelize rows
        for x0 in range(W):
            Fy, Fx = 0.0, 0.0
            for y in range(H):
                for x in range(W):
                    if y == y0 and x == x0:
                        continue
                    weight = 1.0 - u[y, x]
                    if weight == 0.0:
                        continue
                    dy = y - y0
                    dx = x - x0
                    dist = (dy*dy + dx*dx)**0.5
                    if dist > 1e-12:
                        Fy += k * weight * (dy / (dist*dist))
                        Fx += k * weight * (dx / (dist*dist))
            F_field[y0, x0, 0] = Fy
            F_field[y0, x0, 1] = Fx
    return F_field



# -----------------------------
# Bilinear interpolation (Numba)
# -----------------------------
@njit(cache=True)
def bilinear_interpolate(F_field, pos):
    H, W, _ = F_field.shape
    y, x = pos
    y0 = int(np.floor(y))
    x0 = int(np.floor(x))
    y1 = min(y0 + 1, H - 1)
    x1 = min(x0 + 1, W - 1)
    dy = y - y0
    dx = x - x0

    F00 = F_field[y0, x0]
    F01 = F_field[y0, x1]
    F10 = F_field[y1, x0]
    F11 = F_field[y1, x1]

    F0 = F00 * (1 - dx) + F01 * dx
    F1 = F10 * (1 - dx) + F11 * dx
    F = F0 * (1 - dy) + F1 * dy
    return F

# -----------------------------
# Repulsion with parallelization
# -----------------------------
@njit(cache=True, parallel=True, fastmath=True)
def repulsion_force_all(P, k=1.0):
    N = P.shape[0]
    F = np.zeros_like(P, dtype=np.float64)
    for i in prange(N):  # parallel outer loop
        Fy, Fx = 0.0, 0.0
        for j in range(N):
            # if i != j:
            #     dy = P[j, 0] - P[i, 0]
            #     dx = P[j, 1] - P[i, 1]
            #     dist = np.sqrt(dy*dy + dx*dx)
            #     if dist > 1e-12:
            #         Fy += -k * (dy / (dist*dist))
            #         Fx += -k * (dx / (dist*dist))
            
            # omiting i != j check, only slows down, but dy,dx is zero anyway
            dy = P[j, 0] - P[i, 0]
            dx = P[j, 1] - P[i, 1]
            dist2 = dy*dy + dx*dx + 1e-8
            Fy += -k * (dy / dist2)
            Fx += -k * (dx / dist2)


        F[i, 0] = Fy
        F[i, 1] = Fx
    return F

def init_particles(u, N=None, seed=0):
    """
    Initialize particle positions with probability proportional to 1 - u(x).
    """
    rng = np.random.default_rng(seed)
    H, W = u.shape
    weights = (1 - u).ravel()
    weights /= weights.sum()
    coords = np.array([(y, x) for y in range(H) for x in range(W)])
    
    if N is None:
        N = int(round((1 - u).sum()))  # preserve mean gray value
    
    idx = rng.choice(len(coords), size=N, replace=True, p=weights)
    P = coords[idx].astype(float)  # positions are floats
    return P

@njit(cache=True, parallel=True)
def attraction_force_all(P, F_field):
    #return np.array([bilinear_interpolate(F_field, p) for p in P])
    forces = np.zeros_like(P)
    for i in prange(P.shape[0]):
        forces[i] = bilinear_interpolate(F_field, P[i])
    return forces

def step_particles(P, F_field, u, tau=0.1, k=1.0):
    """
    Particle update step using precomputed attraction field.
    """
    H, W = u.shape
    # Attraction via bilinear interpolation
    F_a = attraction_force_all(P,F_field)
    # Repulsion
    F_r = repulsion_force_all(P, k)
    # Update
    P_new = P + tau * (F_a + F_r)
    # Clamp
    P_new[:, 0] = np.clip(P_new[:, 0], 0, H-1)
    P_new[:, 1] = np.clip(P_new[:, 1], 0, W-1)
    return P_new


def simulate_halftoning(u, steps=300, tau=0.1, k=1.0, seed=0,
                        shake_interval=10, shake_strength=1.0, shake_stop=200, tol=0.001):
    rng = np.random.default_rng(seed)
    P = init_particles(u, seed=seed)
    F_field = precompute_attraction_field(u, k=k)
    H, W = u.shape


    for t in range(steps):
        P_new = step_particles(P, F_field, u, tau=tau, k=k)
        displacement = np.mean(np.linalg.norm(P_new - P, axis=1))
        P = P_new

        # Shaking only before shake_stop
        if (t+1) % shake_interval == 0 and (t+1) <= shake_stop:
            factor = max(0.0, 1.0 - (t / shake_stop))
            jitter = rng.uniform(-shake_strength, shake_strength, size=P.shape) * factor

            P += jitter
            P[:, 0] = np.clip(P[:, 0], 0, H-1)
            P[:, 1] = np.clip(P[:, 1], 0, W-1)

        #print(displacement)
        # # Convergence check
        if displacement < tol:
            print(f"Converged at iteration {t+1}, avg displacement={displacement:.5f}")
            break

    return P

def plot_stipples(distributions,
                  n_samples=3000,
                  resolution=128,
                  seed=55,
                  show_plot=False,
                  steps=600):

    samples = distributions[0].sample(n_samples, seed)
    x = np.linspace(np.min(samples[:, 0]), np.max(samples[:, 0]), resolution)
    y = np.linspace(np.min(samples[:, 1]), np.max(samples[:, 1]), resolution)
    X, Y = np.meshgrid(x, y)
    coords = np.dstack((X, Y)).reshape(-1, 2)

    # PDF on grid
    Z = distributions[0].pdf(coords).reshape(X.shape)

    # Normalize to [0,1]
    Z_norm = (Z - Z.min()) / (Z.max() - Z.min())
    u = 1 - Z_norm  # high pdf → black (attraction)

    # Run halftoning
    P = simulate_halftoning(u, steps=steps, tau=0.1, seed=seed)

    # Map stipple indices back to real coords
    x_coords = np.interp(P[:, 1], (0, resolution-1), (x[0], x[-1]))
    y_coords = np.interp(P[:, 0], (0, resolution-1), (y[0], y[-1]))

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x_coords, y_coords, s=5, c="black", edgecolors="none")
    ax.set_title("Stippling from Distribution")
    if show_plot:
        plt.tight_layout()
        plt.show()

    return fig, ax


def main(steps = 200):
    # Example usage
    from sklearn import datasets
    from uadapy.dr import uamds

    def _load_iris():
        iris = datasets.load_iris()
        dists = []
        for c in np.unique(iris.target):
            dists.append(Distribution(iris.data[iris.target == c]))
        return dists, iris.target_names

    distribs_hi, labels = _load_iris()
    distribs_lo = uamds(distribs_hi, n_dims=2)
    fig, axs = plot_stipples(distribs_lo, n_samples=3000, resolution=256, seed=55, show_plot=False, steps=steps)
    return fig, axs

main()