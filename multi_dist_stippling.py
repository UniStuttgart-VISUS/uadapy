import numpy as np
import glasbey as gb
from collections import deque
import matplotlib.pyplot as plt
from numba import njit, prange
from uadapy import Distribution
from uadapy.plotting import utils

def _init_particles(u, seed=55, stippling_scale=1.0):
    """
    Initialize particle (stipple) positions from a tone image u using importance sampling.

    In electrostatic halftoning, the expected number of particles is proportional to
    the total 'ink demand' ∑(1 - u). This initializer samples pixel coordinates with
    probability ∝ (1 - u) and sets the particle count to round(∑(1 - u) * stippling_scale).

    Parameters
    ----------
    u : np.ndarray (H, W), dtype=float
        Tone image with values in [0, 1], where 0 = dark (high ink), 1 = light.
    seed : int, optional
        Random seed for reproducibility. Default is 55.
    stippling_scale : float, optional
        Multiplicative scale applied to the baseline particle count ∑(1 - u).
        Values < 1 reduce the number of stipples; values > 1 increase it. Default is 1.0.

    Returns
    -------
    np.ndarray, shape (N, 2), dtype=float
        Initial particle coordinates in image (row=y, col=x) pixel units.
        Returns an empty array if no particles are requested.
    """

    H, W = u.shape
    rng = np.random.default_rng(seed)
    weights = (1 - u).ravel()
    s = weights.sum()
    if s <= 0:
        return np.empty((0,2), dtype=float)
    weights /= s
    coords = np.array([(y, x) for y in range(H) for x in range(W)])

    N = int(round((1 - u).sum() * stippling_scale))

    if N <= 0:
        return np.empty((0,2), dtype=float)
    idx = rng.choice(len(coords), size=N, replace=True, p=weights)
    return coords[idx].astype(float)

@njit(cache=True, parallel=True, fastmath=True)
def _precompute_attraction_field(u, k=1.0):
    """
    Precompute the attractive image force field F(x) induced by the tone image u.

    Each pixel x contributes a positive 'image charge' of magnitude (1 - u[x]).
    For a probe location p, the attractive force is the superposition of 2-D Coulomb-like
    forces with 1/r decay (discretized on the grid), pointing from p towards x.

    Parameters
    ----------
    u : np.ndarray (H, W), dtype=float
        Tone image (0 = dark, 1 = light). Darker pixels attract particles more strongly.
    k : float, optional
        Global proportionality constant (bundles Coulomb constant and unit charges) used
        in the discrete 2-D 1/r interaction. Default is 1.0.

    Returns
    -------
    np.ndarray, shape (H, W, 2), dtype=float
        Vector field of attractive image forces for sampling via bilinear interpolation.
        The last axis stores (Fy, Fx) in pixel units.
    """
    H, W = u.shape
    F_field = np.zeros((H, W, 2), dtype=np.float64)
    rho = 1.0 - u  # image "charge"
    eps = 1e-12

    for y0 in prange(H):
        for x0 in range(W):
            Fy = 0.0
            Fx = 0.0
            for y in range(H):
                for x in range(W):
                    if y == y0 and x == x0:
                        continue
                    weight = rho[y, x]
                    if weight == 0.0:
                        continue
                    dy = float(y - y0)
                    dx = float(x - x0)
                    dist2 = dy*dy + dx*dx
                    if dist2 <= eps:
                        continue
                    invr2 = 1.0 / (dist2 + eps)
                    Fy += k * weight * dy * invr2
                    Fx += k * weight * dx * invr2
            F_field[y0, x0, 0] = Fy
            F_field[y0, x0, 1] = Fx
    return F_field

@njit(cache=True)
def _bilinear_interpolate(F_field, pos):
    """
    Sample a vector field at subpixel precision using bilinear interpolation.

    Parameters
    ----------
    F_field : np.ndarray (H, W, 2), dtype=float
        Precomputed vector field (e.g., attractive image force) on the pixel grid.
    pos : array-like, shape (2,)
        Continuous position (y, x) in pixel coordinates at which to interpolate F_field.

    Returns
    -------
    np.ndarray, shape (2,), dtype=float
        Interpolated vector value (Fy, Fx) at the given subpixel position.
    """

    H, W, _ = F_field.shape
    y, x = pos
    y0 = int(np.floor(y)); x0 = int(np.floor(x))
    y1 = min(y0 + 1, H - 1); x1 = min(x0 + 1, W - 1)
    dy = y - y0; dx = x - x0
    F00 = F_field[y0, x0]
    F01 = F_field[y0, x1]
    F10 = F_field[y1, x0]
    F11 = F_field[y1, x1]
    F0 = F00 * (1 - dx) + F01 * dx
    F1 = F10 * (1 - dx) + F11 * dx
    return F0 * (1 - dy) + F1 * dy


@njit(cache=True, parallel=True, fastmath=True)
def _repulsion_force_multi(P_list, I, pair_scale=1.0):
    """
    Compute cross-class electrostatic repulsion between all particles (2-D 1/r law).

    For all classes, flattens particle arrays, applies an interaction matrix I to weight
    inter-class and intra-class repulsion, and returns per-class force vectors. The overall
    strength is further scaled by 'pair_scale' which corresponds to q^2 when per-particle
    charge is scaled (e.g., due to stipple-count changes).

    Parameters
    ----------
    P_list : list[np.ndarray]
        List of particle arrays per class; each has shape (Nc, 2) in (y, x) pixels.
    I : np.ndarray (C, C), dtype=float
        Symmetric interaction matrix; I[c, d] scales repulsion between class c and d.
    pair_scale : float, optional
        Global multiplicative factor for pairwise repulsion, typically (q_scale**2),
        where q_scale is the per-particle charge scaling. Default is 1.0.

    Returns
    -------
    list[np.ndarray]
        List of repulsive force arrays per class, each of shape (Nc, 2) in (Fy, Fx).
    """
    C = len(P_list)
    sizes = np.empty(C+1, dtype=np.int64)
    sizes[0] = 0
    for c in range(C):
        sizes[c+1] = sizes[c] + P_list[c].shape[0]
    N = sizes[-1]

    P_all = np.zeros((N, 2), dtype=np.float64)
    cls = np.zeros(N, dtype=np.int64)
    for c in range(C):
        n0, n1 = sizes[c], sizes[c+1]
        Pc = P_list[c]
        P_all[n0:n1, 0] = Pc[:, 0]
        P_all[n0:n1, 1] = Pc[:, 1]
        cls[n0:n1] = c

    F_all = np.zeros_like(P_all, dtype=np.float64)
    eps = 1e-12

    for i in prange(N):
        yi, xi = P_all[i, 0], P_all[i, 1]
        ci = cls[i]
        row_I = I[ci]
        Fy = 0.0
        Fx = 0.0
        for j in range(N):
            if i == j:
                continue
            dy = P_all[j, 0] - yi
            dx = P_all[j, 1] - xi
            dist2 = dy*dy + dx*dx
            if dist2 <= eps:
                continue
            invr2 = 1.0 / (dist2 + eps)
            w = row_I[cls[j]] * pair_scale
            Fy += -w * dy * invr2
            Fx += -w * dx * invr2
        F_all[i, 0] = Fy
        F_all[i, 1] = Fx

    F_list = []
    for c in range(C):
        F_list.append(F_all[sizes[c]:sizes[c+1], :])
    return F_list

def _step_particles_multi(P_list, F_fields, u_list, tau=0.1, I=None, q_scale=1.0, max_step=0.5):
    """
    Advance all particles by one explicit-Euler step with step-length clamping.

    Builds the mixed attractive image fields per class (sum_d I[c,d] F_d), samples them
    at current particle positions (scaled by q_scale ∝ charge), adds cross-class repulsion
    (scaled by q_scale^2), and performs a clamped update:
        p_{k+1} = p_k + clamp( τ * (F_attr + F_rep), max_step )
    Positions are kept inside the image domain.

    Parameters
    ----------
    P_list : list[np.ndarray]
        Current particle positions per class; each array is (Nc, 2) in (y, x) pixels.
    F_fields : list[np.ndarray]
        Precomputed attractive image force fields per class, each (H, W, 2).
    u_list : list[np.ndarray]
        Tone images per class (only used for shape / bounds); each (H, W).
    tau : float, optional
        Artificial time step for explicit Euler integration. Default is 0.1.
    I : np.ndarray or None, optional
        Interaction matrix (C x C) weighting field mixing and repulsion. Defaults to ones.
    q_scale : float, optional
        Per-particle charge scaling applied to attraction (∝ q) and repulsion (via q^2).
        For stipple-count changes, q_scale = 1 / stippling_scale. Default is 1.0.
    max_step : float, optional
        Maximum displacement per particle and iteration, in pixel units (L2 norm).
        Prevents overshoot and improves stability. Default is 0.5.

    Returns
    -------
    list[np.ndarray]
        New particle positions per class after one iteration; shapes match inputs.
    """

    C = len(P_list)
    H, W = u_list[0].shape
    if I is None:
        I = np.ones((C, C), dtype=np.float64)

    Fmix = []
    for c in range(C):
        Fm = np.zeros_like(F_fields[0])
        for d in range(C):
            Fm += I[c, d] * F_fields[d]
        Fmix.append(Fm)

    F_attr = []
    for c in range(C):
        P = P_list[c]
        F = np.empty_like(P, dtype=np.float64)
        for i in range(P.shape[0]):
            F[i] = _bilinear_interpolate(Fmix[c], P[i]) * q_scale
        F_attr.append(F)

    F_rep = _repulsion_force_multi(P_list, I, pair_scale=(q_scale*q_scale))

    P_next = []
    for c in range(C):
        P = P_list[c]
        F_tot = F_attr[c] + F_rep[c]
        Pn = np.empty_like(P)
        for i in range(P.shape[0]):
            step = tau * F_tot[i]
            nrm = (step[0]*step[0] + step[1]*step[1])**0.5
            if nrm > max_step:
                step *= (max_step / (nrm + 1e-12))
            Pn[i,0] = min(max(P[i,0] + step[0], 0.0), H-1.0)
            Pn[i,1] = min(max(P[i,1] + step[1], 0.0), W-1.0)
        P_next.append(Pn)
    return P_next

def _simulate_halftoning_multi(u_list, steps=600, tau=0.1, seed=55,
                              shake_interval=10, shake_strength=1,
                              shake_stop=200, tol=1e-3, I=None,
                              stippling_scale=1.0,
                              max_step_start=0.5,
                              max_step_min=0.01,
                              clamp_decay_strength=3.0,
                              enable_plateau_stop=False,
                              plateau_start_factor=2.0,
                              plateau_window=10,
                              plateau_abs_change=2e-5
                              ):
    """
    Run multi-class electrostatic halftoning with adaptive step clamp and optional plateau stop.

    Parameters
    ----------
    u_list : list[np.ndarray]
        List of C tone images (0 = dark, 1 = light) defining the desired densities per class.
    steps : int, optional
        Maximum number of Euler iterations.
    tau : float, optional
        Artificial time step size for the Euler update.
    seed : int, optional
        RNG seed for reproducible initialization and shaking.
    shake_interval : int, optional
        Apply simulated annealing “shaking” every this many iterations.
    shake_strength : float, optional
        Amplitude of the shaking jitter (uniform in [-strength, +strength]).
    shake_stop : int, optional
        Disable shaking after this iteration index (controls annealing schedule).
    tol : float, optional
        Convergence tolerance on average per-particle displacement (pixels/iter).
        Loop terminates when displacement < tol (or plateau criterion is met).
    I : np.ndarray or None, optional
        Interaction matrix (C, C) that mixes per-class attraction fields and sets cross-class repulsion.
        If None, uses ones.
    stippling_scale : float, optional
        Scale applied to the default particle count per class; use q_scale = 1/stippling_scale
        (internally) to preserve total charge neutrality.
    max_step_start : float, optional
        Initial per-iteration step clamp (in pixels). Decays over time.
    max_step_min : float, optional
        Lower bound on the decayed step clamp (in pixels).
    clamp_decay_strength : float, optional
        Exponential decay strength for the clamp: max_step_t = max(max_step_min,
        max_step_start * exp(-clamp_decay_strength * t / shake_stop)).
    enable_plateau_stop : bool, optional
        If True, enable a simple plateau stop based on a small spread of recent displacements.
    plateau_start_factor : float, optional
        Start checking the plateau condition after plateau_start_factor * shake_stop iterations.
    plateau_window : int, optional
        Number of recent displacement values to keep for the plateau check.
    plateau_abs_change : float, optional
        Absolute spread threshold for plateau stop: if max-min over the window < threshold, stop.

    Returns
    -------
    list[np.ndarray]
        Final particle positions per class; list of C arrays (Nc, 2) in grid coordinates.
    """

    rng = np.random.default_rng(seed)
    C = len(u_list)
    H, W = u_list[0].shape
    if I is None:
        I = np.ones((C, C), dtype=np.float64)

    S = max(stippling_scale, 1e-12)
    q_scale = 1.0 / S

    P_list = [_init_particles(u_list[c], seed=seed+c, stippling_scale=S) for c in range(C)]
    F_fields = [_precompute_attraction_field(u_list[c]) for c in range(C)]

    disp_hist = deque(maxlen=plateau_window)
    plateau_start_iter = int(plateau_start_factor * max(1, shake_stop))

    for t in range(steps):
        decay = 0.0 if shake_stop <= 0 else np.exp(-clamp_decay_strength * (t / float(shake_stop)))
        max_step_t = max(max_step_min, max_step_start * decay)

        P_new = _step_particles_multi(P_list, F_fields, u_list,
                                     tau=tau, I=I, q_scale=q_scale,
                                     max_step=max_step_t)

        # --- convergence (top-K displacement mean) ---
        # collect per-particle displacement magnitudes across all classes
        if P_list:
            disp_chunks = []
            totalN = 0
            for Po, Pn in zip(P_list, P_new):
                if Po.shape[0] > 0:
                    d = Pn - Po
                    # per-particle L2 displacement
                    disp_chunks.append(np.sqrt((d * d).sum(axis=1)))
                    totalN += Po.shape[0]
            if totalN > 0:
                disp_all = np.concatenate(disp_chunks)
                k = min(10, disp_all.size)
                # take largest k efficiently
                topk = np.partition(disp_all, -k)[-k:]

                # mean of top-k displacements
                displacement_topk_mean = float(topk.mean())
            else:
                displacement_topk_mean = 0.0
        else:
            displacement_topk_mean = 0.0

        P_list = P_new

        if (t+1) % shake_interval == 0 and (t+1) <= shake_stop:
            factor = max(0.0, 1.0 - (t / shake_stop))
            for c in range(C):
                if P_list[c].shape[0] == 0:
                    continue
                jitter = rng.uniform(-shake_strength, shake_strength, size=P_list[c].shape) * factor
                P_list[c] += jitter
                P_list[c][:,0] = np.clip(P_list[c][:,0], 0, H-1)
                P_list[c][:,1] = np.clip(P_list[c][:,1], 0, W-1)

        #print(displacement_topk_mean)
        if displacement_topk_mean < tol:
            #print(t)
            break

        if enable_plateau_stop and (t+1) >= plateau_start_iter:
            disp_hist.append(displacement_topk_mean)
            if len(disp_hist) == disp_hist.maxlen:
                r = (max(disp_hist) - min(disp_hist))
                if r < plateau_abs_change:
                    break

    return P_list

def plot_stipples(distributions,
                  n_samples=3000,
                  resolution=128,
                  steps=600,
                  tau=0.1,
                  I=None,
                  stipple_size=4,
                  stippling_scale=1,
                  seed=55,
                  fig=None,
                  axs=None,
                  distrib_labels=None,
                  distrib_colors=None,
                  colorblind_safe=False,
                  show_plot=False):
    """
    Plot multi-class electrostatic halftoning (stippling) from continuous distributions.

    This function produces a multi-class stippling of 2D data by implementing the
    electrostatic halftoning model (charged particles in an attraction field induced by image
    tone) with the multi-class interaction extension:
      - A tone (or "image charge") field u_c is built for each class by rasterizing
        the class PDF onto a common grid; darker tone (1 - u_c) induces stronger attraction.
      - Particles (stipples) repel each other (Coulomb-like repulsion), and are attracted
        to dark regions (image-induced attraction).
      - A user-defined interaction matrix I (CxC) both mixes class-specific attraction fields
        (global field mixing; ∝ = I) and weights cross-class repulsion, enabling second-order
        screening / multi-class placement behavior.

    Internally, each distribution is rasterized, the attraction field per class is precomputed,
    and particles are initialized by importance sampling (1 - u_c). Particle evolution then runs
    an explicit Euler solver with (i) per-particle step clamping for stability (max displacement
    per iteration in pixel units) and (ii) an optional simple plateau stop for very small
    stipple counts. To preserve total “charge neutrality” when the number of particles is
    changed, forces are rescaled by an internal factor q_scale = 1 / stippling_scale:
      - If you halve the number of particles (stippling_scale=0.5), each particle's effective
        charge is doubled, keeping the global balance consistent with the model.

    Parameters
    ----------
    distributions : list[Distribution] or Distribution
        A list of uadapy.Distribution objects (or a single one) representing classes.
    n_samples : int, optional
        Number of samples per class used to estimate a tight common bounding box for all
        distributions.
    resolution : int, optional
        Side length (pixels) of the square raster grid used to build tone images u_c and
        their attraction fields. The solver operates in grid coordinates (pixels).
    steps : int, optional
        Maximum number of Euler iterations for the particle evolution.
    tau : float, optional
        Artificial time step for the explicit Euler integrator. Together with force magnitudes,
        this controls per-iteration displacement before clamping.
    I : np.ndarray or None, optional
        Interaction matrix of shape (C, C). Diagonal terms control intra-class repulsion and
        self-mixing; off-diagonals control cross-class repulsion and field mixing strength.
        If None, uses all-ones (uniform mixing and equal repulsion among classes).
    stipple_size : float, optional
        Matplotlib scatter size for stipples in the output figure.
    stippling_scale : float, optional
        Multiplicative scale on the default particle count per class (derived from sum(1 - u_c)).
        Must be in [0.1, 1.0] in this implementation. Internally the solver rescales forces
        with q_scale = 1 / stippling_scale to preserve total charge when the count changes.
        Example: 0.5: half as many stipples; 2x per-particle charge (via force scaling).
    seed : int, optional
        RNG seed used for particle initialization (importance sampling) and early shaking.
    fig : matplotlib.figure.Figure or None, optional
        Figure to plot into. If None, a new figure is created.
    axs : matplotlib.axes.Axes or None, optional
        Axes to plot into. If None, a new axes (or the first axes of `fig`) is used.
    distrib_labels : list[str] or None, optional
        Legend labels per class. If None, defaults to "Class i".
    distrib_colors : list or None, optional
        List of colors to use for each distribution. If None, Matplotlib Set2 and glasbey colors will be used.
    colorblind_safe : bool, optional
        If True, the plot will use colors suitable for colorblind individuals.
        Default is False.
    show_plot : bool, optional
        If True, display the plot.
        Default is False.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the stippling plot.
    matplotlib.axes.Axes
        The axes with drawn stipples.

    """

    if not isinstance(distributions, (list, tuple)):
        distributions = [distributions]

    C = len(distributions)
    enable_plateau_stop = False

    if axs is None:
        if fig is None:
            fig, axs = plt.subplots(figsize=(10, 8))
        else:
            if fig.axes is not None:
                axs = fig.axes[0]
            else:
                raise ValueError("The provided figure has no axes. Pass an Axes or create subplots first.")
    else:
        if fig is None:
            fig = axs.figure

    rng = np.random.default_rng(seed)
    all_samples = np.vstack([d.sample(n_samples, seed) for d in distributions])
    xmin, xmax = np.min(all_samples[:, 0]), np.max(all_samples[:, 0])
    ymin, ymax = np.min(all_samples[:, 1]), np.max(all_samples[:, 1])

    x = np.linspace(xmin, xmax, resolution)
    y = np.linspace(ymin, ymax, resolution)
    X, Y = np.meshgrid(x, y)
    coords = np.dstack((X, Y)).reshape(-1, 2)

    if distrib_colors is None:
        if colorblind_safe:
            palette = gb.create_palette(palette_size=len(distributions), colorblind_safe=colorblind_safe)
        else:
            palette =  utils.get_colors(len(distributions))
    else:
        if len(distrib_colors) < len(distributions):
            if colorblind_safe:
                additional_colors = gb.create_palette(palette_size=len(distributions) - len(distrib_colors), colorblind_safe=colorblind_safe)
            else:
                additional_colors = utils.get_colors(len(distributions) - len(distrib_colors))
            distrib_colors.extend(additional_colors)
        palette = distrib_colors

    if distrib_labels is None:
        distrib_labels = [f"Class {i+1}" for i in range(C)]

    u_list = []
    for d in distributions:
        Z = d.pdf(coords).reshape(X.shape)
        Zmin, Zmax = np.min(Z), np.max(Z)
        Z_norm = (Z - Zmin) / (Zmax - Zmin + 1e-12)
        u_c = 1.0 - Z_norm
        u_list.append(u_c.astype(np.float64))

    if I is None:
        I = np.ones((C, C), dtype=np.float64)

    if stippling_scale > 1.0 or stippling_scale < 0.1:
        raise ValueError("Stippling scale should be in range 1.0 - 0.1")
    
    if stippling_scale <= 0.5:
        enable_plateau_stop = True

    P_list = _simulate_halftoning_multi(
        u_list,
        steps=steps,
        tau=tau,
        seed=seed,
        shake_interval=10,
        shake_strength=1,
        shake_stop=max(200, steps // 3),
        tol=1e-2,
        I=I,
        stippling_scale=stippling_scale,
        enable_plateau_stop=enable_plateau_stop
    )

    for _, (P, color, label) in enumerate(zip(P_list, palette, distrib_labels)):
        if P.shape[0] == 0:
            continue
        x_coords = np.interp(P[:, 1], (0, resolution - 1), (xmin, xmax))
        y_coords = np.interp(P[:, 0], (0, resolution - 1), (ymin, ymax))

        axs.scatter(x_coords, y_coords, color=color, s=stipple_size, label=label, edgecolors="none")

    axs.set_xlim(xmin, xmax)
    axs.set_ylim(ymin, ymax)
    axs.set_aspect("equal", adjustable="box")
    axs.legend()

    if show_plot:
        plt.tight_layout()
        plt.show()

    return fig, axs

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
fig, axs = plot_stipples(distribs_lo, n_samples=3000, resolution=256, seed=55, show_plot=True)

from scipy.stats import multivariate_normal

mean1 = [0, 0]
cov1 = [[0.05, 0], [0, 0.05]]

mean2 = [1, 1]
cov2 = [[0.05, 0], [0, 0.05]]

mean3 = [-1, 1]
cov3 = [[0.05, 0], [0, 0.05]]

distribs_low = [
    Distribution(multivariate_normal(mean1, cov1), name="Normal"),
    Distribution(multivariate_normal(mean2, cov2), name="Normal"),
    Distribution(multivariate_normal(mean3, cov3), name="Normal"),
]

mean1 = [0, 0]
cov1 = [[0.5, 0.1], [0.1, 0.5]]

mean2 = [1.5, 1]
cov2 = [[0.6, 0], [0, 0.3]]

mean3 = [-1, 1.5]
cov3 = [[0.4, -0.1], [-0.1, 0.4]]

distribs_high = [
    Distribution(multivariate_normal(mean1, cov1), name="Normal"),
    Distribution(multivariate_normal(mean2, cov2), name="Normal"),
    Distribution(multivariate_normal(mean3, cov3), name="Normal"),
]

mean1 = [0, 0]
cov1 = [[0.05, 0], [0, 0.05]]  # Tight

mean2 = [1.5, 1]
cov2 = [[0.5, 0], [0, 0.3]]    # Broad

mean3 = [-1.5, -0.5]
cov3 = [[0.6, 0.1], [0.1, 0.4]]  # Broad

distribs_mixed = [
    Distribution(multivariate_normal(mean1, cov1), name="Normal"),
    Distribution(multivariate_normal(mean2, cov2), name="Normal"),
    Distribution(multivariate_normal(mean3, cov3), name="Normal"),
]

# Horizontal (X-axis stretched)
mean_a = [0, 0]
cov_a = [[0.5, 0], [0, 0.01]]  # Wide in X, narrow in Y

# Vertical (Y-axis stretched)
mean_b = [0, 0]
cov_b = [[0.01, 0], [0, 0.5]]  # Narrow in X, tall in Y

# Fat roundish distribution
mean_c = [0.5, 0.5]
cov_c = [[0.3, 0], [0, 0.3]]  # Uniform spread in both

distribs_stretch = [
    Distribution(multivariate_normal(mean_a, cov_a), name="X-axis Stretch"),
    Distribution(multivariate_normal(mean_b, cov_b), name="Y-axis Stretch"),
    Distribution(multivariate_normal(mean_c, cov_c), name="Fat Round"),
]


plot_stipples(distribs_low, n_samples=3000, resolution=256, seed=42, distrib_labels=["Tight A", "Tight B", "Tight C"], show_plot=True)
plot_stipples(distribs_high, n_samples=3000, resolution=256, seed=42, distrib_labels=["Wide A", "Wide B", "Wide C"], show_plot=True)
plot_stipples(distribs_mixed, n_samples=3000, resolution=256, seed=42, distrib_labels=["Tight", "Wide1", "Wide2"], show_plot=True)
plot_stipples(distribs_stretch, n_samples=3000, resolution=256, seed=42, distrib_labels=["X-Stretch", "Y-Stretch", "Fat"], show_plot=True)
