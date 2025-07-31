import numba
import glasbey as gb
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from scipy.spatial import cKDTree
from uadapy import Distribution
from uadapy.plotting import utils

def remove_overlapping_stipples_all_classes(points, class_labels, min_dist=1.0):
    """
    Remove overlapping stipples across all classes by enforcing a global minimum spacing.

    Parameters
    ----------
    points : np.ndarray
        All stipple positions (N, 2).
    class_labels : np.ndarray
        Class index for each stipple (N,).
    min_dist : float
        Minimum distance allowed between any two stipples.

    Returns
    -------
    np.ndarray, np.ndarray
        Filtered points and corresponding class labels.
    """
    if len(points) == 0:
        return points, class_labels

    tree = cKDTree(points)
    keep = np.ones(len(points), dtype=bool)

    for i in range(len(points)):
        if not keep[i]:
            continue
        neighbors = tree.query_ball_point(points[i], r=min_dist)
        for j in neighbors:
            if j != i:
                keep[j] = False

    return points[keep], class_labels[keep]

class StippleClass:
    """
    Container for class-specific stipple data used in multi-class halftoning.

    Attributes
    ----------
    dist : Distribution
        The distribution object associated with the class.
    Z : np.ndarray
        The raw density grid (PDF values) for the class over the domain.
    Z_norm : np.ndarray
        Normalized version of Z for use in force scaling.
    color : tuple
        Color assigned to this class's stipples.
    points : np.ndarray or None
        Sampled stipple point positions in image space.
    """

    def __init__(self, dist, Z, Z_norm, color):
        self.dist = dist
        self.Z = Z
        self.Z_norm = Z_norm
        self.color = color
        self.points = None

@numba.njit(cache=True, parallel=True)
def compute_image_forces_numba(Z):
    """
    Compute the attraction forces as gradients of the density field.

    Parameters
    ----------
    Z : np.ndarray
        A 2D array representing the normalized density grid.

    Returns
    -------
    np.ndarray
        A 3D array of shape (h, w, 2), containing force vectors [fx, fy] at each pixel.
    """

    h, w = Z.shape
    image_forces = np.zeros((h, w, 2), dtype=np.float32)

    for y_idx in numba.prange(1, h - 1):
        for x_idx in numba.prange(1, w - 1):
            grad_x = Z[y_idx, x_idx + 1] - Z[y_idx, x_idx - 1]
            grad_y = Z[y_idx + 1, x_idx] - Z[y_idx - 1, x_idx]
            image_forces[y_idx, x_idx, 0] = grad_x * Z[y_idx, x_idx]
            image_forces[y_idx, x_idx, 1] = grad_y * Z[y_idx, x_idx]

    return image_forces


@numba.njit(cache=True)
def bilinear_interpolate_numba(image, x, y):
    """
    Perform bilinear interpolation at floating point coordinates on a 2D grid.

    Parameters
    ----------
    image : np.ndarray
        The 2D grid/image from which to interpolate values.
    x : float
        X-coordinate (can be fractional).
    y : float
        Y-coordinate (can be fractional).

    Returns
    -------
    float
        Interpolated value at (x, y).
    """

    x0 = int(np.floor(x))
    x1 = x0 + 1
    y0 = int(np.floor(y))
    y1 = y0 + 1

    x0 = max(0, min(x0, image.shape[1] - 1))
    x1 = max(0, min(x1, image.shape[1] - 1))
    y0 = max(0, min(y0, image.shape[0] - 1))
    y1 = max(0, min(y1, image.shape[0] - 1))

    Ia = image[y0, x0]
    Ib = image[y0, x1]
    Ic = image[y1, x0]
    Id = image[y1, x1]

    wa = (x1 - x) * (y1 - y)
    wb = (x - x0) * (y1 - y)
    wc = (x1 - x) * (y - y0)
    wd = (x - x0) * (y - y0)

    return wa * Ia + wb * Ib + wc * Ic + wd * Id


@numba.njit(cache=True, parallel=True)
def calculate_attraction_forces_numba(points, image_forces, w, h):
    """
    Compute attraction forces for all stipple points based on image gradients.

    Parameters
    ----------
    points : np.ndarray
        Array of shape (N, 2), containing point coordinates.
    image_forces : np.ndarray
        Precomputed attraction force field (h, w, 2).
    w : int
        Image width.
    h : int
        Image height.

    Returns
    -------
    np.ndarray
        Array of shape (N, 2), containing attraction forces on each point.
    """

    num_points = points.shape[0]
    forces = np.zeros((num_points, 2), dtype=np.float32)

    for i in numba.prange(num_points):
        x_pt, y_pt = points[i]
        if 1 <= x_pt < w - 1 and 1 <= y_pt < h - 1:
            fx = bilinear_interpolate_numba(image_forces[:, :, 0], x_pt, y_pt)
            fy = bilinear_interpolate_numba(image_forces[:, :, 1], x_pt, y_pt)
            forces[i, 0] = fx
            forces[i, 1] = fy

    return forces


@numba.njit(cache=True)
def calculate_class_repulsion(points, classes, neighbors, Z_repulsions, interaction_matrix):
    """
    Compute repulsion forces between stipple points, modulated by class interactions.

    Parameters
    ----------
    points : np.ndarray
        Coordinates of stipple points.
    classes : np.ndarray
        Array of class labels corresponding to each point.
    neighbors : np.ndarray
        Indices of nearest neighbors for each point.
    Z_repulsions : list of np.ndarray
        List of repulsion weight maps for each class.
    interaction_matrix : np.ndarray
        Matrix defining inter-class repulsion strengths.

    Returns
    -------
    np.ndarray
        Repulsion forces acting on each point.
    """

    num_points = points.shape[0]
    forces = np.zeros((num_points, 2), dtype=np.float32)

    for i in range(num_points):
        xi = min(int(points[i, 0]), Z_repulsions[0].shape[1] - 1)
        yi = min(int(points[i, 1]), Z_repulsions[0].shape[0] - 1)
        class_i = classes[i]
        repulsion_scale_i = Z_repulsions[class_i][yi, xi]

        for j in neighbors[i]:
            if j != i:
                class_j = classes[j]
                delta = points[j] - points[i]
                dist = np.sqrt(delta[0] ** 2 + delta[1] ** 2) + 1e-6
                interaction = interaction_matrix[class_i, class_j]
                force = interaction * repulsion_scale_i * (delta / dist) / dist
                forces[i] -= force

    return forces


def multi_class_halftoning(classes,
                           x,
                           y,
                           resolution,
                           seed,
                           interaction_matrix,
                           iterations=30,
                           tau=0.1):
    """
    Perform multi-class electrostatic halftoning optimization on stipple points.

    Parameters
    ----------
    classes : list of StippleClass
        List of stipple classes, each containing a distribution and density field.
    x : np.ndarray
        Linearly spaced X-axis grid values.
    y : np.ndarray
        Linearly spaced Y-axis grid values.
    resolution : int
        Width/height of the output density grid.
    seed : int
        Random seed for reproducibility.
    interaction_matrix : np.ndarray
        Matrix controlling repulsion strength between classes.
    iterations : int, optional
        Number of optimization steps. Default is 30.
    tau : float, optional
        Time step scale factor for gradient descent. Default is 0.1.

    Returns
    -------
    np.ndarray
        Final positions of stipple points.
    np.ndarray
        Class labels corresponding to each point.
    """

    h, w = resolution, resolution
    all_points, all_classes, Z_repulsions = [], [], []

    for idx, sc in enumerate(classes):
        num_points = int(np.sum(sc.Z_norm))
        pts = sc.dist.sample(num_points, seed + idx)
        pts[:, 0] = np.interp(pts[:, 0], (x[0], x[-1]), (0, w - 1))
        pts[:, 1] = np.interp(pts[:, 1], (y[0], y[-1]), (0, h - 1))
        sc.points = pts
        all_points.append(pts)
        all_classes.append(np.full((pts.shape[0],), idx))
        Z_repulsions.append(1.0 - 0.7 * sc.Z_norm)

    points = np.vstack(all_points)
    class_labels = np.concatenate(all_classes)

    for iteration in range(iterations):
        attraction = np.zeros_like(points, dtype=np.float32)
        for idx, sc in enumerate(classes):
            image_forces = compute_image_forces_numba(sc.Z)
            idxs = np.where(class_labels == idx)[0]
            attraction[idxs] = calculate_attraction_forces_numba(
                points[idxs], image_forces, w, h)

        tree = KDTree(points)
        _, neighbors = tree.query(points, k=15)

        repulsion = calculate_class_repulsion(points, class_labels, neighbors, Z_repulsions, interaction_matrix)

        forces = attraction + repulsion
        points += tau * forces

        # Add shaking every 10 iterations
        if iteration % 10 == 0:
            max_disp = max(0.1, 0.5 * np.exp(-iteration / 100.0))
            noise = np.random.uniform(-max_disp, max_disp, size=points.shape)
            points += noise

        # Remove overcrowded stipples exactly in the middle
        if iteration == (iterations // 2) - 1:
            points, class_labels = remove_overlapping_stipples_all_classes(
                points, class_labels, min_dist=0.8
            )

    return points, class_labels


def plot_stipples(distributions,
                  n_samples,
                  resolution=128,
                  seed=55,
                  distrib_colors=None,
                  colorblind_safe=False,
                  distrib_labels=None,
                  show_plot=False):
    """
    Plot stipples from multiple distributions using multi-class halftoning.

    Parameters
    ----------
    distributions : list of Distribution
        List of distribution objects to sample from.
    n_samples : int
        Number of points to sample from each distribution.
    resolution : int, optional
        Grid resolution for PDF computation (default is 128).
    seed : int, optional
        Random seed for reproducibility (default is 55).
    distrib_colors : list of tuple, optional
        Custom color palette for each class. If None, colors are autogenerated.
    colorblind_safe : bool, optional
        Use a colorblind-friendly palette. Default is False.
    distrib_labels : list or None, optional
        Labels for each distribution.
    show_plot : bool, optional
        If True, display the final plot. Default is False.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object with the stipple plot.
    matplotlib.axes.Axes
        The axis object containing the scatter plot.
    """

    if isinstance(distributions, Distribution):
        distributions = [distributions]

    distrib_samples = [d.sample(n_samples, seed) for d in distributions]
    all_samples = np.vstack(distrib_samples)

    # Generate colors
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

    x = np.linspace(np.min(all_samples[:, 0]), np.max(all_samples[:, 0]), resolution)
    y = np.linspace(np.min(all_samples[:, 1]), np.max(all_samples[:, 1]), resolution)
    X, Y = np.meshgrid(x, y)
    coords = np.dstack((X, Y)).reshape(-1, 2)

    classes = []
    for i, (d, color) in enumerate(zip(distributions, palette)):
        Z = d.pdf(coords).reshape(X.shape)
        Z_log = np.log1p(Z)
        Z_norm = (Z_log - Z_log.min()) / (Z_log.max() - Z_log.min())
        classes.append(StippleClass(d, Z, Z_norm, color))

    interaction_matrix = np.ones((len(classes), len(classes)), dtype=np.float32)

    points, class_labels = multi_class_halftoning(classes, x, y, resolution, seed, interaction_matrix)

    fig, ax = plt.subplots(figsize=(10, 8))
    for idx, cls in enumerate(classes):
        pts = points[class_labels == idx]
        x_coords = np.interp(pts[:, 0], (0, resolution - 1), (x[0], x[-1]))
        y_coords = np.interp(pts[:, 1], (0, resolution - 1), (y[0], y[-1]))
        label = distrib_labels[idx] if distrib_labels is not None else f"Class {idx}"
        ax.scatter(x_coords, y_coords, color=cls.color, s=5, label=label, edgecolors='none')

    ax.legend()
    if show_plot:
        plt.tight_layout()
        plt.show()

    return fig, ax

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
fig, axs = plot_stipples(distribs_lo, n_samples=3000, resolution=256, seed=55, distrib_labels=labels, show_plot=True)
