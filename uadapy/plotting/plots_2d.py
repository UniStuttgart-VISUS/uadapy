from dataclasses import dataclass
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2
from uadapy import Distribution
from matplotlib.colors import ListedColormap
import uadapy.plotting.utils as utils
import glasbey as gb


@dataclass
class OrientedGrid:
    center: np.ndarray
    axes: np.ndarray
    half_extents: np.ndarray
    resolution: int

    @classmethod
    def axis_aligned(cls, ranges, resolution):
        bounds = np.asarray(ranges, dtype=float)
        return cls(bounds.mean(axis=1), np.eye(2), (bounds[:, 1] - bounds[:, 0]) / 2, resolution)

    def coordinates(self):
        u = np.linspace(-self.half_extents[0], self.half_extents[0], self.resolution)
        v = np.linspace(-self.half_extents[1], self.half_extents[1], self.resolution)
        uu, vv = np.meshgrid(u, v)
        x = self.center[0] + self.axes[0, 0] * uu + self.axes[0, 1] * vv
        y = self.center[1] + self.axes[1, 0] * uu + self.axes[1, 1] * vv
        return x, y

    def cell_area(self):
        spacing = 2 * self.half_extents / (self.resolution - 1)
        return float(spacing[0] * spacing[1])

    def aabb(self):
        radii = np.abs(self.axes) @ self.half_extents
        return [(self.center[i] - radii[i], self.center[i] + radii[i]) for i in range(2)]


def plot_samples(distributions,
                 n_samples,
                 seed=55,
                 point_size=None,
                 alpha=1,
                 fig=None,
                 axs=None,
                 x_label=None,
                 y_label=None,
                 title=None,
                 distrib_colors=None,
                 colorblind_safe=False,
                 show_plot=False):
    """
    Plot samples from the given distribution. If several distributions should be
    plotted together, an array can be passed to this function.

    Parameters
    ----------
    distributions : list
        List of distributions to plot.
    n_samples : int
        Number of samples per distribution.
    seed : int
        Seed for the random number generator for reproducibility. It defaults to 55 if not provided.
    point_size : float or None, optional
        Marker size (area in points^2). If None, matplotlib's default is used.
    alpha : float, optional
        Opacity value if the samples in the scatter plots. By default 1 (fully opaque)
    fig : matplotlib.figure.Figure or None, optional
        Figure object to use for plotting. If None, a new figure will be created.
    axs : matplotlib.axes.Axes or None, optional
        Axes object to use for plotting. If None, new axes will be created.
    x_label : string, optional
        label for x-axis.
    y_label : string, optional
        label for y-axis.
    title : string, optional
        title for the plot.
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
        The figure object containing the plot.
    list
        List of Axes objects used for plotting.
    """

    if isinstance(distributions, Distribution):
        distributions = [distributions]

    for d in distributions:
        if d.n_dims != 2:
            raise ValueError("All distributions must have 2 dimensions.")

    if axs is None:
        if fig is None:
            fig, axs = plt.subplots()
        else:
            if fig.axes is not None:
                axs = fig.axes[0]
            else:
                raise ValueError("The provided figure has no axes. Pass an Axes or create subplots first.")
    else:
        if fig is None:
            fig = axs.figure

    # Generate colors
    palette = _get_color_palette(len(distributions), distrib_colors, colorblind_safe)

    for i, d in enumerate(distributions):
        samples = d.sample(n_samples, seed)
        axs.scatter(x=samples[:,0], y=samples[:,1], color=palette[i], s=point_size, alpha=alpha)
    if x_label:
        axs.set_xlabel(x_label)
    if y_label:
        axs.set_ylabel(y_label)
    if title:
        axs.set_title(title)

    if show_plot:
        fig.tight_layout()
        plt.show()

    return fig, axs


def plot_contour(distributions,
                 resolution=128,
                 ranges=None,
                 quantiles=[25, 75, 95],
                 fig=None,
                 axs=None,
                 distrib_colors=None,
                 colorblind_safe=False,
                 show_plot=False):
    """
    Plot contour plots for given distributions.

    Parameters
    ----------
    distributions : Distribution or list of Distribution
        Distribution(s) to plot.
    resolution : int, optional
        The resolution of the plot. Default is 128.
    ranges : list of tuple or None, optional
        The ranges for the x and y axes as [(x_min, x_max), (y_min, y_max)]. 
        If None, a separate grid is calculated for each distribution and oriented
        along its principal axes. Supplying ranges forces a shared axis-aligned grid.
        Invalid moments fall back to 2000 samples with a fixed seed; covariance
        regularization is used only to construct the grid and does not alter the density.
    quantiles : list of float or None, optional
        List of quantiles to use for determining isovalues. Default is [25, 75, 95].
    fig : matplotlib.figure.Figure or None, optional
        Figure object to use for plotting. If None, a new figure will be created.
    axs : matplotlib.axes.Axes or None, optional
        Axes object to use for plotting. If None, new axes will be created.
    distrib_colors : list or None, optional
        List of colors to use for each distribution. If None, Matplotlib Set2 and glasbey colors will be used.
    colorblind_safe : bool, optional
        If True, the plot will use colors suitable for colorblind individuals.
        Default is False.
    show_plot : bool, optional
        If True, display the plot.
        Default is False.

    Examples
    --------
    >>> from scipy.stats import multivariate_normal
    >>> from uadapy import Distribution
    >>> thin = Distribution(
    ...     multivariate_normal(mean=[0, 0], cov=[[1, 0], [0, 1e-6]]),
    ...     name="Normal",
    ... )
    >>> fig, ax = plot_contour(thin)

    Returns
    -------
    matplotlib.figure.Figure
        The figure object containing the plot.
    matplotlib.axes.Axes
        The axes object used for plotting.

    Raises
    ------
    ValueError
        If a quantile is not between 0 and 100 (exclusive).
    """
    if isinstance(distributions, Distribution):
        distributions = [distributions]

    for d in distributions:
        if d.n_dims != 2:
            raise ValueError("All distributions must have 2 dimensions.")

    if axs is None:
        if fig is None:
            fig, axs = plt.subplots()
        else:
            if fig.axes is not None:
                axs = fig.axes[0]
            else:
                raise ValueError("The provided figure has no axes. Pass an Axes or create subplots first.")
    else:
        if fig is None:
            fig = axs.figure

    # Generate colors
    palette = _get_color_palette(len(distributions), distrib_colors, colorblind_safe)

    # Plot contours for each distribution
    for i, d in enumerate(distributions):
        grid = (OrientedGrid.axis_aligned(ranges, resolution) if ranges is not None
                else _calculate_oriented_grid(d, max(quantiles), resolution))
        xv, yv = grid.coordinates()
        coordinates = np.stack((xv, yv), axis=-1)
        coordinates = coordinates.reshape((-1, 2))
        pdf = d.pdf(coordinates)
        pdf = pdf.reshape(xv.shape)
        color = palette[i]

        isovalues = _calculate_isovalues(pdf, grid.cell_area(), quantiles)

        axs.contour(xv, yv, pdf, levels=isovalues, colors=[color])

    if show_plot:
        fig.tight_layout()
        plt.show()

    return fig, axs


def plot_contour_bands(distributions,
                       resolution=128,
                       ranges=None,
                       quantiles=[25, 75, 95],
                       fig=None,
                       axs=None,
                       show_plot=False):
    """
    Plot contour bands for given distributions.

    Parameters
    ----------
    distributions : Distribution or list of Distribution
        Distribution(s) to plot.
    resolution : int, optional
        The resolution of the plot. Default is 128.
    ranges : list of tuple or None, optional
        The ranges for the x and y axes as [(x_min, x_max), (y_min, y_max)]. 
        If None, a separate grid is calculated for each distribution and oriented
        along its principal axes. Supplying ranges forces a shared axis-aligned grid.
        Invalid moments fall back to 2000 samples with a fixed seed; covariance
        regularization is used only to construct the grid and does not alter the density.
    quantiles : list of float or None, optional
        List of quantiles to use for determining isovalues. Default is [25, 75, 95].
    fig : matplotlib.figure.Figure or None, optional
        Figure object to use for plotting. If None, a new figure will be created.
    axs : matplotlib.axes.Axes or None, optional
        Axes object to use for plotting. If None, new axes will be created.
    show_plot : bool, optional
        If True, display the plot.
        Default is False.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object containing the plot.
    matplotlib.axes.Axes
        The axes object used for plotting.

    Raises
    ------
    ValueError
        If a quantile is not between 0 and 100 (exclusive).
    """
    if isinstance(distributions, Distribution):
        distributions = [distributions]

    for d in distributions:
        if d.n_dims != 2:
            raise ValueError("All distributions must have 2 dimensions.")

    if axs is None:
        if fig is None:
            fig, axs = plt.subplots()
        else:
            if fig.axes is not None:
                axs = fig.axes[0]
            else:
                raise ValueError("The provided figure has no axes. Pass an Axes or create subplots first.")
    else:
        if fig is None:
            fig = axs.figure

    n_quantiles = len(quantiles)
    alpha_values = np.linspace(1/n_quantiles, 1.0, n_quantiles)
    custom_cmap = utils.create_shaded_set2_colormap(alpha_values)

    # Plot contour bands for each distribution
    for i, d in enumerate(distributions):
        grid = (OrientedGrid.axis_aligned(ranges, resolution) if ranges is not None
                else _calculate_oriented_grid(d, max(quantiles), resolution))
        xv, yv = grid.coordinates()
        coordinates = np.stack((xv, yv), axis=-1)
        coordinates = coordinates.reshape((-1, 2))
        pdf = d.pdf(coordinates)
        pdf = pdf.reshape(xv.shape)
        if not np.any(pdf > 0):
            warnings.warn(f"Skipping {d.name}: the PDF is zero on the plotting grid.", RuntimeWarning)
            continue
        pdf = np.ma.masked_where(pdf <= 0, pdf)

        isovalues = _calculate_isovalues(pdf, grid.cell_area(), quantiles)
        max_val = np.max(pdf[pdf > 0])
        if not isovalues or max_val > isovalues[-1]:
            isovalues.append(max_val)

        # Extract color subset for this distribution
        start_idx = i * n_quantiles
        end_idx = start_idx + n_quantiles
        color_subset = custom_cmap.colors[start_idx:end_idx]
        cmap_subset = ListedColormap(color_subset)

        axs.contourf(xv, yv, pdf, levels=isovalues, cmap=cmap_subset)

    if show_plot:
        fig.tight_layout()
        plt.show()

    return fig, axs


# Helper Functions

def _get_color_palette(n_distributions, distrib_colors=None, colorblind_safe=False):
    """
    Generate or extend a color palette for distributions.

    Parameters
    ----------
    n_distributions : int
        Number of distributions needing colors.
    distrib_colors : list or None, optional
        Existing colors to use/extend.
    colorblind_safe : bool, optional
        Whether to use colorblind-safe colors.

    Returns
    -------
    list
        Color palette with at least n_distributions colors.
    """
    if distrib_colors is None:
        if colorblind_safe:
            palette = gb.create_palette(palette_size=n_distributions, colorblind_safe=colorblind_safe)
        else:
            palette = utils.get_colors(n_distributions)
    else:
        if len(distrib_colors) < n_distributions:
            if colorblind_safe:
                additional_colors = gb.create_palette(
                    palette_size=n_distributions - len(distrib_colors),
                    colorblind_safe=colorblind_safe
                )
            else:
                additional_colors = utils.get_colors(n_distributions - len(distrib_colors))
            distrib_colors.extend(additional_colors)
        palette = distrib_colors

    return palette


def _calculate_isovalues(pdf_grid, cell_area, quantiles):
    """
    Calculate density isovalues using cumulative probability.

    Parameters
    ----------
    pdf_grid : np.ndarray
        2D array of PDF values on the grid.
    cell_area : float
        Area represented by each grid cell.
    quantiles : list of float
        List of quantile percentages.

    Returns
    -------
    list of float
        List of density threshold values corresponding to the quantiles.

    Raises
    ------
    ValueError
        If a quantile is not between 0 and 100 (exclusive).
    """
    pdf_sum = np.sum(pdf_grid) * cell_area
    pdf_normalized = pdf_grid / pdf_sum if pdf_sum > 0 else pdf_grid

    # Sort density values in descending order
    sorted_pdf = np.sort(pdf_normalized.flatten())[::-1]

    # Calculate cumulative probability
    cumulative_prob = np.cumsum(sorted_pdf) * cell_area

    # Process quantiles and find density thresholds
    isovalues = []
    sorted_quantiles = sorted(quantiles, reverse=True)

    for quantile in sorted_quantiles:
        if not 0 < quantile < 100:
            raise ValueError(f"Invalid quantile: {quantile}. Quantiles must be between 0 and 100 (exclusive).")

        # Find the density threshold at which cumulative probability reaches this level
        idx = np.searchsorted(cumulative_prob, quantile / 100.0)
        if idx < len(sorted_pdf):
            threshold = sorted_pdf[idx]
            isovalues.append(threshold)

    isovalues.sort()

    unique_isovalues = []
    for val in isovalues:
        if not unique_isovalues or val > unique_isovalues[-1]:
            unique_isovalues.append(val)

    return unique_isovalues


def _calculate_plot_ranges(distributions, quantiles, resolution=128):
    """
    Return axis-aligned bounds of the per-distribution plotting grids.

    Parameters
    ----------
    distributions : list of Distribution
        Distribution(s) to determine ranges for.
    quantiles : list of float
        List of quantiles (percentages) to include in the plot.
    resolution : int, optional
        Grid resolution for numerical refinement. Default is 128.

    Returns
    -------
    tuple of list of tuple
        A pair ``(combined_ranges, all_ranges)``:

        - ``combined_ranges`` contains the overall (min, max) bounds for each
          dimension across all distributions, e.g.,
          ``[(x_min, x_max), (y_min, y_max)]``.
        - ``all_ranges`` contains one list of per-dimension (min, max) bounds
          for each distribution.
    """
    if isinstance(distributions, Distribution):
        distributions = [distributions]

    all_ranges = [
        _calculate_oriented_grid(distribution, max(quantiles), resolution).aabb()
        for distribution in distributions
    ]

    # Combine ranges from all distributions
    combined_ranges = []
    n_dims = len(all_ranges[0])

    for dim in range(n_dims):
        min_vals = [r[dim][0] for r in all_ranges]
        max_vals = [r[dim][1] for r in all_ranges]
        combined_ranges.append((min(min_vals), max(max_vals)))

    return combined_ranges, all_ranges


def _get_moments(distribution):
    """Get finite moments, estimating only invalid moments from reproducible samples."""
    dims = getattr(distribution, "n_dims", 2)
    mean = cov = None
    mean_valid = cov_valid = False
    try:
        value = distribution.mean()
        if value is not None:
            value = np.atleast_1d(np.asarray(value, dtype=float))
            mean_valid = value.shape == (dims,) and np.all(np.isfinite(value))
            if mean_valid:
                mean = value
    except Exception:
        pass
    try:
        value = distribution.cov()
        if value is not None:
            value = np.asarray(value, dtype=float)
            if value.ndim == 0:
                value = np.eye(dims) * value
            elif value.ndim == 1:
                value = np.diag(value)
            cov_valid = value.shape == (dims, dims) and np.all(np.isfinite(value))
            if cov_valid:
                cov = value
    except Exception:
        pass

    if mean_valid and cov_valid:
        return mean, cov

    try:
        samples = np.asarray(distribution.sample(2000, seed=55), dtype=float)
        if samples.ndim == 1 and dims == 1:
            samples = samples[:, None]
        if samples.ndim != 2 or samples.shape[1] != dims:
            raise ValueError("unexpected sample shape")
        samples = samples[np.all(np.isfinite(samples), axis=1)]
        if len(samples) < 2:
            raise ValueError("fewer than two finite samples")
        warnings.warn(
            f"Estimating invalid moments for {getattr(distribution, 'name', type(distribution).__name__)} "
            "from 2000 samples (seed=55).",
            RuntimeWarning,
        )
        if not mean_valid:
            mean = samples.mean(axis=0)
        if not cov_valid:
            cov = np.atleast_2d(np.cov(samples.T))
        return mean, cov
    except Exception as error:
        name = getattr(distribution, "name", type(distribution).__name__)
        raise ValueError(f"Could not estimate moments for distribution {name}.") from error


def _regularize_cov(cov):
    cov = np.asarray(cov, dtype=float)
    cov = np.where(np.isfinite(cov), cov, 0.0)
    cov = (cov + cov.T) / 2
    n = cov.shape[0]
    eps = max(1e-9 * np.trace(cov) / n, 1e-12)
    eigenvalues, eigenvectors = np.linalg.eigh(cov + eps * np.eye(n))
    eigenvalues = np.maximum(eigenvalues, eps)
    order = np.argsort(eigenvalues)[::-1]
    return eigenvalues[order], eigenvectors[:, order]


def _principal_axes(distribution):
    mean, cov = _get_moments(distribution)
    eigenvalues, eigenvectors = _regularize_cov(cov)
    return mean, eigenvalues, eigenvectors


def _calculate_oriented_grid(distribution, largest_quantile, resolution, padding=0.05):
    if distribution.name in ["Normal", "GMM", "multivariate_normal_frozen"]:
        return _calculate_ranges_analytical(distribution, largest_quantile, resolution, padding)
    return _calculate_ranges_numerical(distribution, largest_quantile, resolution=resolution, padding=padding)


def _component_covariances(distribution):
    model = distribution.model
    means = np.asarray(model.means_, dtype=float)
    covariances = np.asarray(model.covariances_, dtype=float)
    cov_type = getattr(model, "covariance_type", "full")
    dims = means.shape[1]
    if covariances.ndim == 3:
        return means, covariances
    if cov_type == "tied":
        covariances = np.repeat(covariances[None, :, :], len(means), axis=0)
    elif cov_type == "diag":
        covariances = np.array([np.diag(c) for c in covariances])
    elif cov_type == "spherical":
        covariances = covariances.reshape(-1)
        covariances = np.array([np.eye(dims) * c for c in covariances])
    return means, covariances


def _calculate_ranges_analytical(distribution, largest_quantile, resolution=128, padding=0.05):
    center, eigenvalues, axes = _principal_axes(distribution)
    chi2_val = chi2.ppf(largest_quantile / 100.0, df=len(center))
    if distribution.name == "GMM":
        means, covariances = _component_covariances(distribution)
        projected_means = (means - center) @ axes
        widths = np.array([
            np.sqrt(chi2_val * np.maximum(np.einsum("i,kij,j->k", axes[:, i], covariances, axes[:, i]), 0))
            for i in range(2)
        ]).T
        lower = np.min(projected_means - widths, axis=0)
        upper = np.max(projected_means + widths, axis=0)
        center = center + axes @ ((lower + upper) / 2)
        half_extents = (upper - lower) * (0.5 + padding)
    else:
        half_extents = np.sqrt(chi2_val * eigenvalues) * (1 + 2 * padding)
    return OrientedGrid(center, axes, np.maximum(half_extents, 1e-6), resolution)


def _calculate_ranges_numerical(
    distribution,
    largest_quantile,
    factor=2.0,
    max_range=1e3,
    threshold=1e-6,
    resolution=128,
    padding=0.05,
):
    """
    Calculate an oriented plotting grid using numerical PDF evaluation.

    Parameters
    ----------
    distribution : Distribution
        Distribution to calculate ranges for.
    largest_quantile : float
        Largest quantile percentage to include.
    factor : float, optional
        Expansion factor for coarse search. Default is 2.0.
    max_range : float, optional
        Maximum search radius. Default is 1000.
    threshold : float, optional
        PDF threshold for determining when we've gone far enough. Default is 1e-6.
    resolution : int, optional
        Grid resolution for fine search. Default is 128.
    padding : float, optional
        Padding to add to each side of the range. Default is 0.05.

    Returns
    -------
    OrientedGrid
        Grid bounds aligned with the principal covariance axes.
    """
    mean, eigenvalues, axes = _principal_axes(distribution)
    extents = np.minimum(2 * np.sqrt(eigenvalues), max_range)

    for axis in range(2):
        while extents[axis] < max_range:
            endpoints = mean + np.array([-1, 1])[:, None] * extents[axis] * axes[:, axis]
            values = np.asarray(distribution.pdf(endpoints)).ravel()
            if np.all(np.isfinite(values)) and np.all(values < threshold):
                break
            extents[axis] = min(extents[axis] * factor, max_range)

    coarse_extents = extents.copy()

    def evaluate(current_extents):
        grid = OrientedGrid(mean.copy(), axes, current_extents.copy(), resolution)
        x, y = grid.coordinates()
        values = np.asarray(distribution.pdf(np.stack((x, y), axis=-1).reshape(-1, 2)))
        return grid, np.nan_to_num(values.reshape(x.shape), nan=0.0, posinf=0.0, neginf=0.0)

    grid, pdf = evaluate(extents)
    for _ in range(10):
        if np.any(pdf > 0) and np.sum(pdf) > 0:
            break
        extents[np.argmin(extents)] *= 0.5
        grid, pdf = evaluate(extents)

    if not np.any(pdf > 0) or np.sum(pdf) <= 0:
        center_pdf = np.asarray(distribution.pdf(mean[None, :])).ravel()
        if not np.any(center_pdf > 0):
            warnings.warn(
                f"Could not refine plotting grid for {getattr(distribution, 'name', type(distribution).__name__)}; "
                "using the coarse grid.",
                RuntimeWarning,
            )
            return OrientedGrid(mean, axes, np.maximum(coarse_extents, 1e-6), resolution)

    cell_area = grid.cell_area()
    ordered = np.sort(pdf.ravel())[::-1]
    cumulative = np.cumsum(ordered) * cell_area
    index = min(np.searchsorted(cumulative, largest_quantile / 100.0), len(ordered) - 1)
    mask = (pdf >= ordered[index]) & (pdf > 0)
    if not np.any(mask):
        warnings.warn("Could not refine plotting grid; using the coarse grid.", RuntimeWarning)
        return OrientedGrid(mean, axes, np.maximum(coarse_extents, 1e-6), resolution)

    u = np.linspace(-extents[0], extents[0], resolution)
    v = np.linspace(-extents[1], extents[1], resolution)
    rows, columns = np.where(mask)
    lower = np.array([u[columns.min()], v[rows.min()]])
    upper = np.array([u[columns.max()], v[rows.max()]])
    span = upper - lower
    padding_distance = np.maximum(span * padding, np.array([2 * extents[0] / (resolution - 1),
                                                             2 * extents[1] / (resolution - 1)]))
    center_offset = (lower + upper) / 2
    refined_center = mean + axes @ center_offset
    refined_extents = np.maximum((span / 2) + padding_distance, 1e-6)
    return OrientedGrid(refined_center, axes, refined_extents, resolution)