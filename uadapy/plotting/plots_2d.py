import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2
from uadapy import Distribution
from matplotlib.colors import ListedColormap
import uadapy.plotting.utils as utils
import glasbey as gb


def plot_samples(distributions,
                 n_samples,
                 seed=55,
                 point_size=None,
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
        axs.scatter(x=samples[:,0], y=samples[:,1], color=palette[i], s=point_size)
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
        If None, ranges are calculated based on the distributions.
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

    # Calculate ranges
    if ranges is None:
        ranges = _calculate_plot_ranges(distributions, quantiles, resolution)

    range_x = ranges[0]
    range_y = ranges[1]

    # Plot contours for each distribution
    for i, d in enumerate(distributions):
        x = np.linspace(range_x[0], range_x[1], resolution)
        y = np.linspace(range_y[0], range_y[1], resolution)
        xv, yv = np.meshgrid(x, y)
        coordinates = np.stack((xv, yv), axis=-1)
        coordinates = coordinates.reshape((-1, 2))
        pdf = d.pdf(coordinates)
        pdf = pdf.reshape(xv.shape)
        color = palette[i]

        isovalues = _calculate_isovalues(pdf, x, y, quantiles)

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
        If None, ranges are calculated based on the distributions.
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

    # Calculate ranges
    if ranges is None:
        ranges = _calculate_plot_ranges(distributions, quantiles, resolution)

    range_x = ranges[0]
    range_y = ranges[1]

    # Plot contour bands for each distribution
    for i, d in enumerate(distributions):
        x = np.linspace(range_x[0], range_x[1], resolution)
        y = np.linspace(range_y[0], range_y[1], resolution)
        xv, yv = np.meshgrid(x, y)
        coordinates = np.stack((xv, yv), axis=-1)
        coordinates = coordinates.reshape((-1, 2))
        pdf = d.pdf(coordinates)
        pdf = pdf.reshape(xv.shape)
        pdf = np.ma.masked_where(pdf <= 0, pdf)

        isovalues = _calculate_isovalues(pdf, x, y, quantiles)
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


def _calculate_isovalues(pdf_grid, grid_x, grid_y, quantiles):
    """
    Calculate density isovalues using cumulative probability.

    Parameters
    ----------
    pdf_grid : np.ndarray
        2D array of PDF values on the grid.
    grid_x : np.ndarray
        1D array of x-coordinates.
    grid_y : np.ndarray
        1D array of y-coordinates.
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
    # Normalize to create a proper PDF (integrate to 1)
    dx = grid_x[1] - grid_x[0]
    dy = grid_y[1] - grid_y[0]
    pdf_sum = np.sum(pdf_grid) * dx * dy
    pdf_normalized = pdf_grid / pdf_sum if pdf_sum > 0 else pdf_grid

    # Sort density values in descending order
    sorted_pdf = np.sort(pdf_normalized.flatten())[::-1]

    # Calculate cumulative probability
    cumulative_prob = np.cumsum(sorted_pdf) * dx * dy

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
    Calculate plotting ranges for distributions.

    For Normal and GMM distributions, uses analytical methods (mean + covariance).
    For other distributions, uses a coarse-to-fine PDF-based approach.

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
    list of tuple
        List of (min, max) tuples for each dimension, e.g., [(x_min, x_max), (y_min, y_max)].
    """
    if isinstance(distributions, Distribution):
        distributions = [distributions]

    largest_quantile = max(quantiles)
    all_ranges = []

    for distribution in distributions:
        # Check if we can use analytical methods
        if distribution.name in ["Normal", "GMM", "multivariate_normal_frozen"]:
            ranges = _calculate_ranges_analytical(distribution, largest_quantile)
        else:
            # Use numerical PDF-based approach
            ranges = _calculate_ranges_numerical(
                distribution,
                largest_quantile,
                resolution=resolution
            )

        all_ranges.append(ranges)

    # Combine ranges from all distributions
    combined_ranges = []
    n_dims = len(all_ranges[0])

    for dim in range(n_dims):
        min_vals = [r[dim][0] for r in all_ranges]
        max_vals = [r[dim][1] for r in all_ranges]
        combined_ranges.append((min(min_vals), max(max_vals)))

    return combined_ranges


def _ellipsoid_ranges(mean, cov, chi2_val, padding):
    """
    Calculate ranges for an ellipsoid defined by mean and covariance.

    Parameters
    ----------
    mean : np.ndarray
        Mean vector of the distribution.
    cov : np.ndarray
        Covariance matrix of the distribution.
    chi2_val : float
        Chi-squared value for the desired quantile.
    padding : float
        Fraction of padding to add to the ranges.

    Returns
    -------
    list of tuple
        List of (min, max) tuples for each dimension.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Calculate radii along each principal axis
    extents = (eigenvectors**2) @ eigenvalues
    radii = np.sqrt(chi2_val * extents)

    mins = mean - radii
    maxs = mean + radii

    span = maxs - mins
    mins -= padding * span
    maxs += padding * span

    return list(zip(mins, maxs))


def _calculate_ranges_analytical(distribution, largest_quantile, padding=0.05):
    """
    Calculate plotting ranges using analytical methods using mean and covariance.

    Parameters
    ----------
    distribution : Distribution
        Distribution to calculate ranges for.
    largest_quantile : float
        Largest quantile percentage to include.
    padding : float, optional
        Padding to add to each side of the range. Default is 0.05.

    Returns
    -------
    list of tuple
        List of (min, max) tuples for each dimension.
    """
    # Normalize quantile once
    largest_quantile /= 100.0

    # Handle scalar mean edge case
    mean = distribution.mean()
    if np.isscalar(mean):
        mean = np.array([mean])

    n_dims = len(mean)
    chi2_val = chi2.ppf(largest_quantile, df=n_dims)

    # GMM case
    if distribution.name == "GMM":
        component_ranges = []

        for mean, cov in zip(
            distribution.model.means_,
            distribution.model.covariances_,
        ):
            component_ranges.append(
                _ellipsoid_ranges(mean, cov, chi2_val, padding=0.0)
            )

        # Combine component-wise ranges
        ranges = []
        for dim in range(n_dims):
            min_val = min(r[dim][0] for r in component_ranges)
            max_val = max(r[dim][1] for r in component_ranges)

            span = max_val - min_val
            min_val -= padding * span
            max_val += padding * span

            ranges.append((min_val, max_val))

        return ranges

    # Single Gaussian case
    return _ellipsoid_ranges(
        mean,
        distribution.cov(),
        chi2_val,
        padding,
    )


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
    Calculate plotting ranges using numerical PDF evaluation.

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
    list of tuple
        List of (min, max) tuples for each dimension.
    """
    # Step 1: Coarse search
    mean = distribution.mean()
    if mean.shape == ():
        mean = np.array([mean])
    if len(mean.shape) == 0:
        mean = np.array([mean])

    n_dims = len(mean)

    # Set initial radius based on covariance
    cov = distribution.cov()
    if len(cov.shape) == 1:
        std_max = np.sqrt(np.max(cov))
    else:
        std_max = np.sqrt(np.max(np.diag(cov)))
    r = 2.0 * std_max

    # Expand radius until PDF at boundary points is below threshold
    while r < max_range:
        # Sample points at the boundaries
        test_points = []
        for dim in range(n_dims):
            point_neg = mean.copy()
            point_neg[dim] -= r
            point_pos = mean.copy()
            point_pos[dim] += r
            test_points.extend([point_neg, point_pos])

        test_points = np.array(test_points)
        pdf_vals = distribution.pdf(test_points)

        if np.all(pdf_vals < threshold):
            break

        r *= factor

    # Initial coarse ranges
    coarse_ranges = [(mean[dim] - r, mean[dim] + r) for dim in range(n_dims)]

    # Step 2: Fine search - refine to find tight bounding box
    if n_dims == 2:
        ranges = _refine_ranges_2d(
            distribution,
            largest_quantile,
            coarse_ranges,
            resolution,
            padding
        )
    else:
        # For higher dimensions, use the coarse ranges with padding
        ranges = []
        for dim_range in coarse_ranges:
            min_val, max_val = dim_range
            range_span = max_val - min_val
            ranges.append((
                min_val + padding * range_span,
                max_val - padding * range_span
            ))

    return ranges


def _refine_ranges_2d(
    distribution,
    largest_quantile,
    coarse_ranges,
    resolution=128,
    padding=0.05,
):
    """
    Refine ranges for 2D distributions by finding contour bounding box.

    Parameters
    ----------
    distribution : Distribution
        2D distribution to calculate ranges for.
    largest_quantile : float
        Largest quantile percentage to include.
    coarse_ranges : list of tuple
        Coarse ranges from initial search.
    resolution : int, optional
        Grid resolution. Default is 128.
    padding : float, optional
        Padding to add to each side of the range. Default is 0.05.

    Returns
    -------
    list of tuple
        Refined (min, max) tuples for each dimension.
    """
    # Create grid over coarse ranges
    x = np.linspace(coarse_ranges[0][0], coarse_ranges[0][1], resolution)
    y = np.linspace(coarse_ranges[1][0], coarse_ranges[1][1], resolution)
    xv, yv = np.meshgrid(x, y)

    # Evaluate PDF on grid
    coords = np.stack((xv, yv), axis=-1).reshape(-1, 2)
    pdf = distribution.pdf(coords).reshape(xv.shape)

    # Normalize to create a proper PDF
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    pdf_sum = np.sum(pdf) * dx * dy
    if pdf_sum > 0:
        pdf /= pdf_sum

    # Sort PDF values in descending order
    sorted_pdf = np.sort(pdf.flatten())[::-1]

    # Calculate cumulative probability
    cumulative = np.cumsum(sorted_pdf) * dx * dy

    # Find the density threshold for the desired quantile
    idx = np.searchsorted(cumulative, largest_quantile / 100.0)
    if idx >= len(sorted_pdf):
        idx = len(sorted_pdf) - 1
    iso_min = sorted_pdf[idx]

    # Find all points above this threshold
    mask = pdf >= iso_min
    xs = xv[mask]
    ys = yv[mask]

    if len(xs) == 0:
        # Fallback to coarse ranges if no points found
        return coarse_ranges

    # Calculate bounding box
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    # Add padding as percentage of range
    x_range = x_max - x_min
    y_range = y_max - y_min

    return [
        (x_min - padding * x_range, x_max + padding * x_range),
        (y_min - padding * y_range, y_max + padding * y_range),
    ]