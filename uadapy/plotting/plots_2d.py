import matplotlib.pyplot as plt
import numpy as np
from uadapy import Distribution
from matplotlib.colors import ListedColormap
import uadapy.plotting.utils as utils
import glasbey as gb

def plot_samples(distributions,
                 n_samples,
                 seed=55,
                 xlabel=None,
                 ylabel=None,
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
    xlabel : string, optional
        label for x-axis.
    ylabel : string, optional
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

    for i, d in enumerate(distributions):
        samples = d.sample(n_samples, seed)
        plt.scatter(x=samples[:,0], y=samples[:,1], color=palette[i])
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if title:
        plt.title(title)

    # Get the current figure and axes
    fig = plt.gcf()
    axs = plt.gca()

    if show_plot:
        fig.tight_layout()
        plt.show()

    return fig, axs

def plot_contour(distributions,
                 resolution=128,
                 ranges=None,
                 quantiles:list=None,
                 seed=55,
                 distrib_colors=None,
                 colorblind_safe=False,
                 show_plot=False):
    """
    Plot contour plots for samples drawn from given distributions.

    Parameters
    ----------
    distributions : list
        List of distributions to plot.
    resolution : int, optional
        The resolution of the plot. Default is 128.
    ranges : list or None, optional
        The ranges for the x and y axes. If None, the ranges are calculated based on the distributions.
    quantiles : list or None, optional
        List of quantiles to use for determining isovalues. If None, the 95%, 75%, and 25% quantiles are used.
    seed : int
        Seed for the random number generator for reproducibility. It defaults to 55 if not provided.
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

    Raises
    ------
    ValueError
        If a quantile is not between 0 and 100 (exclusive), or if a quantile results in an index that is out of bounds.
    """

    if isinstance(distributions, Distribution):
        distributions = [distributions]

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

    # Determine default quantiles: 25%, 75%, and 95%
    if quantiles is None:
        quantiles = [25, 75, 95]
    largest_quantile = max(quantiles)

    distrib_samples = []
    n_samples = 10_000  #TODO: cleverly determine how many samples are needed based on the largest quantile
    for d in distributions:
        samples = d.sample(n_samples, seed)
        distrib_samples.append(samples)

    # Dynamically determine ranges using samples
    if ranges is None:
        all_samples = np.concatenate(distrib_samples, axis=0)
        initial_ranges = [
            (np.percentile(all_samples[:, dim], 0), np.percentile(all_samples[:, dim], 100))
            for dim in range(all_samples.shape[1])
        ]

        # Dynamically adjust the expansion factor based on the largest quantile
        ranges = []
        base_expansion = 0.05  # Base expansion factor for moderate quantiles
        if largest_quantile >= 99.999:
            expansion_factor = 0.15  # Larger expansion for extreme quantiles
        elif largest_quantile >= 99.9:
            expansion_factor = 0.10
        elif largest_quantile >= 99:
            expansion_factor = 0.08
        else:
            expansion_factor = base_expansion

        # Expand the range slightly based on the data spread to ensure no cutoff
        for dim_range in initial_ranges:
            min_val, max_val = dim_range
            range_span = max_val - min_val
            expanded_min = min_val - expansion_factor * range_span
            expanded_max = max_val + expansion_factor * range_span
            ranges.append((expanded_min, expanded_max))

    range_x = ranges[0]
    range_y = ranges[1]

    for i, d in enumerate(distributions):
        x = np.linspace(range_x[0], range_x[1], resolution)
        y = np.linspace(range_y[0], range_y[1], resolution)
        xv, yv = np.meshgrid(x, y)
        coordinates = np.stack((xv, yv), axis=-1)
        coordinates = coordinates.reshape((-1, 2))
        pdf = d.pdf(coordinates)
        pdf = pdf.reshape(xv.shape)
        color = palette[i]

        # Monte Carlo approach for determining isovalues
        isovalues = []
        samples = distrib_samples[i]
        densities = d.pdf(samples)
        densities.sort()
        quantiles.sort(reverse=True)
        for quantile in quantiles:
            if not 0 < quantile < 100:
                raise ValueError(f"Invalid quantile: {quantile}. Quantiles must be between 0 and 100 (exclusive).")
            elif int((1 - quantile/100) * n_samples) >= n_samples:
                raise ValueError(f"Quantile {quantile} results in an index that is out of bounds.")
            isovalues.append(densities[int((1 - quantile/100) * n_samples)])

        plt.contour(xv, yv, pdf, levels=isovalues, colors = [color])

    # Get the current figure and axes
    fig = plt.gcf()
    axs = plt.gca()

    if show_plot:
        fig.tight_layout()
        plt.show()

    return fig, axs

def plot_contour_bands(distributions,
                       n_samples,
                       resolution=128,
                       ranges=None,
                       quantiles: list = None,
                       seed=55,
                       show_plot=False):
    """
    Plot contour bands for samples drawn from given distributions.

    Parameters
    ----------
    distributions : list
        List of distributions to plot.
    n_samples : int
        Number of samples per distribution.
    resolution : int, optional
        The resolution of the plot. Default is 128.
    ranges : list or None, optional
        The ranges for the x and y axes. If None, the ranges are calculated based on the distributions.
    quantiles : list or None, optional
        List of quantiles to use for determining isovalues. If None, the 95%, 75%, and 25% quantiles are used.
    seed : int
        Seed for the random number generator for reproducibility. It defaults to 55 if not provided.
    show_plot : bool, optional
        If True, display the plot.
        Default is False.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object containing the plot.
    list
        List of Axes objects used for plotting.

    Raises
    ------
    ValueError
        If a quantile is not between 0 and 100 (exclusive), or if a quantile results in an index that is out of bounds.
    """

    if isinstance(distributions, Distribution):
        distributions = [distributions]

    # Determine default quantiles: 25%, 75%, and 95%
    if quantiles is None:
        quantiles = [25, 75, 95]
    largest_quantile = max(quantiles)

    n_quantiles = len(quantiles)
    alpha_values = np.linspace(1/n_quantiles, 1.0, n_quantiles)  # Creates alpha values from 1/n to 1.0
    custom_cmap = utils.create_shaded_set2_colormap(alpha_values)

    distrib_samples = []
    for d in distributions:
        samples = d.sample(n_samples, seed)
        distrib_samples.append(samples)

    # Dynamically determine ranges using samples
    if ranges is None:
        all_samples = np.concatenate(distrib_samples, axis=0)
        initial_ranges = [
            (np.percentile(all_samples[:, dim], 0), np.percentile(all_samples[:, dim], 100))
            for dim in range(all_samples.shape[1])
        ]

        # Dynamically adjust the expansion factor based on the largest quantile
        ranges = []
        base_expansion = 0.05  # Base expansion factor for moderate quantiles
        if largest_quantile >= 99.999:
            expansion_factor = 0.15  # Larger expansion for extreme quantiles
        elif largest_quantile >= 99.9:
            expansion_factor = 0.10
        elif largest_quantile >= 99:
            expansion_factor = 0.08
        else:
            expansion_factor = base_expansion

        # Expand the range slightly based on the data spread to ensure no cutoff
        for dim_range in initial_ranges:
            min_val, max_val = dim_range
            range_span = max_val - min_val
            expanded_min = min_val - expansion_factor * range_span
            expanded_max = max_val + expansion_factor * range_span
            ranges.append((expanded_min, expanded_max))

    range_x = ranges[0]
    range_y = ranges[1]

    for i, d in enumerate(distributions):
        x = np.linspace(range_x[0], range_x[1], resolution)
        y = np.linspace(range_y[0], range_y[1], resolution)
        xv, yv = np.meshgrid(x, y)
        coordinates = np.stack((xv, yv), axis=-1)
        coordinates = coordinates.reshape((-1, 2))
        pdf = d.pdf(coordinates)
        pdf = pdf.reshape(xv.shape)
        pdf = np.ma.masked_where(pdf <= 0, pdf)  # Mask non-positive values to avoid log scale issues

        # Monte Carlo approach for determining isovalues
        isovalues = []
        samples = distrib_samples[i]
        densities = d.pdf(samples)
        densities.sort()
        
        quantiles.sort(reverse=True)
        for quantile in quantiles:
            if not 0 < quantile < 100:
                raise ValueError(f"Invalid quantile: {quantile}. Quantiles must be between 0 and 100 (exclusive).")
            elif int((1 - quantile/100) * n_samples) >= n_samples:
                raise ValueError(f"Quantile {quantile} results in an index that is out of bounds.")
            isovalues.append(densities[int((1 - quantile/100) * n_samples)])
        isovalues.append(densities[-1])  # Minimum density value

        # Extract the subset of colors corresponding to the current Set2 color and its 3 alpha variations
        start_idx = i * n_quantiles
        end_idx = start_idx + n_quantiles
        color_subset = custom_cmap.colors[start_idx:end_idx]

        # Create a ListedColormap for the current color and its alpha variations
        cmap_subset = ListedColormap(color_subset)

        # Generate the filled contour plot with transparency and better visibility
        plt.contourf(xv, yv, pdf, levels=isovalues, cmap=cmap_subset)

    # Get the current figure and axes
    fig = plt.gcf()
    axs = plt.gca()

    if show_plot:
        fig.tight_layout()
        plt.show()

    return fig, axs
