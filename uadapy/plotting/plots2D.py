import matplotlib.pyplot as plt
import numpy as np
from uadapy import Distribution
from matplotlib.colors import ListedColormap
import uadapy.plotting.utils as utils

def plot_samples(distributions, n_samples, seed=55, xlabel=None, ylabel=None, title=None, show_plot=False):
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

    colors = utils.get_colors(len(distributions))

    for i, d in enumerate(distributions):
        samples = d.sample(n_samples, seed)
        plt.scatter(x=samples[:,0], y=samples[:,1], color=colors[i])
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

def plot_contour(distributions, resolution=128, ranges=None, quantiles:list=None, seed=55, show_plot=False):
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
        List of quantiles to use for determining isovalues. If None, the 99.7%, 95%, and 68% quantiles are used.
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

    contour_colors = utils.get_colors(len(distributions))

    if ranges is None:
        min_val = np.zeros(distributions[0].mean().shape)+1000
        max_val = np.zeros(distributions[0].mean().shape)-1000
        cov_max = np.zeros(distributions[0].mean().shape)
        for d in distributions:
            min_val=np.min(np.array([d.mean(), min_val]), axis=0)
            max_val=np.max(np.array([d.mean(), max_val]), axis=0)
            cov_max = np.max(np.array([np.diagonal(d.cov()), cov_max]), axis=0)
        cov_max = np.sqrt(cov_max)
        ranges = [(mi-3*co, ma+3*co) for mi,ma, co in zip(min_val, max_val, cov_max)]
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
        color = contour_colors[i]

        # Monte Carlo approach for determining isovalues
        isovalues = []
        n_samples = 10_000  #TODO: cleverly determine how many samples are needed based on the largest quantile
        samples = d.sample(n_samples, seed)
        densities = d.pdf(samples)
        densities.sort()
        if quantiles is None:
            isovalues.append(densities[int((1 - 99.7/100) * n_samples)]) # 99.7% quantile
            isovalues.append(densities[int((1 - 95/100) * n_samples)]) # 95% quantile
            isovalues.append(densities[int((1 - 68/100) * n_samples)]) # 68% quantile
        else:
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

def plot_contour_bands(distributions, n_samples, resolution=128, ranges=None, quantiles: list = None, seed=55,
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
        List of quantiles to use for determining isovalues. If None, the 99.7%, 95%, and 68% quantiles are used.
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

    n_quantiles = len(quantiles)
    alpha_values = np.linspace(1/n_quantiles, 1.0, n_quantiles)  # Creates alpha values from 1/n to 1.0
    custom_cmap = utils.create_shaded_set2_colormap(alpha_values)

    if ranges is None:
        min_val = np.zeros(distributions[0].mean().shape)+1000
        max_val = np.zeros(distributions[0].mean().shape)-1000
        cov_max = np.zeros(distributions[0].mean().shape)
        for d in distributions:
            min_val=np.min(np.array([d.mean(), min_val]), axis=0)
            max_val=np.max(np.array([d.mean(), max_val]), axis=0)
            cov_max = np.max(np.array([np.diagonal(d.cov()), cov_max]), axis=0)
        cov_max = np.sqrt(cov_max)
        ranges = [(mi-3*co, ma+3*co) for mi,ma, co in zip(min_val, max_val, cov_max)]
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
        samples = d.sample(n_samples, seed)
        densities = d.pdf(samples)
        densities.sort()
        if quantiles is None:
            isovalues.append(densities[int((1 - 99.7/100) * n_samples)]) # 99.7% quantile
            isovalues.append(densities[int((1 - 95/100) * n_samples)]) # 95% quantile
            isovalues.append(densities[int((1 - 68/100) * n_samples)]) # 68% quantile
        else:
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
