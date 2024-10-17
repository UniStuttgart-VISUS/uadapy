import matplotlib.pyplot as plt
import numpy as np
from uadapy import Distribution
from numpy import ma
from matplotlib import ticker

def plot_samples(distributions, n_samples, seed=55, **kwargs):
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
    **kwargs : additional keyword arguments
        Additional optional plotting arguments.
        - xlabel : string, optional
            label for x-axis.
        - ylabel : string, optional
            label for y-axis.
        - show_plot : bool, optional
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
        samples = d.sample(n_samples, seed)
        plt.scatter(x=samples[:,0], y=samples[:,1])
    if 'xlabel' in kwargs:
        plt.xlabel(kwargs['xlabel'])
    if 'ylabel' in kwargs:
        plt.ylabel(kwargs['ylabel'])
    if 'title' in kwargs:
        plt.title(kwargs['title'])

    # Get the current figure and axes
    fig = plt.gcf()
    axs = plt.gca()

    show_plot = kwargs.get('show_plot', False)
    if show_plot:
        fig.tight_layout()
        plt.show()

    return fig, axs

def plot_contour(distributions, resolution=128, ranges=None, quantiles:list=None, seed=55, **kwargs):
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
    **kwargs : additional keyword arguments
        Additional optional plotting arguments.
        - show_plot : bool, optional
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
    contour_colors = generate_spectrum_colors(len(distributions))

    # Determine default quantiles: 25%, 75%, and 95%
    if quantiles is None:
        quantiles = [25, 75, 95]
    largest_quantile = max(quantiles)

    if ranges is None:
        min_val = np.zeros(distributions[0].mean().shape)+1000
        max_val = np.zeros(distributions[0].mean().shape)-1000
        cov_max = np.zeros(distributions[0].mean().shape)

        # Dynamically set the expansion factor based on the largest quantile
        if largest_quantile >= 99.99:
            ef = 6  # 6 standard deviations for 99.9% quantiles
        elif largest_quantile >= 99.9:
            ef = 5  # 5 standard deviations for 99.9% quantiles
        elif largest_quantile >= 99:
            ef = 4  # 4 standard deviations for 99% quantiles
        else:
            ef = 3  # Default to 3 standard deviations

        for d in distributions:
            min_val=np.min(np.array([d.mean(), min_val]), axis=0)
            max_val=np.max(np.array([d.mean(), max_val]), axis=0)
            cov_max = np.max(np.array([np.diagonal(d.cov()), cov_max]), axis=0)
        cov_max = np.sqrt(cov_max)
        ranges = [(mi-ef*co, ma+ef*co) for mi,ma, co in zip(min_val, max_val, cov_max)]

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

    show_plot = kwargs.get('show_plot', False)
    if show_plot:
        fig.tight_layout()
        plt.show()

    return fig, axs

def plot_contour_bands(distributions, n_samples, resolution=128, ranges=None, quantiles: list = None, seed=55,
                       **kwargs):
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
    **kwargs : additional keyword arguments
        Additional optional plotting arguments.
        - show_plot : bool, optional
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

    # Sequential and perceptually uniform colormaps
    colormaps = [
        'Reds', 'Blues', 'Greens', 'Greys', 'Oranges', 'Purples',
        'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
        'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn',
        'viridis', 'plasma', 'inferno', 'magma', 'cividis'
    ]

    # Determine default quantiles: 25%, 75%, and 95%
    if quantiles is None:
        quantiles = [25, 75, 95]
    largest_quantile = max(quantiles)

    if ranges is None:
        min_val = np.zeros(distributions[0].mean().shape)+1000
        max_val = np.zeros(distributions[0].mean().shape)-1000
        cov_max = np.zeros(distributions[0].mean().shape)

        # Dynamically set the expansion factor based on the largest quantile
        if largest_quantile >= 99.99:
            ef = 6  # 6 standard deviations for 99.9% quantiles
        elif largest_quantile >= 99.9:
            ef = 5  # 5 standard deviations for 99.9% quantiles
        elif largest_quantile >= 99:
            ef = 4  # 4 standard deviations for 99% quantiles
        else:
            ef = 3  # Default to 3 standard deviations

        for d in distributions:
            min_val=np.min(np.array([d.mean(), min_val]), axis=0)
            max_val=np.max(np.array([d.mean(), max_val]), axis=0)
            cov_max = np.max(np.array([np.diagonal(d.cov()), cov_max]), axis=0)
        cov_max = np.sqrt(cov_max)
        ranges = [(mi-ef*co, ma+ef*co) for mi,ma, co in zip(min_val, max_val, cov_max)]

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
        quantiles.sort(reverse=True)
        for quantile in quantiles:
            if not 0 < quantile < 100:
                raise ValueError(f"Invalid quantile: {quantile}. Quantiles must be between 0 and 100 (exclusive).")
            elif int((1 - quantile/100) * n_samples) >= n_samples:
                raise ValueError(f"Quantile {quantile} results in an index that is out of bounds.")
            isovalues.append(densities[int((1 - quantile/100) * n_samples)])

        # Generate logarithmic levels and create the contour plot with different colormap for each distribution
        plt.contourf(xv, yv, pdf, levels=isovalues, locator=ticker.LogLocator(), cmap=colormaps[i % len(colormaps)])

    # Get the current figure and axes
    fig = plt.gcf()
    axs = plt.gca()

    show_plot = kwargs.get('show_plot', False)
    if show_plot:
        fig.tight_layout()
        plt.show()

    return fig, axs

# HELPER FUNCTIONS
def generate_random_colors(length):
    return ["#"+''.join([np.random.choice('0123456789ABCDEF') for j in range(6)]) for _ in range(length)]

def generate_spectrum_colors(length):
    cmap = plt.cm.get_cmap('viridis', length)  # You can choose different colormaps like 'jet', 'hsv', 'rainbow', etc.
    return np.array([cmap(i) for i in range(length)])