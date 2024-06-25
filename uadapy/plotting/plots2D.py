import matplotlib.pyplot as plt
import numpy as np
import uadapy.distribution as dist
from numpy import ma
from matplotlib import ticker

def plot_samples(distribution, num_samples, **kwargs):
    """
    Plot samples from the given distribution. If several distributions should be
    plotted together, an array can be passed to this function
    :param distribution: Distributions to plot
    :param num_samples: Number of samples per distribution
    :param kwargs: Optional other arguments to pass:
        xlabel for label of x-axis
        ylabel for label of y-axis
    :return:
    """
    if isinstance(distribution, dist.distribution):
        distribution = [distribution]
    for d in distribution:
        samples = d.sample(num_samples)
        plt.scatter(x=samples[:,0], y=samples[:,1])
    if 'xlabel' in kwargs:
        plt.xlabel(kwargs['xlabel'])
    if 'ylabel' in kwargs:
        plt.ylabel(kwargs['ylabel'])
    plt.show()

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
        List of quantiles to use for determining isovalues. If None, the 99.7%, 95%, and 68% quantiles are used.
    seed : int
        Seed for the random number generator for reproducibility. It defaults to 55 if not provided.
    **kwargs : additional keyword arguments
        Additional optional plotting arguments.

    Returns
    -------
    None
        This function does not return a value. It displays a plot using plt.show().

    Raises
    ------
    ValueError
        If a quantile is not between 0 and 100 (exclusive), or if a quantile results in an index that is out of bounds.
    """

    if isinstance(distributions, dist.distribution):
        distributions = [distributions]
    contour_colors = generate_spectrum_colors(len(distributions))

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
        num_samples = 10_000
        samples = d.sample(num_samples, seed)
        densities = d.pdf(samples)
        densities.sort()
        if quantiles is None:
            isovalues.append(densities[int((1 - 99.7/100) * num_samples)]) # 99.7% quantile
            isovalues.append(densities[int((1 - 95/100) * num_samples)]) # 95% quantile
            isovalues.append(densities[int((1 - 68/100) * num_samples)]) # 68% quantile
        else:
            quantiles.sort(reverse=True)
            for quantile in quantiles:
                if not 0 < quantile < 100:
                    raise ValueError(f"Invalid quantile: {quantile}. Quantiles must be between 0 and 100 (exclusive).")
                elif int((1 - quantile/100) * num_samples) >= num_samples:
                    raise ValueError(f"Quantile {quantile} results in an index that is out of bounds.")
                isovalues.append(densities[int((1 - quantile/100) * num_samples)])

        plt.contour(xv, yv, pdf, levels=isovalues, colors = [color])
    plt.show()

def plot_contour_bands(distributions, num_samples, resolution=128, ranges=None, quantiles:list=None, seed=55, **kwargs):
    """
    Plot contour bands for samples drawn from given distributions.

    Parameters
    ----------
    distributions : list
        List of distributions to plot.
    num_samples : int
        Number of samples per distribution.
    resolution : int, optional
        The resolution of the plot. Default is 128.
    ranges : list or None, optional
        The ranges for the x and y axes. If None, the ranges are calculated based on the distributions.
    quantiles : list or None, optional
        List of quantiles to use for determining isovalues. If None, the 99.7%, 95%, and 68% quantiles are used.
    seed : int
        Seed for the random number generator for reproducibility. It defaults to 55 if not provided.
    **kwargs : additional keyword arguments
        Additional optional plotting arguments.

    Returns
    -------
    None
        This function does not return a value. It displays a plot using plt.show().

    Raises
    ------
    ValueError
        If a quantile is not between 0 and 100 (exclusive), or if a quantile results in an index that is out of bounds.
    """

    if isinstance(distributions, dist.distribution):
        distributions = [distributions]

    # Sequential and perceptually uniform colormaps
    colormaps = [
        'Reds', 'Blues', 'Greens', 'Greys', 'Oranges', 'Purples',
        'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
        'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn',
        'viridis', 'plasma', 'inferno', 'magma', 'cividis'
    ]

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
        pdf = ma.masked_where(pdf <= 0, pdf)  # Mask non-positive values to avoid log scale issues

        # Monte Carlo approach for determining isovalues
        isovalues = []
        samples = d.sample(num_samples, seed)
        densities = d.pdf(samples)
        densities.sort()
        if quantiles is None:
            isovalues.append(densities[int((1 - 99.7/100) * num_samples)]) # 99.7% quantile
            isovalues.append(densities[int((1 - 95/100) * num_samples)]) # 95% quantile
            isovalues.append(densities[int((1 - 68/100) * num_samples)]) # 68% quantile
        else:
            quantiles.sort(reverse=True)
            for quantile in quantiles:
                if not 0 < quantile < 100:
                    raise ValueError(f"Invalid quantile: {quantile}. Quantiles must be between 0 and 100 (exclusive).")
                elif int((1 - quantile/100) * num_samples) >= num_samples:
                    raise ValueError(f"Quantile {quantile} results in an index that is out of bounds.")
                isovalues.append(densities[int((1 - quantile/100) * num_samples)])

        # Generate logarithmic levels and create the contour plot with different colormap for each distribution
        plt.contourf(xv, yv, pdf, levels=isovalues, locator=ticker.LogLocator(), cmap=colormaps[i % len(colormaps)])

    plt.show()

# HELPER FUNCTIONS
def generate_random_colors(length):
    return ["#"+''.join([np.random.choice('0123456789ABCDEF') for j in range(6)]) for _ in range(length)]

def generate_spectrum_colors(length):
    cmap = plt.cm.get_cmap('viridis', length)  # You can choose different colormaps like 'jet', 'hsv', 'rainbow', etc.
    return np.array([cmap(i) for i in range(length)])