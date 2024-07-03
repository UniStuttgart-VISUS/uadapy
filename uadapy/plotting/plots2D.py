import matplotlib.pyplot as plt
import numpy as np
import uadapy.distribution as dist
import uadapy.plotting.utils as utils

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

def plot_contour(distributions, resolution=128, ranges=None, quantiles:list=None, seed=55, distrib_colors=None, **kwargs):
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
    if distrib_colors is None:
        distrib_colors = utils.generate_spectrum_colors(len(distributions))

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
    fig, ax = plt.subplots()
    for i, d in enumerate(distributions):
        x = np.linspace(range_x[0], range_x[1], resolution)
        y = np.linspace(range_y[0], range_y[1], resolution)
        xv, yv = np.meshgrid(x, y)
        coordinates = np.stack((xv, yv), axis=-1)
        coordinates = coordinates.reshape((-1, 2))
        pdf = d.pdf(coordinates)
        pdf = pdf.reshape(xv.shape)
        color = distrib_colors[i]

        # Monte Carlo approach for determining isovalues
        isovalues = []
        num_samples = 10_000  #TODO: cleverly determine how many samples are needed based on the largest quantile
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

        ax.contour(xv, yv, pdf, levels=isovalues, colors = [color])
    return fig,ax

def plot_contour_bands(distributions, resolution=128, ranges=None, quantiles:list=None, seed=55, distrib_colors=None, fig=None, ax=None, **kwargs):
    """
    Plot contour bands for samples drawn from given distributions.

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
    if distrib_colors is None:
        distrib_colors = utils.generate_spectrum_colors(len(distributions))

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
        ranges = [(min-3*co, max+3*co) for min,max, co in zip(min_val, max_val, cov_max)]
    range_x = ranges[0]
    range_y = ranges[1]

    if ax is None:
        fig, ax = plt.subplots()
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
        num_samples = 10_000
        isovalues = []
        samples = d.sample(num_samples, seed)
        densities = d.pdf(samples)
        densities.sort()
        if quantiles is None:
            isovalues.append(densities[int((1 - 95/100) * num_samples)]) # 95% quantile
            isovalues.append(densities[int((1 - 75/100) * num_samples)]) # 75% quantile
            isovalues.append(densities[int((1 - 25/100) * num_samples)]) # 25% quantile
        else:
            quantiles.sort(reverse=True)
            for quantile in quantiles:
                if not 0 < quantile < 100:
                    raise ValueError(f"Invalid quantile: {quantile}. Quantiles must be between 0 and 100 (exclusive).")
                elif int((1 - quantile/100) * num_samples) >= num_samples:
                    raise ValueError(f"Quantile {quantile} results in an index that is out of bounds.")
                isovalues.append(densities[int((1 - quantile/100) * num_samples)])
        colors = [
            distrib_colors[i],
            utils.scale_saturation(utils.scale_brightness(utils.any_color_to_rgb(distrib_colors[i]), 0.8), 0.9)
        ]
        if isovalues[-1] != np.infty:
            isovalues.append(np.infty)
        # Generate logarithmic levels and create the contour plot with different colormap for each distribution
        ax.contourf(xv, yv, pdf, levels=isovalues, colors=colors, alpha=0.5)
    return fig, ax
