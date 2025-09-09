import matplotlib.pyplot as plt
import numpy as np
from uadapy import Distribution
import uadapy.plotting.utils as utils
import glasbey as gb

def plot_samples(distributions,
                 n_samples,
                 seed=55,
                 fig=None,
                 axs=None,
                 distrib_colors=None,
                 colorblind_safe=False,
                 show_plot=False):
    """
    Plot samples from the multivariate distribution as a SLOM.

    Parameters
    ----------
    distributions : list
        List of distributions to plot.
    n_samples : int
        Number of samples per distribution.
    seed : int
        Seed for the random number generator for reproducibility. It defaults to 55 if not provided.
    fig : matplotlib.figure.Figure or None, optional
        Figure object to use for plotting. If None, a new figure will be created.
    axs : Array of matplotlib.axes.Axes or None, optional
        Axes objects to use for plotting. If None, new axes will be created.
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

    n_dims = distributions[0].n_dims

    if axs is None:
        if fig is None:
            fig, axs = plt.subplots(nrows=n_dims, ncols=n_dims)
        else:
            if fig.axes is not None:
                axs = np.array(fig.axes).reshape(n_dims, n_dims)
            else:
                raise ValueError("The provided figure has no axes. Pass an Axes or create subplots first.")
    else:
        if fig is None:
            fig = axs[0, 0].figure if isinstance(axs, np.ndarray) else axs.figure

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

    for ax in axs.flat:
        # Hide all ticks and labels
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

    # Fill matrix with data
    for k, d in enumerate(distributions):
        if d.n_dims < 2:
            raise Exception('Wrong dimension of distribution')
        samples = d.sample(n_samples, seed)
        for i, j in zip(*np.triu_indices_from(axs, k=1)):
            for x, y in [(i, j), (j, i)]:
                axs[x,y].scatter(samples[:,y], y=samples[:,x], color=palette[k])

        # Fill diagonal
        for i in range(n_dims):
            axs[i,i].hist(samples[:,i], histtype='stepfilled', fill=False, alpha=1.0, density=True, ec=palette[k])
            axs[i,i].xaxis.set_visible(True)
            axs[i,i].yaxis.set_visible(True)

        for i in range(n_dims):
            axs[-1,i].xaxis.set_visible(True)
            axs[i,0].yaxis.set_visible(True)
        axs[0,1].yaxis.set_visible(True)

    if show_plot:
        fig.tight_layout()
        plt.show()

    return fig, axs

def plot_contour(distributions,
                 n_samples,
                 resolution=128,
                 ranges=None,
                 quantiles: list = None,
                 seed=55,
                 fig=None,
                 axs=None,
                 distrib_colors=None,
                 colorblind_safe=False,
                 show_plot=False):
    """
    Visualizes a multidimensional distribution in a matrix of contour plots.

    Parameters
    ----------
    distributions : list
        List of distributions to plot.
    n_samples : int
        Number of samples per distribution.
    resolution : int, optional
        The resolution for the pdf. Default is 128.
    ranges : list or None, optional
        Array of ranges for all dimensions. If None, the ranges are calculated based on the distributions.
    quantiles : list or None, optional
        List of quantiles to use for determining isovalues. If None, the 95%, 75%, and 25% quantiles are used.
    seed : int
        Seed for the random number generator for reproducibility. It defaults to 55 if not provided.
    fig : matplotlib.figure.Figure or None, optional
        Figure object to use for plotting. If None, a new figure will be created.
    axs : Array of matplotlib.axes.Axes or None, optional
        Axes objects to use for plotting. If None, new axes will be created.
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
    Exception
        If the dimension of the distribution is less than 2.
    """

    if isinstance(distributions, Distribution):
        distributions = [distributions]

    n_dims = distributions[0].n_dims

    if axs is None:
        if fig is None:
            fig, axs = plt.subplots(nrows=n_dims, ncols=n_dims)
        else:
            if fig.axes is not None:
                axs = np.array(fig.axes).reshape(n_dims, n_dims)
            else:
                raise ValueError("The provided figure has no axes. Pass an Axes or create subplots first.")
    else:
        if fig is None:
            fig = axs[0, 0].figure if isinstance(axs, np.ndarray) else axs.figure

    # Determine default quantiles: 25%, 75%, and 95%
    if quantiles is None:
        quantiles = [25, 75, 95]
    largest_quantile = max(quantiles)

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

    for i, ax in enumerate(axs.flat):
        # Hide all ticks and labels
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

    # Fill matrix with data
    for k, d in enumerate(distributions):
        if d.n_dims < 2:
            raise Exception('Wrong dimension of distribution')
        dims = ()
        test = ()
        for i in range(d.n_dims):
            test = (*test, i)
            x = np.linspace(ranges[i][0], ranges[i][1], resolution)
            dims = (*dims, x)
        coordinates = np.array(np.meshgrid(*dims)).transpose(tuple(range(1, n_dims+1)) + (0,))
        pdf = d.pdf(coordinates.reshape((-1, coordinates.shape[-1])))
        pdf = pdf.reshape(coordinates.shape[:-1])
        pdf = pdf.transpose((1,0)+tuple(range(2,n_dims)))

        # Monte Carlo approach for determining isovalues
        isovalues = []
        samples = distrib_samples[k]
        densities = d.pdf(samples)
        densities.sort()
        quantiles.sort(reverse=True)
        for quantile in quantiles:
            if not 0 < quantile < 100:
                raise ValueError(f"Invalid quantile: {quantile}. Quantiles must be between 0 and 100 (exclusive).")
            elif int((1 - quantile/100) * n_samples) >= n_samples:
                raise ValueError(f"Quantile {quantile} results in an index that is out of bounds.")
            isovalues.append(densities[int((1 - quantile/100) * n_samples)])

        for i, j in zip(*np.triu_indices_from(axs, k=1)):
            for x, y in [(i, j), (j, i)]:
                color = palette[k]
                indices = list(np.arange(d.n_dims))
                indices.remove(x)
                indices.remove(y)
                pdf_agg = np.sum(pdf, axis=tuple(indices))
                if x > y:
                    pdf_agg = pdf_agg.T
                axs[x,y].contour(dims[y], dims[x], pdf_agg, levels=isovalues, colors=[color])

        # Fill diagonal
        for i in range(n_dims):
            indices = list(np.arange(d.n_dims))
            indices.remove(i)
            axs[i,i].plot(dims[i], np.sum(pdf, axis=tuple(indices)), color=color)
            axs[i,i].xaxis.set_visible(True)
            axs[i,i].yaxis.set_visible(True)

        for i in range(n_dims):
            axs[-1,i].xaxis.set_visible(True)
            axs[i,0].yaxis.set_visible(True)
        axs[0,1].yaxis.set_visible(True)

    if show_plot:
        fig.tight_layout()
        plt.show()

    return fig, axs

def plot_contour_samples(distributions,
                         n_samples,
                         resolution=128,
                         ranges=None,
                         quantiles: list = None,
                         seed=55,
                         fig=None,
                         axs=None,
                         distrib_colors=None,
                         colorblind_safe=False,
                         show_plot=False):
    """
    Visualizes a multidimensional distribution in a matrix visualization where the
    upper diagonal contains contour plots and the lower diagonal contains scatterplots.

    Parameters
    ----------
    distributions : list
        List of distributions to plot.
    n_samples : int
        Number of samples for the scatterplot.
    resolution : int, optional
        The resolution for the pdf. Default is 128.
    ranges : list or None, optional
        Array of ranges for all dimensions. If None, the ranges are calculated based on the distributions.
    quantiles : list or None, optional
        List of quantiles to use for determining isovalues. If None, the 95%, 75%, and 25% quantiles are used.
    seed : int
        Seed for the random number generator for reproducibility. It defaults to 55 if not provided.
    fig : matplotlib.figure.Figure or None, optional
        Figure object to use for plotting. If None, a new figure will be created.
    axs : Array of matplotlib.axes.Axes or None, optional
        Axes objects to use for plotting. If None, new axes will be created.
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
    Exception
        If the dimension of the distribution is less than 2.
    """

    if isinstance(distributions, Distribution):
        distributions = [distributions]

    n_dims = distributions[0].n_dims

    if axs is None:
        if fig is None:
            fig, axs = plt.subplots(nrows=n_dims, ncols=n_dims)
        else:
            if fig.axes is not None:
                axs = np.array(fig.axes).reshape(n_dims, n_dims)
            else:
                raise ValueError("The provided figure has no axes. Pass an Axes or create subplots first.")
    else:
        if fig is None:
            fig = axs[0, 0].figure if isinstance(axs, np.ndarray) else axs.figure

    # Determine default quantiles: 25%, 75%, and 95%
    if quantiles is None:
        quantiles = [25, 75, 95]
    largest_quantile = max(quantiles)

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

    for i, ax in enumerate(axs.flat):
        # Hide all ticks and labels
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

    # Fill matrix with data
    for k, d in enumerate(distributions):
        if d.n_dims < 2:
            raise Exception('Wrong dimension of distribution')
        dims = ()
        for i in range(d.n_dims):
            x = np.linspace(ranges[i][0], ranges[i][1], resolution)
            dims = (*dims, x)
        coordinates = np.array(np.meshgrid(*dims)).transpose(tuple(range(1, n_dims+1)) + (0,))
        pdf = d.pdf(coordinates.reshape((-1, coordinates.shape[-1])))
        pdf = pdf.reshape(coordinates.shape[:-1])
        pdf = pdf.transpose((1,0)+tuple(range(2,n_dims)))

        # Monte Carlo approach for determining isovalues
        isovalues = []
        samples = distrib_samples[k]
        densities = d.pdf(samples)
        densities.sort()
        quantiles.sort(reverse=True)
        for quantile in quantiles:
            if not 0 < quantile < 100:
                raise ValueError(f"Invalid quantile: {quantile}. Quantiles must be between 0 and 100 (exclusive).")
            elif int((1 - quantile/100) * n_samples) >= n_samples:
                raise ValueError(f"Quantile {quantile} results in an index that is out of bounds.")
            isovalues.append(densities[int((1 - quantile/100) * n_samples)])

        for i, j in zip(*np.triu_indices_from(axs, k=1)):
            for x, y in [(i, j), (j, i)]:
                color = palette[k]
                indices = list(np.arange(d.n_dims))
                indices.remove(x)
                indices.remove(y)
                pdf_agg = np.sum(pdf, axis=tuple(indices))
                if x < y:
                    axs[x,y].contour(dims[y], dims[x], pdf_agg, levels=isovalues, colors=[color])
                else:
                    axs[x, y].scatter(samples[:, y], y=samples[:, x], color=palette[k])

                axs[x, y].set_xlim(ranges[y][0], ranges[y][1])
                axs[x, y].set_ylim(ranges[x][0], ranges[x][1])

        # Fill diagonal
        for i in range(n_dims):
            indices = list(np.arange(d.n_dims))
            indices.remove(i)
            axs[i,i].plot(dims[i], np.sum(pdf, axis=tuple(indices)), color=color)
            axs[i,i].set_xlim(ranges[i][0], ranges[i][1])
            axs[i,i].xaxis.set_visible(True)
            axs[i,i].yaxis.set_visible(True)

        for i in range(n_dims):
            axs[-1,i].xaxis.set_visible(True)
            axs[i,0].yaxis.set_visible(True)
        axs[0,1].yaxis.set_visible(True)

    if show_plot:
        fig.tight_layout()
        plt.show()

    return fig, axs