import matplotlib.pyplot as plt
import numpy as np
from uadapy import Distribution
import uadapy.plotting.utils as utils

def plot_samples(distributions, n_samples, seed=55, show_plot=False):
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
    # Create matrix
    n_dims = distributions[0].n_dims
    fig, axes = plt.subplots(nrows=n_dims, ncols=n_dims)
    contour_colors = utils.generate_spectrum_colors(len(distributions))
    for ax in axes.flat:
        # Hide all ticks and labels
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

    # Fill matrix with data
    for k, d in enumerate(distributions):
        if d.n_dims < 2:
            raise Exception('Wrong dimension of distribution')
        samples = d.sample(n_samples, seed)
        for i, j in zip(*np.triu_indices_from(axes, k=1)):
            for x, y in [(i, j), (j, i)]:
                axes[x,y].scatter(samples[:,y], y=samples[:,x], color=contour_colors[k])

        # Fill diagonal
        for i in range(n_dims):
            axes[i,i].hist(samples[:,i], histtype='stepfilled', fill=False, alpha=1.0, density=True, ec=contour_colors[k])
            axes[i,i].xaxis.set_visible(True)
            axes[i,i].yaxis.set_visible(True)

        for i in range(n_dims):
            axes[-1,i].xaxis.set_visible(True)
            axes[i,0].yaxis.set_visible(True)
        axes[0,1].yaxis.set_visible(True)

    # Get the current figure and axes
    fig = plt.gcf()
    axs = plt.gca()

    if show_plot:
        fig.tight_layout()
        plt.show()

    return fig, axs

def plot_contour(distributions, n_samples, resolution=128, ranges=None, quantiles: list = None, seed=55, show_plot=False):
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
    Exception
        If the dimension of the distribution is less than 2.
    """

    if isinstance(distributions, Distribution):
        distributions = [distributions]
    contour_colors = utils.generate_spectrum_colors(len(distributions))
    # Create matrix
    n_dims = distributions[0].n_dims
    if ranges is None:
        min_val = np.zeros(distributions[0].mean().shape)+1000
        max_val = np.zeros(distributions[0].mean().shape)-1000
        cov_max = np.zeros(np.diagonal(distributions[0].cov()).shape)
        for d in distributions:
            min_val=np.min(np.array([d.mean(), min_val]), axis=0)
            max_val=np.max(np.array([d.mean(), max_val]), axis=0)
            cov_max = np.max(np.array([np.diagonal(d.cov()), cov_max]), axis=0)
        cov_max = np.sqrt(cov_max)
        ranges = [(mi-3*co, ma+3*co) for mi,ma, co in zip(min_val, max_val, cov_max)]
    fig, axes = plt.subplots(nrows=n_dims, ncols=n_dims)
    for i, ax in enumerate(axes.flat):
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

        for i, j in zip(*np.triu_indices_from(axes, k=1)):
            for x, y in [(i, j), (j, i)]:
                color = contour_colors[k]
                indices = list(np.arange(d.n_dims))
                indices.remove(x)
                indices.remove(y)
                pdf_agg = np.sum(pdf, axis=tuple(indices))
                if x > y:
                    pdf_agg = pdf_agg.T
                axes[x,y].contour(dims[y], dims[x], pdf_agg, levels=isovalues, colors=[color])

        # Fill diagonal
        for i in range(n_dims):
            indices = list(np.arange(d.n_dims))
            indices.remove(i)
            axes[i,i].plot(dims[i], np.sum(pdf, axis=tuple(indices)), color=color)
            axes[i,i].xaxis.set_visible(True)
            axes[i,i].yaxis.set_visible(True)

        for i in range(n_dims):
            axes[-1,i].xaxis.set_visible(True)
            axes[i,0].yaxis.set_visible(True)
        axes[0,1].yaxis.set_visible(True)

    # Get the current figure and axes
    fig = plt.gcf()
    axs = plt.gca()

    if show_plot:
        fig.tight_layout()
        plt.show()

    return fig, axs

def plot_contour_samples(distributions, n_samples, resolution=128, ranges=None, quantiles: list = None, seed=55,
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
    Exception
        If the dimension of the distribution is less than 2.
    """

    if isinstance(distributions, Distribution):
        distributions = [distributions]
    contour_colors = utils.generate_spectrum_colors(len(distributions))
    # Create matrix
    n_dims = distributions[0].n_dims
    if ranges is None:
        min_val = np.zeros(distributions[0].mean().shape)+1000
        max_val = np.zeros(distributions[0].mean().shape)-1000
        cov_max = np.zeros(np.diagonal(distributions[0].cov()).shape)
        for d in distributions:
            min_val=np.min(np.array([d.mean(), min_val]), axis=0)
            max_val=np.max(np.array([d.mean(), max_val]), axis=0)
            cov_max = np.max(np.array([np.diagonal(d.cov()), cov_max]), axis=0)
        cov_max = np.sqrt(cov_max)
        ranges = [(mi-3*co, ma+3*co) for mi,ma, co in zip(min_val, max_val, cov_max)]
    fig, axes = plt.subplots(nrows=n_dims, ncols=n_dims)
    for i, ax in enumerate(axes.flat):
        # Hide all ticks and labels
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

    # Fill matrix with data
    for k, d in enumerate(distributions):
        samples = d.sample(n_samples, seed)
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

        for i, j in zip(*np.triu_indices_from(axes, k=1)):
            for x, y in [(i, j), (j, i)]:
                color = contour_colors[k]
                indices = list(np.arange(d.n_dims))
                indices.remove(x)
                indices.remove(y)
                pdf_agg = np.sum(pdf, axis=tuple(indices))
                if x < y:
                    axes[x,y].contour(dims[x], dims[y], pdf_agg, levels=isovalues, colors=[color])
                else:
                    axes[x, y].scatter(samples[:, y], y=samples[:, x], color=contour_colors[k])
                    axes[x, y].set_xlim(ranges[x][0], ranges[x][1])
                    axes[x, y].set_ylim(ranges[y][0], ranges[y][1])

        # Fill diagonal
        for i in range(n_dims):
            indices = list(np.arange(d.n_dims))
            indices.remove(i)
            axes[i,i].plot(dims[i], np.sum(pdf, axis=tuple(indices)), color=color)
            axes[i,i].xaxis.set_visible(True)
            axes[i,i].yaxis.set_visible(True)

        for i in range(n_dims):
            axes[-1,i].xaxis.set_visible(True)
            axes[i,0].yaxis.set_visible(True)
        axes[0,1].yaxis.set_visible(True)

    # Get the current figure and axes
    fig = plt.gcf()
    axs = plt.gca()

    if show_plot:
        fig.tight_layout()
        plt.show()

    return fig, axs