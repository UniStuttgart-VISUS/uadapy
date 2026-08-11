import matplotlib.pyplot as plt
import numpy as np
from uadapy import Distribution
import uadapy.plotting.utils as utils
import glasbey as gb

def plot_samples(distributions,
                 n_samples=100,
                 seed=55,
                 point_size=1,
                 alpha=1,
                 fig=None,
                 axs=None,
                 distrib_colors=None,
                 colorblind_safe=False,
                 show_plot=False,
                 plot_mask=None
):
    """
    Plot samples from the multivariate distribution as a SPLOM.

    Parameters
    ----------
    distributions : list
        List of distributions to plot.
    n_samples : int
        Number of samples per distribution.
    seed : int
        Seed for the random number generator for reproducibility. It defaults to 55 if not provided.
    point_size : float or None, optional
        Marker size (area in points^2). If None, matplotlib's default is used. By default 1.
    alpha : float, optional
        opacity value if the samples in the scatter plots. By default 1 (fully opaque)
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
    plot_mask: 2D boolean array or function (i,j)->bool, optional 
        mask that specifies which subplots will be generated. By default every plot will be generated.

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


    # default plot mask: all True
    if plot_mask is None:
        plot_mask = np.ones((n_dims,n_dims)) == 1
    import inspect
    if inspect.isfunction(plot_mask):
        plot_mask = [[plot_mask(i,j) for j in range(n_dims)] for i in range(n_dims)]
    if not isinstance(plot_mask, np.ndarray):
        plot_mask = np.array(plot_mask)


    for i in range(n_dims):
        for j in range(n_dims):
            if not plot_mask[i,j]:
                continue
            # Hide all ticks and labels
            axs[i,j].xaxis.set_visible(False)
            axs[i,j].yaxis.set_visible(False)


    # Fill matrix with data
    for k, d in enumerate(distributions):
        if d.n_dims < 2:
            raise Exception('Wrong dimension of distribution')
        samples = d.sample(n_samples, seed)
        for i, j in zip(*np.triu_indices_from(axs, k=1)):
            for x, y in [(i, j), (j, i)]:
                if not plot_mask[y,x]:
                    continue
                axs[y,x].scatter(samples[:,x], y=samples[:,y], color=palette[k], s=point_size, alpha=alpha)

        # Fill diagonal
        for i in range(n_dims):
            if not plot_mask[i,i]:
                    continue
            axs[i,i].hist(samples[:,i], histtype='stepfilled', fill=False, alpha=1.0, density=True, ec=palette[k])
            axs[i,i].yaxis.set_visible(True)

        for i in range(n_dims):
            if plot_mask[-1,i]:
                axs[-1,i].xaxis.set_visible(True)
            if plot_mask[i,0]:    
                axs[i,0].yaxis.set_visible(True)
        if plot_mask[0,1]:
            axs[0,1].yaxis.set_visible(True)

    # maximize axis limits for off diagonals and setup axis sharing
    maximize_axes_limits(axs, plot_mask=plot_mask)

    if show_plot:
        fig.tight_layout()
        plt.show()

    return fig, axs


def maximize_axes_limits(axs, plot_mask=None):
    n_dims = len(axs)
    # default plot mask: all True
    if plot_mask is None:
        plot_mask = np.ones((n_dims,n_dims)) == 1
    import inspect
    if inspect.isfunction(plot_mask):
        plot_mask = [[plot_mask(i,j) for j in range(n_dims)] for i in range(n_dims)]
    if not isinstance(plot_mask, np.ndarray):
        plot_mask = np.array(plot_mask)
    # get maximized limits per row and column
    limits = []
    for i in range(n_dims):
        minx,maxx = (np.inf, -np.inf)
        miny,maxy = (np.inf, -np.inf)
        for j in range(n_dims):
            if plot_mask[j,i]:
                minx_,maxx_ = axs[j,i].get_xlim()
                minx = min(minx,minx_)
                maxx = max(maxx,maxx_)
            if plot_mask[i,j] and i != j:
                miny_,maxy_ = axs[i,j].get_ylim()
                miny = min(miny,miny_)
                maxy = max(maxy,maxy_)
        limits.append([(minx,maxx),(miny,maxy)])

    # distribute maximized limits
    for i in range(n_dims):
        for j in range(n_dims):
            if plot_mask[j,i]:
                axs[j, i].set_xlim(limits[i][0])
            if plot_mask[i,j] and i != j:
                axs[i, j].set_ylim(limits[i][1])


def plot_matrix_share_axes(axs):
    """
    Sets up the axis sharing for a plot matrix.
    I.e. all axes in the same column share X, and all axes in the same row (except for the diagonal) share Y. 
    """
    n_dims = len(axs)
    # share axes 
    for i in range(n_dims):
        axs[i,i].sharex(axs[(i+1)%n_dims, i])
        for j in range(n_dims-2):
            axs[(i+2+j)%n_dims, i].sharex(axs[(i+1)%n_dims, i])
            axs[i, (i+2+j)%n_dims].sharey(axs[i, (i+1)%n_dims])


def plot_contour(distributions,
                 n_samples_kde=1000,
                 resolution=128,
                 ranges=None,
                 quantiles: list = None,
                 seed=55,
                 fig=None,
                 axs=None,
                 distrib_colors=None,
                 colorblind_safe=False,
                 show_plot=False,
                 plot_mask=None):
    """
    Visualizes a multidimensional distribution in a matrix of contour plots.
    For this, the 2D/1D marginal distributions have to be estimated for each combination of dimensions.
    This is done via KDE on samples drawn from the distribution.

    Parameters
    ----------
    distributions : list
        List of distributions to plot.
    n_samples_kde : int
        Number of samples drawn per distribution to estimate the marginal 2D distributions via KDE. Default 1000.
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
    plot_mask: 2D boolean array or function (i,j)->bool, optional 
        mask that specifies which subplots will be generated. By default every plot will be generated.

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
            fig, axs = plt.subplots(nrows=n_dims, ncols=n_dims, figsize=((0.5+n_dims)*2,n_dims*2))
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

    # Generate colors
    if distrib_colors is None:
        if colorblind_safe:
            distrib_colors = gb.create_palette(palette_size=len(distributions), colorblind_safe=colorblind_safe)
        else:
            distrib_colors = utils.get_colors(len(distributions))
    else:
        if len(distrib_colors) < len(distributions):
            if colorblind_safe:
                additional_colors = gb.create_palette(palette_size=len(distributions) - len(distrib_colors), colorblind_safe=colorblind_safe)
            else:
                additional_colors = utils.get_colors(len(distributions) - len(distrib_colors))
            distrib_colors.extend(additional_colors)

    distrib_samples = []
    for d in distributions:
        samples = d.sample(n_samples_kde, seed)
        # need to add noise for zero variance since KDE does not like that
        variances = np.var(samples, axis=0)
        from scipy import stats
        noise = stats.uniform.rvs(loc=0, scale=2e-8, size=samples.shape, random_state=seed) - 1e-8
        noise *= (variances == 0)[None,:]
        samples += noise
        distrib_samples.append(samples)


    # default plot mask: all True
    if plot_mask is None:
        plot_mask = np.ones((n_dims,n_dims)) == 1
    import inspect
    if inspect.isfunction(plot_mask):
        plot_mask = [[plot_mask(i,j) for j in range(n_dims)] for i in range(n_dims)]
    if not isinstance(plot_mask, np.ndarray):
        plot_mask = np.array(plot_mask)


    for i in range(n_dims):
        for j in range(n_dims):
            if not plot_mask[i,j]:
                continue
            # Hide all ticks and labels
            axs[i,j].xaxis.set_visible(False)
            axs[i,j].yaxis.set_visible(False)
    

    # Fill matrix with data
    from . import plots_2d
    for i, j in zip(*np.triu_indices_from(axs, k=1)):
        for x, y in [(i, j), (j, i)]:
            if not plot_mask[y,x]:
                continue # skip
            #dists2d = [Distribution(distrib_samples[k][:,[x,y]]) for k in range(len(distributions))]
            dists2d = [d[x,y, 'KDE?', n_samples_kde, 2e-8, seed] for d in distributions]
            [d[[x,y]] for d in distributions]
            plots_2d.plot_contour(dists2d, resolution=resolution, ranges=ranges, quantiles=quantiles, 
                fig=fig, axs=axs[y,x],distrib_colors=distrib_colors, colorblind_safe=colorblind_safe,
                show_plot=False)

    # Fill diagonal
    for i in range(n_dims):
        if not plot_mask[i,i]:
            continue
        #dists1d = [Distribution(distrib_samples[k][:,i]) for k in range(len(distributions))]
        dists1d = [d[i, 'KDE?', n_samples_kde, 2e-8, seed] for d in distributions]
        range_global = np.vstack(distrib_samples)[:,i].min(), np.vstack(distrib_samples)[:,i].max()
        x_global = np.linspace(range_global[0],range_global[1], num=resolution)
        for k in range(len(distributions)):
            xmin, xmax = distrib_samples[k][:,i].min(), distrib_samples[k][:,i].max()
            xs = np.linspace(xmin, xmax, resolution)
            xs = np.sort(np.concatenate([x_global,xs]))
            ys = dists1d[k].pdf(xs)
            axs[i,i].plot(xs, ys, color=distrib_colors[k])
            axs[i,i].yaxis.set_visible(True)
        

    # maximize axis limits for off diagonals and setup axis sharing
    maximize_axes_limits(axs, plot_mask=plot_mask)

    for i in range(n_dims):
        if plot_mask[-1,i]:
            axs[-1,i].xaxis.set_visible(True)
        if plot_mask[i,0]:
            axs[i,0].yaxis.set_visible(True)
    if plot_mask[0,1]:
        axs[0,1].yaxis.set_visible(True)

    if show_plot:
        fig.tight_layout()
        plt.show()

    return fig, axs

def plot_contour_samples(distributions,
                         n_samples=100,
                         n_samples_kde=1000,
                         resolution=128,
                         point_size=1,
                         alpha=1,
                         ranges=None,
                         quantiles: list = None,
                         seed=55,
                         fig=None,
                         axs=None,
                         distrib_colors=None,
                         colorblind_safe=False,
                         show_plot=False
                         ):
    """
    Visualizes a multidimensional distribution in a matrix visualization where the
    upper triangle contains contour plots and the lower triangle contains scatterplots.

    Parameters
    ----------
    distributions : list
        List of distributions to plot.
    n_samples : int
        Number of samples for the scatterplots.
    n_samples_kde : int
        Number of samples drawn per distribution to estimate the marginal 2D distributions via KDE. Default 1000. 
    resolution : int, optional
        The resolution for the pdf. Default is 128.
    point_size : float or None, optional
        Marker size (area in points^2). If None, matplotlib's default is used. By default 1.
    alpha : float, optional
        opacity value if the samples in the scatter plots. By default 1 (fully opaque)
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

    plot_contour(
        distributions, 
        n_samples_kde=n_samples_kde, 
        ranges=ranges, 
        resolution=resolution,
        quantiles=quantiles,
        seed=seed,
        fig=fig,
        axs=axs,
        distrib_colors=distrib_colors,
        colorblind_safe=colorblind_safe,
        show_plot=False,
        plot_mask=lambda row,col: row <= col
        )
    plot_samples(
        distributions, 
        n_samples=n_samples,
        point_size=point_size,
        alpha=alpha,
        seed=seed,
        fig=fig,
        axs=axs,
        distrib_colors=distrib_colors,
        colorblind_safe=colorblind_safe,
        show_plot=False,
        plot_mask=lambda row,col: row > col
        )

    maximize_axes_limits(axs,plot_mask=None)
    plot_matrix_share_axes(axs)

    #for i in range(n_dims):
     #   axs[i,i].yaxis.set_visible(True)

    if show_plot:
        fig.tight_layout()
        plt.show()

    return fig, axs
