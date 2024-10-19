import numpy as np
from uadapy import Distribution
import matplotlib.pyplot as plt
from math import ceil, sqrt
import glasbey as gb
import seaborn as sns
from matplotlib.patches import Ellipse


def _calculate_freedman_diaconis_bins(data):
    q25, q75 = np.percentile(data, [25, 75])
    iqr = q75 - q25
    bin_width = 2 * iqr / np.cbrt(len(data))
    n_bins = int((np.max(data) - np.min(data)) / bin_width)
    return n_bins

def _calculate_offsets(count, max_count):
    occupancy = (count/max_count)
    return np.linspace(-0.45 * occupancy, 0.45 * occupancy, count)

def _calculate_dot_size(n_samples, scale_factor):
    if n_samples < 100:
        dot_size = scale_factor * 3.125
    else:
        dot_size = scale_factor * (50 / (4 ** np.log10(n_samples)))
    return dot_size

def _setup_plot(distributions, n_samples, seed, fig=None, axs=None, colors=None, colorblind_safe=False):
    """
    Set up the plot for samples drawn from given distributions.

    Parameters
    ----------
    distributions : list
        List of distributions to plot. If a single distribution is passed, it will be converted into a list.
    n_samples : int
        Number of samples per distribution.
    seed : int
        Seed for the random number generator for reproducibility.
    fig : matplotlib.figure.Figure or None, optional
        Figure object to use for plotting. If None, a new figure will be created.
    axs : matplotlib.axes.Axes or array of Axes or None, optional
        Axes object(s) to use for plotting. If None, new axes will be created.
    colors : list or None, optional
        List of colors to use for each distribution. If None, Glasbey colors will be used.
    colorblind_safe : bool, optional
        If True, the plot will use colors suitable for colorblind individuals.
        Default is False.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the plot.
    axs : list
        List of Axes objects used for plotting.
    samples : list
        List of samples drawn from the distributions.
    palette : list
        List of colors to use for each distribution.
    n_plots : int
        Number of subplots.
    n_cols : int
        Number of columns in the subplot layout.
    """

    samples = []

    if isinstance(distributions, Distribution):
        distributions = [distributions]

    # Calculate the layout of subplots
    if axs is None:
        n_plots = distributions[0].n_dims
        n_rows = ceil(sqrt(n_plots))
        n_cols = ceil(n_plots / n_rows)
        fig, axs = plt.subplots(n_rows, n_cols)
    else:
        # Case 1: axs is a 2D array (multiple rows and columns)
        if isinstance(axs, np.ndarray):
            dim = axs.shape
            if (len(axs.shape) == 1):
                n_rows, n_cols = 1, dim[0]
            else:
                n_rows, n_cols = dim
        # Case 2: axs is not an array (single subplot)
        else:
            n_rows, n_cols = 1, 1
        n_plots = n_rows + n_cols

    # Ensure axs is a 2D array even if there's only one row or column
    if n_rows == 1:
        axs = [axs]
    if n_cols == 1:
        axs = [[ax] for ax in axs]

    for d in distributions:
        samples.append(d.sample(n_samples, seed))

    # Generate Glasbey colors
    if colors is None:
        palette = gb.create_palette(palette_size=len(samples), colorblind_safe=colorblind_safe)
    else:
        # If colors are provided but fewer than the number of samples, add more colors from Glasbey palette
        if len(colors) < len(samples):
            additional_colors = gb.create_palette(palette_size=len(samples) - len(colors), colorblind_safe=colorblind_safe)
            colors.extend(additional_colors)
        palette = colors

    return fig, axs, samples, palette, n_plots, n_cols

def plot_1d_distribution(
        distributions,
        n_samples,
        plot_types: list,
        seed=55,
        fig=None,
        axs=None,
        distrib_labels=None,
        dim_labels=None,
        distrib_colors=None,
        vert=True,
        colorblind_safe=False,
        show_plot=False,
        dot_size=0,
        **kwargs):
    """
    Plot box plots, violin plots and dot plots for samples drawn from given distributions.

    Parameters
    ----------
    distributions : list
        List of distributions to plot.
    n_samples : int
        Number of samples per distribution.
    plot_types : list
        List of plot types to plot. Valid values are 'boxplot','violinplot', 'stripplot', 'swarmplot' and 'dotplot'.
    seed : int
        Seed for the random number generator for reproducibility. It defaults to 55 if not provided.
    fig : matplotlib.figure.Figure or None, optional
        Figure object to use for plotting. If None, a new figure will be created.
    axs : matplotlib.axes.Axes or array of Axes or None, optional
        Axes object(s) to use for plotting. If None, new axes will be created.
    distrib_labels : list or None, optional
        Labels for each distribution.
    dim_labels : list or None, optional
        Titles for each subplot.
    distrib_colors : list or None, optional
        List of colors to use for each distribution. If None, Glasbey colors will be used.
    vert : bool, optional
        If True, plots will be drawn vertically. If False, plots will be drawn horizontally.
        Default is True.
    colorblind_safe : bool, optional
        If True, the plot will use colors suitable for colorblind individuals.
        Default is False.
    show_plot : bool, optional
        If True, display the plot.
        Default is False.
    dot_size : float, optional
        This parameter determines the size of the dots used in the 'stripplot','swarmplot' and 'dotplot'.
        If not provided, the size is calculated based on the number of samples and the type of plot.
    **kwargs : additional matplotlib keyword arguments
        Additional optional plotting arguments.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object containing the plot.
    list
        List of Axes objects used for plotting.
    """

    fig, axs, samples, palette, n_plots, n_cols = _setup_plot(distributions, n_samples, seed, fig, axs, distrib_colors, colorblind_safe)

    # Check number of attributes
    n_attributes = 1
    if np.ndim(samples) > 2:
        n_attributes = np.shape(samples)[2]

    if distrib_labels:
        ticks = range(len(distrib_labels))
    else:
        ticks = range(len(samples))

    for i, ax_row in enumerate(axs):
        for j, ax in enumerate(ax_row):
            index = i * n_cols + j
            if index < n_plots and index < n_attributes:
                y_min = 9999
                y_max = -9999
                for k, sample in enumerate(samples):
                    if np.ndim(sample) == 1:
                        sample = np.array(sample).reshape(1,-1)
                    if 'boxplot' in plot_types:
                        boxprops = dict(facecolor=palette[k % len(palette)], edgecolor='black')
                        whiskerprops = dict(color='black', linestyle='--')
                        capprops = dict(color='black')
                        if vert == False:
                            kwargs['vert'] = False
                        if 'widths' not in kwargs:
                            kwargs['widths'] = 0.5
                        if 'patch_artist' not in kwargs:
                            kwargs['patch_artist'] = True
                        if 'showfliers' not in kwargs:
                            kwargs['showfliers'] = False
                        if 'whiskerprops' not in kwargs:
                            kwargs['whiskerprops'] = whiskerprops
                        if 'capprops' not in kwargs:
                            kwargs['capprops'] = capprops
                        if 'showmeans' not in kwargs:
                            kwargs['showmeans'] = True
                        if 'meanline' not in kwargs:
                            kwargs['meanline'] = True
                        if 'meanprops' not in kwargs:
                            kwargs['meanprops'] = dict(color="black", linestyle='-')
                        if 'medianprops' not in kwargs:
                            kwargs['medianprops'] = dict(linewidth=0)
                        ax.boxplot(sample[:, index], positions=[k], boxprops=boxprops, **kwargs)
                    if 'violinplot' in plot_types:
                        if vert == False:
                            kwargs['vert'] = False
                        if 'widths' not in kwargs:
                            kwargs['widths'] = 0.75
                        parts =  ax.violinplot(sample[:,index], positions=[k], **kwargs)
                        for pc in parts['bodies']:
                            pc.set_facecolor(palette[k % len(palette)])
                            pc.set_edgecolor(palette[k % len(palette)])
                            pc.set_alpha(0.5)
                        parts['cbars'].remove()
                        parts['cmaxes'].remove()
                        parts['cmins'].remove()
                        if kwargs.get('showmeans', False):
                          parts['cmeans'].set_edgecolor('black')
                    if 'stripplot' in plot_types:
                        if dot_size  == 0:
                            if n_samples < 100:
                                scale_factor = 1
                            else:
                                scale_factor = 1 + np.log10(n_samples / 100)
                            dot_size = _calculate_dot_size(len(sample[:,index]), scale_factor)
                        if vert:
                            sns.stripplot(x=[k]*len(sample[:,index]), y=sample[:,index], color=palette[k % len(palette)], size=dot_size, jitter=0.25, ax=ax)
                        else:
                            sns.stripplot(x=sample[:,index], y=[k]*len(sample[:,index]), color=palette[k % len(palette)], size=dot_size, jitter=0.25, ax=ax, orient='h')
                    if 'swarmplot' in plot_types:
                        if dot_size == 0:
                            dot_size = _calculate_dot_size(len(sample[:,index]), 1)
                        if vert:
                            sns.swarmplot(x=[k]*len(sample[:,index]), y=sample[:,index], color=palette[k % len(palette)], size=dot_size, ax=ax)
                        else:
                            sns.swarmplot(x=sample[:,index], y=[k]*len(sample[:,index]), color=palette[k % len(palette)], size=dot_size, ax=ax, orient='h')
                    if 'dotplot' in plot_types:
                        if dot_size == 0:
                            dot_size = _calculate_dot_size(len(sample[:,index]), 0.005)
                        flat_sample = np.ravel(sample[:,index])
                        ticks = [x + 0.5 for x in range(len(samples))]
                        if y_min > np.min(flat_sample):
                            y_min = np.min(flat_sample)
                        if y_max < np.max(flat_sample):
                            y_max = np.max(flat_sample)
                        n_bins = _calculate_freedman_diaconis_bins(flat_sample)
                        bin_width = (np.max(flat_sample) - np.min(flat_sample)) / n_bins
                        bins = np.arange(np.min(flat_sample), np.max(flat_sample) + bin_width, bin_width)
                        binned_data, bin_edges = np.histogram(flat_sample, bins=bins)
                        max_count = np.max(binned_data)
                        for bin_idx in range(len(binned_data)):
                            count = binned_data[bin_idx]
                            if count > 0:
                                bin_center = (bin_edges[bin_idx] + bin_edges[bin_idx + 1]) / 2
                                # Calculate symmetrical offsets
                                if count == 1:
                                    offsets = [0]  # Single dot in the center
                                else:
                                    offsets = _calculate_offsets(count, max_count)
                                for offset in offsets:
                                    if vert:
                                        ellipse = Ellipse((ticks[k] + offset, bin_center), width=dot_size, height=dot_size, color=palette[k % len(palette)])
                                    else:
                                        ellipse = Ellipse((bin_center, ticks[k] + offset), width=dot_size, height=dot_size, color=palette[k % len(palette)])
                                    ax.add_patch(ellipse)
                if 'dotplot' in plot_types:
                    if vert:
                        ax.set_xlim(0, len(samples))
                        ax.set_ylim(y_min - 1, y_max + 1)
                    else:
                        ax.set_xlim(y_min - 1, y_max + 1)
                        ax.set_ylim(0, len(samples))
                if distrib_labels:
                    if vert:
                        ax.set_xticks(ticks)
                        ax.set_xticklabels(distrib_labels, rotation=45, ha='right')
                    else:
                        ax.set_yticks(ticks)
                        ax.set_yticklabels(distrib_labels, rotation=45, ha='right')
                if dim_labels:
                    ax.set_title(dim_labels[index] if dim_labels and index < len(dim_labels) else 'Distribution ' + str(index + 1))
                ax.yaxis.set_ticks_position('none')
                ax.grid(True, linestyle=':', linewidth='0.5', color='gray')
            else:
                ax.set_visible(False)  # Hide unused subplots

    if show_plot:
        fig.tight_layout()
        plt.show()

    return fig, axs

def generate_boxplot(distributions,
        n_samples=10000,
        seed=55,
        fig=None,
        axs=None,
        distrib_labels=None,
        dim_labels=None,
        distrib_colors=None,
        vert=True,
        colorblind_safe=False,
        show_plot=False,
        **kwargs):
    """
    Plot box plots for samples drawn from given distributions.

    Parameters
    ----------
    distributions : list
        List of distributions to plot.
    n_samples : int, optional
        Number of samples per distribution.
    seed : int
        Seed for the random number generator for reproducibility. It defaults to 55 if not provided.
    fig : matplotlib.figure.Figure or None, optional
        Figure object to use for plotting. If None, a new figure will be created.
    axs : matplotlib.axes.Axes or array of Axes or None, optional
        Axes object(s) to use for plotting. If None, new axes will be created.
    distrib_labels : list or None, optional
        Labels for each distribution.
    dim_labels : list or None, optional
        Titles for each subplot.
    distrib_colors : list or None, optional
        List of colors to use for each distribution. If None, Glasbey colors will be used.
    vert : bool, optional
        If True, plots will be drawn vertically. If False, plots will be drawn horizontally.
        Default is True.
    colorblind_safe : bool, optional
        If True, the plot will use colors suitable for colorblind individuals.
        Default is False.
    show_plot : bool, optional
        If True, display the plot.
        Default is False.
    **kwargs : additional matplotlib.pyplot.boxplot keyword arguments
        Additional optional plotting arguments.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object containing the plot.
    list
        List of Axes objects used for plotting.
    """

    plot_type = 'boxplot'
    fig, axs = plot_1d_distribution(distributions, n_samples, plot_type, seed, fig, axs,
                                    distrib_labels, dim_labels, distrib_colors, vert,
                                    colorblind_safe, show_plot, 0, **kwargs)
    return fig, axs

def generate_violinplot(distributions,
        n_samples=10000,
        seed=55,
        fig=None,
        axs=None,
        distrib_labels=None,
        dim_labels=None,
        distrib_colors=None,
        vert=True,
        colorblind_safe=False,
        show_plot=False,
        **kwargs):
    """
    Plot violin plots for samples drawn from given distributions.

    Parameters
    ----------
    distributions : list
        List of distributions to plot.
    n_samples : int, optional
        Number of samples per distribution.
    seed : int
        Seed for the random number generator for reproducibility. It defaults to 55 if not provided.
    fig : matplotlib.figure.Figure or None, optional
        Figure object to use for plotting. If None, a new figure will be created.
    axs : matplotlib.axes.Axes or array of Axes or None, optional
        Axes object(s) to use for plotting. If None, new axes will be created.
    distrib_labels : list or None, optional
        Labels for each distribution.
    dim_labels : list or None, optional
        Titles for each subplot.
    distrib_colors : list or None, optional
        List of colors to use for each distribution. If None, Glasbey colors will be used.
    vert : bool, optional
        If True, plots will be drawn vertically. If False, plots will be drawn horizontally.
        Default is True.
    colorblind_safe : bool, optional
        If True, the plot will use colors suitable for colorblind individuals.
        Default is False.
    show_plot : bool, optional
        If True, display the plot.
        Default is False.
    **kwargs : additional matplotlib.pyplot.violinplot keyword arguments
        Additional optional plotting arguments.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object containing the plot.
    list
        List of Axes objects used for plotting.
    """

    plot_type = 'violinplot'
    fig, axs = plot_1d_distribution(distributions, n_samples, plot_type, seed, fig, axs,
                                    distrib_labels, dim_labels, distrib_colors, vert,
                                    colorblind_safe, show_plot, 0, **kwargs)
    return fig, axs

def generate_dotplot(distributions,
        n_samples=20,
        fig=None,
        axs=None,
        distrib_labels=None,
        dim_labels=None,
        distrib_colors=None,
        vert=True,
        colorblind_safe=False,
        show_plot=False,
        dot_size=0):
    """
    Plot dot plots for samples drawn from given distributions.

    Parameters
    ----------
    distributions : list
        List of distributions to plot.
    n_samples : int, optional
        Number of samples per distribution.
    fig : matplotlib.figure.Figure or None, optional
        Figure object to use for plotting. If None, a new figure will be created.
    axs : matplotlib.axes.Axes or array of Axes or None, optional
        Axes object(s) to use for plotting. If None, new axes will be created.
    distrib_labels : list or None, optional
        Labels for each distribution.
    dim_labels : list or None, optional
        Titles for each subplot.
    distrib_colors : list or None, optional
        List of colors to use for each distribution. If None, Glasbey colors will be used.
    vert : bool, optional
        If True, plots will be drawn vertically. If False, plots will be drawn horizontally.
        Default is True.
    colorblind_safe : bool, optional
        If True, the plot will use colors suitable for colorblind individuals.
        Default is False.
    show_plot : bool, optional
        If True, display the plot.
        Default is False.
    dot_size : float, optional
        This parameter determines the size of the dots.
        If not provided, the size is calculated based on the number of samples and the type of plot.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object containing the plot.
    list
        List of Axes objects used for plotting.
    """

    plot_type = 'dotplot'
    fig, axs = plot_1d_distribution(distributions, n_samples, plot_type, 55, fig, axs,
                                    distrib_labels, dim_labels, distrib_colors, vert,
                                    colorblind_safe, show_plot, dot_size)
    return fig, axs

def generate_stripplot(distributions,
        n_samples=20,
        seed=55,
        fig=None,
        axs=None,
        distrib_labels=None,
        dim_labels=None,
        distrib_colors=None,
        vert=True,
        colorblind_safe=False,
        show_plot=False,
        dot_size=0):
    """
    Plot strip plots for samples drawn from given distributions.

    Parameters
    ----------
    distributions : list
        List of distributions to plot.
    n_samples : int, optional
        Number of samples per distribution.
    seed : int
        Seed for the random number generator for reproducibility. It defaults to 55 if not provided.
    fig : matplotlib.figure.Figure or None, optional
        Figure object to use for plotting. If None, a new figure will be created.
    axs : matplotlib.axes.Axes or array of Axes or None, optional
        Axes object(s) to use for plotting. If None, new axes will be created.
    distrib_labels : list or None, optional
        Labels for each distribution.
    dim_labels : list or None, optional
        Titles for each subplot.
    distrib_colors : list or None, optional
        List of colors to use for each distribution. If None, Glasbey colors will be used.
    vert : bool, optional
        If True, plots will be drawn vertically. If False, plots will be drawn horizontally.
        Default is True.
    colorblind_safe : bool, optional
        If True, the plot will use colors suitable for colorblind individuals.
        Default is False.
    show_plot : bool, optional
        If True, display the plot.
        Default is False.
    dot_size : float, optional
        This parameter determines the size of the dots.
        If not provided, the size is calculated based on the number of samples and the type of plot.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object containing the plot.
    list
        List of Axes objects used for plotting.
    """

    plot_type = 'stripplot'
    fig, axs = plot_1d_distribution(distributions, n_samples, plot_type, seed, fig, axs,
                                    distrib_labels, dim_labels, distrib_colors, vert,
                                    colorblind_safe, show_plot, dot_size)
    return fig, axs

def generate_swarmplot(distributions,
        n_samples=20,
        seed=55,
        fig=None,
        axs=None,
        distrib_labels=None,
        dim_labels=None,
        distrib_colors=None,
        vert=True,
        colorblind_safe=False,
        show_plot=False,
        dot_size=0):
    """
    Plot swarm plots for samples drawn from given distributions.

    Parameters
    ----------
    distributions : list
        List of distributions to plot.
    n_samples : int, optional
        Number of samples per distribution.
    seed : int
        Seed for the random number generator for reproducibility. It defaults to 55 if not provided.
    fig : matplotlib.figure.Figure or None, optional
        Figure object to use for plotting. If None, a new figure will be created.
    axs : matplotlib.axes.Axes or array of Axes or None, optional
        Axes object(s) to use for plotting. If None, new axes will be created.
    distrib_labels : list or None, optional
        Labels for each distribution.
    dim_labels : list or None, optional
        Titles for each subplot.
    distrib_colors : list or None, optional
        List of colors to use for each distribution. If None, Glasbey colors will be used.
    vert : bool, optional
        If True, plots will be drawn vertically. If False, plots will be drawn horizontally.
        Default is True.
    colorblind_safe : bool, optional
        If True, the plot will use colors suitable for colorblind individuals.
        Default is False.
    show_plot : bool, optional
        If True, display the plot.
        Default is False.
    dot_size : float, optional
        This parameter determines the size of the dots.
        If not provided, the size is calculated based on the number of samples and the type of plot.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object containing the plot.
    list
        List of Axes objects used for plotting.
    """

    plot_type = 'swarmplot'

    fig, axs = plot_1d_distribution(distributions, n_samples, plot_type, seed, fig, axs,
                                    distrib_labels, dim_labels, distrib_colors, vert,
                                    colorblind_safe, show_plot, dot_size)
    return fig, axs
