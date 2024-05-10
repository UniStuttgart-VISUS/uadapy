import numpy as np
import uadapy.distribution as dist
import matplotlib.pyplot as plt
from math import ceil, sqrt
import glasbey as gb
import seaborn as sns
from matplotlib.patches import Ellipse


def calculate_freedman_diaconis_bins(data):
    q25, q75 = np.percentile(data, [25, 75])
    iqr = q75 - q25
    bin_width = 2 * iqr / np.cbrt(len(data))
    num_bins = int((np.max(data) - np.min(data)) / bin_width)
    return num_bins

def calculate_offsets(count, max_count):
    occupancy = (count/max_count)
    return np.linspace(-0.45 * occupancy, 0.45 * occupancy, count)

def calculate_dot_size(num_samples, scale_factor):
    if num_samples < 100:
        dot_size = 3.125
    else:
        dot_size = scale_factor * (50 /(4 ** np.log10(num_samples)))
    return dot_size

def setup_plot(distributions, num_samples, seed, fig=None, axs=None, colors=None, **kwargs):
    """
    Set up the plot for samples drawn from given distributions.

    Parameters
    ----------
    distributions : list
        List of distributions to plot. If a single distribution is passed, it will be converted into a list.
    num_samples : int
        Number of samples per distribution.
    seed : int
        Seed for the random number generator for reproducibility.
    fig : matplotlib.figure.Figure or None, optional
        Figure object to use for plotting. If None, a new figure will be created.
    axs : matplotlib.axes.Axes or array of Axes or None, optional
        Axes object(s) to use for plotting. If None, new axes will be created.
    colors : list or None, optional
        List of colors to use for each distribution. If None, Glasbey colors will be used.
    **kwargs : additional keyword arguments
        Additional optional arguments.
        - colorblind_safe : bool, optional
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
    num_plots : int
        Number of subplots.
    num_cols : int
        Number of columns in the subplot layout.
    """

    samples = []

    if isinstance(distributions, dist.distribution):
        distributions = [distributions]

    # Calculate the layout of subplots
    if axs is None:
        num_plots = distributions[0].dim
        num_rows = ceil(sqrt(num_plots))
        num_cols = ceil(num_plots / num_rows)
        fig, axs = plt.subplots(num_rows, num_cols)
    else:
        # Case 1: axs is a 2D array (multiple rows and columns)
        if isinstance(axs, np.ndarray):
            dim = axs.shape
            if (len(axs.shape) == 1):
                num_rows, num_cols = 1, dim[0]
            else:
                num_rows, num_cols = dim
        # Case 2: axs is not an array (single subplot)
        else:
            num_rows, num_cols = 1, 1
        num_plots = num_rows + num_cols

    # Ensure axs is a 2D array even if there's only one row or column
    if num_rows == 1:
        axs = [axs]
    if num_cols == 1:
        axs = [[ax] for ax in axs]

    for d in distributions:
        samples.append(d.sample(num_samples, seed))

    # Generate Glasbey colors
    if colors is None:
        palette = gb.create_palette(palette_size=len(samples), colorblind_safe=kwargs.get('colorblind_safe', False))
    else:
        # If colors are provided but fewer than the number of samples, add more colors from Glasbey palette
        if len(colors) < len(samples):
            additional_colors = gb.create_palette(palette_size=len(samples) - len(colors), colorblind_safe=kwargs.get('colorblind_safe', True))
            colors.extend(additional_colors)
        palette = colors

    return fig, axs, samples, palette, num_plots, num_cols

def plot_1d_distribution(distributions, num_samples, plot_types:list, seed=55, fig=None, axs=None, labels=None, titles=None, colors=None, **kwargs):
    """
    Plot box plots, violin plots and dot plots for samples drawn from given distributions.

    Parameters
    ----------
    distributions : list
        List of distributions to plot.
    num_samples : int
        Number of samples per distribution.
    plot_types : list
        List of plot types to plot. Valid values are 'boxplot','violinplot', 'stripplot', 'swarmplot' and 'dotplot'.
    seed : int
        Seed for the random number generator for reproducibility. It defaults to 55 if not provided.
    fig : matplotlib.figure.Figure or None, optional
        Figure object to use for plotting. If None, a new figure will be created.
    axs : matplotlib.axes.Axes or array of Axes or None, optional
        Axes object(s) to use for plotting. If None, new axes will be created.
    labels : list or None, optional
        Labels for each distribution.
    titles : list or None, optional
        Titles for each subplot.
    colors : list or None, optional
        List of colors to use for each distribution. If None, Glasbey colors will be used.
    **kwargs : additional keyword arguments
        Additional optional plotting arguments.
        - vert : bool, optional
            If True, boxes will be drawn vertically. If False, boxes will be drawn horizontally.
            Default is True.
        - colorblind_safe : bool, optional
            If True, the plot will use colors suitable for colorblind individuals.
            Default is False.
        - dot_size : float, optional
            This parameter determines the size of the dots used in the 'stripplot' and 'swarmplot'.
            If not provided, the size is calculated based on the number of samples and the type of plot.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object containing the plot.
    list
        List of Axes objects used for plotting.
    """

    fig, axs, samples, palette, num_plots, num_cols = setup_plot(distributions, num_samples, seed, fig, axs, colors, **kwargs)

    # Check number of attributes
    num_attributes = 1
    if np.ndim(samples) > 2:
        num_attributes = np.shape(samples)[2]

    if labels:
        ticks = range(len(labels))
    else:
        ticks = range(len(samples))

    for i, ax_row in enumerate(axs):
        for j, ax in enumerate(ax_row):
            index = i * num_cols + j
            if index < num_plots and index < num_attributes:
                y_min = 9999
                y_max = -9999
                for k, sample in enumerate(samples):
                    if np.ndim(sample) == 1:
                        sample = np.array(sample).reshape(1,-1)
                    if 'boxplot' in plot_types:
                        boxprops = dict(facecolor=palette[k % len(palette)], edgecolor='black')
                        whiskerprops = dict(color='black', linestyle='--')
                        capprops = dict(color='black')
                        ax.boxplot(sample[:, index], positions=[k], patch_artist=True, boxprops=boxprops,
                                   showfliers=False, whiskerprops=whiskerprops, capprops=capprops,
                                   showmeans=True, meanline=True, meanprops=dict(color="black", linestyle='-'),
                                   medianprops=dict(linewidth=0), vert=kwargs.get('vert', True))
                    if 'violinplot' in plot_types:
                        parts =  ax.violinplot(sample[:,index], positions=[k], showmeans=True, vert=kwargs.get('vert', True))
                        for pc in parts['bodies']:
                            pc.set_facecolor(palette[k % len(palette)])
                            pc.set_edgecolor(palette[k % len(palette)])
                            pc.set_alpha(0.5)
                        parts['cbars'].remove()
                        parts['cmaxes'].remove()
                        parts['cmins'].remove()
                        parts['cmeans'].set_edgecolor('black')
                    if 'stripplot' in plot_types or 'swarmplot' in plot_types or 'dotplot' in plot_types: 
                        if 'dot_size' in kwargs:
                            dot_size = kwargs['dot_size']
                    if 'stripplot' in plot_types:
                        if 'dot_size' not in kwargs:
                            scale_factor = 1 + np.log10(num_samples/100)
                            dot_size = calculate_dot_size(len(sample[:,index]), scale_factor)
                        if kwargs.get('vert',True):
                            sns.stripplot(x=[k]*len(sample[:,index]), y=sample[:,index], color=palette[k % len(palette)], size=dot_size, jitter=0.25, ax=ax)
                        else:
                            sns.stripplot(x=sample[:,index], y=[k]*len(sample[:,index]), color=palette[k % len(palette)], size=dot_size, jitter=0.25, ax=ax, orient='h')
                    if 'swarmplot' in plot_types:
                        if 'dot_size' not in kwargs:
                            dot_size = calculate_dot_size(len(sample[:,index]), 1)
                        if kwargs.get('vert',True):
                            sns.swarmplot(x=[k]*len(sample[:,index]), y=sample[:,index], color=palette[k % len(palette)], size=dot_size, ax=ax)
                        else:
                            sns.swarmplot(x=sample[:,index], y=[k]*len(sample[:,index]), color=palette[k % len(palette)], size=dot_size, ax=ax, orient='h')
                    if 'dotplot' in plot_types:
                        if 'dot_size' not in kwargs:
                            dot_size = calculate_dot_size(len(sample[:,index]), 0.005)
                        flat_sample = np.ravel(sample[:,index])
                        ticks = [x + 0.5 for x in range(len(samples))]
                        if y_min > np.min(flat_sample):
                            y_min = np.min(flat_sample)
                        if y_max < np.max(flat_sample):
                            y_max = np.max(flat_sample)
                        num_bins = calculate_freedman_diaconis_bins(flat_sample)
                        bin_width = kwargs.get('bin_width', (np.max(flat_sample) - np.min(flat_sample)) / num_bins)
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
                                    offsets = calculate_offsets(count, max_count)
                                for offset in offsets:
                                    if kwargs.get('vert',True):
                                        ellipse = Ellipse((ticks[k] + offset, bin_center), width=dot_size, height=dot_size, color=palette[k % len(palette)])
                                    else:
                                        ellipse = Ellipse((bin_center, ticks[k] + offset), width=dot_size, height=dot_size, color=palette[k % len(palette)])
                                    ax.add_patch(ellipse)
                if 'dotplot' in plot_types:
                    if kwargs.get('vert',True):
                        ax.set_xlim(0, len(samples))
                        ax.set_ylim(y_min - 1, y_max + 1)
                    else:
                        ax.set_xlim(y_min - 1, y_max + 1)
                        ax.set_ylim(0, len(samples))
                if labels:
                    if kwargs.get('vert', True):
                        ax.set_xticks(ticks)
                        ax.set_xticklabels(labels, rotation=45, ha='right')
                    else:
                        ax.set_yticks(ticks)
                        ax.set_yticklabels(labels, rotation=45, ha='right')
                if titles:
                    ax.set_title(titles[index] if titles and index < len(titles) else 'Distribution ' + str(index + 1))
                ax.yaxis.set_ticks_position('none')
                ax.grid(True, linestyle=':', linewidth='0.5', color='gray')
            else:
                ax.set_visible(False)  # Hide unused subplots

    return fig, axs
