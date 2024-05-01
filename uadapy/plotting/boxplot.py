import matplotlib.pyplot as plt
import uadapy.distribution as dist
from math import ceil, sqrt
import numpy as np
import glasbey as gb

def plot_boxplot(distributions, num_samples, fig=None, axs=None, labels=None, titles=None, colors=None, **kwargs):
    """
    Plot box plots for samples drawn from given distributions.

    Parameters
    ----------
    distributions : list
        List of distributions to plot.
    num_samples : int
        Number of samples per distribution.
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

    Returns
    -------
    matplotlib.figure.Figure
        The figure object containing the plot.
    list
        List of Axes objects used for plotting.
    """
    samples = []
    vertical = True
    colorblind_safe = False

    if isinstance(distributions, dist.distribution):
        distributions = [distributions]

    if kwargs:
        for key, value in kwargs.items():
            if key == 'vert':
                vertical = value
            if key == 'colorblind_safe':
                colorblind_safe = value

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
        samples.append(d.sample(num_samples))

    # Generate Glasbey colors
    if colors is None:
        palette = gb.create_palette(palette_size=len(samples), colorblind_safe=colorblind_safe)
    else:
        # If colors are provided but fewer than the number of samples, add more colors from Glasbey palette
        if len(colors) < len(samples):
            additional_colors = gb.create_palette(palette_size=len(samples) - len(colors), colorblind_safe=colorblind_safe)
            colors.extend(additional_colors)
        palette = colors

    for i, ax_row in enumerate(axs):
        for j, ax in enumerate(ax_row):
            index = i * num_cols + j
            if index < num_plots:
                for k, sample in enumerate(samples):
                    boxprops = dict(facecolor=palette[k % len(palette)], edgecolor='black')
                    whiskerprops = dict(color='black', linestyle='--')
                    capprops = dict(color='black')
                    ax.boxplot(sample[index], positions=[k], patch_artist=True, boxprops=boxprops,
                               showfliers=False, whiskerprops=whiskerprops, capprops=capprops,
                               showmeans=True, meanline=True, meanprops=dict(color="black", linestyle='-'),
                               medianprops=dict(linewidth=0), vert=vertical)
                if labels:
                    if vertical:
                        ax.set_xticks(range(len(labels)))
                        ax.set_xticklabels(labels, rotation=45, ha='right')
                    else:
                        ax.set_yticks(range(len(labels)))
                        ax.set_yticklabels(labels, rotation=45, ha='right')
                if titles:
                    ax.set_title(titles[index] if titles and index < len(titles) else 'Distribution ' + str(index + 1))
                ax.yaxis.set_ticks_position('none')
                ax.grid(True, linestyle=':', linewidth='0.5', color='gray')
            else:
                ax.set_visible(False)  # Hide unused subplots

    return fig, axs