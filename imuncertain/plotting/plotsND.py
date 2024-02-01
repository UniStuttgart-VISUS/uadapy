import itertools

import matplotlib.pyplot as plt
import numpy as np
import imuncertain.distribution as dist
import imuncertain.plotting.utils as utils

def plot_samples(distributions, num_samples, **kwargs):
    """
    Plot samples from the multivariate distribution as a SLOM
    :param distribution: The multivariate distributions
    :param num_samples: Number of samples to draw
    :param kwargs: Optional other arguments to pass:
    :return:
    """
    if isinstance(distributions, dist.distribution):
        distributions = [distributions]
    # Create matrix
    numvars = distributions[0].dim
    fig, axes = plt.subplots(nrows=numvars, ncols=numvars)
    contour_colors = utils.generate_spectrum_colors(len(distributions))
    for ax in axes.flat:
        # Hide all ticks and labels
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

    # Fill matrix with data
    for k, d in enumerate(distributions):
        if d.dim < 2:
            raise Exception('Wrong dimension of distribution')
        samples = d.sample(num_samples)
        for i, j in zip(*np.triu_indices_from(axes, k=1)):
            for x, y in [(i, j), (j, i)]:
                axes[x,y].scatter(samples[:,y], y=samples[:,x], color=contour_colors[k])

        # Fill diagonal
        for i in range(numvars):
            axes[i,i].hist(samples[:,i], histtype='stepfilled', fill=False, alpha=1.0, density=True, ec=contour_colors[k])
            axes[i,i].xaxis.set_visible(True)
            axes[i,i].yaxis.set_visible(True)

        for i in range(numvars):
            axes[-1,i].xaxis.set_visible(True)
            axes[i,0].yaxis.set_visible(True)
        axes[0,1].yaxis.set_visible(True)
    fig.tight_layout()
    plt.show()

def plot_contour(distributions, resolution=(128, 128), ranges=None, **kwargs):
    """
    Visualizes a multidimensional distribution in a matrix of contour plot
    :param distributions: Distributions to plot
    :param resolution: The resolution for the pdf
    :param ranges: Array of ranges for all dimensions
    :return:
    """
    if isinstance(distributions, dist.distribution):
        distributions = [distributions]
    contour_colors = utils.generate_spectrum_colors(len(distributions))
    # Create matrix
    numvars = distributions[0].dim
    if ranges is None:
        ranges = [(0,1)]*numvars
    fig, axes = plt.subplots(nrows=numvars, ncols=numvars)
    for i, ax in enumerate(axes.flat):
        # Hide all ticks and labels
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

    # Fill matrix with data
    for k, d in enumerate(distributions):
        if d.dim < 2:
            raise Exception('Wrong dimension of distribution')
        dims = ()
        test = ()
        for i in range(d.dim):
            test = (*test, i)
            x = np.linspace(ranges[i][0], ranges[i][1], resolution[0])
            dims = (*dims, x)
        coordinates = np.array(np.meshgrid(*dims)).transpose(tuple(range(1, numvars+1)) + (0,))
        pdf = d.pdf(coordinates.reshape((-1, coordinates.shape[-1])))
        pdf = pdf.reshape(coordinates.shape[:-1])
        pdf = pdf.transpose((1,0)+tuple(range(2,numvars)))
        for i, j in zip(*np.triu_indices_from(axes, k=1)):
            for x, y in [(i, j), (j, i)]:
                color = contour_colors[k]
                indices = list(np.arange(d.dim))
                indices.remove(x)
                indices.remove(y)
                pdf_agg = np.sum(pdf, axis=tuple(indices))
                if x > y:
                    pdf_agg = pdf_agg.T
                axes[x,y].contour(dims[y], dims[x], pdf_agg, colors=[color])

        # Fill diagonal
        for i in range(numvars):
            indices = list(np.arange(d.dim))
            indices.remove(i)
            axes[i,i].plot(dims[i], np.sum(pdf, axis=tuple(indices)), color=color)
            axes[i,i].xaxis.set_visible(True)
            axes[i,i].yaxis.set_visible(True)

        for i in range(numvars):
            axes[-1,i].xaxis.set_visible(True)
            axes[i,0].yaxis.set_visible(True)
        axes[0,1].yaxis.set_visible(True)
    fig.tight_layout()
    plt.show()

def plot_contour_samples(distributions, num_samples, resolution=(128, 128), ranges=None, **kwargs):
    """
    Visualizes a multidimensional distribution in a matrix visualization where the
    upper diagonal contains contour plots and the lower diagonal normal scatterplots
    :param distributions: Distributions to plot
    :param num_samples: Number of samples for the scatterplot
    :param resolution: The resolution for the pdf
    :param ranges: Array of ranges for all dimensions
    :return:
    """
    if isinstance(distributions, dist.distribution):
        distributions = [distributions]
    contour_colors = utils.generate_spectrum_colors(len(distributions))
    # Create matrix
    numvars = distributions[0].dim
    if ranges is None:
        ranges = [(0,1)]*numvars
    fig, axes = plt.subplots(nrows=numvars, ncols=numvars)
    for i, ax in enumerate(axes.flat):
        # Hide all ticks and labels
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

    # Fill matrix with data
    for k, d in enumerate(distributions):
        samples = d.sample(num_samples)
        if d.dim < 2:
            raise Exception('Wrong dimension of distribution')
        dims = ()
        for i in range(d.dim):
            x = np.linspace(ranges[i][0], ranges[i][1], resolution[0])
            dims = (*dims, x)
        coordinates = np.array(np.meshgrid(*dims)).transpose(tuple(range(1, numvars+1)) + (0,))
        pdf = d.pdf(coordinates.reshape((-1, coordinates.shape[-1])))
        pdf = pdf.reshape(coordinates.shape[:-1])
        pdf = pdf.transpose((1,0)+tuple(range(2,numvars)))
        for i, j in zip(*np.triu_indices_from(axes, k=1)):
            for x, y in [(i, j), (j, i)]:
                color = contour_colors[k]
                indices = list(np.arange(d.dim))
                indices.remove(x)
                indices.remove(y)
                pdf_agg = np.sum(pdf, axis=tuple(indices))
                if x < y:
                    axes[x,y].contour(dims[x], dims[y], pdf_agg, colors=[color])
                else:
                    axes[x, y].scatter(samples[:, y], y=samples[:, x], color=contour_colors[k])
                    axes[x, y].set_xlim(ranges[x][0], ranges[x][1])
                    axes[x, y].set_ylim(ranges[y][0], ranges[y][1])

        # Fill diagonal
        for i in range(numvars):
            indices = list(np.arange(d.dim))
            indices.remove(i)
            axes[i,i].plot(dims[i], np.sum(pdf, axis=tuple(indices)), color=color)
            axes[i,i].xaxis.set_visible(True)
            axes[i,i].yaxis.set_visible(True)

        for i in range(numvars):
            axes[-1,i].xaxis.set_visible(True)
            axes[i,0].yaxis.set_visible(True)
        axes[0,1].yaxis.set_visible(True)
    fig.tight_layout()
    plt.show()