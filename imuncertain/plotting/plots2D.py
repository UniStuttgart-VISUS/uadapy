import matplotlib.pyplot as plt
import numpy as np
import imuncertain.distribution as dist
import imuncertain.plotting.utils as utils

def plot_samples(distributions, num_samples, **kwargs):
    """
    Plot samples from the given distribution. If several distributions should be
    plotted together, an array can be passed to this function
    :param distributions: Distributions to plot
    :param num_samples: Number of samples per distribution
    :param kwargs: Optional other arguments to pass:
        xlabel for label of x-axis
        ylabel for label of y-axis
    :return:
    """
    if isinstance(distributions, dist.distribution):
        distributions = [distributions]
    for d in distributions:
        if d.dim != 2:
            raise Exception('Wrong dimension of distribution')
        samples = d.sample(num_samples)
        plt.scatter(x=samples[:,0], y=samples[:,1])
    if 'xlabel' in kwargs:
        plt.xlabel(kwargs['xlabel'])
    if 'ylabel' in kwargs:
        plt.ylabel(kwargs['ylabel'])
    plt.show()


def plot_pdf_contour(distributions, resolution=(128, 128), range_x=(0, 1), range_y=(0, 1), **kwargs):
    """
    :param distributions: Distributions to plot
    :param resolution: The resolution for the pdf
    :param range_x: The range for the x-axis
    :param range_y: The range for the y-axis
    :param kwargs: Optional other arguments to pass:
        xlabel for label of x-axis
        ylabel for label of y-axis
    :return:
    """
    if isinstance(distributions, dist.distribution):
        distributions = [distributions]
    contour_colors = utils.generate_spectrum_colors(distributions[0].dim)
    for i, d in enumerate(distributions):
        if d.dim != 2:
            raise Exception('Wrong dimension of distribution')
        x = np.linspace(range_x[0], range_x[1], resolution[0])
        y = np.linspace(range_y[0], range_y[1], resolution[1])
        xv, yv = np.meshgrid(x, y)
        coordinates = np.stack((xv, yv), axis=-1)
        pdf = d.pdf(coordinates)
        color = contour_colors[i]
        plt.contour(xv, yv, pdf, colors = [color])
    if 'xlabel' in kwargs:
        plt.xlabel(kwargs['xlabel'])
    if 'ylabel' in kwargs:
        plt.ylabel(kwargs['ylabel'])
    plt.show()