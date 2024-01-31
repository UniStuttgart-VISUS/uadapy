import matplotlib.pyplot as plt
import numpy as np
import imuncertain.distribution as dist

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
        if d.dim != 2:
            raise Exception('Wrong dimension of distribution')
        samples = d.sample(num_samples)
        plt.scatter(x=samples[:,0], y=samples[:,1])
    if 'xlabel' in kwargs:
        plt.xlabel(kwargs['xlabel'])
    if 'ylabel' in kwargs:
        plt.ylabel(kwargs['ylabel'])
    plt.show()


def plot_pdf_contour(distribution, resolution=(128,128), range_x=(0,1), range_y=(0,1), **kwargs):
    """
    :param distribution: Distributions to plot
    :param resolution: The resolution for the pdf
    :param range_x: The range for the x-axis
    :param range_y: The range for the y-axis
    :param kwargs: Optional other arguments to pass:
        xlabel for label of x-axis
        ylabel for label of y-axis
    :return:
    """
    if isinstance(distribution, dist.distribution):
        distribution = [distribution]
    contour_colors = generate_spectrum_colors(distribution[0].dim)
    print(contour_colors)
    for i, d in enumerate(distribution):
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

# HELPER FUNCTIONS
def generate_random_colors(length):
    return ["#"+''.join([np.random.choice('0123456789ABCDEF') for j in range(6)]) for _ in range(length)]

def generate_spectrum_colors(length):
    cmap = plt.cm.get_cmap('viridis', length)  # You can choose different colormaps like 'jet', 'hsv', 'rainbow', etc.
    return np.array([cmap(i) for i in range(length)])