import matplotlib.pyplot as plt
import numpy as np
import uadapy.distribution as dist

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


def plot_contour(distributions, resolution=128, ranges=None, **kwargs):
    if isinstance(distributions, dist.distribution):
        distributions = [distributions]
    contour_colors = generate_spectrum_colors(len(distributions))
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
    for i, d in enumerate(distributions):
        x = np.linspace(range_x[0], range_x[1], resolution)
        y = np.linspace(range_y[0], range_y[1], resolution)
        xv, yv = np.meshgrid(x, y)
        coordinates = np.stack((xv, yv), axis=-1)
        pdf = d.pdf(coordinates)
        color = contour_colors[i]
        plt.contour(xv, yv, pdf, colors = [color])
    plt.show()

# HELPER FUNCTIONS
def generate_random_colors(length):
    return ["#"+''.join([np.random.choice('0123456789ABCDEF') for j in range(6)]) for _ in range(length)]

def generate_spectrum_colors(length):
    cmap = plt.cm.get_cmap('viridis', length)  # You can choose different colormaps like 'jet', 'hsv', 'rainbow', etc.
    return np.array([cmap(i) for i in range(length)])