from uadapy import distribution
from uadapy.dr import uamds
import uadapy.data as data
import uadapy.plotting.plots1D as plots1D
import uadapy.plotting.plots2D as plots2D
import uadapy.plotting.plotsND as plotsND
import numpy as np
import matplotlib.pyplot as plt


def example_uamds():
    distribs_hi = data.load_iris_normal()
    plots1D.plot_1d_distribution(distribs_hi, num_samples=100, plot_types='boxplot')
    distribs_lo = uamds(distribs_hi, dims=2)
    plots2D.plot_contour(distribs_lo)
    #plots2D.plot_contour_bands(distribs_lo, 10000, 128, None, [5, 25, 55, 75, 95])
    #plotsND.plot_contour(distribs_lo, 10000, 128, None, [5, 25, 50, 75, 95])
    #plotsND.plot_contour_samples(distribs_lo, 10000, 128, None, [5, 25, 50, 75, 95])

def example_kde():
    samples = np.random.randn(1000,2)
    distr = distribution(samples)
    plots2D.plot_contour(distr)

def example_uamds_1d():
    distribs_hi = data.load_iris_normal()
    distribs_lo = uamds(distribs_hi, dims=4)
    labels = ['setosa','versicolor','virginica']
    titles = ['sepal length','sepal width','petal length','petal width']
    colors = ['red','green', 'blue']
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig, axs = plots1D.plot_1d_distribution(distribs_lo, 10000, ['boxplot','violinplot'], 444, fig, axs, labels, titles, colors, vert=True, colorblind_safe=False)
    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
    example_uamds()
   # example_kde()
   # example_uamds_1d()
