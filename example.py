import uadapy as ua
import uadapy.data as data
import uadapy.dr.uamds as uamds
import uadapy.dr.uapca as uapca
import uadapy.plotting.plots1D as plots1D
import uadapy.plotting.plots2D as plots2D
import uadapy.plotting.plotsND as plotsND
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl



def example_uamds():
    distribs_hi = data.load_iris_normal()
    colors = mpl.colormaps['Set2'].colors[0:3]
    attrib_names = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width']
    fig, ax = plots1D.plot_1d_distribution(distribs_hi, num_samples=10_000, plot_types=['boxplot', 'violinplot'], colors=colors, titles=attrib_names)
    #fig.show()
    fig.savefig("iris-1d.svg", bbox_inches='tight')
    fig.savefig("iris-1d.pdf", bbox_inches='tight')
    fig, ax = plotsND.contour_plot_matrix(distribs_hi, distrib_colors=colors, attrib_names=attrib_names)
    #fig.show()
    fig.savefig("iris-plotmatrix.svg", bbox_inches='tight')
    fig.savefig("iris-plotmatrix.pdf", bbox_inches='tight')
    fig, ax = plt.subplots(nrows=2, ncols=1)
    distribs_lo = uamds.uamds(distribs_hi, dims=2)
    plots2D.plot_contour_bands(distribs_lo, distrib_colors=colors, ax=ax[0])
    ax[0].set_title("Uncertainty-aware MDS")
    ax[0].axis('equal')
    distribs_lo = uapca.uapca(distribs_hi, dims=2)
    plots2D.plot_contour_bands(distribs_lo, distrib_colors=colors, ax=ax[1])
    ax[1].set_title("Uncertainty-aware PCA")
    ax[1].axis('equal')
    fig.tight_layout()
    #fig.show()
    fig.savefig("iris-dr.svg", bbox_inches='tight')
    fig.savefig("iris-dr.pdf", bbox_inches='tight')
    #plots2D.plot_contour_bands(distribs_lo, 10000, 128, None, [5, 25, 55, 75, 95])
    #plotsND.plot_contour(distribs_lo, 10000, 128, None, [5, 25, 50, 75, 95])
    #plotsND.plot_contour_samples(distribs_lo, 10000, 128, None, [5, 25, 50, 75, 95])


def example_kde():
    samples = np.random.randn(1000,2)
    distr = ua.distribution.distribution(samples)
    plots2D.plot_contour(distr)

def example_uamds_1d():
    distribs_hi = data.load_iris_normal()
    distribs_lo = uamds.uamds(distribs_hi, dims=4)
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
