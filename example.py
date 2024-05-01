import uadapy as ua
import uadapy.data as data
import uadapy.dr.uamds as uamds
import uadapy.plotting.plots2D as plots2D
import uadapy.plotting.boxplot as boxplot
import numpy as np
import matplotlib.pyplot as plt



def example_uamds():
    distribs_hi = data.load_iris_normal()
    distribs_lo = uamds.uamds(distribs_hi, dims=2)
    plots2D.plot_contour(distribs_lo)

def example_kde():
    samples = np.random.randn(1000,2)
    distr = ua.distribution.distribution(samples)
    plots2D.plot_contour(distr)

def example_uamds_boxplot():
    distribs_hi = data.load_iris_normal()
    distribs_lo = uamds.uamds(distribs_hi, dims=4)
    labels = ['setosa','versicolor','virginica']
    titles = ['sepal length','sepal width','petal length','petal width']
    colors = ['red','green', 'blue']
    fig, axs = plt.subplots(2, 2)
    fig, axs = boxplot.plot_boxplot(distribs_lo, 10000, fig, axs, labels, titles, colors, vert=True, colorblind_safe=False)
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    example_uamds()
   # example_kde()
   # example_uamds_boxplot()
