import uadapy as ua
import uadapy.data as data
import uadapy.dr.uamds as uamds
import uadapy.plotting.plots2D as plots2D
import uadapy.plotting.boxplot as boxplot
import numpy as np



def example_uamds():
    distribs_hi = data.load_iris_normal()
    distribs_lo = uamds.uamds(distribs_hi, dims=2)
    plots2D.plot_contour(distribs_lo)

def example_kde():
    samples = np.random.randn(1000,2)
    distr = ua.distribution.distribution(samples)
    plots2D.plot_contour(distr)

def example_uamds_scatter():
    distribs_hi = data.load_iris_normal()
    print(distribs_hi)
    distribs_lo = uamds.uamds(distribs_hi, dims=2)
    plots2D.plot_samples(distribs_lo, 10)

def example_kde_scatter():
    samples = np.random.randn(1000,2)
    distr = ua.distribution.distribution(samples)
    plots2D.plot_samples(distr, 10)

def example_uamds_boxplot():
    distribs_hi = data.load_iris_normal()
    distribs_lo = uamds.uamds(distribs_hi, dims=2)
    labels = ['setosa','versicolor','virginica']
    titles = ['sepal length','sepal width','petal length','petal width']
    boxplot.plot_boxplot(distribs_lo, 100, labels, titles)

def example_kde_boxplot():
    samples = np.random.randn(1000,2)
    distr = ua.distribution.distribution(samples)
    boxplot.plot_boxplot(distr, 100)



if __name__ == '__main__':
   # example_uamds()
   # example_kde()
    #example_uamds_scatter()
    #example_kde_scatter()
    example_uamds_boxplot()
    #example_kde_boxplot()
