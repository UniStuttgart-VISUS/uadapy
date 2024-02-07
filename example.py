import uadapy as ua
import uadapy.data as data
import uadapy.dr.uamds as uamds
import uadapy.plotting.plots2D as plots2D
import numpy as np



def example_uamds():
    distribs_hi = data.load_iris_normal()
    distribs_lo = uamds.uamds(distribs_hi, dims=2)
    plots2D.plot_contour(distribs_lo)

def example_kde():
    samples = np.random.randn(1000,2)
    distr = ua.distribution.distribution(samples)
    plots2D.plot_contour(distr)



if __name__ == '__main__':
    #example_uamds()
    example_kde()
