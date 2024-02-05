import uadapy.data as data
import uadapy.dr.uamds as uamds
import uadapy.plotting.plots2D as plots2D



def example_uamds():
    distribs_hi = data.load_iris_normal()
    distribs_lo = uamds.uamds(distribs_hi, dims=2)
    plots2D.plot_contour(distribs_lo)



if __name__ == '__main__':
    example_uamds()
