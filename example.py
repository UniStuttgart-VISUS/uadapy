import imuncertain.data as data
import imuncertain.dr.uamds as uamds
import imuncertain.plotting.plots2D as plots2D



def example_uamds():
    distribs_hi = data.load_iris_normal()
    distribs_lo = uamds.uamds(distribs_hi, dims=2)
    plots2D.plot_contour(distribs_lo)



if __name__ == '__main__':
    example_uamds()
