# This file is to test consistency of uadapy's API.
# It calls most commonly used functions with default arguments and named arguments
# as the corrsponding documentation suggests. If calling a function errors, it is
# due to a change in the API or changes that result in breaking previously working
# functionality. If an API change was intentional, the test calls in this file need
# to be updated accordingly. In any case, this script should never error.

# make script aware of parent directory where uadapy is located
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_data_module():
    import uadapy.data
    uadapy.data.load_iris()
    uadapy.data.load_iris_normal()


def test_dr_module():
    import uadapy.dr
    import uadapy.distribution
    import numpy as np
    # list of distributions (normal distributions estimated from random data
    distribs = [uadapy.distribution(np.random.rand(10, 3), name='Normal') for _ in range(4)]
    uadapy.dr.uapca(distributions=distribs, n_dims=2)
    uadapy.dr.uamds(distributions=distribs, n_dims=2)


def test_plotting_module():
    # TODO
    pass


def main():
    test_data_module()
    test_dr_module()


if __name__ == '__main__':
    main()
