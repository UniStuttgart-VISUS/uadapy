import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend

import pytest
from uadapy.dr import uamds
import uadapy.data as data
from uadapy.plotting import plots_nd

@pytest.fixture
def sample_distributions():
    """Fixture to create sample distributions."""
    distribs_hi = data.load_iris_normal()
    distribs_lo = uamds(distribs_hi, n_dims=2)
    return distribs_lo

@pytest.mark.mpl_image_compare(baseline_dir="baseline")
def test_plot_samples_nd(sample_distributions):
    """Test plot_samples function."""
    fig, axs = plots_nd.plot_samples(sample_distributions, n_samples=10000)
    return fig

@pytest.mark.mpl_image_compare(baseline_dir="baseline")
def test_plot_contour_nd(sample_distributions):
    """Test plot_contour function."""
    fig, axs = plots_nd.plot_contour(sample_distributions,n_samples=10000)
    return fig

@pytest.mark.mpl_image_compare(baseline_dir="baseline")
def test_plot_contour_samples_nd(sample_distributions):
    """Test plot_contour_samples function."""
    fig, axs = plots_nd.plot_contour_samples(sample_distributions, n_samples=10000)
    return fig