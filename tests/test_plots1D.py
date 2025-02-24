import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend

import pytest
from uadapy.dr import uamds
import uadapy.data as data
from uadapy.plotting import plots1D

@pytest.fixture
def sample_distribution():
    """Fixture to create a sample distribution."""
    distribs_hi = data.load_iris_normal()
    distribs_lo = uamds(distribs_hi, n_dims=1)
    return distribs_lo

@pytest.fixture
def sample_distributions():
    """Fixture to create sample distributions."""
    distribs_hi = data.load_iris_normal()
    distribs_lo = uamds(distribs_hi, n_dims=4)
    return distribs_lo

@pytest.mark.mpl_image_compare(baseline_dir="baseline")
def test_generate_boxplot(sample_distribution):
    """Test generate_boxplot function."""
    fig, axs = plots1D.generate_boxplot(sample_distribution)
    return fig

@pytest.mark.mpl_image_compare(baseline_dir="baseline")
def test_generate_violinplot(sample_distribution):
    """Test generate_violinplot function."""
    fig, axs = plots1D.generate_violinplot(sample_distribution)
    return fig

@pytest.mark.mpl_image_compare(baseline_dir="baseline")
def test_generate_dotplot(sample_distribution):
    """Test generate_dotplot function."""
    fig, axs = plots1D.generate_dotplot(sample_distribution)
    return fig

@pytest.mark.mpl_image_compare(baseline_dir="baseline")
def test_generate_stripplot(sample_distribution):
    """Test generate_stripplot function."""
    fig, axs = plots1D.generate_stripplot(sample_distribution)
    return fig

@pytest.mark.mpl_image_compare(baseline_dir="baseline")
def test_generate_swarmplot(sample_distribution):
    """Test generate_swarmplot function."""
    fig, axs = plots1D.generate_swarmplot(sample_distribution)
    return fig

@pytest.mark.mpl_image_compare(baseline_dir="baseline")
def test_generate_multidim_boxplot(sample_distributions):
    """Test generate_multidim_boxplot function."""
    fig, axs = plots1D.generate_multidim_boxplot(sample_distributions)
    return fig

@pytest.mark.mpl_image_compare(baseline_dir="baseline")
def test_generate_multidim_violinplot(sample_distributions):
    """Test generate_multidim_violinplot function."""
    fig, axs = plots1D.generate_multidim_violinplot(sample_distributions)
    return fig

@pytest.mark.mpl_image_compare(baseline_dir="baseline")
def test_generate_multidim_dotplot(sample_distributions):
    """Test generate_multidim_dotplot function."""
    fig, axs = plots1D.generate_multidim_dotplot(sample_distributions)
    return fig

@pytest.mark.mpl_image_compare(baseline_dir="baseline")
def test_generate_multidim_stripplot(sample_distributions):
    """Test generate_multidim_stripplot function."""
    fig, axs = plots1D.generate_multidim_stripplot(sample_distributions)
    return fig

@pytest.mark.mpl_image_compare(baseline_dir="baseline")
def test_generate_multidim_swarmplot(sample_distributions):
    """Test generate_multidim_swarmplot function."""
    fig, axs = plots1D.generate_multidim_swarmplot(sample_distributions)
    return fig
