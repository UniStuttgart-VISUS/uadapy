import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend

import pytest
from uadapy.temporal.uastl import uastl
from uadapy.plotting.plots_timeseries import plot_timeseries, plot_correlated_timeseries, plot_correlation_matrix, plot_corr_length, plot_correlated_corr_length
from uadapy.data import generate_synthetic_timeseries

@pytest.fixture
def synthetic_timeseries():
    """Fixture to generate a synthetic timeseries."""
    return generate_synthetic_timeseries(200)

@pytest.fixture
def correlated_timeseries(synthetic_timeseries):
    """Fixture to generate a correlated timeseries using UASTL."""
    return uastl(synthetic_timeseries, [50, 150])

@pytest.mark.mpl_image_compare(baseline_dir="baseline")
def test_plot_timeseries(synthetic_timeseries):
    """Test plot_timeseries function."""
    fig, axs = plot_timeseries(synthetic_timeseries, n_samples=3)
    return fig

@pytest.mark.mpl_image_compare(baseline_dir="baseline")
def test_plot_corr_length(synthetic_timeseries):
    """Test plot_corr_length function."""
    fig, axs = plot_corr_length(synthetic_timeseries)
    return fig

@pytest.mark.mpl_image_compare(baseline_dir="baseline")
def test_plot_correlated_timeseries(correlated_timeseries):
    """Test plot_correlated_timeseries function."""
    fig, axs = plot_correlated_timeseries(correlated_timeseries, n_samples=3, co_point=100)
    return fig

@pytest.mark.mpl_image_compare(baseline_dir="baseline")
def test_plot_correlation_matrix(correlated_timeseries):
    """Test plot_correlation_matrix function."""
    fig, axs = plot_correlation_matrix(correlated_timeseries)
    return fig

@pytest.mark.mpl_image_compare(baseline_dir="baseline")
def test_plot_correlated_corr_length(correlated_timeseries):
    """Test plot_correlated_corr_length function."""
    fig, axs = plot_correlated_corr_length(correlated_timeseries)
    return fig
