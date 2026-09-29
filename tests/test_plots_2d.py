import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend

import numpy as np
import pytest
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from uadapy import Distribution
from uadapy.distributions import MultivariateGMM
from uadapy.dr import uapca
import uadapy.data as data
from uadapy.plotting import plots_2d

@pytest.fixture
def sample_distributions():
    """Fixture to create sample distributions."""
    distribs_hi = data.load_iris_normal()
    distribs_lo = uapca(distribs_hi, n_dims=2)
    return distribs_lo

@pytest.mark.mpl_image_compare(baseline_dir="baseline")
def test_plot_samples(sample_distributions):
    """Test plot_samples function."""
    fig, axs = plots_2d.plot_samples(sample_distributions, n_samples=10000)
    return fig

@pytest.mark.mpl_image_compare(baseline_dir="baseline")
def test_plot_contour(sample_distributions):
    """Test plot_contour function."""
    fig, axs = plots_2d.plot_contour(sample_distributions)
    return fig

@pytest.mark.mpl_image_compare(baseline_dir="baseline")
def test_plot_contour_bands(sample_distributions):
    """Test plot_contour_bands function."""
    fig, axs = plots_2d.plot_contour_bands(sample_distributions)
    return fig


def _normal_distribution(angle=0, stds=(1, 1e-3)):
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])
    covariance = rotation @ np.diag(np.square(stds)) @ rotation.T
    return Distribution(multivariate_normal(mean=[0, 0], cov=covariance), name="Normal")


@pytest.mark.parametrize("angle", [0, np.pi / 4, 0.37])
def test_thin_normal_contours_are_visible_and_autoscale(angle):
    distribution = _normal_distribution(angle)
    _, axes = plots_2d.plot_contour(distribution, resolution=128)
    assert len(axes.collections[0].get_paths()) == 3

    covariance = distribution.cov()
    radius = np.sqrt(plots_2d.chi2.ppf(0.95, 2) * np.diag(covariance))
    xlim, ylim = axes.get_xlim(), axes.get_ylim()
    assert xlim[0] <= -radius[0] and xlim[1] >= radius[0]
    assert ylim[0] <= -radius[1] and ylim[1] >= radius[1]
    assert axes.dataLim.x0 <= -radius[0] and axes.dataLim.x1 >= radius[0]
    assert axes.dataLim.y0 <= -radius[1] and axes.dataLim.y1 >= radius[1]


def test_thin_and_wide_distributions_each_get_visible_contours():
    thin = _normal_distribution()
    wide = Distribution(multivariate_normal(mean=[4, 0], cov=np.eye(2)), name="Normal")
    _, axes = plots_2d.plot_contour([thin, wide], resolution=128)
    assert len(axes.collections) == 2
    assert all(len(collection.get_paths()) == 3 for collection in axes.collections)
    assert axes.get_xlim()[0] < -2 and axes.get_xlim()[1] > 6


def test_explicit_ranges_force_shared_axis_aligned_grid():
    class RecordingDistribution:
        name = "recording"
        n_dims = 2

        def __init__(self):
            self.points = None

        def mean(self):
            return np.zeros(2)

        def cov(self):
            return np.eye(2)

        def pdf(self, points):
            self.points = np.asarray(points)
            return np.exp(-np.sum(self.points ** 2, axis=1) / 2)

    distribution = Distribution(RecordingDistribution(), name="recording")
    plots_2d.plot_contour(distribution, ranges=[(-2, 3), (-4, 5)], resolution=16)
    assert np.allclose(distribution.model.points.min(axis=0), [-2, -4])
    assert np.allclose(distribution.model.points.max(axis=0), [3, 5])


class _FallbackDistribution:
    name = "fallback"
    n_dims = 2

    def mean(self):
        return np.array([1.0, 2.0])

    def cov(self):
        return None

    def sample(self, n, seed=None):
        return np.random.default_rng(seed).normal(size=(n, 2))


def test_invalid_moment_fallback_warns_once_and_is_reproducible():
    distribution = _FallbackDistribution()
    with pytest.warns(RuntimeWarning, match="2000 samples") as caught:
        moments_a = plots_2d._get_moments(distribution)
    with pytest.warns(RuntimeWarning):
        moments_b = plots_2d._get_moments(distribution)
    assert len(caught) == 1
    assert np.array_equal(moments_a[0], moments_b[0])
    assert np.array_equal(moments_a[1], moments_b[1])
    assert np.array_equal(moments_a[0], [1, 2])


def test_moment_fallback_failure_names_distribution():
    class Broken(_FallbackDistribution):
        name = "broken"

        def sample(self, n, seed=None):
            raise RuntimeError("sampling failed")

    with pytest.raises(ValueError, match="broken"):
        plots_2d._get_moments(Broken())


@pytest.mark.parametrize("bad_moment", ["mean_raises", "mean_nonfinite", "cov_raises", "cov_nonfinite"])
def test_only_invalid_moment_is_replaced(bad_moment):
    class PartiallyInvalid(_FallbackDistribution):
        def mean(self):
            if bad_moment == "mean_raises":
                raise RuntimeError("mean unavailable")
            if bad_moment == "mean_nonfinite":
                return [np.nan, 2]
            return [1, 2]

        def cov(self):
            if bad_moment == "cov_raises":
                raise RuntimeError("covariance unavailable")
            if bad_moment == "cov_nonfinite":
                return [[np.inf, 0], [0, 1]]
            return np.eye(2) * 3

    with pytest.warns(RuntimeWarning):
        mean, covariance = plots_2d._get_moments(PartiallyInvalid())
    if bad_moment.startswith("mean_"):
        assert np.allclose(covariance, np.eye(2) * 3)
        assert not np.array_equal(mean, [1, 2])
    else:
        assert np.array_equal(mean, [1, 2])
        assert np.allclose(covariance, np.cov(np.random.default_rng(55).normal(size=(2000, 2)).T))


def test_singular_covariance_produces_finite_positive_grid_extents():
    class Singular(_FallbackDistribution):
        def cov(self):
            return np.array([[1.0, 1.0], [1.0, 1.0]])

    grid = plots_2d._calculate_ranges_analytical(Singular(), 95, resolution=32)
    assert np.all(np.isfinite(grid.half_extents))
    assert np.all(grid.half_extents > 0)
    assert np.all(np.isfinite(grid.coordinates()[0]))


def test_numeric_rotated_ridge_contours():
    class Ridge(_FallbackDistribution):
        name = "ridge"
        angle = 0.41
        direction = np.array([np.cos(angle), np.sin(angle)])
        covariance = np.outer(direction, direction) + 1e-6 * np.outer(
            [-direction[1], direction[0]], [-direction[1], direction[0]]
        )

        def mean(self):
            return np.zeros(2)

        def cov(self):
            return self.covariance

        def pdf(self, points):
            points = np.asarray(points)
            return multivariate_normal.pdf(points, mean=[0, 0], cov=self.covariance)

    _, axes = plots_2d.plot_contour(Distribution(Ridge(), name="ridge"), resolution=128)
    assert len(axes.collections[0].get_paths()) == 3


@pytest.mark.parametrize("covariance_type", ["full", "diag", "tied", "spherical"])
def test_gmm_analytical_grid_supports_covariance_types(covariance_type):
    model = GaussianMixture(n_components=2, covariance_type=covariance_type)
    model.means_ = np.array([[-1.0, 0], [1.0, 0]])
    if covariance_type == "full":
        model.covariances_ = np.array([np.diag([0.4, 0.01]), np.diag([0.01, 0.4])])
    elif covariance_type == "diag":
        model.covariances_ = np.array([[0.4, 0.01], [0.01, 0.4]])
    elif covariance_type == "tied":
        model.covariances_ = np.diag([0.2, 0.2])
    else:
        model.covariances_ = np.array([0.2, 0.3])
    model.weights_ = np.array([0.5, 0.5])
    gmm = MultivariateGMM(model)
    distribution = Distribution(gmm, name="GMM")
    grid = plots_2d._calculate_ranges_analytical(distribution, 95)
    assert np.all(np.isfinite(grid.coordinates()[0]))
    assert np.all(grid.half_extents > 0)


def test_isovalue_contour_contains_requested_probability_mass():
    grid = plots_2d.OrientedGrid.axis_aligned([(-4, 4), (-4, 4)], 200)
    x, y = grid.coordinates()
    points = np.stack((x, y), axis=-1)
    density = multivariate_normal.pdf(points, mean=[0, 0], cov=np.eye(2))
    threshold = plots_2d._calculate_isovalues(density, grid.cell_area(), [95])[0]
    samples = np.random.default_rng(55).normal(size=(10000, 2))
    sample_density = multivariate_normal.pdf(samples, mean=[0, 0], cov=np.eye(2))
    assert np.mean(sample_density >= threshold) == pytest.approx(0.95, abs=0.02)


def test_contour_bands_handle_thin_and_zero_pdf():
    thin = _normal_distribution()
    _, axes = plots_2d.plot_contour_bands(thin, resolution=128)
    assert len(axes.collections) > 0

    class Zero(_FallbackDistribution):
        def pdf(self, points):
            return np.zeros(len(points))

    with pytest.warns(RuntimeWarning, match="PDF is zero"):
        _, axes = plots_2d.plot_contour_bands(Distribution(Zero(), name="zero"), resolution=32)
    assert len(axes.collections) == 0