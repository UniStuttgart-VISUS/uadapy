from sklearn import datasets
from sklearn.mixture import GaussianMixture
import numpy as np
from uadapy import Distribution, TimeSeries
from uadapy.distributions import MultivariateGMM
from scipy import stats
from typing import List, Iterable, Optional

def _to_class_distributions(X: np.ndarray,
                            y: np.ndarray,
                            normal: bool = False,
                            classes: Optional[Iterable[int]] = None) -> List[Distribution]:
    """
    Convert labeled samples into one `Distribution` per class.

    If `normal` is False, each class is represented by a `Distribution`
    initialized with raw samples (KDE-backed). If `normal` is True, each class
    is represented as a multivariate Normal via `Distribution(samples, "Normal")`.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : ndarray of shape (n_samples,)
        Class labels for each sample in `X`.
    normal : bool, optional
        If True, fit a multivariate Normal for each class.
        If False, store raw samples for KDE-based density and sampling.
        Default is False.
    classes : iterable of int, optional
        Subset of class labels to include. If None, all unique labels in `y`
        are used. Default is None.

    Returns
    -------
    list of Distribution
        One `Distribution` object per selected class, in ascending class-label order.

    """

    if classes is None:
        classes = np.unique(y)
    dists = []
    for c in classes:
        Xc = np.asarray(X[y == c])
        if normal:
            dists.append(Distribution(Xc, "Normal"))
        else:
            dists.append(Distribution(Xc))
    return dists

def load_iris(normal: bool = False):
    """
    Load the Iris dataset as class-conditional distributions (KDE-based by default).
    This function creates one nonparametric, KDE-backed Distribution per Iris species
    (setosa, versicolor, virginica).

    Parameters
    ----------
    normal : bool, optional
        If False (default), represent each species with a KDE over raw samples
        (nonparametric). If True, fit a multivariate Normal per species.

    Returns
    -------
    list of Distribution
        Three distributions, one for each Iris species (setosa, versicolor, virginica).
    """

    iris = datasets.load_iris()
    return _to_class_distributions(iris.data, iris.target, normal=normal)

def load_iris_normal():
    """
    Load the Iris dataset as class-conditional multivariate Normal distributions.
    This function fits a multivariate Normal distribution to the feature vectors
    of each Iris species (setosa, versicolor, virginica).

    Returns
    -------
    list of Distribution
        Three multivariate Normal distributions, one for each Iris species.

    Returns
    -------
    list of Distribution
        Three multivariate Normal distributions, one per Iris species.
    """

    return load_iris(normal=True)

def load_wine(normal: bool = False):
    """
    Load the Wine dataset as class-conditional distributions (KDE-based by default).
    Builds one KDE-backed, nonparametric Distribution per wine cultivar using the
    13-dimensional chemical analysis features.

    Parameters
    ----------
    normal : bool, optional
        If False (default), use KDE per class. If True, use multivariate Normal.

    Returns
    -------
    list of Distribution
        Three distributions, one for each wine cultivar.
    """

    wine = datasets.load_wine()
    return _to_class_distributions(wine.data, wine.target, normal=normal)

def load_wine_normal():
    """
    Load the Wine dataset as class-conditional multivariate Normal distributions.
    Fits a multivariate Normal to the chemical analysis features for each of the
    three wine cultivars.

    Returns
    -------
    list of Distribution
        Three multivariate Normal distributions, one per wine cultivar.
    """

    return load_wine(normal=True)

def load_breast_cancer(normal: bool = False):
    """
    Load the Breast Cancer Wisconsin dataset as class-conditional distributions (KDE-based by default).
    Creates one nonparametric, KDE-backed Distribution for each label (benign, malignant)
    from the 30-dimensional features.

    Parameters
    ----------
    normal : bool, optional
        If False (default), use KDE per class. If True, use multivariate Normal.

    Returns
    -------
    list of Distribution
        Two distributions, one for each class.
    """
    ds = datasets.load_breast_cancer()
    return _to_class_distributions(ds.data, ds.target, normal=normal)

def load_breast_cancer_normal():
    """
    Load the Breast Cancer Wisconsin dataset as multivariate Normal distributions.
    Fits a multivariate Normal to the feature vectors for each class label
    (benign vs. malignant).

    Returns
    -------
    list of Distribution
        Two multivariate Normal distributions, one per class.
    """
    return load_breast_cancer(normal=True)

def load_digits_normal():
    """
    Load the Digits dataset as class-conditional multivariate Normal distributions.
    Fits a multivariate Normal to the flattened 8x8 image vectors for each digit
    (0-9).

    Returns
    -------
    list of Distribution
        Ten multivariate Normal distributions, one per digit.
    """
    ds = datasets.load_digits()
    return _to_class_distributions(ds.data, ds.target, normal=True)

def fetch_covtype_distributions_normal():
    """
    Fetch the Forest Cover Type dataset as class-conditional multivariate Normals.
    Fits a multivariate Normal distribution for each of the 7 cover types over
    the 54-dimensional feature space (terrain and one-hot indicators).

    Returns
    -------
    list of Distribution
        Seven multivariate Normal distributions, one per forest cover type.
    """

    ds = datasets.fetch_covtype()
    return _to_class_distributions(ds.data, ds.target, normal=True)


def make_blobs_distributions(n_samples: int = 1500,
                             centers: int | np.ndarray = 3,
                             n_features: int = 2,
                             cluster_std: float | Iterable[float] = 1.0,
                             random_state: int | None = 42,
                             normal: bool = False):
    """
    Generate Gaussian blobs and return KDE-based class-conditional distributions (by default).
    Uses `make_blobs` to synthesize clustered data and builds one Distribution per
    cluster label.

    Parameters
    ----------
    n_samples : int, optional
        Total number of samples. Default is 1500.
    centers : int or ndarray, optional
        Number of clusters or explicit centers. Default is 3.
    n_features : int, optional
        Dimensionality of the feature space. Default is 2.
    cluster_std : float or iterable, optional
        Standard deviation(s) of clusters. Default is 1.0.
    random_state : int or None, optional
        Seed for reproducibility. Default is 42.
    normal : bool, optional
        If False (default), KDE per cluster. If True, multivariate Normal per cluster.

    Returns
    -------
    list of Distribution
        One distribution for each generated blob.
    """

    X, y = datasets.make_blobs(
        n_samples=n_samples,
        centers=centers,
        n_features=n_features,
        cluster_std=cluster_std,
        random_state=random_state,
    )
    return _to_class_distributions(X, y, normal=normal)

def make_blobs_distributions_normal(**kwargs):
    """
    Generate Gaussian blobs and fit class-conditional multivariate Normals.
    Creates synthetic clusters with `make_blobs` and fits a multivariate Normal
    to each cluster's samples.

    Parameters
    ----------
    **kwargs
        Forwarded to `make_blobs_distributions` (e.g., n_samples, centers,
        n_features, cluster_std, random_state).

    Returns
    -------
    list of Distribution
        One multivariate Normal distribution per blob.
    """

    return make_blobs_distributions(normal=True, **kwargs)

def make_moons_distributions(n_samples: int = 2000,
                             noise: float = 0.15,
                             random_state: Optional[int] = 42,
                             normal: bool = False):
    """
    Generate the two-moons dataset and return KDE-based class-conditional distributions (by default).
    Produces two interleaving half-circles with label 0/1 and builds a KDE-backed
    Distribution for each.

    Parameters
    ----------
    n_samples : int, optional
        Total number of samples. Default is 2000.
    noise : float, optional
        Standard deviation of Gaussian noise. Default is 0.15.
    random_state : int or None, optional
        Seed for reproducibility. Default is 42.
    normal : bool, optional
        If False (default), KDE per class. If True, multivariate Normal per class.

    Returns
    -------
    list of Distribution
        Two distributions, one for each moon.
    """

    X, y = datasets.make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    return _to_class_distributions(X, y, normal=normal)

def make_moons_distributions_normal(**kwargs):
    """
    Generate the two-moons dataset and fit multivariate Normal distributions.
    Produces two interleaving half-circles with `make_moons` and fits a
    multivariate Normal to each moon's samples.

    Parameters
    ----------
    **kwargs
        Forwarded to `make_moons_distributions` (e.g., n_samples, noise,
        random_state).

    Returns
    -------
    list of Distribution
        Two multivariate Normal distributions, one per moon.
    """

    return make_moons_distributions(normal=True, **kwargs)

def make_circles_distributions(n_samples: int = 2000,
                               noise: float = 0.1,
                               factor: float = 0.5,
                               random_state: Optional[int] = 42,
                               normal: bool = False):
    """
    Generate concentric circles and return KDE-based class-conditional distributions (by default).
    Creates two rings (inner/outer) and fits a KDE-backed Distribution to each.

    Parameters
    ----------
    n_samples : int, optional
        Total number of samples. Default is 2000.
    noise : float, optional
        Standard deviation of Gaussian noise. Default is 0.1.
    factor : float, optional
        Scale between inner/outer circle (0 < factor < 1). Default is 0.5.
    random_state : int or None, optional
        Seed for reproducibility. Default is 42.
    normal : bool, optional
        If False (default), KDE per class. If True, multivariate Normal per class.

    Returns
    -------
    list of Distribution
        Two distributions, one for the inner circle and one for the outer circle.
    """

    X, y = datasets.make_circles(n_samples=n_samples, noise=noise, factor=factor, random_state=random_state)
    return _to_class_distributions(X, y, normal=normal)

def make_circles_distributions_normal(**kwargs):
    """
    Generate concentric circles and fit multivariate Normal distributions.
    Uses `make_circles` to create two concentric rings and fits a multivariate
    Normal to each ring's samples.

    Parameters
    ----------
    **kwargs
        Forwarded to `make_circles_distributions` (e.g., n_samples, noise,
        factor, random_state).

    Returns
    -------
    list of Distribution
        Two multivariate Normal distributions, one per circle.
    """

    return make_circles_distributions(normal=True, **kwargs)

def make_gaussian_quantiles_data(normal: bool = False,
                                 n_samples: int = 2000,
                                 n_features: int = 2,
                                 n_classes: int = 3,
                                 mean: np.ndarray | None = None,
                                 cov: float = 1.0,
                                 random_state: int = 42):
    """
    Generate Gaussian-quantiles data and return KDE-based class-conditional distributions (by default).
    Samples from a multivariate Gaussian are split into labels by quantile thresholds.

    Parameters
    ----------
    normal : bool, optional
        If False (default), KDE per class. If True, multivariate Normal per class.
    n_samples : int, optional
        Total number of samples. Default is 2000.
    n_features : int, optional
        Feature dimensionality. Default is 2.
    n_classes : int, optional
        Number of quantile-based classes. Default is 3.
    mean : ndarray or None, optional
        Mean vector of the underlying Gaussian. Default is None (zeros).
    cov : float, optional
        Covariance scaling factor. Default is 1.0.
    random_state : int, optional
        Seed for reproducibility. Default is 42.

    Returns
    -------
    list of Distribution
        One distribution per class created by the quantile split.
    """

    X, y = datasets.make_gaussian_quantiles(mean=mean, cov=cov, n_samples=n_samples,
                                   n_features=n_features, n_classes=n_classes,
                                   random_state=random_state)

    return _to_class_distributions(X, y, normal=normal)


def make_gaussian_quantiles_normal(**kwargs):
    """
    Generate Gaussian-quantiles data and fit multivariate Normal distributions.
    Samples from an underlying multivariate Gaussian are split into classes by
    quantile thresholds (`make_gaussian_quantiles`).

    Parameters
    ----------
    **kwargs
        Forwarded to `make_gaussian_quantiles_data` (e.g., n_samples, n_features,
        n_classes, mean, cov, random_state).

    Returns
    -------
    list of Distribution
        One multivariate Normal distribution per quantile-defined class.
    """

    return make_gaussian_quantiles_data(normal=True, **kwargs)

def make_classification_distributions_normal(n_samples: int = 2000,
                                             n_features: int = 10,
                                             n_informative: int = 5,
                                             n_redundant: int = 2,
                                             n_repeated: int = 0,
                                             n_classes: int = 3,
                                             class_sep: float = 1.0,
                                             flip_y: float = 0.01,
                                             weights: Optional[Iterable[float]] = None,
                                             random_state: Optional[int] = 42):
    """
    Generate a synthetic classification problem and fit multivariate Normals.
    Calls `make_classification` to synthesize an n-class dataset with controllable
    informative/redundant features and class separability, then fits a multivariate
    Normal per class for a parametric density view.

    Parameters
    ----------
    n_samples : int, optional
        Total number of samples to generate. Default is 2000.
    n_features : int, optional
        Total number of features. Default is 10.
    n_informative : int, optional
        Number of informative features. Default is 5.
    n_redundant : int, optional
        Number of redundant features (linear combinations of informative).
        Default is 2.
    n_repeated : int, optional
        Number of duplicated features. Default is 0.
    n_classes : int, optional
        Number of target classes. Default is 3.
    class_sep : float, optional
        Controls separability between classes (larger = easier). Default is 1.0.
    flip_y : float, optional
        Fraction of samples whose class is randomly exchanged (label noise).
        Default is 0.01.
    weights : iterable of float, optional
        Class weights that sum to 1. If None, classes are balanced. Default is None.
    random_state : int or None, optional
        Seed for reproducibility. Default is 42.

    Returns
    -------
    list of Distribution
        One distribution per generated class.

    """

    X, y = datasets.make_classification(n_samples=n_samples, n_features=n_features,
                               n_informative=n_informative, n_redundant=n_redundant,
                               n_repeated=n_repeated, n_classes=n_classes,
                               class_sep=class_sep, flip_y=flip_y,
                               weights=weights, random_state=random_state)
    return _to_class_distributions(X, y, normal=True)

def make_hastie_10_2_distributions(n_samples: int = 12000,
                                   random_state: Optional[int] = 42,
                                   normal: bool = False):
    """
    Generate the Hastie 10-2 binary classification dataset and return class-conditional
    distributions (KDE-based by default).
    This classic synthetic set has 10 features and labels y ∈ {-1, +1}. By default,
    this function builds one nonparametric, KDE-backed Distribution per class.

    Parameters
    ----------
    n_samples : int, optional
        Total number of samples to generate. Default is 12000 (as in sklearn).
    random_state : int or None, optional
        Seed for reproducibility. Default is 42.
    normal : bool, optional
        If False (default), use KDE per class. If True, use multivariate Normal.

    Returns
    -------
    list of Distribution
        Two distributions, one for the class y = -1 and one for y = +1.
    """
    X, y = datasets.make_hastie_10_2(n_samples=n_samples, random_state=random_state)
    y = (y > 0).astype(int)
    return _to_class_distributions(X, y, normal=normal)

def make_hastie_10_2_distributions_normal(**kwargs):
    """
    Generate the Hastie 10-2 dataset and fit multivariate Normal distributions.
    Creates the 10-feature binary dataset and fits a multivariate Normal to each
    class (y ∈ {-1, +1} mapped to {0, 1}).

    Parameters
    ----------
    **kwargs
        Forwarded to `make_hastie_10_2_distributions` (e.g., n_samples, random_state).

    Returns
    -------
    list of Distribution
        Two multivariate Normal distributions, one per class.
    """
    return make_hastie_10_2_distributions(normal=True, **kwargs)

def load_iris_gmm(n_components=2, random_state=0):
    """
    Uses the iris dataset and fits a Gaussian Mixture Model for each class.
    
    Parameters
    ----------
    n_components : int, optional
        Number of mixture components for each GMM.
        Default value is 2.
    random_state : int, optional
        Random seed for reproducibility.
        Default value is 0.
    
    Returns
    -------
    list
        List of Distribution objects, each wrapping a MultivariateGMM model
        fitted to one class of the iris dataset.
    """
    iris = datasets.load_iris()
    dists = []
    
    for c in np.unique(iris.target):
        # Get data for this class
        class_data = iris.data[iris.target == c]
        
        # Fit GMM to the class data
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type='full',
            random_state=random_state
        )
        gmm.fit(class_data)
        
        # Wrap and store
        dists.append(Distribution(MultivariateGMM(gmm), name="GMM"))
    
    return dists

def generate_synthetic_timeseries(timesteps=200, trend = 0.1):
    """
    Generates synthetic time series data by modeling a combination of trend,
    periodic patterns, and noise using a multivariate normal distribution
    with an exponential quadratic kernel for covariance.

    Parameters
    ----------
    timesteps : int
        The time steps of the time series.
        Default value is 200.

    Returns
    -------
    timeseries : Timeseries object
        An instance of the TimeSeries class, which represents a univariate time series.
    """
    np.random.seed(0)
    t = np.arange(1, timesteps + 1)
    trend = t * trend
    periodic = 10 * np.sin(2 * np.pi * t / 100)
    noise = 2 * (np.random.rand(timesteps) - 0.5)
    mu = trend + periodic + noise
    sigma2 = 20 * np.ones(timesteps)
    sigma_sq = np.sqrt(sigma2)
    sigma = np.zeros((timesteps, timesteps))

    def ex_qu_kernel(x, y, sigma_i, sigma_j, l):
        return sigma_i * sigma_j * np.exp(-0.5 * np.linalg.norm(x - y)**2 / l**2)
    
    for i in range(timesteps):
        for j in range(timesteps):
            sigma[i, j] = ex_qu_kernel(t[i], t[j], sigma_sq[i], sigma_sq[j], 5)

    # Ensure symmetry
    sigma = (sigma + sigma.T) / 2
    
    # Ensure positive definiteness
    epsilon = 1e-6
    sigma += np.eye(sigma.shape[0]) * epsilon
    model = stats.multivariate_normal(mu, sigma)
    timeseries = TimeSeries(model, None)

    return timeseries

def generate_synthetic_gmm(n_classes=3, n_dims=4, random_state=0):
    """
    Generates synthetic Gaussian Mixture Model distributions.
    Creates multiple classes, each represented as a GMM with
    random number of components (1 to 10). Per component, random means
    and covariances are generated to form the GMM.
    
    Parameters
    ----------
    n_classes : int, optional
        Number of classes to generate.
        Default value is 3.
    n_dims : int, optional
        Dimensionality of the original data space.
        Default value is 4.
    random_state : int, optional
        Random seed for reproducibility.
        Default value is 0.
    
    Returns
    -------
    list
        List of Distribution objects, each wrapping a MultivariateGMM model.
    """
    np.random.seed(random_state)

    distributions = []
    
    for class_idx in range(n_classes):
        # Vary the number of components per class
        n_components = np.random.randint(1, 10)
        
        # Generate synthetic data for this class
        samples_per_component = 100
        class_data = []
        
        for comp_idx in range(n_components):
            # Random mean
            mean = np.random.randn(n_dims) * 3 + class_idx * 5
            
            # Random covariance
            A = np.random.randn(n_dims, n_dims)
            cov = A @ A.T + np.eye(n_dims) * 0.5
            
            # Generate samples
            samples = np.random.multivariate_normal(mean, cov, samples_per_component)
            class_data.append(samples)
        
        # Combine all component samples
        class_data = np.vstack(class_data)
        
        # Fit GMM
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type='full',
            random_state=random_state + class_idx
        )
        gmm.fit(class_data)
        
        # Wrap and store
        distributions.append(Distribution(MultivariateGMM(gmm), name="GMM"))
    
    return distributions
