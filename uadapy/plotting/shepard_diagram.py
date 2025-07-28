import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
from sklearn import datasets
from itertools import combinations
from uadapy import Distribution
from uadapy.dr import uamds

METRIC_LABELS = {
    "kl": "KL Divergence",
    "wasserstein": "Wasserstein Distance",
    "mean": "Euclidean Distance (Means)"
}

def custom_wasserstein(dist1, dist2):
    n_samples = 10000
    seed = 55
    samples1 = dist1.sample(n_samples, seed)
    samples2 = dist2.sample(n_samples, seed)
    return np.mean([wasserstein_distance(samples1[:, k], samples2[:, k])
                    for k in range(samples1.shape[1])])

def _load_iris():
    iris = datasets.load_iris()
    dists = []
    for c in np.unique(iris.target):
        dists.append(Distribution(iris.data[iris.target == c]))
    return dists, iris.target_names

def _get_metric_name(metric):
    if isinstance(metric, str):
        return METRIC_LABELS.get(metric, metric)
    elif callable(metric):
        return getattr(metric, "__name__", "custom")
    else:
        return str(metric)

def _kl_gaussian(p_mean, p_cov, q_mean, q_cov):
    p_var = np.diag(p_cov)
    q_var = np.diag(q_cov)
    term1 = np.sum(np.log(q_var / p_var))
    term2 = np.sum((p_var + (p_mean - q_mean) ** 2) / q_var)
    return 0.5 * (term1 + term2 - len(p_mean))

def _compute_distances(dists, metric="kl", n_samples=1000, seed=55):
    distances = []
    for i, j in combinations(range(len(dists)), 2):
        dist1, dist2 = dists[i], dists[j]
        if callable(metric):
            d = metric(dist1, dist2)
        elif metric == "kl":
            mA, cA = dist1.mean(), dist1.cov()
            mB, cB = dist2.mean(), dist2.cov()
            d = _kl_gaussian(mA, cA, mB, cB)
        elif metric == "wasserstein":
            samples1 = dist1.sample(n_samples, seed)
            samples2 = dist2.sample(n_samples, seed)
            d = np.mean([wasserstein_distance(samples1[:, k], samples2[:, k]) 
                        for k in range(samples1.shape[1])])
        elif metric == "mean":
            d = np.linalg.norm(dist1.mean() - dist2.mean())
        else:
            raise ValueError("Unknown metric")
        distances.append(d)
    return distances

def plot_shepard_diagram(distributions_hi,
                         distributions_lo,
                         n_samples=1000,
                         seed=55,
                         metric="kl",
                         labels=None,
                         show_plot=False):
    
    """
    Plot a Shepard diagram to assess how well pairwise distances between distributions 
    in a high-dimensional space are preserved in a reduced (low-dimensional) space.
    Supports predefined or custom distance metrics.

    Parameters
    ----------
    distributions_hi : list of uadapy.Distribution or uadapy.Distribution
        The original (high-dimensional) distributions.
    distributions_lo : list of uadapy.Distribution or uadapy.Distribution
        The reduced (low-dimensional) distributions obtained via dimensionality reduction.
    n_samples : int, optional
        Number of samples to draw per distribution when using sample-based distance metrics.
        Default is 10000.
    seed : int, optional
        Seed for the random number generator to ensure reproducibility of sampling. 
        Default is 55.
    metric : str or callable, optional
        Distance metric to use for computing pairwise distances between distributions.
        If a string, choose from:
        - "kl"           : KL divergence between Gaussian approximations.
        - "wasserstein"  : Average Wasserstein distance across dimensions.
        - "mean"         : Euclidean distance between distribution means.
        If a callable, it must accept the following arguments:
            - dist1: uadapy.Distribution
            - dist2: uadapy.Distribution
        and return a scalar distance value.
        Default is "kl".
    labels : list of str, optional
        List of class names for the distributions. If provided, labels like 
        "class A vs class B" will be used to annotate points in the plot.
        If None, numeric indices will be used.
    show_plot : bool, optional
        If True, displays the plot.
        Default is False.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object containing the Shepard diagram.
    matplotlib.axes.Axes
        The Axes object used for the plot.
    """

    if isinstance(distributions_hi, Distribution):
        distributions_hi = [distributions_hi]
    if isinstance(distributions_lo, Distribution):
        distributions_lo = [distributions_lo]

    orig_dists = _compute_distances(distributions_hi, metric, n_samples, seed)
    red_dists = _compute_distances(distributions_lo, metric, n_samples, seed)

    if labels is not None:
        pairs = list(combinations(range(len(labels)), 2))
        pair_labels = [f"{labels[i]} vs {labels[j]}" for i, j in pairs]
    else:
        pairs = list(combinations(range(len(distributions_hi)), 2))
        pair_labels = [f"{i} vs {j}" for i, j in pairs]
    

    plt.figure(figsize=(6, 5))
    plt.scatter(orig_dists, red_dists, s=100)
    for xi, yi, label in zip(orig_dists, red_dists, pair_labels):
        plt.annotate(label, (xi, yi), textcoords="offset points", xytext=(5, 5), fontsize=9)
    plt.plot([min(orig_dists), max(orig_dists)],
             [min(orig_dists), max(orig_dists)], 'r--', label="Ideal: y = x")
    metric_name = _get_metric_name(metric)
    plt.xlabel(f"{metric_name} in Original Space")
    plt.ylabel(f"{metric_name} in Reduced Space")
    plt.title(f"Shepard Diagram ({metric_name})")
    plt.grid(True)
    plt.legend()

    fig = plt.gcf()
    axs = plt.gca()

    if show_plot:
        fig.tight_layout()
        plt.show()

    return fig, axs

# Example usage
distribs_hi, labels = _load_iris()
distribs_lo = uamds(distribs_hi, n_dims=2)
plot_shepard_diagram(distribs_hi,
                     distribs_lo,
                     n_samples=10000,
                     seed=55,
                     metric=custom_wasserstein,
                     labels=labels,
                     show_plot=True)
