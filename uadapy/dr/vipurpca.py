"""
This module provides helper and plotting functions built on top of the
VIPurPCA library for uncertainty-aware principal component analysis (PCA).

It enables the computation of eigenvectors under uncertainty and the
visualization of distribution trajectories in principal component space.

For details on the VIPurPCA method, see the corresponding paper:
https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10368349

"""


import jax
import jax.numpy as np
from jax import vmap
import numpy
import matplotlib.pyplot as plt
from sklearn import preprocessing
import uadapy.plotting.utils as utils
import glasbey as gb
from scipy.linalg import block_diag
from vipurpca import PCA
from vipurpca.helper_functions import equipotential_standard_normal


def _prepare_pca_inputs(dists):
    """
    Prepare stacked means, block-diagonal covariance, and labels from distributions.

    Parameters
    ----------
    dists : list of Distribution
        List of distribution objects. Each distribution must provide
        `.mean()`, `.cov()`, and `.name`.

    Returns
    -------
    Y : ndarray of shape (n, p)
        Stacked mean vectors.
    C : ndarray of shape (n*p, n*p)
        Block-diagonal covariance matrix.
    labels : ndarray of shape (n,)
        Distribution labels repeated for each sample.
    """
    means = []
    cov_blocks = []
    labels = []

    for dist in dists:
        mu = numpy.atleast_2d(dist.mean())
        cov = numpy.asarray(dist.cov())
        means.append(mu)
        cov_blocks.append(cov)
        labels.append(dist.name)

    Y = numpy.vstack(means).astype(numpy.float32)
    C = block_diag(*cov_blocks).astype(numpy.float32)

    return np.asarray(Y), np.asarray(C), numpy.asarray(labels)


def _effective_rank_from_X(X, tol=None):
    """
    Compute the effective numerical rank of a matrix using SVD.

    Parameters
    ----------
    X : ndarray of shape (n, p)
        Input data matrix.
    tol : float, optional
        Numerical tolerance. If None, a default based on matrix size
        and precision is used.

    Returns
    -------
    rank : int
        Effective rank of the input matrix.
    """
    Xc = X - np.mean(X, axis=0, keepdims=True)
    s = np.linalg.svd(Xc, compute_uv=False)
    if tol is None:
        tol = max(Xc.shape) * np.finfo(s.dtype).eps * s[0]
    return int(np.sum(s > tol))


def _fit_pca_with_uncertainty(Y, cov_Y, n_components=3):
    """
    Fit PCA model with uncertainty propagation.

    Parameters
    ----------
    Y : ndarray of shape (n, p)
        Stacked mean vectors.
    cov_Y : ndarray of shape (n*p, n*p)
        Covariance matrix (full).
    n_components : int, default=3
        Number of principal components to retain.

    Returns
    -------
    model : PCA
        PCA model containing eigenvalues, eigenvectors, and
        covariance of eigenvectors.
    """
    r = _effective_rank_from_X(Y)
    if n_components > r:
        print(f"Reducing n_components from {n_components} to {r} due to rank deficiency.")
        n_components = r

    model = PCA(matrix=Y,
                sample_cov=None,
                feature_cov=None,
                full_cov=cov_Y,
                n_components=n_components,
                axis=0)

    model.pca_value()
    model.compute_cov_eigenvectors()
    return model


def compute_distribution_eigenvectors(dists, n_components=3):
    """
    Compute eigenvectors of PCA fitted on distributions with uncertainty.

    Parameters
    ----------
    dists : list of Distribution
        List of distribution objects with `.mean()`, `.cov()`, and `.name`.
    n_components : int, default=3
        Number of principal components to retain.

    Returns
    -------
    eigenvectors : ndarray of shape (p, n_components)
        Principal component eigenvectors.
    """
    Y, cov_Y, _ = _prepare_pca_inputs(dists)
    model = _fit_pca_with_uncertainty(Y, cov_Y, n_components)
    return model.eigenvectors


def plot_distribution_trajectories(dists,
                                   n_components=3,
                                   pcx=1,
                                   pcy=2,
                                   n_frames=10,
                                   seed=55,
                                   fig=None,
                                   axs=None,
                                   distrib_colors=None,
                                   colorblind_safe=False,
                                   show_plot=False):
    """
    Plot static trajectories of distributions under PCA with uncertainty.

    For each distribution, PCA directions are sampled according to the
    propagated covariance of eigenvectors. Trajectories in the principal
    component space are then plotted for each sample.

    Parameters
    ----------
    dists : list of Distribution
        List of distribution objects with `.mean()`, `.cov()`, and `.name`.
    n_components : int, default=3
        Number of principal components to retain.
    pcx : int, default=1
        Principal component index for x-axis.
    pcy : int, default=2
        Principal component index for y-axis.
    n_frames : int, default=10
        Number of trajectory samples to draw.
    seed : int, default=42
        Random seed for reproducibility.
    fig : matplotlib.figure.Figure, optional
        Figure object to draw into. If None, a new figure is created.
    axs : matplotlib.axes.Axes, optional
        Axes object to draw into. If None, new axes are created.
    distrib_colors : list or None, optional
        List of colors to use for each distribution. If None, Matplotlib Set2 and glasbey colors will be used.
    colorblind_safe : bool, optional
        If True, the plot will use colors suitable for colorblind individuals.
        Default is False.
    show_plot : bool, optional
        If True, display the plot.
        Default is False.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object containing the plot.
    axs : matplotlib.axes.Axes
        Axes object containing the plot.
    """
    numpy.random.seed(seed)

    Y, cov_Y, labels = _prepare_pca_inputs(dists)
    model = _fit_pca_with_uncertainty(Y, cov_Y, n_components)

    if model.cov_eigenvectors is None:
        raise Exception('Uncertainty of eigenvectors has not been computed.')
    if pcx > model.n_components or pcy > model.n_components:
        raise Exception('pcx and pcy must be <= n_components')

    S = equipotential_standard_normal(model.size[1] * model.n_components, n_frames + 1)
    L, _ = jax.scipy.linalg.cho_factor(
        model.cov_eigenvectors + 1e-5 * np.eye(model.cov_eigenvectors.shape[0]), lower=True
    )
    eigv_samples = np.transpose(np.dot(L, S)) + np.ravel(model.eigenvectors, 'F')

    samples_reshaped = vmap(
        lambda s: np.transpose(np.reshape(s, (min(model.size[0], model.n_components), model.size[1]), 'C'))
    )(eigv_samples)

    samples = np.array([model.X_unflattener(model.X_flat) @ i for i in samples_reshaped])
    samples = np.array([s[:, [pcx - 1, pcy - 1]] for s in samples])  # shape: [frames, samples, 2]

    n_frames, n_samples, _ = samples.shape

    if axs is None:
        if fig is None:
            fig, axs = plt.subplots(figsize=(8, 7))
        else:
            if fig.axes is not None:
                axs = fig.axes[0]
            else:
                raise ValueError("The provided figure has no axes. Pass an Axes or create subplots first.")
    else:
        if fig is None:
            fig = axs.figure

    if distrib_colors is None:
        if colorblind_safe:
            palette = gb.create_palette(
                palette_size=len(set(labels)), 
                colorblind_safe=colorblind_safe
            )
        else:
            palette = utils.get_colors(len(set(labels)))
    else:
        if len(distrib_colors) < len(set(labels)):
            if colorblind_safe:
                additional_colors = gb.create_palette(
                    palette_size=len(set(labels)) - len(distrib_colors), 
                    colorblind_safe=colorblind_safe
                )
            else:
                additional_colors = utils.get_colors(len(set(labels)) - len(distrib_colors))
            distrib_colors.extend(additional_colors)
        palette = distrib_colors

    # Map colors to samples
    if labels is not None:
        le = preprocessing.LabelEncoder()
        labels_as_int = le.fit_transform(labels)
        color_list = [palette[i] for i in labels_as_int]
    else:
        color_list = ['black'] * n_samples

    # Plot each sample's trajectory
    for i in range(n_samples):
        traj = samples[:, i, :]
        c = color_list[i]
        axs.plot(traj[:, 0], traj[:, 1], lw=1.5, color=c, alpha=0.7)
        axs.plot(traj[0, 0], traj[0, 1], 'o', color=c, markersize=4)
        axs.plot(traj[-1, 0], traj[-1, 1], 'X', color=c, markersize=5)

    axs.set_xlabel(f"PC {pcx}", fontsize=12)
    axs.set_ylabel(f"PC {pcy}", fontsize=12)
    axs.set_title("Distribution trajectories under PCA with uncertainty", fontsize=14)
    axs.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    axs.set_aspect('equal')
    axs.set_ylim(axs.get_xlim())

    if labels is not None:
        handles = []
        for idx, name in enumerate(le.classes_):
            handles.append(
                plt.Line2D(
                    [0], [0],
                    marker='o',
                    color='w',
                    label=name,
                    markerfacecolor=palette[idx],
                    markersize=6
                )
            )
        axs.legend(handles=handles, loc='best', title='Labels')

    if show_plot:
        fig.tight_layout()
        plt.show()

    return fig, axs



# Example usage

from uadapy import Distribution
from sklearn import datasets

def load_iris():
    iris = datasets.load_iris()
    dists = []
    for c in np.unique(iris.target):
        dists.append(Distribution(iris.data[iris.target == c], name=iris.target_names[c]))
    return dists

dists = load_iris()

eigenvectors = compute_distribution_eigenvectors(dists, n_components=2)
print(eigenvectors)
print("Eigenvectors shape:", eigenvectors.shape)

fig, axs = plot_distribution_trajectories(dists, n_components=2, pcx=1, pcy=2, n_frames=10,
                                          distrib_colors=None, colorblind_safe=False, show_plot=True)
