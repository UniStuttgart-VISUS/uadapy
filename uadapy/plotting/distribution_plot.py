import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


def confidence_ellipse(mean: np.ndarray, cov: np.ndarray, ax: plt.Axes, n_std: float = 3.0,
                       facecolor='blue', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    assert len(mean) == 2
    assert cov.shape[-1] == 2

    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, edgecolor="black", **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = mean[0]

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = mean[1]

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


class InteractiveNormal:
    def __init__(self, mean: np.ndarray, cov: np.ndarray, ax: plt.Axes, n_std: float = 1.0, extends: float = 10,
                 epsilon: float = 10, x_label: str = "", y_label: str = ""):
        self.mean = mean
        self.cov = cov
        self.ax = ax
        self.n_std = n_std
        self.extends = extends
        self.epsilon = epsilon
        self.x_label = x_label
        self.y_label = y_label

        self.init_extends = False

        # init points
        self.points = None
        self.eigenvectors = None
        self.init_points()
        self.update()

    def init_points(self):
        eigenvalues, eigenvectors = np.linalg.eig(self.cov)
        self.points = self.mean + (np.sign(eigenvalues) * np.sqrt(np.abs(eigenvalues)) * eigenvectors).T
        self.points = np.row_stack([self.points, 2 * self.mean - self.points])

        if not self.init_extends:
            self.init_extends = True
            self.extends = self.extends + 4 * np.linalg.norm(self.points - self.mean, axis=0).max()

    def update(self, plot_int: bool = False):
        self.ax.clear()

        # self.ax.autoscale(False)

        # clear
        # self.ax.set_yticklabels([])
        # self.ax.set_xticklabels([])
        # self.ax.set_xticks([])
        # self.ax.set_yticks([])

        self.ax.set_xlabel(self.x_label)
        self.ax.set_ylabel(self.y_label)

        # plot ellipse
        # confidence_ellipse(self.mean, self.cov, self.ax, self.n_std)
        confidence_ellipse(self.mean, self.cov, self.ax, self.n_std, facecolor="white")

        # plot lines of eigenvectors
        if plot_int:
            line_color = "gray"
            point_color = "orange"
            linestyle = 'dashed'
            self.ax.plot([self.mean[0], self.points[0, 0]], [self.mean[1], self.points[0, 1]], c=line_color,
                         linestyle=linestyle)
            self.ax.plot([self.mean[0], self.points[1, 0]], [self.mean[1], self.points[1, 1]], c=line_color,
                         linestyle=linestyle)

            self.ax.plot([self.mean[0], self.points[2, 0]], [self.mean[1], self.points[2, 1]], c=line_color,
                         linestyle=linestyle)
            self.ax.plot([self.mean[0], self.points[3, 0]], [self.mean[1], self.points[3, 1]], c=line_color,
                         linestyle=linestyle)

            # plot points of eigenvectors
            self.ax.scatter(self.points[:, 0], self.points[:, 1], c=point_color)

            # plot mean center
            self.ax.scatter(self.mean[0], self.mean[1], c="black")

        extends = self.extends  # 2 * np.abs(points).max()

        self.ax.axis('equal')
        self.ax.set_ylim(self.mean[1] - extends, self.mean[1] + extends)
        self.ax.set_xlim(self.mean[0] - extends, self.mean[0] + extends)



        # self.ax.autoscale(False)

        # self.ax.set_title(f"{[self.mean[0] - extends, self.mean[0] + extends]}, {[-self.mean[1] + extends, self.mean[0] + extends]}")

        # self.ax.get_figure().canvas.draw_idle()

    def get_ind_under_point(self, event):
        'get the index of the vertex under point if within epsilon tolerance'
        tinv = self.ax.transData
        xyt = tinv.transform(self.points)
        xt, yt = xyt[:, 0], xyt[:, 1]
        d = np.hypot(xt - event.x, yt - event.y)
        indseq, = np.nonzero(d == d.min())
        ind = indseq[0]

        if d[ind] >= self.epsilon:
            ind = None

        return ind
