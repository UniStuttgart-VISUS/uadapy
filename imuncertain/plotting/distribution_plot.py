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
                      facecolor=facecolor, **kwargs)

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
                 epsilon: float = 10):
        self.mean = mean
        self.cov = cov
        self.ax = ax
        self.n_std = n_std
        self.extends = extends
        self.epsilon = epsilon

        # init points
        self.points = None
        self.eigenvectors = None
        self.init_points()
        self.update()


    def init_points(self):
        eigenvalues, eigenvectors = np.linalg.eig(self.cov)

        # indices = eigenvalues.argsort()[::-1]
        # eigenvalues = eigenvalues[indices]
        # eigenvectors = eigenvectors.T[indices].T
        print("evl", eigenvalues)
        print("o", self.points)

        if self.eigenvectors is None:
            self.eigenvectors = eigenvectors
            self.points = (np.sign(eigenvalues) * np.sqrt(np.abs(eigenvalues)) * eigenvectors).T
        else:
            if self.eigenvectors[:, 0].T @ eigenvectors[:, 0] < 0:
                indices = np.array([1, 0])
                eigenvalues = eigenvalues[indices]
                eigenvectors = eigenvectors.T[indices].T

        self.eigenvectors = eigenvectors
        self.points = (np.sign(eigenvalues) * np.sqrt(np.abs(eigenvalues)) * eigenvectors).T


        print("a", self.points)

    def update(self):
        self.ax.clear()

        # plot ellipse
        confidence_ellipse(self.mean, self.cov, self.ax, self.n_std)

        # plot lines of eigenvectors
        self.ax.plot([self.mean[0], self.points[0, 0]], [self.mean[1], self.points[0, 1]])
        self.ax.plot([self.mean[0], self.points[1, 0]], [self.mean[1], self.points[1, 1]])

        # plot points of eigenvectors
        self.ax.scatter(self.points[:, 0], self.points[:, 1])

        # plot mean center
        self.ax.scatter(self.mean[0], self.mean[1], c="black")

        extends = self.extends  # 2 * np.abs(points).max()

        self.ax.axis('equal')
        self.ax.set_xlim([-extends, extends])
        self.ax.set_ylim([-extends, extends])
        self.ax.get_figure().canvas.draw_idle()



    def get_ind_under_point(self, event):
        'get the index of the vertex under point if within epsilon tolerance'
        # display coords
        # print('display x is: {0}; display y is: {1}'.format(event.x,event.y))
        tinv = self.ax.transData
        # print('data x is: {0}; data y is: {1}'.format(xy[0],xy[1]))
        xyt = tinv.transform(self.points)
        xt, yt = xyt[:, 0], xyt[:, 1]
        d = np.hypot(xt - event.x, yt - event.y)
        indseq, = np.nonzero(d == d.min())
        ind = indseq[0]

        if d[ind] >= self.epsilon:
            ind = None

        return ind

    def adjust_points(self, point_index: int, new_value: np.ndarray):
        new_point = new_value
        new_point_length = np.linalg.norm(new_point)
        points = self.points.copy()
        points[point_index] = new_point

        old_other_point_length = np.linalg.norm(points[1 - point_index])

        other_point = (-1 + (2 * point_index)) * np.array([new_point[1], -new_point[0]])
        other_point_length = np.linalg.norm(other_point)
        points[1 - point_index] = other_point / other_point_length
        points[1 - point_index] *= old_other_point_length

        new_eigenvalues = np.zeros(2, dtype=float)
        new_eigenvalues[point_index] = new_point_length
        new_eigenvalues[1 - point_index] = old_other_point_length
        new_eigenvalues = new_eigenvalues ** 2

        points_norm = points.copy()
        points_norm[0] = points_norm[0] / np.linalg.norm(points_norm[0])
        points_norm[1] = points_norm[1] / np.linalg.norm(points_norm[1])  # np.linalg.norm(points[1])])

        # print("points_norm: ", points_norm)
        # print("dd:", points_norm[0].T @ points_norm[1])

        cov = points_norm.T @ (np.eye(2) * new_eigenvalues) @ points_norm
        self.cov = cov
        self.points = points
