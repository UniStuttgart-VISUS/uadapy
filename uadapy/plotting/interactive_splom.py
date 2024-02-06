import matplotlib

from uadapy.plotting.distribution_plot import InteractiveNormal

matplotlib.use("TkAgg")

import numpy as np
import matplotlib.pyplot as plt


def compute_scaling_factor_at_axis(axis: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    # return the scaling factor s for a = s * b along the specified axis

    # angle of axis
    angle = compute_angle(axis, np.array([1, 0]))

    R = make_2d_rotation_matrix(angle)

    # rotate a and b to be aligned with x-axis
    ar = R @ a
    br = R @ b

    # print("a:", np.rad2deg(angle), axis, R, a, b, ar, br)

    # x-value is length on axis
    an = ar[0]
    bn = br[0]

    if bn < 1e-16 or an < 1e-16:
        return 1.0

    return an / bn


def compute_rotation_scaling_matrix__along_axis(axis: np.ndarray, a: np.ndarray, b: np.ndarray):
    """
    Returns a rotation_scale matrix that transforms a
        vector/matrix by the scale factor s for a = s * b along the specified axis.
    """
    # angle of axis
    angle = compute_angle(axis, np.array([1, 0]))

    R = make_2d_rotation_matrix(angle)  # rotates axis to x-axis

    # rotate a and b to be aligned with x-axis
    ar = R @ a
    br = R @ b

    # print("a:", np.rad2deg(angle), axis, R, a, b, ar, br)

    # x-value is length on axis
    an = ar[0]
    bn = br[0]

    if bn < 1e-16 or an < 1e-16:
        scale_factor = 1.0
    else:
        scale_factor = an / bn

    S = np.array([[np.sqrt(scale_factor), 0.0], [0.0, 1.0]])

    return S @ R


def compute_scaling(a: np.ndarray, b: np.ndarray) -> float:
    # return the scaling factor s for a = s * b

    # x-value is length on axis
    an = np.linalg.norm(a)
    bn = np.linalg.norm(b)

    if bn < 1e-16 or an < 1e-16:
        return 1.0

    return an / bn



def compute_angle(a: np.ndarray, b: np.ndarray) -> float:
    dot = np.dot(a, b)
    det = np.cross(a, b)
    angle = np.arctan2(det, dot)
    # print(np.rad2deg(angle))
    return angle


def make_2d_rotation_matrix(angle: float):
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)

    # rotation matrix
    R = np.array([[cos_angle, -sin_angle],
                  [sin_angle, cos_angle]])
    return R


class InteractiveSplom:
    def __init__(self, mean: np.ndarray, cov: np.ndarray, fig_args: dict = None, n_std: float = 1.0,
                 extends: float = 10, epsilon: float = 5):
        self.mean = mean
        self.cov = cov

        self.n_std = n_std
        self.extends = extends
        self.epsilon = epsilon

        self.dim = len(mean)

        assert self.dim > 1

        if fig_args is None:
            fig_args = {}
        fig, axes = plt.subplots(self.dim-1, self.dim-1, **fig_args)
        if self.dim <= 2:
            axes = np.array([axes])

        self.fig = fig

        self.axes = axes
        self.subplots = np.empty(axes.shape, dtype=object)

        for row_i, row in enumerate(axes):
            row_i += 1
            for col_i, ax in enumerate(row):
                if col_i >= row_i:
                    self.subplots[row_i, col_i] = None

                    ax.axis("off")
                    continue

                x_label = ""
                y_label = ""

                if row_i-1 == len(self.subplots) - 1:
                    x_label = f"dim={col_i}"

                if col_i == 0:
                    y_label = f"dim={row_i}"

                mean_ij = self.mean[[row_i, col_i]]
                cov_ij = np.array([[self.cov[col_i, col_i], self.cov[col_i, row_i]],
                                   [self.cov[row_i, col_i], self.cov[row_i, row_i]]])
                self.subplots[row_i-1, col_i] = InteractiveNormal(mean_ij, cov_ij, ax, self.n_std,
                                                                  extends=self.extends, epsilon=self.epsilon,
                                                                  y_label=y_label,
                                                                  x_label=x_label)

        fig.canvas.mpl_connect('button_press_event', self.button_press_callback)
        fig.canvas.mpl_connect('button_release_event', self.button_release_callback)
        fig.canvas.mpl_connect('motion_notify_event', self.motion_notify_callback)

        self.update_all_plots()
        self.current_pressed_subplot = None
        self.last_local_mouse_pos = np.zeros((2, ), dtype=float)
        self.currently_selected_point = None

    def get_current_subplot(self, event) -> InteractiveNormal:
        for subplot in self.subplots.flat:
            if subplot is not None and subplot.ax is event.inaxes:
                return subplot
        return None

    def get_current_subplot_idx(self, current_subplot):
        for row_i, row in enumerate(self.subplots):
            for col_i, subplot in enumerate(row):
                if subplot is current_subplot:
                    return row_i, col_i

        return None, None

    def button_press_callback(self, event):
        'whenever a mouse button is pressed'
        if event.inaxes is None:
            return
        if event.button != 1:
            return

        subplot = self.get_current_subplot(event)

        self.current_pressed_subplot = subplot
        self.last_local_mouse_pos = np.array([event.xdata, event.ydata])
        point_idx = subplot.get_ind_under_point(event)
        self.currently_selected_point = point_idx

    def button_release_callback(self, event):
        'whenever a mouse button is released'
        if event.button != 1:
            return
        self.current_pressed_subplot = None
        self.last_local_mouse_pos = np.zeros((2,), dtype=float)
        self.currently_selected_point = None
        self.update_all_plots()

    def update_mean_cov(self):
        for row_i, row in enumerate(self.axes):
            row_i += 1
            for col_i, ax in enumerate(row):
                if col_i >= row_i:
                    continue

                mean_ij = self.mean[[row_i, col_i]]
                cov_ij = np.array([[self.cov[col_i, col_i], self.cov[col_i, row_i]],
                                   [self.cov[row_i, col_i], self.cov[row_i, row_i]]])

                self.subplots[row_i-1, col_i].mean = mean_ij
                self.subplots[row_i-1, col_i].cov = cov_ij

                if self.current_pressed_subplot is not self.subplots[row_i-1, col_i] or True:
                    self.subplots[row_i-1, col_i].init_points()
                    # self.subplots[row_i - 1, col_i].adjust_points(0, self.subplots[row_i - 1, col_i].points[0])

    def update_plots(self, row_i_, col_i_):
        updated = []
        for i in range(len(self.subplots)):
            if self.subplots[i, col_i_] is not None and (i, col_i_) not in updated:
                self.subplots[i, col_i_].update(i == row_i_)
                updated.append((i, col_i_))
            if self.subplots[row_i_, i] is not None and (row_i_, i) not in updated:
                if i == col_i_:
                    continue
                self.subplots[row_i_, i].update()
                updated.append((row_i_, i))

            if row_i_ < len(self.subplots) - 1:
                if self.subplots[i, row_i_+1] is not None and (i, row_i_+1) not in updated:
                    self.subplots[i, row_i_+1].update()
                    updated.append((i, row_i_+1))

            if col_i_ > 0:
                if self.subplots[col_i_-1, i] is not None and (col_i_-1, i) not in updated:
                    self.subplots[col_i_-1, i].update()
                    updated.append((col_i_-1, i))

    def update_all_plots(self):
        for subplot in self.subplots.flat:
            if subplot is not None:
                subplot.update(True)

    def motion_notify_callback(self, event):
        'on mouse movement'
        if event.inaxes is None:
            return
        if event.button != 1:
            return

        new_current_subplot = self.get_current_subplot(event)
        if new_current_subplot is self.current_pressed_subplot:
            if new_current_subplot is not None:
                row_i, col_i = self.get_current_subplot_idx(new_current_subplot)
                assert row_i is not None and col_i is not None

                row_i += 1
                if self.currently_selected_point is None:
                    angle = compute_angle(self.last_local_mouse_pos - new_current_subplot.mean,
                                               np.array([event.xdata, event.ydata]) - new_current_subplot.mean)
                    R_sub_ = make_2d_rotation_matrix(angle)
                    # print(R_sub_)

                    self.mean[row_i] = new_current_subplot.mean[0]
                    self.mean[col_i] = new_current_subplot.mean[1]

                    # apply rotation of everything
                    R_ = np.eye(len(self.cov), dtype=float)
                    R_[col_i, col_i] = R_sub_[0, 0]
                    R_[col_i, row_i] = R_sub_[0, 1]
                    R_[row_i, col_i] = R_sub_[1, 0]
                    R_[row_i, row_i] = R_sub_[1, 1]

                    self.cov = R_ @ self.cov @ R_.T
                else:
                    scaling_factor = compute_scaling_factor_at_axis(
                        self.current_pressed_subplot.points[self.currently_selected_point] -
                        self.current_pressed_subplot.mean,
                        self.current_pressed_subplot.points[self.currently_selected_point] -
                        self.current_pressed_subplot.mean,
                        np.array([event.xdata, event.ydata]) - new_current_subplot.mean)

                    ev = (self.current_pressed_subplot.points[self.currently_selected_point] -
                          self.current_pressed_subplot.mean)

                    angle = compute_angle(ev, np.array([1.0, 0.0]))
                    R = make_2d_rotation_matrix(angle)

                    R_ = np.eye(len(self.cov))
                    R_[col_i, col_i] = R[0, 0]
                    R_[col_i, row_i] = R[0, 1]
                    R_[row_i, col_i] = R[1, 0]
                    R_[row_i, row_i] = R[1, 1]

                    S = np.eye(len(self.cov))
                    S[col_i, col_i] = 1 / np.sqrt(scaling_factor)

                    self.cov = (R_.T @ S @ R_) @ self.cov @ (R_.T @ S @ R_).T

                self.cov[np.abs(self.cov) < 1e-16] = 0.0
                self.update_mean_cov()
                self.update_plots(row_i - 1, col_i)

        else:
            self.current_pressed_subplot = None
            self.last_local_mouse_pos = np.zeros((2,), dtype=float)
            self.update_all_plots()

        self.fig.canvas.draw_idle()

        self.last_local_mouse_pos = np.array([event.xdata, event.ydata])

    def show(self):
        self.fig.show()


def main():
    # dim = 7
    # mean = np.zeros(dim)
    # cov = np.eye(dim, dim)

    from uadapy.data import load_iris_normal

    dist = load_iris_normal()[0]

    isplom = InteractiveSplom(dist.mean(), dist.cov(), epsilon=20, extends=0.1)
    # isplom = InteractiveSplom(mean, cov, epsilon=20, extends=20)
    isplom.show()
    plt.show()

    # print("old cov:\n", dist.cov())
    new_cov = isplom.cov
    print("new cov:\n", new_cov)


if __name__ == '__main__':
    main()
