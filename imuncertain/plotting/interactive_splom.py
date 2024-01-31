import matplotlib

from imuncertain.plotting.distribution_plot import InteractiveNormal

matplotlib.use("TkAgg")


import numpy as np

from matplotlib.widgets import Button
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


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

        self.subplots = np.empty(axes.shape, dtype=object)

        for row_i, row in enumerate(axes):
            row_i += 1
            for col_i, ax in enumerate(row):
                if col_i >= row_i:
                    self.subplots[row_i, col_i] = None
                    continue

                mean_ij = self.mean[[row_i, col_i]]
                cov_ij = np.array([[self.cov[col_i, col_i], self.cov[col_i, row_i]],
                                   [self.cov[row_i, col_i], self.cov[row_i, row_i]]])
                self.subplots[row_i-1, col_i] = InteractiveNormal(mean_ij, cov_ij, ax, self.n_std,
                                                                  self.extends, self.epsilon)

        fig.canvas.mpl_connect('button_press_event', self.button_press_callback)
        fig.canvas.mpl_connect('button_release_event', self.button_release_callback)
        fig.canvas.mpl_connect('motion_notify_event', self.motion_notify_callback)

        self.current_pressed_subplot = None

    def get_current_subplot(self, event):
        for subplot in self.subplots.flat:
            if subplot is not None and subplot.ax is event.inaxes:
                return subplot
        return None

    def button_press_callback(self, event):
        'whenever a mouse button is pressed'
        if event.inaxes is None:
            return
        if event.button != 1:
            return

        subplot = self.get_current_subplot(event)

        self.current_pressed_subplot = subplot

        # print(pind)
        # pind = self.get_ind_under_point(event)

    def button_release_callback(self, event):
        'whenever a mouse button is released'
        if event.button != 1:
            return
        self.current_pressed_subplot = None

    def motion_notify_callback(self, event):
        'on mouse movement'
        if event.inaxes is None:
            return
        if event.button != 1:
            return

        new_current_subplot = self.get_current_subplot(event)
        if new_current_subplot is self.current_pressed_subplot:
            if new_current_subplot is not None:
                point_index = new_current_subplot.get_ind_under_point(event)
                print(point_index, np.array([event.xdata, event.ydata]))
                if point_index is not None:
                    new_current_subplot.adjust_points(point_index, np.array([event.xdata, event.ydata]))
                    new_current_subplot.update()
        else:
            self.current_pressed_subplot = None

    def show(self):
        self.fig.show()


def main():
    dim = 3
    mean = np.zeros(dim)
    cov = np.eye(dim, dim)
    isplom = InteractiveSplom(mean, cov, epsilon=10)
    isplom.show()
    plt.show()


if __name__ == '__main__':
    main()
