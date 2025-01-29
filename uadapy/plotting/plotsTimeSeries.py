import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
from matplotlib.colors import Normalize
import glasbey as gb

def _compute_correlation_matrix(sigma):
    c_inv_mult = np.zeros(sigma.shape[0])
    for k in range(sigma.shape[0]):
        c = sigma[k, k]
        if c < 1e-12:
            c_inv_mult[k] = 0
        else:
            c_inv_mult[k] = 1 / np.sqrt(c)
    cor_mat = np.diag(c_inv_mult) @ sigma @ np.diag(c_inv_mult)
    return cor_mat

def _plot_data(data, plot_type, n_samples, samples_colored, colorblind_safe, line_width):

    smpl_width = 0.5
    sigmalvl = [0, 0.674490, 2.575829]

    cmap_gradient = plt.get_cmap('Blues')
    n_levels = len(sigmalvl) + 2
    color_indices = np.linspace(0, 155, n_levels).astype(int)
    col = cmap_gradient(np.flip(color_indices) / 256)[:, :3]

    if plot_type == "isoband":
        x = np.arange(len(data['mu']))
        for j in range(len(sigmalvl) - 1, 0, -1):
            pos_cont = data['mu'] + sigmalvl[j] * data['sigma_sq']
            neg_cont = data['mu'] - sigmalvl[j] * data['sigma_sq']
            for i in range(len(data['mu']) - 1):
                xp = [x[i], x[i + 1], x[i + 1], x[i]]
                yp = [neg_cont[i], neg_cont[i + 1], pos_cont[i + 1], pos_cont[i]]
                plt.fill(xp, yp, color=col[j], edgecolor='none')
        plt.plot(x, data['mu'], color=col[0], linewidth=line_width)
    
    elif plot_type == "spaghetti":
        if colorblind_safe:
            colors = gb.create_palette(palette_size=n_samples, colorblind_safe=True)
        else :
            cmap_discrete = plt.get_cmap('tab10')
            colors = cmap_discrete(np.linspace(0, 1, max(n_samples, 10)))

        samples_2_plot = data['samples'][:, :n_samples]
        if data['samples'].shape[1] == 2 * n_samples:
            samples_2_plot_shift = data['samples'][:, n_samples:2 * n_samples]
            if not samples_colored:
                h2a = plt.plot(samples_2_plot, color=colors[1], linewidth=line_width * smpl_width)
                h2a_shift = plt.plot(samples_2_plot_shift, color=colors[1], linewidth=line_width * smpl_width * 0.2, linestyle='-')
                h2a_shift_dot = plt.plot(samples_2_plot_shift, color=colors[1], linewidth=line_width * smpl_width, linestyle=':')
            else:
                for i in range(n_samples):
                    plt.plot(samples_2_plot[:, i], linewidth=line_width * smpl_width, color=colors[i])
                    plt.plot(samples_2_plot_shift[:, i], linewidth=line_width * smpl_width * 0.2, color=colors[i], linestyle='-')
                    plt.plot(samples_2_plot_shift[:, i], linewidth=line_width * smpl_width, color=colors[i], linestyle=':')
            if not samples_colored and n_samples > 1:
                for h in h2a:
                    h.set_alpha(0.5)
                for h in h2a_shift:
                    h.set_alpha(0.5)
                for h in h2a_shift_dot:
                    h.set_alpha(0.5)
        else:
            if not samples_colored:
                h2a = plt.plot(samples_2_plot, color=colors[1], linewidth=line_width * smpl_width)
            else:
                for i in range(n_samples):
                    plt.plot(samples_2_plot[:, i], linewidth=line_width * smpl_width, color=colors[i])
            if not samples_colored and n_samples > 1:
                for h in h2a:
                    h.set_alpha(0.5)

    elif plot_type == "comb":
        x = np.arange(len(data['mu']))
        _plot_data(data, "isoband", n_samples, samples_colored, colorblind_safe, line_width)
        plt.plot(x, data['mu'], color=col[0], linewidth=line_width)
        _plot_data(data, "spaghetti", n_samples, samples_colored, colorblind_safe, line_width)

def _plot_correlation_length_data(
        timeseries,
        multi_distr_present=False,
        fig=None,
        axs=None,
        show_plot=False):
    """
    Plot correlation length.

    Parameters
    ----------
    timeseries : Timeseries object
        An instance of the TimeSeries class, which represents a univariate time series.
    multi_distr_present : bool, optional
        if True, plots correlation length for correlated uncertain timeseries data.
        Default is False.
    fig : matplotlib.figure.Figure or None, optional
        Figure object to use for plotting. If None, a new figure will be created.
    axs : matplotlib.axes.Axes or array of Axes or None, optional
        Axes object(s) to use for plotting. If None, new axes will be created.
    show_plot : bool, optional
        If True, display the plot.
        Default is False.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object containing the plot.
    list
        List of Axes objects used for plotting.
    """

    mu = timeseries.mean()
    sigma = timeseries.cov()
    num_periods = 1

    if multi_distr_present:
        sb_plot_nmb = num_periods + 3
    else:
        sb_plot_nmb = 1
    length_data = len(mu) // sb_plot_nmb
    cor_mat = _compute_correlation_matrix(sigma)
    discr_nmb = 13
    time_stamp = [0, timeseries.timesteps]
    stamp_length = time_stamp[1] - time_stamp[0]
    stamp_indices = list(range(len(mu)))

    if fig is None:
        if multi_distr_present:
            fig = plt.figure(figsize=(12, 8))
        else:
            fig = plt.figure(figsize=(15, 3))
    plt.suptitle('Correlation Length')

    sign_cor_mat = np.sign(cor_mat)
    maxdim = len(sign_cor_mat)
    full_cor_length = np.zeros(maxdim)
    vert_r = np.zeros(maxdim)
    vert_l = np.zeros(maxdim)
    for i in range(maxdim):
        cur_sign_el = sign_cor_mat[i, i]
        if abs(cur_sign_el) > 0:
            vert1 = 0
            vert2 = 0
            dec_b = (i + 1) % length_data

            if dec_b == 0:
                var_r = 0
                var_l = length_data
            elif (dec_b < length_data / 2) or (dec_b > length_data / 2):
                var_r = length_data - dec_b
                var_l = dec_b
            else:
                var_r = length_data // 2
                var_l = length_data // 2
            for j in range(0, var_r-1):
                if cur_sign_el == sign_cor_mat[i, i + j]:
                    vert1 += 1
                else:
                    break
            for j in range(0, var_l-1):
                if cur_sign_el == sign_cor_mat[i, i - j]:
                    vert2 += 1
                else:
                    break
            vert_r[i] = vert1
            vert_l[i] = vert2
            full_cor_length[i] = vert1 + vert2 - 1

    nmb_colors = discr_nmb // 2 + 1
    out = [None] * len(cor_mat)
    vert_l = vert_l.astype(int)
    vert_r = vert_r.astype(int)
    for i in range(len(cor_mat)):
        col_ind = np.full(length_data, np.nan)
        k = 1
        if (vert_l[i] >= 0) or (vert_r[i] >= 0):
            for j in range(vert_l[i] - 1, 0, -1):
                col_ind[k] = min(int(np.ceil(cor_mat[i, i - j] * nmb_colors)), nmb_colors)
                k += 1
            col_ind[k] = min(int(np.ceil(cor_mat[i, i] * nmb_colors)), nmb_colors)
            k += 1
            for j in range(0, vert_r[i] - 1):
                col_ind[k] = min(int(np.ceil(cor_mat[i, i + j] * nmb_colors)), nmb_colors)
                k += 1
            col_ind = col_ind[~np.isnan(col_ind)]
            out[i] = col_ind

    plot_mat = np.zeros((len(cor_mat), length_data))
    ysize = np.zeros(len(out))
    for i in range(len(out)):
        if out[i] is not None:
            insert = out[i]
            ysize[i] = len(insert)
            plot_mat[i, :len(insert)] = insert
    plot_mat = np.flipud(plot_mat.T)
    cmap_full = cm.get_cmap('coolwarm', 256)
    colors_upper_half = cmap_full(np.linspace(0.5, 1, nmb_colors - 1))
    colors = np.vstack(([1, 1, 1, 1], colors_upper_half))

    # Create a custom discrete colormap
    cmap = mcolors.ListedColormap(colors)

    for i in range(sb_plot_nmb):
        plt.subplot(sb_plot_nmb, 1, i + 1)
        max_i = max(ysize[stamp_indices[(i * stamp_length):((i + 1) * stamp_length)]])
        max_i = max_i.astype(int)
        image2plot = plot_mat[-max_i:, stamp_indices[(i * stamp_length):((i + 1) * stamp_length)]]
        plt.imshow(image2plot, cmap=cmap, aspect='auto')
        vert_ls = np.repeat(vert_l[stamp_indices[(i * stamp_length):((i + 1) * stamp_length)]], 2)
        xs = np.repeat(np.linspace(0, stamp_length, stamp_length + 1), 2)
        xs = xs[1:-1]
        plt.plot(xs - 0.5, max_i - vert_ls, linewidth=2, color='black', linestyle='-')
        if i == 0:
            plt.title("input data")
        elif i == 1:
            plt.title("trend component")
        elif i == sb_plot_nmb - 1:
            plt.title("residual component")
        else:
            plt.title(f"seasonal component {i - 1}")
    plt.xlabel('timesteps')

    fig = plt.gcf()
    axs = plt.gca()

    if show_plot:
        fig.tight_layout()
        plt.show()

    return fig, axs

def plot_timeseries(
        timeseries,
        n_samples,
        seed=55,
        fig=None,
        axs=None,
        colorblind_safe=False,
        show_plot=False):
    """
    Plot single uncertain timeseries data.

    Parameters
    ----------
    timeseries : Timeseries object
        An instance of the TimeSeries class, which represents a univariate time series.
    n_samples : int
        The number of samples to draw from the given timeseries distribution.
    seed : int
        Seed for the random number generator for reproducibility. It defaults to 55 if not provided.
    fig : matplotlib.figure.Figure or None, optional
        Figure object to use for plotting. If None, a new figure will be created.
    axs : matplotlib.axes.Axes or array of Axes or None, optional
        Axes object(s) to use for plotting. If None, new axes will be created.
    colorblind_safe : bool, optional
        If True, the plot will use colors suitable for colorblind individuals.
        Default is False.
    show_plot : bool, optional
        If True, display the plot.
        Default is False.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object containing the plot.
    list
        List of Axes objects used for plotting.
    """

    mu = timeseries.mean()
    sigma = timeseries.cov()

    if fig is None:
        fig = plt.figure(figsize=(15, 3))
    plt.suptitle('Timeseries')

    plot_type = 'comb'
    samples_colored = True
    line_width = 2.5
    time_stamp = [0, timeseries.timesteps]

    samples = timeseries.sample(n_samples, seed).transpose()
    y = {'samples': samples}
    y['mu'] = mu[time_stamp[0]:time_stamp[1]]
    y['sigma_sq'] = np.sqrt(np.maximum(np.diag(sigma[time_stamp[0]:time_stamp[1], time_stamp[0]:time_stamp[1]]), 0))
    y['sigma'] = sigma[time_stamp[0]:time_stamp[1], time_stamp[0]:time_stamp[1]]

    _plot_data(y, plot_type, n_samples, samples_colored, colorblind_safe, line_width)
    plt.xlabel('timesteps')

    axs = plt.gca()
    if show_plot:
        fig.tight_layout()
        plt.show()

    return fig, axs

def plot_correlated_timeseries(
        timeseries,
        n_samples,
        co_point,
        seed=55,
        fig=None,
        axs=None,
        colorblind_safe=False,
        show_plot=False):
    """
    Plot correlated uncertain timeseries data.

    Parameters
    ----------
    timeseries : Timeseries object
        An instance of the TimeSeries class, which represents a univariate time series.
    n_samples : int
        The number of samples to draw from the given timeseries distribution.
    co_point : int
        Interactive point for correlation exploration.
    seed : int
        Seed for the random number generator for reproducibility. It defaults to 55 if not provided.
    fig : matplotlib.figure.Figure or None, optional
        Figure object to use for plotting. If None, a new figure will be created.
    axs : matplotlib.axes.Axes or array of Axes or None, optional
        Axes object(s) to use for plotting. If None, new axes will be created.
    colorblind_safe : bool, optional
        If True, the plot will use colors suitable for colorblind individuals.
        Default is False.
    show_plot : bool, optional
        If True, display the plot.
        Default is False.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object containing the plot.
    list
        List of Axes objects used for plotting.
    """

    mean = timeseries.mean()
    cov = timeseries.cov()
    discr_nmb = 13
    samples_colored=True
    line_width = 2.5
    plot_type = 'comb'
    num_periods = 1

    if fig is None:
        fig = plt.figure(figsize=(12, 8))
    
    plt.suptitle('Uncertainty-Aware Seasonal-Trend Decomposition')

    if (len(mean) == cov.shape[0]) and (len(mean) == cov.shape[1]):
        ylen = len(mean) // (3 + num_periods)
    else:
        raise ValueError("Size of mu and sigma are not concise")

    time_stamp = [0, ylen]
    stamp_indices = list(range(len(mean)))

    samples = timeseries.sample(n_samples, seed).transpose()
    y = {'samples': samples[time_stamp[0]:time_stamp[1], :]}
    LT = {'samples': samples[ylen + time_stamp[0]:ylen + time_stamp[1], :]}
    ST = {'samples': np.zeros((time_stamp[1] - time_stamp[0], n_samples, num_periods))}
    for i in range(num_periods):
        ST['samples'][:, :, i] = samples[(i + 2) * ylen + time_stamp[0]:(i + 2) * ylen + time_stamp[1], :]
    R = {'samples': samples[(2 + num_periods) * ylen + time_stamp[0]:(2 + num_periods) * ylen + time_stamp[1], :]}

    y['mu'] = mean[time_stamp[0]:time_stamp[1]]
    y['sigma_sq'] = np.sqrt(np.maximum(np.diag(cov[time_stamp[0]:time_stamp[1], time_stamp[0]:time_stamp[1]]), 0))
    y['sigma'] = cov[time_stamp[0]:time_stamp[1], time_stamp[0]:time_stamp[1]]

    LT['mu'] = mean[ylen + time_stamp[0]:ylen + time_stamp[1]]
    LT['sigma_sq'] = np.sqrt(np.maximum(np.diag(cov[ylen + time_stamp[0]:ylen + time_stamp[1], ylen + time_stamp[0]:ylen + time_stamp[1]]), 0))
    LT['sigma'] = cov[ylen + time_stamp[0]:ylen + time_stamp[1], ylen + time_stamp[0]:ylen + time_stamp[1]]

    ST['mu'] = np.zeros((time_stamp[1] - time_stamp[0], num_periods))
    ST['sigma_sq'] =  np.zeros((time_stamp[1] - time_stamp[0], num_periods))
    ST['sigma'] = np.zeros((time_stamp[1] - time_stamp[0], time_stamp[1] - time_stamp[0], num_periods))

    for i in range(num_periods):
        ST['mu'][:, i] = mean[(i + 2) * ylen + time_stamp[0]:(i + 2) * ylen + time_stamp[1]].flatten()
        ST['sigma_sq'][:, i] = np.sqrt(np.maximum(np.diag(cov[(i + 2) * ylen + time_stamp[0]:(i + 2) * ylen + time_stamp[1], (i + 2) * ylen + time_stamp[0]:(i + 2) * ylen + time_stamp[1]]), 0))
        ST['sigma'][:, :, i] = cov[(i + 2) * ylen + time_stamp[0]:(i + 2) * ylen + time_stamp[1], (i + 2) * ylen + time_stamp[0]:(i + 2) * ylen + time_stamp[1]]

    R['mu'] = mean[(2 + num_periods) * ylen + time_stamp[0]:(2 + num_periods) * ylen + time_stamp[1]]
    R['sigma_sq'] =  np.sqrt(np.maximum(np.diag(cov[(2 + num_periods) * ylen + time_stamp[0]:(2 + num_periods) * ylen + time_stamp[1], (2 + num_periods) * ylen + time_stamp[0]:(2 + num_periods) * ylen + time_stamp[1]]), 0))
    R['sigma'] = cov[(2 + num_periods) * ylen + time_stamp[0]:(2 + num_periods) * ylen + time_stamp[1], (2 + num_periods) * ylen + time_stamp[0]:(2 + num_periods) * ylen + time_stamp[1]]

    cmap = cm.get_cmap('coolwarm', discr_nmb)
    norm = Normalize(vmin=-(discr_nmb // 2), vmax=(discr_nmb // 2))

    if co_point == 0:
        helper_co_dep = 0
    elif abs(co_point) > len(mean):
        print('Covariance point set too high, plotting without dependency plot.\n')
        helper_co_dep = 0
    elif co_point in stamp_indices:
        helper_co_dep = 1
    else:
        helper_co_dep = 1
        print('Covariance point is out of time_stamp-Range.\n')

    if helper_co_dep == 1:
        interactive_pointer = co_point
        cor_mat = _compute_correlation_matrix(cov)
        plot_back = cor_mat[interactive_pointer, :]

    sigmalvl = [0, 0.674490, 2.575829]

    part_factor = 5
    max_sigmalvl = np.max(sigmalvl)
    ceil_max_sigmalvl = np.ceil(max_sigmalvl)
    lower_limit = np.min(y['mu'] - ceil_max_sigmalvl * y['sigma_sq'])
    upper_limit = np.max(y['mu'] + ceil_max_sigmalvl * y['sigma_sq'])
    y['lims'] = [1, len(y['mu']), lower_limit, upper_limit]
    y['lims'][2] -= helper_co_dep * 1 / part_factor * (y['lims'][3] - y['lims'][2])
    maxyheight = [(y['lims'][3] - y['lims'][2]) / (part_factor + 1) / 2]
    ymid = [y['lims'][2] + maxyheight[0]]

    lower_limit = np.min(LT['mu'] - ceil_max_sigmalvl * LT['sigma_sq'])
    upper_limit = np.max(LT['mu'] + ceil_max_sigmalvl * LT['sigma_sq'])
    LT['lims'] = [1, len(LT['mu']), lower_limit, upper_limit]
    LT['lims'][2] -= helper_co_dep * 1 / part_factor * (LT['lims'][3] - LT['lims'][2])
    maxyheight.append((LT['lims'][3] - LT['lims'][2]) / (part_factor + 1) / 2)
    ymid.append(LT['lims'][2] + maxyheight[1])

    ST['lims'] = np.zeros((4, ST['mu'].shape[1]))
    for i in range(num_periods):
        lower_limit = np.min(ST['mu'][:, i] - ceil_max_sigmalvl * ST['sigma_sq'][:, i])
        upper_limit = np.max(ST['mu'][:, i] + ceil_max_sigmalvl * ST['sigma_sq'][:, i])
        ST['lims'][:, i] = [1, len(ST['mu'][:, i]), lower_limit, upper_limit]
        ST['lims'][2, i] -= helper_co_dep * 1 / part_factor * (ST['lims'][3, i] - ST['lims'][2, i])
        maxyheight.append((ST['lims'][3, i] - ST['lims'][2, i]) / (part_factor + 1) / 2)
        ymid.append(ST['lims'][2, i] + maxyheight[-1])

    lower_limit = np.min(R['mu'] - ceil_max_sigmalvl * R['sigma_sq'])
    upper_limit = np.max(R['mu'] + ceil_max_sigmalvl * R['sigma_sq'])
    R['lims'] = [1, len(R['mu']), lower_limit, upper_limit]
    R['lims'][2] -= helper_co_dep * 1 / part_factor * (R['lims'][3] - R['lims'][2])
    maxyheight.append((R['lims'][3] - R['lims'][2]) / (part_factor + 1) / 2)
    ymid.append(R['lims'][2] + maxyheight[-1])

    x = np.linspace(0, ylen, ylen + 1)

    plt.subplot(3 + num_periods, 1, 1)
    if helper_co_dep == 1:
        j = 0
        for i in range(1, len(plot_back) + 1):
            if i - 1 == co_point and (i % ylen) > time_stamp[0] and (i % ylen) < time_stamp[1]:
                k = (i % ylen) - time_stamp[0]
                thickness = (time_stamp[1] - time_stamp[0]) / 500
                xdif = x[k] - x[k -1]
                xp = [x[k-1] - thickness * xdif, x[k-1] + thickness * xdif, x[k-1] + thickness * xdif, x[k-1] - thickness * xdif]
                yss = ymid[j]
                diff = maxyheight[j]
                yp = [yss - diff, yss - diff, yss + diff + part_factor * diff * 2, yss + diff + part_factor * diff * 2]
                plt.fill(xp, yp, color=[0, 0, 0], edgecolor='none', alpha=1)

            if (i % ylen) != 0 and (i % ylen) > time_stamp[0] and (i % ylen) <= time_stamp[1]:
                maxc = max(abs(plot_back[(j) * ylen: (j + 1) * ylen]))
                if maxc > 0:
                    yss = ymid[j]
                    diff = maxyheight[j]
                    if plot_back[i-1] > 0:
                        neg_cont = yss
                        pos_cont = yss + plot_back[i-1] / maxc * diff
                    else:
                        neg_cont = yss + plot_back[i-1] / maxc * diff
                        pos_cont = yss
                    col_ind = int(np.ceil(plot_back[i-1] / maxc * (discr_nmb // 2)))
                else:
                    yss = ymid[j]
                    neg_cont = yss
                    pos_cont = yss
                    col_ind = 0
                color = cmap(norm(col_ind))
                k = (i % ylen) - time_stamp[0]
                xdif = x[k] - x[k-1]
                xp = [x[k-1] - 1 / 2 * xdif, x[k-1] + 1 / 2 * xdif, x[k-1] + 1 / 2 * xdif, x[k-1] - 1 / 2 * xdif]
                yp = [neg_cont, neg_cont, pos_cont, pos_cont]
                plt.fill(xp, yp, color=color, edgecolor=color, alpha=1)

            elif i % ylen == 0 and ylen == time_stamp[1]:
                maxc = max(abs(plot_back[(j) * ylen: (j + 1) * ylen]))
                if maxc > 0:
                    yss = ymid[j]
                    diff = maxyheight[j]
                    if plot_back[i-1] > 0:
                        neg_cont = yss
                        pos_cont = yss + plot_back[i-1] / maxc * diff
                    else:
                        neg_cont = yss + plot_back[i-1] / maxc * diff
                        pos_cont = yss
                    col_ind = int(np.ceil(plot_back[i-1] / maxc * (discr_nmb // 2)))
                else:
                    yss = ymid[j]
                    neg_cont = yss
                    pos_cont = yss
                    col_ind = 0
                color = cmap(norm(col_ind))
                k = time_stamp[1] - time_stamp[0]
                xdif = x[k] - x[k-1]
                xp = [x[k-1] - 1 / 2 * xdif, x[k-1] + 1 / 2 * xdif, x[k-1] + 1 / 2 * xdif, x[k-1] - 1 / 2 * xdif]
                yp = [neg_cont, neg_cont, pos_cont, pos_cont]
                plt.fill(xp, yp, color=color, edgecolor=color, alpha=1)


            if i % ylen == 0 and i < len(plot_back):
                j += 1
                plt.subplot(3 + num_periods, 1, j + 1)

    plt.subplot(3 + num_periods, 1, 1)
    _plot_data(y, plot_type, n_samples, samples_colored, colorblind_safe, line_width)
    plt.title("input data")

    plt.subplot(3 + num_periods, 1, 2)
    _plot_data(LT, plot_type, n_samples, samples_colored, colorblind_safe, line_width)
    plt.title("trend component")

    for i in range(num_periods):
        plt.subplot(3 + num_periods, 1, 2 + i + 1)
        temp = {'mu': ST['mu'][:, i], 'sigma_sq': ST['sigma_sq'][:, i], 'sigma': ST['sigma'][:, :, i]}
        if plot_type in ["comb", "spaghetti"]:
            temp['samples'] = ST['samples'][:, :, i]
        _plot_data(temp, plot_type, n_samples, samples_colored, colorblind_safe, line_width)
        plt.title(f"seasonal component {i + 1}")

    plt.subplot(3 + num_periods, 1, 3 + num_periods)
    _plot_data(R, plot_type, n_samples, samples_colored, colorblind_safe, line_width)
    plt.title("residual component")

    fig = plt.gcf()
    axs = plt.gca()

    if show_plot:
        fig.tight_layout()
        plt.show()

    return fig, axs

def plot_correlation_matrix(
        timeseries,
        fig=None,
        axs=None,
        show_plot=False):
    """
    Plot correlation matrix for the timeseries data.

    Parameters
    ----------
    timeseries : Timeseries object
        An instance of the TimeSeries class, which represents a univariate time series.
    fig : matplotlib.figure.Figure or None, optional
        Figure object to use for plotting. If None, a new figure will be created.
    axs : matplotlib.axes.Axes or array of Axes or None, optional
        Axes object(s) to use for plotting. If None, new axes will be created.
    show_plot : bool, optional
        If True, display the plot.
        Default is False.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object containing the plot.
    list
        List of Axes objects used for plotting.
    """

    mu = timeseries.mean()
    sigma = timeseries.cov()
    num_periods  = 1
    line_width = 2.5
    cor_mat = _compute_correlation_matrix(sigma)

    if fig is None:
        fig = plt.figure(figsize=(8, 8))

    plt.suptitle('Correlation Matrix')
    plt.imshow(cor_mat, cmap='coolwarm')
    plt.colorbar()

    sublength = len(mu) // (num_periods + 3)

    for i in range(1, num_periods + 3):
        plt.plot([(i - 1) * sublength + 0.5 + sublength] * len(mu), ':', color='white', linewidth=line_width)
        plt.plot([(i - 1) * sublength + sublength + 0.5] * len(mu), np.linspace(0, len(mu) - 1, len(mu)), ':', color='white', linewidth=line_width)

    plt.clim(-1, 1)

    fig = plt.gcf()
    axs = plt.gca()

    if show_plot:
        fig.tight_layout()
        plt.show()

    return fig, axs

def plot_corr_length(
        timeseries,
        fig=None,
        axs=None,
        show_plot=False):
    """
    Plot correlation length for single uncertain timeseries data.

    Parameters
    ----------
    timeseries : Timeseries object
        An instance of the TimeSeries class, which represents a univariate time series.
    fig : matplotlib.figure.Figure or None, optional
        Figure object to use for plotting. If None, a new figure will be created.
    axs : matplotlib.axes.Axes or array of Axes or None, optional
        Axes object(s) to use for plotting. If None, new axes will be created.
    show_plot : bool, optional
        If True, display the plot.
        Default is False.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object containing the plot.
    list
        List of Axes objects used for plotting.
    """
    multi_distr_present = False
    fig, axs = _plot_correlation_length_data(timeseries, multi_distr_present, fig, axs, show_plot)

    return fig, axs

def plot_correlated_corr_length(
        timeseries,
        fig=None,
        axs=None,
        show_plot=False):
    """
    Plot correlation length for correlated uncertain timeseries data.

    Parameters
    ----------
    timeseries : Timeseries object
        An instance of the TimeSeries class, which represents a univariate time series.
    fig : matplotlib.figure.Figure or None, optional
        Figure object to use for plotting. If None, a new figure will be created.
    axs : matplotlib.axes.Axes or array of Axes or None, optional
        Axes object(s) to use for plotting. If None, new axes will be created.
    show_plot : bool, optional
        If True, display the plot.
        Default is False.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object containing the plot.
    list
        List of Axes objects used for plotting.
    """
    multi_distr_present = True
    fig, axs = _plot_correlation_length_data(timeseries, multi_distr_present, fig, axs, show_plot)

    return fig, axs