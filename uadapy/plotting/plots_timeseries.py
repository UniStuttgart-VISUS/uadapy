import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import colormaps
from matplotlib.colors import Normalize
from uadapy import CorrelatedDistributions
import glasbey as gb
from uadapy.temporal.spectralAnalysis import compute_percentiles_complex


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

def _reconstruct_covariance_matrix(block_cov):

    block_rows = [np.concatenate(row, axis=1) for row in block_cov]
    cov_mat = np.concatenate(block_rows, axis=0)
    return cov_mat

def _plot_data(data, plot_type, n_samples, axs, samples_colored, colorblind_safe, line_width):

    smpl_width = 0.5

    cmap_gradient = plt.get_cmap('Blues')
    color_indices = np.linspace(0, 155, 5).astype(int)
    col = cmap_gradient(np.flip(color_indices) / 256)[:, :3]

    if 'x' in data:
        x = data['x']
    elif 'central' in data:
        x = np.arange(len(data['central']))
    elif 'samples' in data:
        x = np.arange(len(data['samples']))
    elif 'mu' in data:
        x = np.arange(len(data['mu']))
    else:
        x = np.arange(100)

    if plot_type == "isoband":
        if 'light_band' in data:
            axs.fill_between(x, data['light_band'][0], data['light_band'][1], color=col[2], alpha=0.5)
            axs.fill_between(x, data['dark_band'][0], data['dark_band'][1], color=col[1], alpha=0.5)
            axs.plot(x, data['central'], color=col[0], linewidth=line_width)
        else:
            sigmalvl = [0, 0.674490, 2.575829]
            #x = np.arange(len(data['mu']))
            for j in range(len(sigmalvl) - 1, 0, -1):
                pos_cont = data['mu'] + sigmalvl[j] * data['sigma_sq']
                neg_cont = data['mu'] - sigmalvl[j] * data['sigma_sq']
                for i in range(len(data['mu']) - 1):
                    xp = [x[i], x[i + 1], x[i + 1], x[i]]
                    yp = [neg_cont[i], neg_cont[i + 1], pos_cont[i + 1], pos_cont[i]]
                    axs.fill(xp, yp, color=col[j], edgecolor='none')
            axs.plot(x, data['mu'], color=col[0], linewidth=line_width)
    
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
                h2a = axs.plot(samples_2_plot, color=colors[1], linewidth=line_width * smpl_width)
                h2a_shift = axs.plot(samples_2_plot_shift, color=colors[1], linewidth=line_width * smpl_width * 0.2, linestyle='-')
                h2a_shift_dot = axs.plot(samples_2_plot_shift, color=colors[1], linewidth=line_width * smpl_width, linestyle=':')
            else:
                for i in range(n_samples):
                    axs.plot(x, samples_2_plot[:, i], linewidth=line_width * smpl_width, color=colors[i])
                    axs.plot(x, samples_2_plot_shift[:, i], linewidth=line_width * smpl_width * 0.2, color=colors[i], linestyle='-')
                    axs.plot(x, samples_2_plot_shift[:, i], linewidth=line_width * smpl_width, color=colors[i], linestyle=':')
            if not samples_colored and n_samples > 1:
                for h in h2a:
                    h.set_alpha(0.5)
                for h in h2a_shift:
                    h.set_alpha(0.5)
                for h in h2a_shift_dot:
                    h.set_alpha(0.5)
        else:
            if not samples_colored:
                h2a = axs.plot(x, samples_2_plot, color=colors[1], linewidth=line_width * smpl_width)
            else:
                for i in range(n_samples):
                    axs.plot(x, samples_2_plot[:, i], linewidth=line_width * smpl_width, color=colors[i])
            if not samples_colored and n_samples > 1:
                for h in h2a:
                    h.set_alpha(0.5)

    elif plot_type == "comb":
        axs = _plot_data(data, "isoband", n_samples, axs, samples_colored, colorblind_safe, line_width)
        axs = _plot_data(data, "spaghetti", n_samples, axs, samples_colored, colorblind_safe, line_width)

    return axs

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
    timeseries : Timeseries object or CorrelatedDistributions object
        An instance of the TimeSeries class or CorrelatedDistributions class.
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

    discr_nmb = 13

    if multi_distr_present:
        cov_mat = timeseries.covariance_matrix
        sb_plot_nmb = timeseries.n_distributions
        length_data = len(timeseries.distributions[0].mean())
        cov_mat = _reconstruct_covariance_matrix(cov_mat)
        cor_mat = _compute_correlation_matrix(cov_mat)
    else:
        mu = timeseries.mean()
        sigma = timeseries.cov()
        sb_plot_nmb = 1
        length_data = len(mu)
        cor_mat = _compute_correlation_matrix(sigma)
    time_stamp = [0, length_data]
    stamp_length = time_stamp[1] - time_stamp[0]
    stamp_indices = list(range(length_data * sb_plot_nmb))

    if axs is None:
        if fig is None:
            if multi_distr_present:
                fig, axs = plt.subplots(sb_plot_nmb, 1, figsize=(12, 8))
            else:
                fig, axs = plt.subplots(figsize=(15, 3))
        else:
            if fig.axes is not None:
                if multi_distr_present:
                    if len(fig.axes) != sb_plot_nmb:
                        raise ValueError(f"Expected {sb_plot_nmb} axes in provided figure, got {len(fig.axes)}.")
                    axs = fig.axes
                else:
                    axs = fig.axes[0]
            else:
                raise ValueError("The provided figure has no axes. Pass an Axes or create subplots first.")
    else:
        if fig is None:
            fig = axs[0].figure if isinstance(axs, np.ndarray) else axs.figure

    axs = np.atleast_1d(axs)
    if axs.size != sb_plot_nmb:
        raise ValueError(f"Expected {sb_plot_nmb} axes in provided figure, got {axs.size}.")

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
    cmap_full = colormaps.get_cmap('coolwarm').resampled(256)
    colors_upper_half = cmap_full(np.linspace(0.5, 1, nmb_colors - 1))
    colors = np.vstack(([1, 1, 1, 1], colors_upper_half))

    # Create a custom discrete colormap
    cmap = mcolors.ListedColormap(colors)

    for i in range(sb_plot_nmb):
        max_i = max(ysize[stamp_indices[(i * stamp_length):((i + 1) * stamp_length)]])
        max_i = max_i.astype(int)
        image2plot = plot_mat[-max_i:, stamp_indices[(i * stamp_length):((i + 1) * stamp_length)]]
        axs[i].imshow(image2plot, cmap=cmap, aspect='auto')
        vert_ls = np.repeat(vert_l[stamp_indices[(i * stamp_length):((i + 1) * stamp_length)]], 2)
        xs = np.repeat(np.linspace(0, stamp_length, stamp_length + 1), 2)
        xs = xs[1:-1]
        axs[i].plot(xs - 0.5, max_i - vert_ls, linewidth=2, color='black', linestyle='-')
        if i == 0:
            axs[i].set_title("input data")
        elif i == 1:
            axs[i].set_title("trend component")
        elif i == sb_plot_nmb - 1:
            axs[i].set_title("residual component")
        else:
            axs[i].set_title(f"seasonal component {i - 1}")
    axs[-1].set_xlabel('timesteps')
    fig.suptitle('Correlation Length')

    if show_plot:
        fig.tight_layout()
        plt.show()

    return fig, axs

def plot_timeseries(
        timeseries,
        n_samples,
        percentiles = False,
        seed=55,
        fig=None,
        axs=None,
        colorblind_safe=False,
        show_plot=False,
        x_label='timesteps',
        y_label='value'):
    """
    Plot single uncertain timeseries data.

    Parameters
    ----------
    timeseries : Timeseries object
        An instance of the TimeSeries class, which represents a univariate time series.
    n_samples : int
        The number of samples to draw from the given timeseries distribution.
    percentiles : bool, optional
        If True, plot the 50th, and 95th percentiles of the distribution.
    seed : int
        Seed for the random number generator for reproducibility. It defaults to 55 if not provided.
    fig : matplotlib.figure.Figure or None, optional
        Figure object to use for plotting. If None, a new figure will be created.
    axs : matplotlib.axes.Axes or None, optional
        Axes object to use for plotting. If None, new axes will be created.
    colorblind_safe : bool, optional
        If True, the plot will use colors suitable for colorblind individuals.
        Default is False.
    show_plot : bool, optional
        If True, display the plot.
        Default is False.
    x_label : str, optional
        Label for the x-axis. Default is 'timesteps'.
    y_label : str, optional
        Label for the y-axis. Default is 'value'.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object containing the plot.
    list
        List of Axes objects used for plotting.
    """

    mu = timeseries.mean()
    sigma = timeseries.cov()

    if axs is None:
        if fig is None:
            fig, axs = plt.subplots(figsize=(15, 3))
        else:
            if fig.axes is not None:
                axs = fig.axes[0]
            else:
                raise ValueError("The provided figure has no axes. Pass an Axes or create subplots first.")
    else:
        if fig is None:
            fig = axs.figure

    plot_type = 'comb'
    samples_colored = True
    line_width = 2.5

    samples = timeseries.sample(n_samples, seed).transpose()
    y = {'samples': samples}
    y['x'] = timeseries.timesteps
    if not percentiles or timeseries.distribution == 'Normal':
        y['central'] = mu
        y['dark_band'] = [mu - 0.674490*np.sqrt(np.maximum(np.diag(sigma), 0)), mu + 0.674490*np.sqrt(np.maximum(np.diag(sigma), 0))]#np.sqrt(np.maximum(np.diag(sigma), 0))
        y['light_band'] = [mu - 2.575829*np.sqrt(np.maximum(np.diag(sigma), 0)), mu + 2.575829*np.sqrt(np.maximum(np.diag(sigma), 0))]#sigma
    else:
        p2_5, p25, median, p75, p97_5 = compute_percentiles_complex(timeseries, [0.025, 0.25, 0.5, 0.75, 0.975])
        y['light_band'] = [p2_5, p97_5]
        y['dark_band'] = [p25, p75]
        y['central'] = median

    axs = _plot_data(y, plot_type, n_samples, axs, samples_colored, colorblind_safe, line_width)
    axs.set_xlabel(x_label)
    axs.set_ylabel(y_label)
    fig.suptitle('Timeseries')

    if show_plot:
        fig.tight_layout()
        plt.show()

    return fig, axs

def plot_correlated_timeseries(
        corr_timeseries  : CorrelatedDistributions,
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
    corr_timeseries : CorrelatedDistributions object
        An instance of the CorrelatedDistributions class.
    n_samples : int
        The number of samples to draw from the given timeseries distribution.
    co_point : int
        Interactive point for correlation exploration.
    seed : int
        Seed for the random number generator for reproducibility. It defaults to 55 if not provided.
    fig : matplotlib.figure.Figure or None, optional
        Figure object to use for plotting. If None, a new figure will be created.
    axs : Array of matplotlib.axes.Axes or None, optional
        Axes objects to use for plotting. If None, new axes will be created.
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

    discr_nmb = 13
    samples_colored=True
    line_width = 2.5
    plot_type = 'comb'
    samples = {}
    mu = {}
    sigma_sq = {}
    sigma = {}
    n_distributions = corr_timeseries.n_distributions
    num_periods = n_distributions - 3
    ts_means = [x.mean() for x in corr_timeseries.distributions]
    cov_mat = corr_timeseries.covariance_matrix

    if axs is None:
        if fig is None:
            fig, axs = plt.subplots(n_distributions, 1, figsize=(12, 8))
        else:
            if fig.axes is not None:
                if len(fig.axes) != n_distributions:
                    raise ValueError(f"Expected {n_distributions} axes in provided figure, got {len(fig.axes)}.")
                axs = fig.axes
            else:
                raise ValueError("The provided figure has no axes. Pass an Axes or create subplots first.")
    else:
        if fig is None:
            fig = axs[0].figure if isinstance(axs, np.ndarray) else axs.figure

    axs = np.atleast_1d(axs)
    if axs.size != n_distributions:
        raise ValueError(f"Expected {n_distributions} axes in provided figure, got {axs.size}.")

    timesteps = len(ts_means[0])
    time_stamp = [0, timesteps]
    stamp_indices = list(range(len(ts_means[0]) * corr_timeseries.n_distributions))

    samples = corr_timeseries.sample(n_samples, seed).T
    samples = np.split(samples, corr_timeseries.n_distributions, axis=0)

    for i in range(corr_timeseries.n_distributions):
        mu[i] = ts_means[i]
        sigma_sq[i] = np.sqrt(np.maximum(np.diag(cov_mat[i][i]), 0))
        sigma[i] = cov_mat[i][i]

    y = {'samples': samples[0], 'mu': mu[0], 'sigma_sq': sigma_sq[0], 'sigma': sigma[0]}
    LT = {'samples': samples[1], 'mu': mu[1], 'sigma_sq': sigma_sq[1], 'sigma': sigma[1]}

    ST = {
        'samples': np.zeros((time_stamp[1] - time_stamp[0], n_samples, num_periods)),
        'mu': np.zeros((time_stamp[1] - time_stamp[0], num_periods)),
        'sigma_sq': np.zeros((time_stamp[1] - time_stamp[0], num_periods)),
        'sigma': np.zeros((time_stamp[1] - time_stamp[0], time_stamp[1] - time_stamp[0], num_periods)),
    }

    for i in range(num_periods):
        ST['samples'][:, :, i] = samples[2 + i]
        ST['mu'][:, i] = mu[2 + i]
        ST['sigma_sq'][:, i] = sigma_sq[2 + i]
        ST['sigma'][:, :, i] = sigma[2 + i]

    R = {'samples': samples[corr_timeseries.n_distributions - 1],
         'mu': mu[corr_timeseries.n_distributions - 1],
         'sigma_sq': sigma_sq[corr_timeseries.n_distributions - 1],
         'sigma': sigma[corr_timeseries.n_distributions - 1]}

    cmap = colormaps.get_cmap('coolwarm').resampled(discr_nmb)
    norm = Normalize(vmin=-(discr_nmb // 2), vmax=(discr_nmb // 2))

    if co_point == 0:
        helper_co_dep = 0
    elif abs(co_point) > len(ts_means[0]) * corr_timeseries.n_distributions:
        print('Covariance point set too high, plotting without dependency plot.\n')
        helper_co_dep = 0
    elif co_point in stamp_indices:
        helper_co_dep = 1
    else:
        helper_co_dep = 1
        print('Covariance point is out of time_stamp-Range.\n')

    if helper_co_dep == 1:
        interactive_pointer = co_point
        cov_mat = _reconstruct_covariance_matrix(cov_mat)
        cor_mat = _compute_correlation_matrix(cov_mat)
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

    x = np.linspace(0, timesteps, timesteps + 1)

    if helper_co_dep == 1:
        j = 0
        for i in range(1, len(plot_back) + 1):
            if i - 1 == co_point and (i % timesteps) > time_stamp[0] and (i % timesteps) < time_stamp[1]:
                k = (i % timesteps) - time_stamp[0]
                thickness = (time_stamp[1] - time_stamp[0]) / 500
                xdif = x[k] - x[k -1]
                xp = [x[k-1] - thickness * xdif, x[k-1] + thickness * xdif, x[k-1] + thickness * xdif, x[k-1] - thickness * xdif]
                yss = ymid[j]
                diff = maxyheight[j]
                yp = [yss - diff, yss - diff, yss + diff + part_factor * diff * 2, yss + diff + part_factor * diff * 2]
                axs[j].fill(xp, yp, color=[0, 0, 0], edgecolor='none', alpha=1)

            if (i % timesteps) != 0 and (i % timesteps) > time_stamp[0] and (i % timesteps) <= time_stamp[1]:
                maxc = max(abs(plot_back[(j) * timesteps: (j + 1) * timesteps]))
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
                k = (i % timesteps) - time_stamp[0]
                xdif = x[k] - x[k-1]
                xp = [x[k-1] - 1 / 2 * xdif, x[k-1] + 1 / 2 * xdif, x[k-1] + 1 / 2 * xdif, x[k-1] - 1 / 2 * xdif]
                yp = [neg_cont, neg_cont, pos_cont, pos_cont]
                axs[j].fill(xp, yp, color=color, edgecolor=color, alpha=1)

            elif i % timesteps == 0 and timesteps == time_stamp[1]:
                maxc = max(abs(plot_back[(j) * timesteps: (j + 1) * timesteps]))
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
                axs[j].fill(xp, yp, color=color, edgecolor=color, alpha=1)


            if i % timesteps == 0 and i < len(plot_back):
                j += 1

    axs[0] = _plot_data(y, plot_type, n_samples, axs[0], samples_colored, colorblind_safe, line_width)
    axs[0].set_title("input data")

    axs[1] = _plot_data(LT, plot_type, n_samples, axs[1], samples_colored, colorblind_safe, line_width)
    axs[1].set_title("trend component")

    for i in range(num_periods):
        temp = {'mu': ST['mu'][:, i], 'sigma_sq': ST['sigma_sq'][:, i], 'sigma': ST['sigma'][:, :, i]}
        if plot_type in ["comb", "spaghetti"]:
            temp['samples'] = ST['samples'][:, :, i]
        axs[2 + i] = _plot_data(temp, plot_type, n_samples, axs[2 + i], samples_colored, colorblind_safe, line_width)
        axs[2 + i].set_title(f"seasonal component {i + 1}")

    axs[n_distributions - 1] = _plot_data(R, plot_type, n_samples, axs[n_distributions - 1], samples_colored, colorblind_safe, line_width)
    axs[n_distributions - 1].set_title("residual component")

    fig.suptitle('Uncertainty-Aware Seasonal-Trend Decomposition')
    if show_plot:
        fig.tight_layout()
        plt.show()

    return fig, axs

def plot_correlation_matrix(
        corr_timeseries : CorrelatedDistributions,
        fig=None,
        axs=None,
        discretize=True,
        show_plot=False):
    """
    Plot correlation matrix for the timeseries data.

    Parameters
    ----------
    corr_timeseries : CorrelatedDistributions object
        An instance of the CorrelatedDistributions class.
    fig : matplotlib.figure.Figure or None, optional
        Figure object to use for plotting. If None, a new figure will be created.
    axs : matplotlib.axes.Axes or None, optional
        Axes object to use for plotting. If None, new axes will be created.
    discretize : bool, optional
        If True, discretize the colormap. Default is True.
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

    line_width = 2.5
    discr_nmb = 13
    ts_length = len(corr_timeseries.distributions[0].mean())
    total_len = ts_length * corr_timeseries.n_distributions
    cov_mat = corr_timeseries.covariance_matrix
    cov_mat = _reconstruct_covariance_matrix(cov_mat)
    cor_mat = _compute_correlation_matrix(cov_mat)

    if axs is None:
        if fig is None:
            fig, axs = plt.subplots(figsize=(8, 8))
        else:
            if fig.axes is not None:
                axs = fig.axes[0]
            else:
                raise ValueError("The provided figure has no axes. Pass an Axes or create subplots first.")
    else:
        if fig is None:
            fig = axs.figure

    if discretize:
        cmap = mcolors.ListedColormap(colormaps.get_cmap('coolwarm')(np.linspace(0, 1, discr_nmb)))
        norm = mcolors.BoundaryNorm(np.linspace(-1, 1, discr_nmb + 1), cmap.N)
        im = axs.imshow(cor_mat, cmap=cmap, norm=norm)
    else:
        cmap = 'coolwarm'
        im = axs.imshow(cor_mat, cmap=cmap)

    fig.colorbar(im, ax=axs)
    fig.suptitle('Correlation Matrix')

    for i in range(1, corr_timeseries.n_distributions):
        axs.plot([(i - 1) * ts_length + 0.5 + ts_length] * total_len, ':', color='white', linewidth=line_width)
        axs.plot([(i - 1) * ts_length + ts_length + 0.5] * total_len, np.linspace(0, total_len - 1, total_len), ':', color='white', linewidth=line_width)

    im.set_clim(-1, 1)

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
    axs : matplotlib.axes.Axes or None, optional
        Axes object to use for plotting. If None, new axes will be created.
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
        corr_timeseries,
        fig=None,
        axs=None,
        show_plot=False):
    """
    Plot correlation length for correlated uncertain timeseries data.

    Parameters
    ----------
    corr_timeseries : CorrelatedDistributions object
        An instance of the CorrelatedDistributions class.
    fig : matplotlib.figure.Figure or None, optional
        Figure object to use for plotting. If None, a new figure will be created.
    axs : Array of matplotlib.axes.Axes or None, optional
        Axes objects to use for plotting. If None, new axes will be created.
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
    fig, axs = _plot_correlation_length_data(corr_timeseries, multi_distr_present, fig, axs, show_plot)

    return fig, axs