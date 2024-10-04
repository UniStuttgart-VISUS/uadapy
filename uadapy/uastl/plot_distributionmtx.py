from plot_dist import plot_dist
from diverging_map import diverging_map
from uastl import uncertain_data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def plot_distributionmtx(y_ltstr : uncertain_data, num_periods, plot_type, **opts):
    """
    This function PLOT_DISTRIBUTIONMTX creates the main visualization plot
    of the distribution for given mean and covariance matrix.
    It is highly dependent on the function plot_dist.

    Plot distribution for time series depending on the Gaussian distributed 
    variable y_ltstr = [y, LT, ST_1, ..., ST_L, R] with
        * y_ltstr.mu: mean vector 
        * y_ltstr.sigma: covariance matrix

    The theory is described in the related manuscript "Uncertainty-Aware
    Seasonal-Trend Decomposition Based on Loess".

    y_ltstr = PLOT_DISTRIBUTIONMTX(y_ltstr, num_periods, type, opts)

    Input:
        * y_ltstr: struct with mean and sigma containing the data y, trend LT, 
                  seasonal ST, and residual R components computed by UASTL
            - y_ltstr.mu: mean vector with components
            - y_ltstr.sigma: covariance matrix with components
        * num_periods: Number of periods to subdivide the y_ltstr cell data
        * type: one of "comb", "isoband", or "spaghetti"
                "comb" plots isobands and spaghetti samples
        * opts: optional input with additional VIS techniques
            - time_stamp (:,1) double = []
            - nmbsamples (1,1) {mustBeInteger} = 5
            - samples_colored (1,1) logical = False
            - line_width {mustBeNumeric} = 1
            - plot_cov (1,1) logical = False
            - plot_cor (1,1) logical = False
            - plot_cor_length (1,:) struct = struct
            - co_point (1,1) {mustBeInteger} = 0
            - delta (:,1) double = []
            - export

    Output: builds Figure(s) containing the desired VIS techniques for UASTL
        * y_ltstr: Same y_ltstr but with added field CorMat if correlation matrix
                  is computed (if co_point or plot_cor options are set)
    """

    print("############################### Visualization ###################################")
    print("##### Uncertainty-Aware Seasonal-Trend Decomposition Based on Loess (UASTL) #####")
    print("#################################################################################")
    print("#####")
    print(f"##### Number of Periods:\t num_periods = {num_periods}")
    print("#####")
    print(f"##### Plotting Type:\t \t type = {plot_type}")
    print(f"##### \t ├── Number of Spaghetthi: \t nmbsamples = {opts['nmbsamples']}")
    print(f"##### \t ├── Spaghetthi colored:\t samples_colored = {opts['samples_colored']}")
    print(f"##### \t ├── Set LineWidth Factor:\t line_width = {opts['line_width']}")
    print(f"##### \t └── Discretization Steps:\t discr_nmb = {opts['discr_nmb']}")
    print("#####")
    time_stamp = opts.get('time_stamp')
    delta = opts.get('delta')
    if time_stamp:
        print(f"##### Plotting Interval:\t time_stamp = [{opts['time_stamp'][0]},{opts['time_stamp'][1]}]")
    else:
        print("##### Plotting Interval:\t Full Time Domain.")
    print("#####")
    print("##### Visualization Techniques:")
    print(f"##### \t ├── Covariance Matrix:\t\t plot_cov = {opts['plot_cov']}")
    print(f"##### \t ├── Correlation Matrix:\t plot_cor = {opts['plot_cor']}")
    print(f"##### \t ├── Correlation Plot:\t \t plot_cor_length = {opts['plot_cor_length']}")
    if opts['co_point'] != 0:
        print(f"##### \t └── Local Corr. Details:\t co_point = {opts['co_point']}")
    else:
        print(f"##### \t └── No Local Corr. Details:\t co_point = {opts['co_point']}")
    print("#####")
    if 'export' in opts and opts['export']['export'] and 'export_path' in opts['export']:
        print(f"##### Subplot Export into Folder: \t export.export_path = {opts['export']['export_path']}")
    elif 'export' in opts and opts['export']['export']:
        print("##### Subplots Export into Current Folder")
    else:
        print("##### Subplots not exported: \t\t export.export = False")
    print("#####")
    print("#################################################################################")
    print("#####")
    print("##### Plotting START")
    print("#####")

    # checks and setup of (optional) variables
    # y_ltstr.mu & y_ltstr.sigma
    if (len(y_ltstr.mu) == y_ltstr.sigma.shape[0]) and (len(y_ltstr.mu) == y_ltstr.sigma.shape[1]):
        ylen = len(y_ltstr.mu) // (3 + num_periods)
    else:
        raise ValueError("Size of mu and sigma are not concise")

    # export
    if 'export' in opts and opts['export']:
        required_export_fields = ["export_path", "export_name", "export", "pbaspect", "yaxis", "dashed_lines", "format"]
        for field in required_export_fields:
            if field not in opts['export']:
                opts['export'][field] = ""
        if 'export_name' in opts['export']:
            export_name = opts['export']['export_name']
        else:
            export_name = "plot_dist"
    else:
        opts['export'] = {'export': False, 'format': "FIG"}
        export_name = ""

    # discr_nmb
    if opts['discr_nmb'] > 257:
        opts['discr_nmb'] = 257
        print('"nmb_colors" set too large, setting to 257')
    elif opts['discr_nmb'] % 2 == 0:
        opts['discr_nmb'] += 1
        print('"nmb_colors" set to the next higher odd number.')

    if time_stamp :
        if opts['time_stamp'][1] <= opts['time_stamp'][0]:
            print('time_stamp not correctly assigned. Ignoring input.')
            opts['time_stamp'] = []

    if not time_stamp:
        tme_stmp = [0, ylen]
        opts['stmp_length'] = ylen
        opts['stamp_indices'] = list(range(len(y_ltstr.mu)))
        opts['time_stamp'] = [1, ylen]
    else:
        tme_stmp = opts['time_stamp']
        opts['stmp_length'] = tme_stmp[1] - tme_stmp[0]
        stamp_indices = list(range(opts['time_stamp'][0], opts['time_stamp'][1] + 1))
        stamp_indices += list(range(ylen + opts['time_stamp'][0], ylen + opts['time_stamp'][1] + 1))
        for i in range(num_periods):
            stamp_indices += list(range((i + 1) * ylen + opts['time_stamp'][0], (i + 1) * ylen + opts['time_stamp'][1] + 1))
        opts['stamp_indices'] = stamp_indices + list(range((num_periods + 2) * ylen + opts['time_stamp'][0], (num_periods + 2) * ylen + opts['time_stamp'][1] + 1))

    # Picking global samples if comb or spaghetti plot
    if plot_type in ["comb", "spaghetti"]:
        all_zeros = np.all(y_ltstr.a_hat == 0)
        if (delta is not None) and (not all_zeros):
            # SHIFTED samples via delta vector
            sigma = y_ltstr.sigma[:ylen, :ylen]
            epsilon = 1e-10  # Small value to add
            sigma = sigma + epsilon * np.eye(sigma.shape[0]) #prevent positive definite matrix error
            T = np.linalg.cholesky(sigma)# returns lower traingular matrix by default #.T
            samp_orig = np.random.randn(opts['nmbsamples'], T.shape[1]).T
            a_hat_e_hat = y_ltstr.a_hat
            t_samp_orig = T @ samp_orig
            samp = a_hat_e_hat @ t_samp_orig + y_ltstr.mu
            samp_new = samp + a_hat_e_hat @ np.diag(opts['delta'] - 1) @ t_samp_orig
            y_ltstr.samples = np.hstack((samp, samp_new))
            sigma_new = np.sqrt(np.sum((a_hat_e_hat @ np.diag(opts['delta']) @ T)**2, axis=1))

            y = {'samples': y_ltstr.samples[tme_stmp[0]:tme_stmp[1], :],
                 'sigma_new': sigma_new[tme_stmp[0]:tme_stmp[1]]}
            LT = {'samples': y_ltstr.samples[ylen + tme_stmp[0]:ylen + tme_stmp[1], :],
                  'sigma_new': sigma_new[ylen + tme_stmp[0]:ylen + tme_stmp[1]]}

            ST = {'samples': np.zeros((tme_stmp[1] - tme_stmp[0], opts['nmbsamples'] * 2 , num_periods)), #y_ltstr.samples.shape so * 2
                  'sigma_new': np.zeros((tme_stmp[1] - tme_stmp[0], num_periods))}
            for i in range(num_periods):
                ST['samples'][:, :, i] = y_ltstr.samples[(i + 2) * ylen + tme_stmp[0]:(i + 2) * ylen + tme_stmp[1], :]
                ST['sigma_new'][:, i] = sigma_new[(i + 2) * ylen + tme_stmp[0]:(i + 2) * ylen + tme_stmp[1]]

            R = {'samples': y_ltstr.samples[(2 + num_periods) * ylen + tme_stmp[0]:(2 + num_periods) * ylen + tme_stmp[1], :],
                 'sigma_new': sigma_new[(2 + num_periods) * ylen + tme_stmp[0]:(2 + num_periods) * ylen + tme_stmp[1]]}
        else:
            if delta:
                print('No y_ltstr.a_hat given, therefore going on without delta.')
            y_ltstr.samples = np.random.multivariate_normal(y_ltstr.mu.flatten(), y_ltstr.sigma, opts['nmbsamples']).T
            y = {'samples': y_ltstr.samples[tme_stmp[0]:tme_stmp[1], :]}
            LT = {'samples': y_ltstr.samples[ylen + tme_stmp[0]:ylen + tme_stmp[1], :]}
            ST = {'samples': np.zeros((tme_stmp[1] - tme_stmp[0], opts['nmbsamples'], num_periods))}
            for i in range(num_periods):
                ST['samples'][:, :, i] = y_ltstr.samples[(i + 2) * ylen + tme_stmp[0]:(i + 2) * ylen + tme_stmp[1], :]
            R = {'samples': y_ltstr.samples[(2 + num_periods) * ylen + tme_stmp[0]:(2 + num_periods) * ylen + tme_stmp[1], :]}

    y['mu'] = y_ltstr.mu[tme_stmp[0]:tme_stmp[1]]
    y['sigma_sq'] = np.sqrt(np.maximum(np.diag(y_ltstr.sigma[tme_stmp[0]:tme_stmp[1], tme_stmp[0]:tme_stmp[1]]), 0))
    y['sigma'] = y_ltstr.sigma[tme_stmp[0]:tme_stmp[1], tme_stmp[0]:tme_stmp[1]]

    LT['mu'] = y_ltstr.mu[ylen + tme_stmp[0]:ylen + tme_stmp[1]]
    LT['sigma_sq'] = np.sqrt(np.maximum(np.diag(y_ltstr.sigma[ylen + tme_stmp[0]:ylen + tme_stmp[1], ylen + tme_stmp[0]:ylen + tme_stmp[1]]), 0))
    LT['sigma'] = y_ltstr.sigma[ylen + tme_stmp[0]:ylen + tme_stmp[1], ylen + tme_stmp[0]:ylen + tme_stmp[1]]

    ST['mu'] = np.zeros((tme_stmp[1] - tme_stmp[0], num_periods))
    ST['sigma_sq'] =  np.zeros((tme_stmp[1] - tme_stmp[0], num_periods))
    ST['sigma'] = np.zeros((tme_stmp[1] - tme_stmp[0], tme_stmp[1] - tme_stmp[0], num_periods))

    for i in range(num_periods):
        ST['mu'][:, i] = y_ltstr.mu[(i + 2) * ylen + tme_stmp[0]:(i + 2) * ylen + tme_stmp[1]].flatten()
        ST['sigma_sq'][:, i] = np.sqrt(np.maximum(np.diag(y_ltstr.sigma[(i + 2) * ylen + tme_stmp[0]:(i + 2) * ylen + tme_stmp[1], (i + 2) * ylen + tme_stmp[0]:(i + 2) * ylen + tme_stmp[1]]), 0))
        ST['sigma'][:, :, i] = y_ltstr.sigma[(i + 2) * ylen + tme_stmp[0]:(i + 2) * ylen + tme_stmp[1], (i + 2) * ylen + tme_stmp[0]:(i + 2) * ylen + tme_stmp[1]]

    R['mu'] = y_ltstr.mu[(2 + num_periods) * ylen + tme_stmp[0]:(2 + num_periods) * ylen + tme_stmp[1]]
    R['sigma_sq'] =  np.sqrt(np.maximum(np.diag(y_ltstr.sigma[(2 + num_periods) * ylen + tme_stmp[0]:(2 + num_periods) * ylen + tme_stmp[1], (2 + num_periods) * ylen + tme_stmp[0]:(2 + num_periods) * ylen + tme_stmp[1]]), 0))
    R['sigma'] = y_ltstr.sigma[(2 + num_periods) * ylen + tme_stmp[0]:(2 + num_periods) * ylen + tme_stmp[1], (2 + num_periods) * ylen + tme_stmp[0]:(2 + num_periods) * ylen + tme_stmp[1]]

    # SET FIGURE
    plt.figure(figsize=(12, 8))
    plt.get_current_fig_manager().resize(1710, 1112)
    plt.suptitle('Uncertainty-Aware Seasonal-Trend Decomposition')

    nmb_colors = opts['discr_nmb']
    colors = color_lut(1)
    colors = colors[np.round(np.linspace(0, 256, nmb_colors)).astype(int), :]

    if opts['co_point'] == 0:
        helper_co_dep = 0
    elif abs(opts['co_point']) > len(y_ltstr.mu):
        print('Covariance point set too high, plotting without dependency plot.\n')
        helper_co_dep = 0
    elif opts['co_point'] in opts['stamp_indices']:
        helper_co_dep = 1
    else:
        helper_co_dep = 1
        print('Covariance point is out of time_stamp-Range.\n')

    if helper_co_dep == 1:
        interactive_pointer = opts['co_point']
        # We choose CorMat for background plot.
        y_ltstr = compute_cor_mat(y_ltstr)
        plot_back = y_ltstr.cor_mat[interactive_pointer, :]

    # Get the ylims for each part by (mean+sigmalvlmax)/3*4 to add below
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

    ST['lims'] = np.zeros((4, ST['mu'].shape[1])) #added for keyerror
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

    x = np.linspace(1, ylen, ylen)

    # Approx. 16:9 pbaspect of plot:
    pbx_length = (3 + num_periods + 1) / 9 * 16
    pby_length = 1

    export_opts = opts['export']
    if 'pbaspect' in export_opts:
        pbx_length, pby_length, _ = export_opts['pbaspect']

    plt.subplot(3 + num_periods, 1, 1)
    if helper_co_dep == 1:
        j = 0
        for i in range(len(plot_back) - 1):
            if i == opts['co_point'] - 1:  # Adjusting for 0-indexing in Python
                k = (i % ylen) - tme_stmp[0] - 1
                thickness = (tme_stmp[1] - tme_stmp[0] - 1) / 500
                xdif = x[k + 1] - x[k]
                xp = [x[k] - thickness * xdif, x[k] + thickness * xdif, x[k] + thickness * xdif, x[k] - thickness * xdif]
                yss = ymid[j]
                if 'yaxis' in opts['export'] :
                    if opts['export']['yaxis'] != "":
                        if opts['export']['yaxis'][1][1] != float('inf'):
                            diff = 5 * maxyheight[j]
                else:
                    diff = maxyheight[j]
                yp = [yss - diff, yss - diff, yss + diff + part_factor * diff * 2, yss + diff + part_factor * diff * 2]
                plt.fill(xp, yp, color=[0, 0, 0], edgecolor='none', alpha=1)

            maxc = max(abs(plot_back[(j) * ylen: (j + 1) * ylen]))
            if maxc > 0:
                yss = ymid[j]
                diff = maxyheight[j]
                if plot_back[i] > 0:
                    neg_cont = yss
                    pos_cont = yss + plot_back[i] / maxc * diff
                else:
                    neg_cont = yss + plot_back[i] / maxc * diff
                    pos_cont = yss
                col_ind = int(np.ceil(plot_back[i] / maxc * (opts['discr_nmb'] // 2))) + (opts['discr_nmb'] // 2)
            else:
                yss = ymid[j]
                neg_cont = yss
                pos_cont = yss
                col_ind = (opts['discr_nmb'] // 2)

            if (i % ylen) != 0 and (i % ylen) >= tme_stmp[0] + 1:
                k = (i % ylen) - tme_stmp[0] - 1
                xdif = x[k + 1] - x[k]
                xp = [x[k] - 1 / 2 * xdif, x[k] + 1 / 2 * xdif, x[k] + 1 / 2 * xdif, x[k] - 1 / 2 * xdif]
                yp = [neg_cont, neg_cont, pos_cont, pos_cont]
                plt.fill(xp, yp, color=colors[col_ind], edgecolor=colors[col_ind], alpha=1)
            elif i % ylen == 0 and i != 0:
                j += 1
                plt.subplot(3 + num_periods, 1, j + 1)
    
    plt.subplot(3 + num_periods, 1, 1)

    opts['export']['export_name'] = export_name + "_y"
    if 'yaxis' in opts['export']:
        if opts['export']['yaxis'] != "":
            y['lims'][2:4] = opts['export']['yaxis'][0]
    if 'dashed_lines' in opts['export'] and opts['export']['dashed_lines'] != "":
        plt.plot(np.ones(len(y['mu'])) * opts['export']['dashed_lines'][0][0], linewidth=opts['line_width'] * 0.5, color=[0.84, 0.84, 0.84], linestyle='--')
        plt.plot(np.ones(len(y['mu'])) * opts['export']['dashed_lines'][0][1], linewidth=opts['line_width'] * 0.5, color=[0.84, 0.84, 0.84], linestyle='--')
        temp_export = opts['export']
        temp_export['dashed_lines'] = opts['export']['dashed_lines'][0]
    else:
        temp_export = opts['export']
    if (delta is not None) and len(delta) > 1:
        if 'yaxis' in opts['export'] and opts['export']['yaxis'] != "":
            maxdel = (opts['export']['yaxis'][0][1] - opts['export']['yaxis'][0][0]) * 0.1
            basevalue = opts['export']['yaxis'][0][1]
        else:
            maxdel = (y['lims'][3] - y['lims'][2]) * 0.1
            basevalue = y['lims'][3]
        y['lims'][3] = basevalue + maxdel
        plotthis = maxdel * (opts['delta'] - 1) / max(opts['delta'] - 1) + basevalue
        plotthis = plotthis[tme_stmp[0]:tme_stmp[1]]
        plt.fill_between(range(len(plotthis)), basevalue, plotthis, facecolor=[0.6, 0.6, 0.6], alpha=0.3, edgecolor='none')

    plot_dist(y, plot_type, samples_colored=opts['samples_colored'], nmbsamples=opts['nmbsamples'], ylim=y['lims'][2:4], pbaspect=[pbx_length, pby_length, 1], line_width=opts['line_width'] * 1, export=temp_export)
    if 'format' in opts['export'] and opts['export']['format'] == "FIG":
        plt.title("input data")

    plt.subplot(3 + num_periods, 1, 2)

    opts['export']['export_name'] = export_name + "_LT"
    if 'yaxis' in opts['export']:
        if opts['export']['yaxis'] != "":
            LT['lims'][2:4] = opts['export']['yaxis'][1]

    if 'dashed_lines' in opts['export'] and opts['export']['dashed_lines'] != "":
        plt.plot(np.ones(len(y['mu'])) * opts['export']['dashed_lines'][1][0], linewidth=opts['line_width'] * 0.5, color=[0.84, 0.84, 0.84], linestyle='--')
        plt.plot(np.ones(len(y['mu'])) * opts['export']['dashed_lines'][1][1], linewidth=opts['line_width'] * 0.5, color=[0.84, 0.84, 0.84], linestyle='--')
        temp_export = opts['export']
        temp_export['dashed_lines'] = opts['export']['dashed_lines'][1]
        plot_dist(LT, plot_type, samples_colored=opts['samples_colored'], nmbsamples=opts['nmbsamples'], ylim=LT['lims'][2:4], pbaspect=[pbx_length, pby_length, 1], line_width=opts['line_width'] * 1, export=temp_export)
    else:
        plot_dist(LT, plot_type, samples_colored=opts['samples_colored'], nmbsamples=opts['nmbsamples'], ylim=LT['lims'][2:4], pbaspect=[pbx_length, pby_length, 1], line_width=opts['line_width'] * 1, export=opts['export'])
    if 'format' in opts['export'] and opts['export']['format'] == "FIG":
        plt.title("trend component")

    for i in range(num_periods):
        plt.subplot(3 + num_periods, 1, 2 + i + 1)
        temp = {'mu': ST['mu'][:, i], 'sigma_sq': ST['sigma_sq'][:, i], 'sigma': ST['sigma'][:, :, i]}
        if plot_type in ["comb", "spaghetti"]:
            temp['samples'] = ST['samples'][:, :, i]
        opts['export']['export_name'] = export_name + "_ST" + str(i + 1)
        if 'yaxis' in opts['export']:
            if opts['export']['yaxis'] != "":
                ST['lims'][2:4, i] = opts['export']['yaxis'][2 + i]
        if 'dashed_lines' in opts['export'] and opts['export']['dashed_lines'] != "":
            plt.plot(np.ones(len(y['mu'])) * opts['export']['dashed_lines'][2 + i][0], linewidth=opts['line_width'] * 0.5, color=[0.84, 0.84, 0.84], linestyle='--')
            plt.plot(np.ones(len(y['mu'])) * opts['export']['dashed_lines'][2 + i][1], linewidth=opts['line_width'] * 0.5, color=[0.84, 0.84, 0.84], linestyle='--')
            temp_export = opts['export']
            temp_export['dashed_lines'] = opts['export']['dashed_lines'][2 + i]
            all_zeros = np.all(y_ltstr.a_hat == 0)
            if (delta is not None) and (not all_zeros):
                temp['sigma_new'] = ST['sigma_new'][:, i]
            plot_dist(temp, plot_type, samples_colored=opts['samples_colored'], nmbsamples=opts['nmbsamples'], ylim=ST['lims'][2:4, i], pbaspect=[pbx_length, pby_length, 1], line_width=opts['line_width'] * 1, export=temp_export)
        else:
            all_zeros = np.all(y_ltstr.a_hat == 0)
            if (delta is not None) and (not all_zeros):
                temp['sigma_new'] = ST['sigma_new'][:, i]
            plot_dist(temp, plot_type, samples_colored=opts['samples_colored'], nmbsamples=opts['nmbsamples'], ylim=ST['lims'][2:4, i], pbaspect=[pbx_length, pby_length, 1], line_width=opts['line_width'] * 1, export=opts['export'])
        if 'format' in opts['export'] and opts['export']['format'] == "FIG":
            plt.title(f"seasonal component {i + 1}")

    plt.subplot(3 + num_periods, 1, 3 + num_periods)

    opts['export']['export_name'] = export_name + "_R"
    if 'yaxis' in opts['export']:
        if opts['export']['yaxis'] != "":
            R['lims'][2:4] = opts['export']['yaxis'][3 + num_periods]
    if 'dashed_lines' in opts['export'] and opts['export']['dashed_lines'] != "":
        plt.plot(np.ones(len(y['mu'])) * opts['export']['dashed_lines'][3 + num_periods][0], linewidth=opts['line_width'] * 0.5, color=[0.84, 0.84, 0.84], linestyle='--')
        plt.plot(np.ones(len(y['mu'])) * opts['export']['dashed_lines'][3 + num_periods][1], linewidth=opts['line_width'] * 0.5, color=[0.84, 0.84, 0.84], linestyle='--')
        temp_export = opts['export']
        temp_export['dashed_lines'] = opts['export']['dashed_lines'][3 + num_periods]
        plot_dist(R, plot_type, samples_colored=opts['samples_colored'], nmbsamples=opts['nmbsamples'], ylim=R['lims'][2:4], pbaspect=[pbx_length, pby_length, 1], line_width=opts['line_width'] * 1, export=temp_export)
    else:
        plot_dist(R, plot_type, samples_colored=opts['samples_colored'], nmbsamples=opts['nmbsamples'], ylim=R['lims'][2:4], pbaspect=[pbx_length, pby_length, 1], line_width=opts['line_width'] * 1, export=opts['export'])
    if 'format' in opts['export'] and opts['export']['format'] == "FIG":
        plt.title("residual component")

    if opts['export']['export'] and 'format' in opts['export'] and opts['export']['format'] == "FIG":
        cur_fig = plt.gcf()
        cur_fig.set_size_inches(16, 9)
        plt.savefig(f"{opts['export']['export_path']}{export_name}.png", bbox_inches='tight', dpi=300)

    fig_1 = plt.gcf()
    axs_1 = plt.gca()
    fig_1.tight_layout()

    # Change export_name to original input opts.export.export_name
    opts['export']['export_name'] = export_name
    if opts['plot_cov']:
        fig_2, axs_2 = plot_cov_mat(y_ltstr, num_periods, opts)
        fig_2.tight_layout()

    if opts['plot_cor']:
        y_ltstr, fig_3, axs_2 = plot_cor_mat(y_ltstr, num_periods, opts)
        fig_3.tight_layout()

    if opts['plot_cor_length']:
        plot_cor_length(y_ltstr, num_periods, opts)
        if opts['export']['export'] and 'format' in opts['export'] and opts['export']['format'] == "FIG":
            cur_fig = plt.gcf()
            cur_fig.set_size_inches(16, 9)
            plt.savefig(f"{opts['export']['export_path']}{opts['export']['export_name']}_cor_length.png", bbox_inches='tight', dpi=300)

    print("#####")
    print("##### Plotting DONE")
    print("#####")
    print("#################################################################################")
    return y_ltstr

# HELPER FUNCTIONS

def plot_cov_mat(y_ltstr, num_periods, opts):
    """
    This function PLOTCOV creates a plot of the covariance matrix
    y_ltstr.sigma
    """
    nmb_colors = opts['discr_nmb']
    color_map = color_lut(1)
    color_map = color_map[np.round(np.linspace(0, 256, nmb_colors)).astype(int), :]

    # Create a custom colormap
    cmap = mcolors.ListedColormap(color_map)

    plt.figure()
    if not opts['export']['export']:
        plt.suptitle('Covariance Matrix')
    plt.imshow(y_ltstr.sigma, cmap=cmap)
    plt.colorbar()

    # White lines between block matrices y, LT, STs, R
    sublength = len(y_ltstr.mu) // (num_periods + 3)

    for i in range(1, num_periods + 3):
        plt.plot([(i - 1) * sublength + 0.5 + sublength] * len(y_ltstr.mu), ':', color='white', linewidth=opts['line_width'])
        plt.plot([(i - 1) * sublength + sublength + 0.5] * len(y_ltstr.mu), np.linspace(0, len(y_ltstr.mu) - 1, len(y_ltstr.mu)), ':', color='white', linewidth=opts['line_width'])

    max_el = np.max(np.abs(y_ltstr.sigma))
    if abs(max_el) > 0:
        plt.clim(-max_el, max_el)

    if opts['export']['export']:
        if 'export_name' in opts['export']:
            export_name = opts['export']['export_name']
        else:
            export_name = "plot_dist"
        opts['export']['export_name'] = f"{export_name}_cov_plot"
        opts['export']['pbaspect'] = [1, 1, 1]
        export_plot(opts)

    fig = plt.gcf()
    axs = plt.gca()

    return fig, axs

def plot_cor_mat(y_ltstr, num_periods, opts):
    """
    This function PLOTCOR creates a plot of the correlation matrix using the
    covariance matrix y_ltstr.sigma and the function compute_cor_mat that transforms
    the covariance matrix to the correlation matrix.
    This is part of the correlation exploration for UASTL.
    """
    y_ltstr = compute_cor_mat(y_ltstr)
    nmb_colors = opts['discr_nmb']
    color_map = color_lut(1)
    color_map = color_map[np.round(np.linspace(0, 256, nmb_colors)).astype(int), :]

    # Create a custom colormap
    cmap = mcolors.ListedColormap(color_map)

    plt.figure()
    if not opts['export']['export']:
        plt.suptitle('Correlation Matrix')
    plt.imshow(y_ltstr.cor_mat, cmap=cmap)
    plt.colorbar()

    # White lines between block matrices y, LT, STs, R
    sublength = len(y_ltstr.mu) // (num_periods + 3)

    for i in range(1, num_periods + 3):
        plt.plot([(i - 1) * sublength + 0.5 + sublength] * len(y_ltstr.mu), ':', color='white', linewidth=opts['line_width'])
        plt.plot([(i - 1) * sublength + sublength + 0.5] * len(y_ltstr.mu), np.linspace(0, len(y_ltstr.mu) - 1, len(y_ltstr.mu)), ':', color='white', linewidth=opts['line_width'])

    plt.clim(-1, 1)
    if opts['export']['export']:
        if 'export_name' in opts['export']:
            export_name = opts['export']['export_name']
        else:
            export_name = "plot_dist"
        opts['export']['export_name'] = f"{export_name}_cor_plot"
        opts['export']['pbaspect'] = [1, 1, 1]
        export_plot(opts)

    fig = plt.gcf()
    axs = plt.gca()

    return y_ltstr, fig, axs

def compute_cor_mat(y_ltstr):
    """
    This function CORCOMP computes the correlation matrix given the
    covariance matrix y_ltstr.sigma.
    """
    c_inv_mult = np.zeros(y_ltstr.sigma.shape[0])
    for k in range(y_ltstr.sigma.shape[0]):
        c = y_ltstr.sigma[k, k]
        if c < 1e-12:
            c_inv_mult[k] = 0
        else:
            c_inv_mult[k] = 1 / np.sqrt(c)
    y_ltstr.cor_mat = np.diag(c_inv_mult) @ y_ltstr.sigma @ np.diag(c_inv_mult)
    return y_ltstr

def plot_cor_length(y_ltstr, num_periods, opts):
    """
    This function PLOTCORLENGTH creates a plot of the correlation length for
    each component of the correlation matrix.
    This is part of the correlation exploration for UASTL.
    """
    sb_plot_nmb = num_periods + 3
    length_data = len(y_ltstr.mu) // sb_plot_nmb
    all_zeros = np.all(y_ltstr.cor_mat == 0)
    if all_zeros:
        y_ltstr = compute_cor_mat(y_ltstr)

    sign_cor_mat = np.sign(y_ltstr.cor_mat)
    maxdim = len(sign_cor_mat)
    full_cor_length = np.zeros(maxdim)
    vert_r = np.zeros(maxdim)
    vert_l = np.zeros(maxdim)
    for i in range(maxdim):
        cur_sign_el = sign_cor_mat[i, i]
        if abs(cur_sign_el) > 0:
            vert1 = 1
            vert2 = 1
            dec_b = i % length_data

            if dec_b == 0:
                var_r = 0
                var_l = length_data
            elif dec_b < length_data / 2 or dec_b > length_data / 2:
                var_r = length_data - dec_b
                var_l = dec_b
            else:
                var_r = length_data // 2
                var_l = length_data // 2

            for j in range(1, var_r):
                if cur_sign_el == sign_cor_mat[i, i + j]:
                    vert1 += 1
                else:
                    break
            for j in range(1, var_l):
                if cur_sign_el == sign_cor_mat[i, i - j]:
                    vert2 += 1
                else:
                    break
            vert_r[i] = vert1
            vert_l[i] = vert2
            full_cor_length[i] = vert1 + vert2 - 1

    nmb_colors = opts['discr_nmb'] // 2 + 1
    out = [None] * len(y_ltstr.cor_mat)
    vert_l = vert_l.astype(int)
    vert_r = vert_r.astype(int)
    for i in range(len(y_ltstr.cor_mat)):
        col_ind = np.full(length_data, np.nan)
        k = 2
        if vert_l[i] >= 1 or vert_r[i] >= 1:
            print(vert_l[i])
            for j in range(vert_l[i] - 1, 0, -1):
                col_ind[k] = min(int(np.ceil(y_ltstr.cor_mat[i, i - j] * nmb_colors)), nmb_colors)
                k += 1
            col_ind[k] = min(int(np.ceil(y_ltstr.cor_mat[i, i] * nmb_colors)), nmb_colors)
            k += 1
            for j in range(0, vert_r[i] - 1):
                col_ind[k] = min(int(np.ceil(y_ltstr.cor_mat[i, i + j] * nmb_colors)), nmb_colors)
                k += 1
            col_ind = col_ind[~np.isnan(col_ind)]
            out[i] = col_ind

    plot_mat = np.zeros((len(y_ltstr.cor_mat), length_data))
    ysize = np.zeros(len(out))
    for i in range(len(out)):
        if out[i] is not None:
            insert = out[i]
            ysize[i] = len(insert)
            plot_mat[i, :len(insert)] = insert

    plot_mat = np.flipud(plot_mat.T)

    colors = color_lut(1)
    colors = colors[128:256, :]
    colors = colors[np.round(np.linspace(0, 127, nmb_colors)).astype(int), :]
    colors = np.vstack(([1, 1, 1], colors))

    plt.figure()
    if not opts['export']['export']:
        plt.suptitle('Correlation Length')
    for i in range(sb_plot_nmb):
        plt.subplot(sb_plot_nmb, 1, i + 1)
        max_i = max(ysize[opts['stamp_indices'][(i * opts['stmp_length']):((i + 1) * opts['stmp_length'])]])
        image2plot = plot_mat[-max_i:, opts['stamp_indices'][(i * opts['stmp_length']):((i + 1) * opts['stmp_length'])]]
        plt.imshow(image2plot, cmap=colors)
        plt.hold(True)
        vert_ls = np.repeat(vert_l[opts['stamp_indices'][(i * opts['stmp_length']):((i + 1) * opts['stmp_length'])]], 2)
        xs = np.repeat(np.linspace(1, opts['stmp_length'] + 1, opts['stmp_length'] + 1), 2)
        xs = xs[:-1]
        plt.plot(xs - 0.5, max_i + 1 - vert_ls, linewidth=2, color='black', linestyle='-')
        if 'format' in opts['export'] and opts['export']['format'] == "FIG":
            if i == 0:
                plt.title("input data")
            elif i == 1:
                plt.title("trend component")
            elif i == sb_plot_nmb - 1:
                plt.title("residual component")
            else:
                plt.title(f"seasonal component {i - 1}")

    if opts['export']['export']:
        if 'export_name' in opts['export']:
            export_name = opts['export']['export_name']
        else:
            export_name = "plot_dist"
        for i in range(sb_plot_nmb):
            plt.subplot(sb_plot_nmb, 1, i + 1)
            opts['export']['export_name'] = f"{export_name}_cor_length_{i + 1}"
            export_sub_plot(opts)

def export_sub_plot(opts):
    """
    This function EXPORTSUBPLOT uses the current axis handle and exports it
    using the specified export options (opts.export.*).
    """
    if 'format' not in opts['export']:
        opts['export']['format'] = "SUBs"
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    ax.set_visible(True)
    if 'pbaspect' in opts['export']:
        ax.set_aspect(opts['export']['pbaspect'])
    else:
        ax.set_aspect([1, 1, 1])
    plt.colorbar().remove()
    if opts['export']['format'] == "SUBs":
        if opts['export']['export']:
            plt.savefig(f"{opts['export']['export_path']}{opts['export']['export_name']}.pdf", format='pdf', bbox_inches='tight', transparent=True)
    elif opts['export']['format'] == "FIG":
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(True)

def export_plot(opts):
    """
    This function EXPORTPLOT uses the current figure handle and exports it
    using the specified export options (opts.export.*).
    """
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    ax.set_visible(True)
    if 'pbaspect' in opts['export']:
        ax.set_aspect(opts['export']['pbaspect'])
    else:
        ax.set_aspect([1, 1, 1])
    plt.colorbar().remove()
    if opts['export']['export']:
        plt.savefig(f"{opts['export']['export_path']}{opts['export']['export_name']}.pdf", format='pdf', bbox_inches='tight', transparent=True)

def color_lut(number):
    """
    Imports the used color scheme generated via the code by Andy Stein and
    based on Kenneth Moreland's code for creating diverging colormaps.
    """
    A = pd.read_csv('CoolWarmFloat257.csv')
    if number == 1:
        return diverging_map(A.iloc[:, 0].values, [0.230, 0.299, 0.754], [0.706, 0.016, 0.150])  # blue-white-red
    return None
