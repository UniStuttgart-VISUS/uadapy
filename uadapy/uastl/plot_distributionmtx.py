from plot_dist import plot_dist
from diverging_map import diverging_map
from uastl import UncertainData
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def plot_distributionmtx(yLTSTR : UncertainData, numPeriods, plot_type, **opts):
    """
    This function PLOT_DISTRIBUTIONMTX creates the main visualization plot
    of the distribution for given mean and covariance matrix.
    It is highly dependent on the function plot_dist.

    Plot distribution for time series depending on the Gaussian distributed 
    variable yLTSTR = [y, LT, ST_1, ..., ST_L, R] with
        * yLTSTR.mu: mean vector 
        * yLTSTR.Sigma: covariance matrix

    The theory is described in the related manuscript "Uncertainty-Aware
    Seasonal-Trend Decomposition Based on Loess".

    yLTSTR = PLOT_DISTRIBUTIONMTX(yLTSTR, numPeriods, type, opts)

    Input:
        * yLTSTR: struct with mean and Sigma containing the data y, trend LT, 
                  seasonal ST, and residual R components computed by UASTL
            - yLTSTR.mu: mean vector with components
            - yLTSTR.Sigma: covariance matrix with components
        * numPeriods: Number of periods to subdivide the yLTSTR cell data
        * type: one of "comb", "isoband", or "spaghetti"
                "comb" plots isobands and spaghetti samples
        * opts: optional input with additional VIS techniques
            - timeStamp (:,1) double = []
            - nmbsamples (1,1) {mustBeInteger} = 5
            - samplesColored (1,1) logical = False
            - lineWidth {mustBeNumeric} = 1
            - plotCov (1,1) logical = False
            - plotCor (1,1) logical = False
            - plotCorLength (1,:) struct = struct
            - coPoint (1,1) {mustBeInteger} = 0
            - delta (:,1) double = []
            - export

    Output: builds Figure(s) containing the desired VIS techniques for UASTL
        * yLTSTR: Same yLTSTR but with added field CorMat if correlation matrix
                  is computed (if coPoint or plotCor options are set)
    """

    print("############################### Visualization ###################################")
    print("##### Uncertainty-Aware Seasonal-Trend Decomposition Based on Loess (UASTL) #####")
    print("#################################################################################")
    print("#####")
    print(f"##### Number of Periods:\t numPeriods = {numPeriods}")
    print("#####")
    print(f"##### Plotting Type:\t \t type = {plot_type}")
    print(f"##### \t ├── Number of Spaghetthi: \t nmbsamples = {opts['nmbsamples']}")
    print(f"##### \t ├── Spaghetthi colored:\t samplesColored = {opts['samplesColored']}")
    print(f"##### \t ├── Set LineWidth Factor:\t lineWidth = {opts['lineWidth']}")
    print(f"##### \t └── Discretization Steps:\t discrNmb = {opts['discrNmb']}")
    print("#####")
    timeStamp = opts.get('timeStamp')
    delta = opts.get('delta')
    if timeStamp:
        print(f"##### Plotting Interval:\t timeStamp = [{opts['timeStamp'][0]},{opts['timeStamp'][1]}]")
    else:
        print("##### Plotting Interval:\t Full Time Domain.")
    print("#####")
    print("##### Visualization Techniques:")
    print(f"##### \t ├── Covariance Matrix:\t\t plotCov = {opts['plotCov']}")
    print(f"##### \t ├── Correlation Matrix:\t plotCor = {opts['plotCor']}")
    print(f"##### \t ├── Correlation Plot:\t \t plotCorLength = {opts['plotCorLength']}")
    if opts['coPoint'] != 0:
        print(f"##### \t └── Local Corr. Details:\t coPoint = {opts['coPoint']}")
    else:
        print(f"##### \t └── No Local Corr. Details:\t coPoint = {opts['coPoint']}")
    print("#####")
    if 'export' in opts and opts['export']['export'] and 'exportPath' in opts['export']:
        print(f"##### Subplot Export into Folder: \t export.exportPath = {opts['export']['exportPath']}")
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
    # yLTSTR.mu & yLTSTR.Sigma
    if (len(yLTSTR.mu) == yLTSTR.Sigma.shape[0]) and (len(yLTSTR.mu) == yLTSTR.Sigma.shape[1]):
        ylen = len(yLTSTR.mu) // (3 + numPeriods)
    else:
        raise ValueError("Size of mu and Sigma are not concise")

    # export
    if 'export' in opts and opts['export']:
        required_export_fields = ["exportPath", "exportName", "export", "pbaspect", "yaxis", "dashedLines", "format"]
        for field in required_export_fields:
            if field not in opts['export']:
                opts['export'][field] = ""
        if 'exportName' in opts['export']:
            exportName = opts['export']['exportName']
        else:
            exportName = "plot_dist"
    else:
        opts['export'] = {'export': False, 'format': "FIG"}
        exportName = ""

    # discrNmb
    if opts['discrNmb'] > 257:
        opts['discrNmb'] = 257
        print('"nmbColors" set too large, setting to 257')
    elif opts['discrNmb'] % 2 == 0:
        opts['discrNmb'] += 1
        print('"nmbColors" set to the next higher odd number.')

    if timeStamp :
        if opts['timeStamp'][1] <= opts['timeStamp'][0]:
            print('TimeStamp not correctly assigned. Ignoring input.')
            opts['timeStamp'] = []

    if not timeStamp:
        TmeStmp = [1, ylen]
        opts['StmpLength'] = ylen
        opts['stamp_indices'] = list(range(len(yLTSTR.mu)))
        opts['timeStamp'] = [1, ylen]
    else:
        TmeStmp = opts['timeStamp']
        opts['StmpLength'] = TmeStmp[1] - TmeStmp[0] + 1
        stamp_indices = list(range(opts['timeStamp'][0], opts['timeStamp'][1] + 1))
        stamp_indices += list(range(ylen + opts['timeStamp'][0], ylen + opts['timeStamp'][1] + 1))
        for i in range(numPeriods):
            stamp_indices += list(range((i + 1) * ylen + opts['timeStamp'][0], (i + 1) * ylen + opts['timeStamp'][1] + 1))
        opts['stamp_indices'] = stamp_indices + list(range((numPeriods + 2) * ylen + opts['timeStamp'][0], (numPeriods + 2) * ylen + opts['timeStamp'][1] + 1))

    # Picking global samples if comb or spaghetti plot
    if plot_type in ["comb", "spaghetti"]:
        all_zeros = np.all(yLTSTR.AHat == 0)
        if (delta is not None) and (not all_zeros):
            # SHIFTED samples via delta vector
            Sigma = yLTSTR.Sigma[:ylen, :ylen]
            epsilon = 1e-10  # Small value to add
            Sigma = Sigma + epsilon * np.eye(Sigma.shape[0]) #prevent positive definite matrix error
            T = np.linalg.cholesky(Sigma)# returns lower traingular matrix by default #.T
            sampOrig = np.random.randn(opts['nmbsamples'], T.shape[1]).T
            AHat_EHat = yLTSTR.AHat
            T_sampOrig = T @ sampOrig
            samp = AHat_EHat @ T_sampOrig + yLTSTR.mu
            sampNeu = samp + AHat_EHat @ np.diag(opts['delta'] - 1) @ T_sampOrig
            yLTSTR.samples = np.hstack((samp, sampNeu))
            sigmaNew = np.sqrt(np.sum((AHat_EHat @ np.diag(opts['delta']) @ T)**2, axis=1))

            y = {'samples': yLTSTR.samples[TmeStmp[0]-1:TmeStmp[1], :],
                 'sigmaNew': sigmaNew[TmeStmp[0]-1:TmeStmp[1]]}
            LT = {'samples': yLTSTR.samples[ylen + TmeStmp[0]-1:ylen + TmeStmp[1], :],
                  'sigmaNew': sigmaNew[ylen + TmeStmp[0]-1:ylen + TmeStmp[1]]}

            ST = {'samples': np.zeros((TmeStmp[1] - TmeStmp[0] + 1, opts['nmbsamples'] * 2 , numPeriods)), #yLTSTR.samples.shape so * 2
                  'sigmaNew': np.zeros((TmeStmp[1] - TmeStmp[0] + 1, numPeriods))}
            for i in range(numPeriods):
                ST['samples'][:, :, i] = yLTSTR.samples[(i + 1) * ylen + TmeStmp[0]-1:(i + 1) * ylen + TmeStmp[1], :]
                ST['sigmaNew'][:, i] = sigmaNew[(i + 1) * ylen + TmeStmp[0]-1:(i + 1) * ylen + TmeStmp[1]]

            R = {'samples': yLTSTR.samples[(2 + numPeriods) * ylen + TmeStmp[0]-1:(2 + numPeriods) * ylen + TmeStmp[1], :],
                 'sigmaNew': sigmaNew[(2 + numPeriods) * ylen + TmeStmp[0]-1:(2 + numPeriods) * ylen + TmeStmp[1]]}
        else:
            if delta:
                print('No yLTSTR.AHat given, therefore going on without delta.')
            yLTSTR.samples = np.random.multivariate_normal(yLTSTR.mu.flatten(), yLTSTR.Sigma, opts['nmbsamples']).T
            y = {'samples': yLTSTR.samples[TmeStmp[0]-1:TmeStmp[1], :]}
            LT = {'samples': yLTSTR.samples[ylen + TmeStmp[0]-1:ylen + TmeStmp[1], :]}
            ST = {'samples': np.zeros((TmeStmp[1] - TmeStmp[0] + 1, opts['nmbsamples'], numPeriods))}
            for i in range(numPeriods):
                ST['samples'][:, :, i] = yLTSTR.samples[(i + 2) * ylen + TmeStmp[0]-1:(i + 2) * ylen + TmeStmp[1], :]
            R = {'samples': yLTSTR.samples[(2 + numPeriods) * ylen + TmeStmp[0]-1:(2 + numPeriods) * ylen + TmeStmp[1], :]}

    y['mu'] = yLTSTR.mu[TmeStmp[0]-1:TmeStmp[1]]
    y['sigma'] = np.sqrt(np.maximum(np.diag(yLTSTR.Sigma[TmeStmp[0]-1:TmeStmp[1], TmeStmp[0]-1:TmeStmp[1]]), 0))
    y['Sigma'] = yLTSTR.Sigma[TmeStmp[0]-1:TmeStmp[1], TmeStmp[0]-1:TmeStmp[1]]

    LT['mu'] = yLTSTR.mu[ylen + TmeStmp[0]-1:ylen + TmeStmp[1]]
    LT['sigma'] = np.sqrt(np.maximum(np.diag(yLTSTR.Sigma[ylen + TmeStmp[0]-1:ylen + TmeStmp[1], ylen + TmeStmp[0]-1:ylen + TmeStmp[1]]), 0))
    LT['Sigma'] = yLTSTR.Sigma[ylen + TmeStmp[0]-1:ylen + TmeStmp[1], ylen + TmeStmp[0]-1:ylen + TmeStmp[1]]

    ST['mu'] = np.zeros((TmeStmp[1] - TmeStmp[0] + 1, numPeriods))
    ST['sigma'] =  np.zeros((TmeStmp[1] - TmeStmp[0] + 1, numPeriods))
    ST['Sigma'] = np.zeros((TmeStmp[1] - TmeStmp[0] + 1, TmeStmp[1] - TmeStmp[0] + 1, numPeriods))

    for i in range(numPeriods):
        ST['mu'][:, i] = yLTSTR.mu[(i + 2) * ylen + TmeStmp[0]-1:(i + 2) * ylen + TmeStmp[1]].flatten()
        ST['sigma'][:, i] = np.sqrt(np.maximum(np.diag(yLTSTR.Sigma[(i + 2) * ylen + TmeStmp[0]-1:(i + 2) * ylen + TmeStmp[1], (i + 2) * ylen + TmeStmp[0]-1:(i + 2) * ylen + TmeStmp[1]]), 0))
        ST['Sigma'][:, :, i] = yLTSTR.Sigma[(i + 2) * ylen + TmeStmp[0]-1:(i + 2) * ylen + TmeStmp[1], (i + 2) * ylen + TmeStmp[0]-1:(i + 2) * ylen + TmeStmp[1]]

    R['mu'] = yLTSTR.mu[(2 + numPeriods) * ylen + TmeStmp[0]-1:(2 + numPeriods) * ylen + TmeStmp[1]]
    R['sigma'] =  np.sqrt(np.maximum(np.diag(yLTSTR.Sigma[(2 + numPeriods) * ylen + TmeStmp[0]-1:(2 + numPeriods) * ylen + TmeStmp[1], (2 + numPeriods) * ylen + TmeStmp[0]-1:(2 + numPeriods) * ylen + TmeStmp[1]]), 0))
    R['Sigma'] = yLTSTR.Sigma[(2 + numPeriods) * ylen + TmeStmp[0]-1:(2 + numPeriods) * ylen + TmeStmp[1], (2 + numPeriods) * ylen + TmeStmp[0]-1:(2 + numPeriods) * ylen + TmeStmp[1]]

    # SET FIGURE
    plt.figure()
    plt.get_current_fig_manager().resize(1710, 1112)
    plt.suptitle('Uncertainty-Aware Seasonal-Trend Decomposition')

    nmbColors = opts['discrNmb']
    colors = color_lut(1)
    colors = colors[np.round(np.linspace(0, 256, nmbColors)).astype(int), :]

    if opts['coPoint'] == 0:
        helperCoDep = 0
    elif abs(opts['coPoint']) > len(yLTSTR.mu):
        print('Covariance point set too high, plotting without dependency plot.\n')
        helperCoDep = 0
    elif opts['coPoint'] in opts['stamp_indices']:
        helperCoDep = 1
    else:
        helperCoDep = 1
        print('Covariance point is out of timeStamp-Range.\n')

    if helperCoDep == 1:
        Interactiv_Pointer = opts['coPoint']
        # We choose CorMat for background plot.
        yLTSTR = corComp(yLTSTR)
        plotBack = yLTSTR.corMat[Interactiv_Pointer, :]

    # Get the ylims for each part by (mean+sigmalvlmax)/3*4 to add below
    sigmalvl = [0, 0.674490, 2.575829]

    part_factor = 5
    max_sigmalvl = np.max(sigmalvl)
    ceil_max_sigmalvl = np.ceil(max_sigmalvl)
    lower_limit = np.min(y['mu'] - ceil_max_sigmalvl * y['sigma'])
    upper_limit = np.max(y['mu'] + ceil_max_sigmalvl * y['sigma'])
    y['lims'] = [1, len(y['mu']), lower_limit, upper_limit]
    y['lims'][2] -= helperCoDep * 1 / part_factor * (y['lims'][3] - y['lims'][2])
    maxyheight = [(y['lims'][3] - y['lims'][2]) / (part_factor + 1) / 2]
    ymid = [y['lims'][2] + maxyheight[0]]

    lower_limit = np.min(LT['mu'] - ceil_max_sigmalvl * LT['sigma'])
    upper_limit = np.max(LT['mu'] + ceil_max_sigmalvl * LT['sigma'])
    LT['lims'] = [1, len(LT['mu']), lower_limit, upper_limit]
    LT['lims'][2] -= helperCoDep * 1 / part_factor * (LT['lims'][3] - LT['lims'][2])
    maxyheight.append((LT['lims'][3] - LT['lims'][2]) / (part_factor + 1) / 2)
    ymid.append(LT['lims'][2] + maxyheight[1])

    ST['lims'] = np.zeros((4, ST['mu'].shape[1])) #added for keyerror
    for i in range(numPeriods):
        lower_limit = np.min(ST['mu'][:, i] - ceil_max_sigmalvl * ST['sigma'][:, i])
        upper_limit = np.max(ST['mu'][:, i] + ceil_max_sigmalvl * ST['sigma'][:, i])
        ST['lims'][:, i] = [1, len(ST['mu'][:, i]), lower_limit, upper_limit]
        ST['lims'][2, i] -= helperCoDep * 1 / part_factor * (ST['lims'][3, i] - ST['lims'][2, i])
        maxyheight.append((ST['lims'][3, i] - ST['lims'][2, i]) / (part_factor + 1) / 2)
        ymid.append(ST['lims'][2, i] + maxyheight[-1])

    lower_limit = np.min(R['mu'] - ceil_max_sigmalvl * R['sigma'])
    upper_limit = np.max(R['mu'] + ceil_max_sigmalvl * R['sigma'])
    R['lims'] = [1, len(R['mu']), lower_limit, upper_limit]
    R['lims'][2] -= helperCoDep * 1 / part_factor * (R['lims'][3] - R['lims'][2])
    maxyheight.append((R['lims'][3] - R['lims'][2]) / (part_factor + 1) / 2)
    ymid.append(R['lims'][2] + maxyheight[-1])

    x = np.linspace(1, ylen, ylen)

    # Approx. 16:9 pbaspect of plot:
    pbxLength = (3 + numPeriods + 1) / 9 * 16
    pbyLength = 1

    export_opts = opts['export']
    if 'pbaspect' in export_opts:
        pbxLength, pbyLength, _ = export_opts['pbaspect']

    plt.subplot(3 + numPeriods, 1, 1)
    if helperCoDep == 1:
        j = 0
        for i in range(len(plotBack) - 1):
            if i == opts['coPoint'] - 1:  # Adjusting for 0-indexing in Python
                k = (i % ylen) - TmeStmp[0]
                thickness = (TmeStmp[1] - TmeStmp[0]) / 500
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

            maxc = max(abs(plotBack[(j) * ylen: (j + 1) * ylen]))
            if maxc > 0:
                yss = ymid[j]
                diff = maxyheight[j]
                if plotBack[i] > 0:
                    negCont = yss
                    posCont = yss + plotBack[i] / maxc * diff
                else:
                    negCont = yss + plotBack[i] / maxc * diff
                    posCont = yss
                colInd = int(np.ceil(plotBack[i] / maxc * (opts['discrNmb'] // 2))) + (opts['discrNmb'] // 2)
            else:
                yss = ymid[j]
                negCont = yss
                posCont = yss
                colInd = (opts['discrNmb'] // 2)

            if (i % ylen) != 0 and (i % ylen) >= TmeStmp[0]:
                k = (i % ylen) - TmeStmp[0]
                xdif = x[k + 1] - x[k]
                xp = [x[k] - 1 / 2 * xdif, x[k] + 1 / 2 * xdif, x[k] + 1 / 2 * xdif, x[k] - 1 / 2 * xdif]
                yp = [negCont, negCont, posCont, posCont]
                plt.fill(xp, yp, color=colors[colInd], edgecolor=colors[colInd], alpha=1)
            elif i % ylen == 0 and i != 0:
                j += 1
                plt.subplot(3 + numPeriods, 1, j + 1)
    
    plt.subplot(3 + numPeriods, 1, 1)

    opts['export']['exportName'] = exportName + "_y"
    if 'yaxis' in opts['export']:
        if opts['export']['yaxis'] != "":
            y['lims'][2:4] = opts['export']['yaxis'][0]
    if 'dashedLines' in opts['export'] and opts['export']['dashedLines'] != "":
        plt.plot(np.ones(len(y['mu'])) * opts['export']['dashedLines'][0][0], linewidth=opts['lineWidth'] * 0.5, color=[0.84, 0.84, 0.84], linestyle='--')
        plt.plot(np.ones(len(y['mu'])) * opts['export']['dashedLines'][0][1], linewidth=opts['lineWidth'] * 0.5, color=[0.84, 0.84, 0.84], linestyle='--')
        tempExport = opts['export']
        tempExport['dashedLines'] = opts['export']['dashedLines'][0]
    else:
        tempExport = opts['export']
    if (delta is not None) and len(delta) > 1:
        if 'yaxis' in opts['export'] and opts['export']['yaxis'] != "":
            maxdel = (opts['export']['yaxis'][0][1] - opts['export']['yaxis'][0][0]) * 0.1
            basevalue = opts['export']['yaxis'][0][1]
        else:
            maxdel = (y['lims'][3] - y['lims'][2]) * 0.1
            basevalue = y['lims'][3]
        y['lims'][3] = basevalue + maxdel
        plotthis = maxdel * (opts['delta'] - 1) / max(opts['delta'] - 1) + basevalue
        plotthis = plotthis[TmeStmp[0]-1:TmeStmp[1]]
        plt.fill_between(range(len(plotthis)), basevalue, plotthis, facecolor=[0.6, 0.6, 0.6], alpha=0.3, edgecolor='none')

    plot_dist(y, plot_type, samplesColored=opts['samplesColored'], nmbsamples=opts['nmbsamples'], ylim=y['lims'][2:4], pbaspect=[pbxLength, pbyLength, 1], lineWidth=opts['lineWidth'] * 1, export=tempExport)
    if 'format' in opts['export'] and opts['export']['format'] == "FIG":
        plt.title("input data")

    plt.subplot(3 + numPeriods, 1, 2)

    opts['export']['exportName'] = exportName + "_LT"
    if 'yaxis' in opts['export']:
        if opts['export']['yaxis'] != "":
            LT['lims'][2:4] = opts['export']['yaxis'][1]

    if 'dashedLines' in opts['export'] and opts['export']['dashedLines'] != "":
        plt.plot(np.ones(len(y['mu'])) * opts['export']['dashedLines'][1][0], linewidth=opts['lineWidth'] * 0.5, color=[0.84, 0.84, 0.84], linestyle='--')
        plt.plot(np.ones(len(y['mu'])) * opts['export']['dashedLines'][1][1], linewidth=opts['lineWidth'] * 0.5, color=[0.84, 0.84, 0.84], linestyle='--')
        tempExport = opts['export']
        tempExport['dashedLines'] = opts['export']['dashedLines'][1]
        plot_dist(LT, plot_type, samplesColored=opts['samplesColored'], nmbsamples=opts['nmbsamples'], ylim=LT['lims'][2:4], pbaspect=[pbxLength, pbyLength, 1], lineWidth=opts['lineWidth'] * 1, export=tempExport)
    else:
        plot_dist(LT, plot_type, samplesColored=opts['samplesColored'], nmbsamples=opts['nmbsamples'], ylim=LT['lims'][2:4], pbaspect=[pbxLength, pbyLength, 1], lineWidth=opts['lineWidth'] * 1, export=opts['export'])
    if 'format' in opts['export'] and opts['export']['format'] == "FIG":
        plt.title("trend component")

    for i in range(numPeriods):
        plt.subplot(3 + numPeriods, 1, 2 + i + 1)
        temp = {'mu': ST['mu'][:, i], 'sigma': ST['sigma'][:, i], 'Sigma': ST['Sigma'][:, :, i]}
        if plot_type in ["comb", "spaghetti"]:
            temp['samples'] = ST['samples'][:, :, i]
        opts['export']['exportName'] = exportName + "_ST" + str(i + 1)
        if 'yaxis' in opts['export']:
            if opts['export']['yaxis'] != "":
                ST['lims'][2:4, i] = opts['export']['yaxis'][2 + i]
        if 'dashedLines' in opts['export'] and opts['export']['dashedLines'] != "":
            plt.plot(np.ones(len(y['mu'])) * opts['export']['dashedLines'][2 + i][0], linewidth=opts['lineWidth'] * 0.5, color=[0.84, 0.84, 0.84], linestyle='--')
            plt.plot(np.ones(len(y['mu'])) * opts['export']['dashedLines'][2 + i][1], linewidth=opts['lineWidth'] * 0.5, color=[0.84, 0.84, 0.84], linestyle='--')
            tempExport = opts['export']
            tempExport['dashedLines'] = opts['export']['dashedLines'][2 + i]
            all_zeros = np.all(yLTSTR.AHat == 0)
            if (delta is not None) and (not all_zeros):
                temp['sigmaNew'] = ST['sigmaNew'][:, i]
            plot_dist(temp, plot_type, samplesColored=opts['samplesColored'], nmbsamples=opts['nmbsamples'], ylim=ST['lims'][2:4, i], pbaspect=[pbxLength, pbyLength, 1], lineWidth=opts['lineWidth'] * 1, export=tempExport)
        else:
            all_zeros = np.all(yLTSTR.AHat == 0)
            if (delta is not None) and (not all_zeros):
                temp['sigmaNew'] = ST['sigmaNew'][:, i]
            plot_dist(temp, plot_type, samplesColored=opts['samplesColored'], nmbsamples=opts['nmbsamples'], ylim=ST['lims'][2:4, i], pbaspect=[pbxLength, pbyLength, 1], lineWidth=opts['lineWidth'] * 1, export=opts['export'])
        if 'format' in opts['export'] and opts['export']['format'] == "FIG":
            plt.title(f"seasonal component {i + 1}")

    plt.subplot(3 + numPeriods, 1, 3 + numPeriods)

    opts['export']['exportName'] = exportName + "_R"
    if 'yaxis' in opts['export']:
        if opts['export']['yaxis'] != "":
            R['lims'][2:4] = opts['export']['yaxis'][3 + numPeriods]
    if 'dashedLines' in opts['export'] and opts['export']['dashedLines'] != "":
        plt.plot(np.ones(len(y['mu'])) * opts['export']['dashedLines'][3 + numPeriods][0], linewidth=opts['lineWidth'] * 0.5, color=[0.84, 0.84, 0.84], linestyle='--')
        plt.plot(np.ones(len(y['mu'])) * opts['export']['dashedLines'][3 + numPeriods][1], linewidth=opts['lineWidth'] * 0.5, color=[0.84, 0.84, 0.84], linestyle='--')
        tempExport = opts['export']
        tempExport['dashedLines'] = opts['export']['dashedLines'][3 + numPeriods]
        plot_dist(R, plot_type, samplesColored=opts['samplesColored'], nmbsamples=opts['nmbsamples'], ylim=R['lims'][2:4], pbaspect=[pbxLength, pbyLength, 1], lineWidth=opts['lineWidth'] * 1, export=tempExport)
    else:
        plot_dist(R, plot_type, samplesColored=opts['samplesColored'], nmbsamples=opts['nmbsamples'], ylim=R['lims'][2:4], pbaspect=[pbxLength, pbyLength, 1], lineWidth=opts['lineWidth'] * 1, export=opts['export'])
    if 'format' in opts['export'] and opts['export']['format'] == "FIG":
        plt.title("residual component")

    if opts['export']['export'] and 'format' in opts['export'] and opts['export']['format'] == "FIG":
        curFig = plt.gcf()
        curFig.set_size_inches(16, 9)
        plt.savefig(f"{opts['export']['exportPath']}{exportName}.png", bbox_inches='tight', dpi=300)

    # Change exportName to original input opts.export.exportName
    opts['export']['exportName'] = exportName
    if opts['plotCov']:
        plotCov(yLTSTR, numPeriods, opts)

    if opts['plotCor']:
        yLTSTR = plotCor(yLTSTR, numPeriods, opts)

    if opts['plotCorLength']:
        plotCorLength(yLTSTR, numPeriods, opts)
        if opts['export']['export'] and 'format' in opts['export'] and opts['export']['format'] == "FIG":
            curFig = plt.gcf()
            curFig.set_size_inches(16, 9)
            plt.savefig(f"{opts['export']['exportPath']}{opts['export']['exportName']}_CorLength.png", bbox_inches='tight', dpi=300)

    plt.tight_layout()
    plt.show()
    print("#####")
    print("##### Plotting DONE")
    print("#####")
    print("#################################################################################")
    return yLTSTR

# HELPER FUNCTIONS

def plotCov(yLTSTR, numPeriods, opts):
    """
    This function PLOTCOV creates a plot of the covariance matrix
    yLTSTR.Sigma
    """
    nmbColors = opts['discrNmb']
    colorLut = color_lut(1)
    colorLut = colorLut[np.round(np.linspace(0, 256, nmbColors)).astype(int), :]
    plt.figure()
    if not opts['export']['export']:
        plt.suptitle('Covariance Matrix')
    plt.imshow(yLTSTR.Sigma, cmap=colorLut)
    plt.colorbar()

    # White lines between block matrices y, LT, STs, R
    sublength = len(yLTSTR.mu) // (numPeriods + 3)
    plt.hold(True)
    for i in range(1, numPeriods + 2):
        plt.plot([(i * sublength + 0.5)] * len(yLTSTR.mu), linestyle=':', color='white', linewidth=opts['lineWidth'])
        plt.plot([(i * sublength + 0.5)] * len(yLTSTR.mu), np.linspace(1, len(yLTSTR.mu), len(yLTSTR.mu)), linestyle=':', color='white', linewidth=opts['lineWidth'])

    max_el = np.max(np.abs(yLTSTR.Sigma))
    if abs(max_el) > 0:
        plt.clim(-max_el, max_el)

    if opts['export']['export']:
        if 'exportName' in opts['export']:
            exportName = opts['export']['exportName']
        else:
            exportName = "plot_dist"
        opts['export']['exportName'] = f"{exportName}_CovPlot"
        opts['export']['pbaspect'] = [1, 1, 1]
        exportPlot(opts)

def plotCor(yLTSTR, numPeriods, opts):
    """
    This function PLOTCOR creates a plot of the correlation matrix using the
    covariance matrix yLTSTR.Sigma and the function corComp that transforms
    the covariance matrix to the correlation matrix.
    This is part of the correlation exploration for UASTL.
    """
    yLTSTR = corComp(yLTSTR)
    nmbColors = opts['discrNmb']
    colorLut = color_lut(1)
    colorLut = colorLut[np.round(np.linspace(0, 256, nmbColors)).astype(int), :]

    # Create a custom colormap
    cmap = mcolors.ListedColormap(colorLut)

    plt.figure()
    if not opts['export']['export']:
        plt.suptitle('Correlation Matrix')
    plt.imshow(yLTSTR.corMat, cmap=cmap)
    plt.colorbar()

    # White lines between block matrices y, LT, STs, R
    sublength = len(yLTSTR.mu) // (numPeriods + 3)

    for i in range(1, numPeriods + 3):
        plt.plot([(i - 1) * sublength + 0.5 + sublength] * len(yLTSTR.mu), np.arange(len(yLTSTR.mu)), ':', color='white', linewidth=opts['lineWidth'])
        plt.plot([(i - 1) * sublength + sublength + 0.5] * len(yLTSTR.mu), np.linspace(0, len(yLTSTR.mu) - 1, len(yLTSTR.mu)), ':', color='white', linewidth=opts['lineWidth'])

    plt.clim(-1, 1)
    if opts['export']['export']:
        if 'exportName' in opts['export']:
            exportName = opts['export']['exportName']
        else:
            exportName = "plot_dist"
        opts['export']['exportName'] = f"{exportName}_CorPlot"
        opts['export']['pbaspect'] = [1, 1, 1]
        exportPlot(opts)
    return yLTSTR

def corComp(yLTSTR):
    """
    This function CORCOMP computes the correlation matrix given the
    covariance matrix yLTSTR.Sigma.
    """
    cInvMult = np.zeros(yLTSTR.Sigma.shape[0])
    for k in range(yLTSTR.Sigma.shape[0]):
        c = yLTSTR.Sigma[k, k]
        if c < 1e-12:
            cInvMult[k] = 0
        else:
            cInvMult[k] = 1 / np.sqrt(c)
    yLTSTR.corMat = np.diag(cInvMult) @ yLTSTR.Sigma @ np.diag(cInvMult)
    return yLTSTR

def plotCorLength(yLTSTR, numPeriods, opts):
    """
    This function PLOTCORLENGTH creates a plot of the correlation length for
    each component of the correlation matrix.
    This is part of the correlation exploration for UASTL.
    """
    sbplotNmb = numPeriods + 3
    lngthData = len(yLTSTR.mu) // sbplotNmb
    if 'corMat' not in yLTSTR:
        yLTSTR = corComp(yLTSTR)

    signCorMat = np.sign(yLTSTR['corMat'])
    maxdim = len(signCorMat)
    FullCorLength = np.zeros(maxdim)
    VertR = np.zeros(maxdim)
    VertL = np.zeros(maxdim)
    for i in range(maxdim):
        curSignEl = signCorMat[i, i]
        if abs(curSignEl) > 0:
            vert1 = 1
            vert2 = 1
            decB = i % lngthData

            if decB == 0:
                varR = 0
                varL = lngthData
            elif decB < lngthData / 2 or decB > lngthData / 2:
                varR = lngthData - decB
                varL = decB
            else:
                varR = lngthData // 2
                varL = lngthData // 2

            for j in range(1, varR):
                if curSignEl == signCorMat[i, i + j]:
                    vert1 += 1
                else:
                    break
            for j in range(1, varL):
                if curSignEl == signCorMat[i, i - j]:
                    vert2 += 1
                else:
                    break
            VertR[i] = vert1
            VertL[i] = vert2
            FullCorLength[i] = vert1 + vert2 - 1

    nmbColors = opts['discrNmb'] // 2 + 1
    out = [None] * len(yLTSTR['corMat'])
    for i in range(len(yLTSTR['corMat'])):
        ColInd = np.full(lngthData, np.nan)
        k = 2
        if VertL[i] >= 1 or VertR[i] >= 1:
            for j in range(VertL[i] - 1, 0, -1):
                ColInd[k] = min(int(np.ceil(yLTSTR['corMat'][i, i - j] * nmbColors)), nmbColors)
                k += 1
            ColInd[k] = min(int(np.ceil(yLTSTR['corMat'][i, i] * nmbColors)), nmbColors)
            k += 1
            for j in range(1, VertR[i]):
                ColInd[k] = min(int(np.ceil(yLTSTR['corMat'][i, i + j] * nmbColors)), nmbColors)
                k += 1
            ColInd = ColInd[~np.isnan(ColInd)]
            out[i] = ColInd

    plotmat = np.zeros((len(yLTSTR['corMat']), lngthData))
    ysize = np.zeros(len(out))
    for i in range(len(out)):
        if out[i] is not None:
            insert = out[i]
            ysize[i] = len(insert)
            plotmat[i, :len(insert)] = insert

    plotmat = np.flipud(plotmat.T)

    colors = color_lut(1)
    colors = colors[128:256, :]
    colors = colors[np.round(np.linspace(0, 127, nmbColors)).astype(int), :]
    colors = np.vstack(([1, 1, 1], colors))

    plt.figure()
    if not opts['export']['export']:
        plt.suptitle('Correlation Length')
    for i in range(sbplotNmb):
        plt.subplot(sbplotNmb, 1, i + 1)
        maxI = max(ysize[opts['stamp_indices'][(i * opts['StmpLength']):((i + 1) * opts['StmpLength'])]])
        image2plot = plotmat[-maxI:, opts['stamp_indices'][(i * opts['StmpLength']):((i + 1) * opts['StmpLength'])]]
        plt.imshow(image2plot, cmap=colors)
        plt.hold(True)
        VertLs = np.repeat(VertL[opts['stamp_indices'][(i * opts['StmpLength']):((i + 1) * opts['StmpLength'])]], 2)
        xs = np.repeat(np.linspace(1, opts['StmpLength'] + 1, opts['StmpLength'] + 1), 2)
        xs = xs[:-1]
        plt.plot(xs - 0.5, maxI + 1 - VertLs, linewidth=2, color='black', linestyle='-')
        if 'format' in opts['export'] and opts['export']['format'] == "FIG":
            if i == 0:
                plt.title("input data")
            elif i == 1:
                plt.title("trend component")
            elif i == sbplotNmb - 1:
                plt.title("residual component")
            else:
                plt.title(f"seasonal component {i - 1}")

    if opts['export']['export']:
        if 'exportName' in opts['export']:
            exportName = opts['export']['exportName']
        else:
            exportName = "plot_dist"
        for i in range(sbplotNmb):
            plt.subplot(sbplotNmb, 1, i + 1)
            opts['export']['exportName'] = f"{exportName}_CorLength_{i + 1}"
            exportSubPlot(opts)

def exportSubPlot(opts):
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
            plt.savefig(f"{opts['export']['exportPath']}{opts['export']['exportName']}.pdf", format='pdf', bbox_inches='tight', transparent=True)
    elif opts['export']['format'] == "FIG":
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(True)

def exportPlot(opts):
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
        plt.savefig(f"{opts['export']['exportPath']}{opts['export']['exportName']}.pdf", format='pdf', bbox_inches='tight', transparent=True)

def color_lut(number):
    """
    Imports the used color scheme generated via the code by Andy Stein and
    based on Kenneth Moreland's code for creating diverging colormaps.
    """
    A = pd.read_csv('CoolWarmFloat257.csv')
    if number == 1:
        return diverging_map(A.iloc[:, 0].values, [0.230, 0.299, 0.754], [0.706, 0.016, 0.150])  # blue-white-red
    return None
