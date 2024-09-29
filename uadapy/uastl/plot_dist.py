import numpy as np
import matplotlib.pyplot as plt

def plot_dist(data, plot_type, **opts):
    """
    Plot the distribution of the given data.
    """

    opts['smpl_width'] = 0.5
    line_fact = opts.get('line_width', 1)
    sigmalvl = [0, 0.674490, 2.575829]
    own_colors = np.array([
        [228, 26, 28], [55, 126, 184], [77, 175, 74], [152, 78, 163],
        [255, 127, 0], [255, 255, 51], [166, 86, 40], [251, 180, 174],
        [179, 205, 227], [204, 235, 197], [222, 203, 228], [254, 217, 166],
        [255, 255, 204], [229, 216, 189]
    ]) / 255.0
    col = np.zeros((len(sigmalvl) + 2, 3))
    col[:, 0] = np.linspace(own_colors[1, 0], 1, len(sigmalvl) + 2)
    col[:, 1] = np.linspace(own_colors[1, 1], 1, len(sigmalvl) + 2)
    col[:, 2] = np.linspace(own_colors[1, 2], 1, len(sigmalvl) + 2)

    if 'ylim' not in opts:
        max_sigmalvl = np.max(sigmalvl)
        ceil_max_sigmalvl = np.ceil(max_sigmalvl + 1)
        lower_bound = np.min(data['mu'] - ceil_max_sigmalvl * data['sigma_sq'])
        upper_bound = np.max(data['mu'] + ceil_max_sigmalvl * data['sigma_sq'])
        axbounds = [1, len(data['mu']), lower_bound, upper_bound]
    if 'ylim' in opts:
        all_zeros = np.all(opts['ylim'] == 0)
        if all_zeros:
            max_sigmalvl = np.max(sigmalvl)
            ceil_max_sigmalvl = np.ceil(max_sigmalvl + 1)
            lower_bound = np.min(data['mu'] - ceil_max_sigmalvl * data['sigma_sq'])
            upper_bound = np.max(data['mu'] + ceil_max_sigmalvl * data['sigma_sq'])
            axbounds = [1, len(data['mu']), lower_bound, upper_bound]
        else:
            axbounds = [1, len(data['mu']), opts['ylim'][0], opts['ylim'][1]]

    if plot_type == "isoband":
        x = np.arange(1, len(data['mu']) + 1)
        for j in range(len(sigmalvl) - 1, 1, -1):
            pos_cont = data['mu'] + sigmalvl[j] * data['sigma_sq']
            neg_cont = data['mu'] - sigmalvl[j] * data['sigma_sq']
            for i in range(len(data['mu']) - 1):
                xp = [x[i], x[i + 1], x[i + 1], x[i]]
                yp = [neg_cont[i], neg_cont[i + 1], pos_cont[i + 1], pos_cont[i]]
                plt.fill(xp, yp, color=col[j, :], edgecolor='none', alpha=0.5)
        plt.plot(x, data['mu'], color=col[0, :], linewidth=line_fact * 1.5)
    elif plot_type == "spaghetti":
        initial_color_order = np.array([
            [0.9290, 0.6940, 0.1250], [0.4660, 0.6740, 0.1880],
            [0.4940, 0.1840, 0.5560], [0.6350, 0.0780, 0.1840],
            [0.8500, 0.3250, 0.0980], [0.0, 0.4470, 0.7410],
            [0.3010, 0.7450, 0.9330]
        ])
        if opts['nmbsamples'] > len(initial_color_order):
            initial_color_order = np.tile(initial_color_order, (int(np.ceil(opts['nmbsamples'] / len(initial_color_order))), 1))
        initial_color_order = initial_color_order[:opts['nmbsamples']]
        colors = initial_color_order

        samples_2_plot = data['samples'][:, :opts['nmbsamples']]
        if data['samples'].shape[1] == 2 * opts['nmbsamples']:
            samples_2_plot_shift = data['samples'][:, opts['nmbsamples']:2 * opts['nmbsamples']]
            if not opts['samples_colored']:
                h2a = plt.plot(samples_2_plot, color=own_colors[1, :], linewidth=line_fact * opts['smpl_width'])
                h2a_shift = plt.plot(samples_2_plot_shift, color=own_colors[1, :], linewidth=line_fact * opts['smpl_width'] * 0.2, linestyle='-')
                h2a_shift_dot = plt.plot(samples_2_plot_shift, color=own_colors[1, :], linewidth=line_fact * opts['smpl_width'], linestyle=':')
            else:
                for i in range(opts['nmbsamples']):
                    plt.plot(samples_2_plot[:, i], linewidth=line_fact * opts['smpl_width'], color=colors[i, :])
                    plt.plot(samples_2_plot_shift[:, i], linewidth=line_fact * opts['smpl_width'] * 0.2, color=colors[i, :], linestyle='-')
                    plt.plot(samples_2_plot_shift[:, i], linewidth=line_fact * opts['smpl_width'], color=colors[i, :], linestyle=':')
            if not opts['samples_colored'] and opts['nmbsamples'] > 1:
                for h in h2a:
                    h.set_alpha(0.5)
                for h in h2a_shift:
                    h.set_alpha(0.5)
                for h in h2a_shift_dot:
                    h.set_alpha(0.5)
        else:
            if not opts['samples_colored']:
                h2a = plt.plot(samples_2_plot, color=own_colors[1, :], linewidth=line_fact * opts['smpl_width'])
            else:
                for i in range(opts['nmbsamples']):
                    plt.plot(samples_2_plot[:, i], linewidth=line_fact * opts['smpl_width'], color=colors[i, :])
            if not opts['samples_colored'] and opts['nmbsamples'] > 1:
                for h in h2a:
                    h.set_alpha(0.5)
    elif plot_type == "comb":
        plot_dist(data, "isoband", **opts)
        if "sigma_new" in data:
            flattened_mu = data['mu'].flatten()
            plt.plot(np.arange(len(data['mu'])), flattened_mu + sigmalvl[1] * data['sigma_new'], color=[*col[1, :], 0.5], linewidth=line_fact * 0.25, linestyle='-')
            plt.plot(np.arange(len(data['mu'])), flattened_mu - sigmalvl[1] * data['sigma_new'], color=[*col[1, :], 0.5], linewidth=line_fact * 0.25, linestyle='-')

            plt.plot(np.arange(len(data['mu'])), flattened_mu + sigmalvl[2] * data['sigma_new'], color=[*col[2, :], 0.5], linewidth=line_fact * 0.25, linestyle='-')
            plt.plot(np.arange(len(data['mu'])), flattened_mu - sigmalvl[2] * data['sigma_new'], color=[*col[2, :], 0.5], linewidth=line_fact * 0.25, linestyle='-')

            plt.plot(np.arange(len(data['mu'])), flattened_mu + sigmalvl[1] * data['sigma_new'], color=[*col[1, :], 0.5], linewidth=line_fact * 0.5, linestyle=':')
            plt.plot(np.arange(len(data['mu'])), flattened_mu - sigmalvl[1] * data['sigma_new'], color=[*col[1, :], 0.5], linewidth=line_fact * 0.5, linestyle=':')

            plt.plot(np.arange(len(data['mu'])), flattened_mu + sigmalvl[2] * data['sigma_new'], color=[*col[2, :], 0.5], linewidth=line_fact * 0.5, linestyle=':')
            plt.plot(np.arange(len(data['mu'])), flattened_mu - sigmalvl[2] * data['sigma_new'], color=[*col[2, :], 0.5], linewidth=line_fact * 0.5, linestyle=':')
        plt.plot(np.arange(len(data['mu'])), data['mu'], color=col[0, :], linewidth=line_fact)
        plot_dist(data, "spaghetti", **opts)
        if opts['export']['export']:
            opts['axbounds'] = plt.axis()
            export_subplot(opts)
        return

    if opts['export']['export']:
        opts['axbounds'] = plt.axis()
        export_subplot(opts)
    if axbounds[2] != axbounds[3]:
        plt.axis(axbounds)

def export_subplot(opts):
    """
    Export the current subplot using the specified export options.
    """
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    if 'pbaspect' in opts['export']:
        ax.set_aspect(opts['export']['pbaspect'])
    else:
        ax.set_aspect('equal')

    ax.set_axis_off()

    if opts['export']['export']:
        if opts['export']['format'] == 'SUBs':
            plt.savefig(f"{opts['export']['export_path']}{opts['export']['export_name']}.pdf", bbox_inches='tight', pad_inches=0, transparent=True)
        elif opts['export']['format'] == 'FIG':
            ax.set_xticks(np.arange(0, 1, step=0.1))
            ax.set_yticks(np.arange(0, 1, step=0.1))
            ax.set_frame_on(True)
            plt.savefig(f"{opts['export']['export_path']}{opts['export']['export_name']}.png", bbox_inches='tight', pad_inches=0)
