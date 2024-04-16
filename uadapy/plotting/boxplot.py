import matplotlib.pyplot as plt
import uadapy.distribution as dist
from math import ceil, sqrt

def plot_boxplot(distribution, num_samples, labels=None, titles=None, **kwargs):
    """
    Plot box plots for samples drawn from given distributions.
    :param distributions: Distributions to plot
    :param num_samples: Number of samples per distribution
    :param labels: Labels for each distribution
    :param titles: Titles for each subplot
    :param kwargs: Additional optional plotting arguments
    """
    samples = []

    if isinstance(distribution, dist.distribution):
        distribution = [distribution]

    # Calculate the layout of subplots
    if (titles == None):
        num_rows = num_cols = 2
        num_plots = 1
    else:
        num_plots = len(titles)
        num_rows = ceil(sqrt(num_plots))
        num_cols = ceil(num_plots / num_rows)

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(10, 10))

    for d in distribution:
        samples.append(d.sample(num_samples))

    for i, ax_row in enumerate(axs):
        for j, ax in enumerate(ax_row):
            index = i * num_cols + j
            if index < num_plots:
                for k, sample in enumerate(samples):
                    ax.boxplot(sample[index], positions=[k], patch_artist=True)
                if labels:
                    ax.set_xticks(range(len(labels)))
                    ax.set_xticklabels(labels)
                if titles:
                    ax.set_title(titles[index] if titles and index < len(titles) else 'Comparison')
            else:
                ax.set_visible(False)  # Hide unused subplots
    
    fig.tight_layout()
    plt.show()