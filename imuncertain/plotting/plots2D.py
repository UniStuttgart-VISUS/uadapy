import plotly.graph_objects as go

def plot_samples(distribution, num_samples, **kwargs):
    """
    Plot samples from the given distribution. If several distributions should be
    plotted together, an array can be passed to this function
    :param distribution: Distributions to plot
    :param num_samples: Number of samples per distribution
    :param kwargs: Any other argument for
    :return:
    """
    fig = go.Figure()
    for d in distribution:
        samples = d.sample(num_samples)
        fig.add_trace(go.Scatter(x=samples[:,0], y=samples[:,1], mode='markers'))
    fig.update_layout(template ='plotly_white')
    fig.update_layout(kwargs)
    fig.show()
