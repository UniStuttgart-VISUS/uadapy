import numpy as np
from uastl import uastl, uncertain_data, options
from plot_distributionmtx import plot_distributionmtx

# Synthetic data generation
def generate_synthetic_data(n=200):

    np.random.seed(0)
    t = np.arange(1, n + 1)
    trend = t / 10
    periodic = 10 * np.sin(2 * np.pi * t / 100)
    noise = 2 * (np.random.rand(n) - 0.5)
    mu = trend + periodic + noise
    sigma2 = 20 * np.ones(n)
    sigma_sq = np.sqrt(sigma2)
    sigma = np.zeros((n, n))
    samples = np.zeros((800,3))
    cor_mat = np.zeros((800,800))
    a_hat = np.zeros((800,200))

    def ex_qu_kernel(x, y, sigma_i, sigma_j, l):
        return sigma_i * sigma_j * np.exp(-0.5 * np.linalg.norm(x - y)**2 / l**2)
    
    for i in range(n):
        for j in range(n):
            sigma[i, j] = ex_qu_kernel(t[i], t[j], sigma_sq[i], sigma_sq[j], 5)

    return uncertain_data(mu=mu, sigma=sigma, samples=samples, cor_mat=cor_mat, a_hat=a_hat), t

def main_demo():
    X, t = generate_synthetic_data()

    # options for UASTL
    opts = options(n_s=np.array([5]), n_l=np.array([5]), n_t=5, post_smoothing_seasonal_n=np.array([5]), post_smoothing_trend_n=5)

    # Perform UASTL
    y_ltstr, a_hat_global = uastl(X, 100, opts) #np.array([100])
    y_ltstr.a_hat = a_hat_global[:, :len(y_ltstr.mu) // (1 + 3)]

    # UASTL and Correlation exploration
    export = {
        'export': False,
        'export_path': "figures/",
        'export_name': "fig_1",
        'pbaspect': [7, 1, 1],
        'format': "FIG"
        # 'yaxis': None,
        # 'dashed_lines': None
    }

    y_ltstr = plot_distributionmtx(
        y_ltstr,
        1,
        'comb',
        samples_colored=True,
        nmbsamples=3,
        plot_cov=False,
        plot_cor=True,
        plot_cor_length=False,
        co_point=100,
        export=export,
        line_width=2.5,
        discr_nmb=13
    )

    # UASTL and Sensitivity Analysis
    pos = 100
    ampl = 1
    bdwidth = 20
    delta = np.ones(len(y_ltstr.mu) // (1 + 3))
    delta[pos - bdwidth:pos + bdwidth + 1] = 1 + ampl * np.exp(-0.5 * np.abs(np.arange(1, 2 * bdwidth + 2) - (bdwidth + 1)) ** 2 / (bdwidth / 3) ** 2)

    export['export_name'] = "fig_2"

    y_ltstr = plot_distributionmtx(
        y_ltstr,
        1,
        'comb',
        samples_colored=True,
        nmbsamples=3,
        plot_cov=False,
        plot_cor=False,
        plot_cor_length=False,
        co_point=pos,
        export=export,
        line_width=2.5,
        discr_nmb=13,
        delta=delta,
        time_stamp=[50, 150]
    )

if __name__ == "__main__":
    main_demo()