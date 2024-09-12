import numpy as np
from uastl import uastl, UncertainData, Options
from plot_distributionmtx import plot_distributionmtx

# Synthetic data generation
def generate_synthetic_data(N=200):

    np.random.seed(0)
    t = np.arange(1, N + 1)
    trend = t / 10
    periodic = 10 * np.sin(2 * np.pi * t / 100)
    noise = 2 * (np.random.rand(N) - 0.5)
    mu = trend + periodic + noise
    sigma2 = 20 * np.ones(N)
    sigma = np.sqrt(sigma2)
    Sigma = np.zeros((N, N))
    samples = np.zeros((800,3))
    corMat = np.zeros((800,800))
    AHat = np.zeros((800,200))

    def ExQuKernel(x, y, sigma_i, sigma_j, l):
        return sigma_i * sigma_j * np.exp(-0.5 * np.linalg.norm(x - y)**2 / l**2)
    
    for i in range(N):
        for j in range(N):
            Sigma[i, j] = ExQuKernel(t[i], t[j], sigma[i], sigma[j], 5)

    return UncertainData(mu=mu, Sigma=Sigma, samples=samples, corMat=corMat, AHat=AHat), t

def main_demo():
    X, t = generate_synthetic_data()

    # Options for UASTL
    opts = Options(n_s=np.array([5]), n_l=np.array([5]), n_t=5, postSmoothingSeasonal_n=np.array([5]), postSmoothingTrend_n=5)

    # Perform UASTL
    yLTSTR, AHatGlobal = uastl(X, 100, opts) #np.array([100])
    yLTSTR.AHat = AHatGlobal[:, :len(yLTSTR.mu) // (1 + 3)]

    # UASTL and Correlation exploration
    export = {
        'export': False,
        'exportPath': "figures/",
        'exportName': "fig_1",
        'pbaspect': [7, 1, 1],
        'format': "FIG"
        # 'yaxis': None,
        # 'dashedLines': None
    }

    yLTSTR = plot_distributionmtx(
        yLTSTR,
        1,
        'comb',
        samplesColored=True,
        nmbsamples=3,
        plotCov=False,
        plotCor=True,
        plotCorLength=False,
        coPoint=100,
        export=export,
        lineWidth=2.5,
        discrNmb=13
    )

    # UASTL and Sensitivity Analysis
    pos = 100
    ampl = 1
    bdwidth = 20
    delta = np.ones(len(yLTSTR.mu) // (1 + 3))
    delta[pos - bdwidth:pos + bdwidth + 1] = 1 + ampl * np.exp(-0.5 * np.abs(np.arange(1, 2 * bdwidth + 2) - (bdwidth + 1)) ** 2 / (bdwidth / 3) ** 2)

    export['exportName'] = "fig_2"

    yLTSTR = plot_distributionmtx(
        yLTSTR,
        1,
        'comb',
        samplesColored=True,
        nmbsamples=3,
        plotCov=False,
        plotCor=False,
        plotCorLength=False,
        coPoint=pos,
        export=export,
        lineWidth=2.5,
        discrNmb=13,
        delta=delta,
        timeStamp=[50, 150]
    )

if __name__ == "__main__":
    main_demo()