import random
from numba import njit
import numpy as np
import matplotlib.pyplot as plt

@njit
def gaussian_random() -> float:
    return random.gauss(0.0, 1.0)

def test_gaussian(generator: callable, n_samples: int=100000) -> tuple[float, float]:
    values = [generator() for _ in range(n_samples)]
    mean = sum(values) / n_samples
    var = sum((x - mean)**2 for x in values) / (n_samples - 1)
    std = var**0.5
    return mean, std

def plot_gaussian_histogram(generator: callable, n_samples: int=10000000, bins: int=100)-> None:
    samples1 = np.array([generator() for _ in range(int(n_samples/10000))])
    samples2 = np.array([generator() for _ in range(int(n_samples/100))])
    samples3 = np.array([generator() for _ in range(n_samples)])

    x = np.linspace(-4, 4, 1000)
    pdf = (1 / np.sqrt(2 * np.pi)) * np.exp(-x**2 / 2)

    mosaic = """
    aaaa
    bbbb
    cccc
    """

    fig,axs = plt.subplot_mosaic(mosaic=mosaic, figsize = (8,9), sharex=True)
    for _ in axs:
        axs[_].set_xticks([-4, -2, 0, 2, 4])
        axs[_].set_yticks([0,0.2,0.4,0.6])

    axs['a'].hist(samples1, bins=bins, density=True, label=f"RNG")
    axs['a'].plot(x, pdf, 'r-', linewidth=2, label="Exact")
    axs['a'].legend(loc=1)


    axs['b'].hist(samples2, bins=bins, density=True,label=f"RNG")
    axs['b'].plot(x, pdf, 'r-', linewidth=2, label = "Exact")
    axs['b'].set_ylabel("$P_x$", fontsize = 24)
    axs['b'].legend(loc=1)


    axs['c'].hist(samples3, bins=bins, density=True,label=f"RNG")
    axs['c'].plot(x, pdf, 'r-', linewidth=2, label="Exact")
    axs['c'].set_xlabel("$x$", fontsize = 24)
    axs['c'].legend(loc=1)

    fig.tight_layout()
    fig.savefig("/Users/torbjornkringeland/Desktop/NTNU/6/TFY4235/DNA-migratios/DNA-report/Images/gaussian_histogram.pgf")