import random
import numpy as np
import matplotlib.pyplot as plt

def gaussian_random() -> float:
    return random.gauss(0.0, 1.0)

def test_gaussian(generator: callable, n_samples: int=100000) -> tuple[float, float]:
    values = [generator() for _ in range(n_samples)]
    mean = sum(values) / n_samples
    var = sum((x - mean)**2 for x in values) / (n_samples - 1)
    std = var**0.5
    return mean, std

def plot_gaussian_histogram(generator: callable, n_samples: int=100000, bins: int=60)-> tuple[float, float]:
    samples = np.array([generator() for _ in range(n_samples)])

    mean = np.mean(samples)
    std = np.std(samples, ddof=1)

    plt.figure()
    plt.hist(samples, bins=bins, density=True)
    plt.xlabel("ξ̂")
    plt.ylabel("Probability density")
    plt.title(f"mean={mean:.3f}, std={std:.3f}")
    plt.tight_layout()
    plt.show()

    return mean, std