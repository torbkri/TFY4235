import random

def gaussian_random() -> float:
    return random.gauss(0.0, 1.0)

def test_gaussian(generator: callable, n_samples: int=100000) -> tuple[float, float]:
    values = [generator() for _ in range(n_samples)]
    mean = sum(values) / n_samples
    var = sum((x - mean)**2 for x in values) / (n_samples - 1)
    std = var**0.5
    return mean, std