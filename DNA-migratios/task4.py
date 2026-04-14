import random

def gaussian_random():
    return random.gauss(0.0, 1.0)

def test_gaussian(generator, n_samples=100000):
    values = [generator() for _ in range(n_samples)]
    mean = sum(values) / n_samples
    var = sum((x - mean)**2 for x in values) / (n_samples - 1)
    std = var**0.5
    return mean, std