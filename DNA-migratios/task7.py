
import numpy as np
import matplotlib.pyplot as plt

def boltzmann_density_reduced(u_hat, D_hat):
    return np.exp(-u_hat / D_hat) / (D_hat * (1.0 - np.exp(-1.0 / D_hat)))

def plot_energy_distribution(run_data, D_hat, bins=60, burn_in_fraction=0.2):
    u_vals = np.array(run_data["u"])
    start = int(burn_in_fraction * len(u_vals))
    u_eq = u_vals[start:]

    plt.figure()
    plt.hist(u_eq, bins=bins, density=True, alpha=0.6, label="Simulation")

    u_grid = np.linspace(0.0, 1.0, 500)
    p_grid = boltzmann_density_reduced(u_grid, D_hat)
    plt.plot(u_grid, p_grid, label="Boltzmann theory")

    plt.xlabel("Û")
    plt.ylabel("Probability density")
    plt.legend()
    plt.tight_layout()
    plt.show()