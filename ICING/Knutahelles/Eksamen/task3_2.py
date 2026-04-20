import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from tqdm import tqdm


def set_seed(seed=69):
    np.random.seed(seed)


def intial_spins(L):
    return np.random.choice([-1, 1], size=(L, L))


@njit
def dE_H(i, j, spins, H, J=1, mu=1):
    nx, ny = spins.shape
    s = -spins[i, j]

    bond_term = -2 * J * s * (
        spins[i-1, j]
        + spins[(i+1) % nx, j]
        + spins[i, j-1]
        + spins[i, (j+1) % ny]
    )

    mag_term = - 2 * mu * H * s

    return bond_term + mag_term


@njit
def sweep_H(spins, T, H, spin_sum, rand_i, rand_j, rand_r):
    N = spins.size

    for k in range(N):

        i = rand_i[k]
        j = rand_j[k]
        r = rand_r[k]

        dE_ij = dE_H(i, j, spins, H)
        if np.exp(-dE_ij / T) > r:
            spin_sum -= 2 * spins[i, j]
            spins[i, j] = -spins[i, j]

    return spins, spin_sum


def run_metropolis_H(L, n_sweeps, T, H, spins=None):
    if spins is None:
        spins = intial_spins(L)

    spin_sum = np.sum(spins)
    spin_sums = np.zeros(n_sweeps)

    for sweep_idx in range(n_sweeps):
        N = spins.size
        rand_i = np.random.randint(0, L, size=N)
        rand_j = np.random.randint(0, L, size=N)
        rand_r = np.random.random(size=N)

        spins, spin_sum = sweep_H(
            spins, T, H, spin_sum, rand_i, rand_j, rand_r)
        spin_sums[sweep_idx] = spin_sum

    return spins, spin_sums/spins.size



# the same error as 3.1, forgot to account for system size in the susceptibility calculation. fixed by scaling later
def find_susceptiblity(L, ave, n_sweeps, T_min, T_max, T_res, H=0.01):
    if ave > n_sweeps - 1000:
        raise ValueError("Average must be less than n_sweeps - 1000")
    T_range = np.linspace(T_min, T_max, T_res)
    chi_values = np.zeros(T_res)
    M_values = np.zeros(T_res)

    for i in tqdm(range(T_res)):
        T = T_range[i]
        _, M = run_metropolis_H(L, n_sweeps, T, H=H)

        M = M[-ave:]
        chi_values[i] = np.var(M) / T
        M_values[i] = np.mean(M)

    return T_range, chi_values, M_values


if __name__ == "__main__":
    set_seed()
