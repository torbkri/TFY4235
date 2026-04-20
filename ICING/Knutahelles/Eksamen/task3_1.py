import numpy as np
from numba import njit
from tqdm import tqdm
import matplotlib.pyplot as plt


# set seed is for testing purposes
def set_seed(seed=69):
    np.random.seed(seed)


def intial_spins(L):
    return np.random.choice([-1, 1], size=(L, L))


@njit
def dE(i, j, spins, J=1):
    nx, ny = spins.shape
    s = -spins[i, j]

    return -2 * J * s * (
        spins[i-1, j]
        + spins[(i+1) % nx, j]
        + spins[i, j-1]
        + spins[i, (j+1) % ny]
    )


@njit
def sweep(spins, T, spin_sum, total_E, rand_i, rand_j, rand_r):
    N = spins.size

    for k in range(N):

        i = rand_i[k]
        j = rand_j[k]
        r = rand_r[k]

        dE_ij = dE(i, j, spins)
        if np.exp(-dE_ij / T) > r:
            spin_sum -= 2 * spins[i, j]
            spins[i, j] = -spins[i, j]
            total_E += dE_ij
    return spins, spin_sum, total_E


@njit
def sweep_forbidden_sites(spins, T, spin_sum, total_E, forbidden_mask,
                          rand_i, rand_j, rand_r):
    N = spins.size

    for k in range(N):
        i = rand_i[k]
        j = rand_j[k]

        if forbidden_mask[i, j]:
            continue

        r = rand_r[k]
        dE_ij = dE(i, j, spins)

        if np.exp(-dE_ij / T) > r:
            spin_sum -= 2 * spins[i, j]
            spins[i, j] = -spins[i, j]
            total_E += dE_ij

    return spins, spin_sum, total_E


def create_forbidden_mask(L, p):
    n_forbidden = int(p * L * L)
    mask = np.zeros((L, L), dtype=bool)
    indices = np.array([(i, j) for i in range(L) for j in range(L)])

    np.random.shuffle(indices)

    for (i, j) in indices[:n_forbidden]:
        mask[i, j] = True

    return mask


@njit
def total_energy(spins, J=1):
    L = spins.shape[0]
    energy = 0

    for i in range(L):
        for j in range(L):
            energy -= J * spins[i, j] * (
                spins[(i+1) % L, j] + spins[i, (j+1) % L]
            )
    return energy


def run_metropolis(L, n_sweeps, T, spins=None):
    if spins is None:
        spins = intial_spins(L)

    total_E = total_energy(spins)
    spin_sum = np.sum(spins)
    spin_sums = np.zeros(n_sweeps)
    total_Es = np.zeros(n_sweeps)

    for sweep_idx in tqdm(range(n_sweeps), leave=False):
        N = spins.size
        rand_i = np.random.randint(0, L, size=N)
        rand_j = np.random.randint(0, L, size=N)
        rand_r = np.random.random(size=N)

        spins, spin_sum, total_E = sweep(
            spins, T, spin_sum, total_E, rand_i, rand_j, rand_r)
        spin_sums[sweep_idx] = spin_sum
        total_Es[sweep_idx] = total_E

    return spins, spin_sums/spins.size, total_Es/spins.size


def run_metropolis_forbidden_sites(L, n_sweeps, T, p, spins=None, forbidden_mask=None):
    if spins is None:
        spins = intial_spins(L)
    if forbidden_mask is None:
        forbidden_mask = create_forbidden_mask(L, p)
    total_E = total_energy(spins)
    spin_sum = np.sum(spins)
    spin_sums = np.zeros(n_sweeps)
    total_Es = np.zeros(n_sweeps)

    for sweep_idx in range(n_sweeps):
        N = spins.size
        rand_i = np.random.randint(0, L, size=N)
        rand_j = np.random.randint(0, L, size=N)
        rand_r = np.random.random(size=N)

        spins, spin_sum, total_E = sweep_forbidden_sites(
            spins, T, spin_sum, total_E, forbidden_mask, rand_i, rand_j, rand_r)
        spin_sums[sweep_idx] = spin_sum
        total_Es[sweep_idx] = total_E

    return spins, spin_sums/spins.size, total_Es/spins.size


# here i forgot that i normalise mag and E, but i multiply by N^2 later to account for the difference in variance
def find_C(L, cutoff=1000, n_sweeps=10000, T_res=300):
    T_range = np.linspace(2, 3, T_res)
    C = np.zeros(T_res)

    for i in tqdm(range(T_res)):
        T = T_range[i]
        _, _, E = run_metropolis(L, n_sweeps, T)

        E = E[cutoff:]
        C[i] = np.var(E) / (T**2)

    T_c = np.argmax(C)
    return T_range, C, T_c


def run_simluated_annealing(L, n_sweeps, T_max, T_min, n_steps, p, forbidden_mask=None):
    spins = intial_spins(L)
    if forbidden_mask is None:
        forbidden_mask = create_forbidden_mask(L, p)
    T_range = np.linspace(T_max, T_min, n_steps)

    for i in tqdm(range(n_steps)):
        T = T_range[i]
        spins, spin_sums, total_Es = run_metropolis_forbidden_sites(
            L, n_sweeps, T, p, spins=spins, forbidden_mask=forbidden_mask)

    return spins, spin_sums[-1], total_Es[-1], forbidden_mask


if __name__ == "__main__":
    set_seed()
