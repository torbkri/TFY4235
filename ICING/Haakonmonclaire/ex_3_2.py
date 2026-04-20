import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from tqdm import tqdm

from plotting import *
from ex_3_1 import create_lattice

@jit(nopython = True)
def calculate_energy_change_w_H(i, j, lattice, H):
    J = 1  # Interaction strength
    mu = 1 # Bohr magneton

    flipped_current_spin = -lattice[i, j]

    up_neighbor = lattice[i, j-1]
    down_neighbor = lattice[i, (j+1) % lattice.shape[1]]
    left_neighbor = lattice[i-1, j]
    right_neighbor = lattice[(i+1) % lattice.shape[0], j]

    neighbor_sum = up_neighbor + down_neighbor + left_neighbor + right_neighbor

    energy_change = -2 * J * flipped_current_spin * neighbor_sum - mu * H * flipped_current_spin
    
    return energy_change 

@jit(nopython = True)
def sweep_lattice_w_H(lattice, T, H):
    for n in range(lattice.size):
        i = np.random.randint(0, lattice.shape[0])
        j = np.random.randint(0, lattice.shape[1])

        r = np.random.uniform(0, 1)

        energy_change = calculate_energy_change_w_H(i, j, lattice, H)

        P_star = np.exp(-energy_change / T)

        if P_star > r:
            lattice[i, j] = -lattice[i, j]

    return lattice

# Sweep the lattice n_sweeps times at a given T
def run_sweeps_w_H(lattice, T, n_sweeps, H):
    new_lattice = lattice.copy()
    for n in range(n_sweeps):
        new_lattice = sweep_lattice_w_H(new_lattice, T, H)
    
    return new_lattice

# For exercise 3.2.1
def calculate_M_time_evolution_snapshots_w_H(lattice, T, n_sweeps, H, snapshotvals):
    new_lattice = lattice.copy()
    M = np.zeros(n_sweeps)

    lattice_snapshots = []

    for n in range(n_sweeps):
        new_lattice = sweep_lattice_w_H(new_lattice, T, H)

        M[n] = np.mean(new_lattice)

        if n in snapshotvals:
            lattice_snapshots.append(new_lattice)

    return M, new_lattice, np.array(lattice_snapshots)

# For exercise 3.2.2
def calculate_M_var_M_w_H(lattice, T, n_sweeps, H):
    new_lattice = lattice.copy()
    M = np.zeros(n_sweeps)

    for n in range(n_sweeps):
        new_lattice = sweep_lattice_w_H(new_lattice, T, H)

        M[n] = np.mean(new_lattice)

    M_var = np.var(M[-2000:]) # Calculate the variance of the last 2000 sweeps where equilibrium is met 

    return M_var, new_lattice

def get_M_var_M_temp_evolution_w_H(lattice, T_values, n_sweeps, H, directory=None, save = False):
    M_lst = np.zeros_like(T_values)
    M_var_lst = np.zeros_like(T_values)
    
    for i, T in tqdm(enumerate(T_values), total=len(T_values)):
        M_var, flipped_lattice = calculate_M_var_M_w_H(lattice, T, n_sweeps, H)
       
        M_lst[i] = np.mean(flipped_lattice) # Calculate the mean magnetization of the final lattice after n_sweeps at given T
        M_var_lst[i] = M_var

    if save:
        np.save(f'{directory}/M_Tvals{T_values.shape[0]}_L{lattice.shape[0]}_Nsweeps{n_sweeps}_H{H}.npy', M_lst)
        np.save(f'{directory}/M_var_Tvals{T_values.shape[0]}_L{lattice.shape[0]}_Nsweeps{n_sweeps}_H{H}.npy', M_var_lst)
        print(f"Saved data for {T_values.shape[0]} temperatures, L = {lattice.shape[0]}, n_sweeps = {n_sweeps} to {directory}")

    return np.abs(M_lst), M_var_lst

def calculate_Chi(M_var_lst, T_values):
    kb = 1
    return M_var_lst / (kb * T_values)

# For exercise 3.2.3
def t(T, T_c):
    return (T - T_c) / T_c

def test(T_values, M_var_lst, gamma, beta, delta, H, T_c):
    chi = calculate_Chi(M_var_lst, T_values)

    t_vals = t(T_values, T_c)

    positive_t_mask = t_vals > 0
    negative_t_mask = t_vals < 0

    chi_t = chi * np.abs(t_vals)**(gamma)
    chi_t_pos = chi_t[positive_t_mask]
    chi_t_neg = chi_t[negative_t_mask]

    h_t = H / (np.abs(t_vals)**(beta * delta))
    h_t_pos = h_t[positive_t_mask]
    h_t_neg = h_t[negative_t_mask]

    plt.scatter(h_t_neg, chi_t_neg, label='Below $T_c$', alpha=0.7, s = 5)
    plt.scatter(h_t_pos, chi_t_pos, label='Above $T_c$', alpha=0.7, s = 5)
    plt.xscale('log')
    plt.yscale('log')
    # plt.loglog(h_t_red, chi_t_red, c = 'red', label='Above $T_c$', alpha=0.7)
    # plt.loglog(h_t_blue, chi_t_blue, c='blue', label='Below $T_c$', alpha=0.7)
    plt.legend()
    plt.xlabel(r'$h_t$', fontsize=20)
    plt.ylabel(r'$\chi_t$', fontsize=20)
    plt.show()



if __name__ == "__main__":
    L = 40
    n_sweeps = 10000

    T = 2.3
    H = 0.05

    lattice = create_lattice(L)

    T_values = np.linspace(1, 4, 300)

    directory = 'simulation_data_w_H'

    # M_lst, M_var_lst = get_M_var_M_temp_evolution_w_H(lattice, T_values, n_sweeps, H, directory=directory, save=True)

    # load data
    M_var_lst = np.load(f'{directory}/M_var_Tvals{T_values.shape[0]}_L{lattice.shape[0]}_Nsweeps{n_sweeps}_H{H}.npy')

    # Constants for task 3.2.3
    gamma = 7 / 4
    beta = 1 / 8
    delta = 15
    # h = H, bcs J and mu are set to 1
    T_c = 2.3 # approximately

    # chi = calculate_Chi(M_var_lst, T_values)
    # plt.scatter(T_values, chi, label = r'$\chi$', s = 5)
    # plt.show()

    test(T_values, M_var_lst, gamma, beta, delta, H, T_c)


   