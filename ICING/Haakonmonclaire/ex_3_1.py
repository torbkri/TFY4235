import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patheffects as patheffects
from numba import jit
from tqdm import tqdm

from plotting import *

def create_lattice(L):
    lattice = np.random.choice([-1, 1], size=(L, L))
    return lattice

@jit(nopython=True)
def calculate_total_energy(lattice):
    J = 1 
    energy = 0.0
    L = lattice.shape[0]
    for i in range(L):
        for j in range(L):
            s = lattice[i, j]

            right = lattice[(i + 1) % L, j]
            down = lattice[i, (j + 1) % L]
            energy += -J * s * (right + down)
    return energy

@jit(nopython = True)
def calculate_energy_change(i, j, lattice):
    J = 1  # Interaction strength

    flipped_current_spin = -lattice[i, j]

    up_neighbor = lattice[i, j-1]
    down_neighbor = lattice[i, (j+1) % lattice.shape[1]]
    left_neighbor = lattice[i-1, j]
    right_neighbor = lattice[(i+1) % lattice.shape[0], j]

    neighbor_sum = up_neighbor + down_neighbor + left_neighbor + right_neighbor

    energy_change = -2 * J * flipped_current_spin * neighbor_sum
    
    return energy_change 

@jit(nopython = True)
def sweep_lattice(lattice, T, energy):
    for n in range(lattice.size):
        # Pick random site
        i = np.random.randint(0, lattice.shape[0])
        j = np.random.randint(0, lattice.shape[1])

        # Get number for metropolis algorithm
        r = np.random.uniform(0, 1)

        energy_change = calculate_energy_change(i, j, lattice)

        P_star = np.exp(-energy_change / T)

        if P_star > r:
            lattice[i, j] = -lattice[i, j]
            energy += energy_change

    return lattice, energy

# Sweep the lattice n_sweeps times at a given T
def run_sweeps(lattice, T, n_sweeps, energy):
    new_lattice = lattice.copy()
    for n in range(n_sweeps):
        new_lattice, energy = sweep_lattice(new_lattice, T, energy)
    
    return new_lattice, energy

def run_sweeps_calculate_M_E_time_evolution(lattice, T, n_sweeps, energy):
    new_lattice = lattice.copy()
    M = np.zeros(n_sweeps)
    E = np.zeros(n_sweeps)

    for n in range(n_sweeps):
        new_lattice, energy = sweep_lattice(new_lattice, T, energy)

        M[n] = np.mean(new_lattice)
        E[n] = energy

    return M, E, new_lattice, energy

def get_M_E_time_evolution_3_temps(lattice, temps, n_sweeps, energy):
    M_lst = []
    E_lst = []
    lattice_lst = []
    for i, T in enumerate(temps):
        M, E, lattice_temp, energy = run_sweeps_calculate_M_E_time_evolution(lattice, T, n_sweeps, energy)
        M_lst.append(M)
        E_lst.append(E)
        lattice_lst.append(lattice_temp)

    return np.array(M_lst), np.array(E_lst), np.array(lattice_lst)

# Sweeps the lattice for a range of temperatures
def get_M_E_temp_evolution(lattice, T_values, n_sweeps, snapshotvals, directory=None, save = False):
    M_lst = np.zeros_like(T_values)
    E_final_lst = np.zeros_like(T_values)
    E_var_lst = np.zeros_like(T_values)

    lattice_snapshots = np.zeros((len(snapshotvals), lattice.shape[0], lattice.shape[1]))

    snapshot_indices = [np.argmin(np.abs(T_values - T_snap)) for T_snap in snapshotvals]

    snapshot_counter = 0 

    for i, T in tqdm(enumerate(T_values), total=len(T_values)):
        energy = calculate_total_energy(lattice)

        M, E, flipped_lattice, energy = run_sweeps_calculate_M_E_time_evolution(lattice, T, n_sweeps, energy) #This M is not used, just from last task

        M_lst[i] = np.mean(flipped_lattice)
        E_final_lst[i] = energy
        E_var_lst[i] = np.var(E[-2000:]) # Calculate the variance of the last 2000 sweeps to ensure equilibrium is met

        if i in snapshot_indices:
            lattice_snapshots[snapshot_counter] = flipped_lattice.copy()
            snapshot_counter += 1
  
    if save:
        np.save(f'{directory}/snapshots{snapshotvals}_Tvals{T_values.shape[0]}_L{lattice.shape[0]}_Nsweeps{n_sweeps}.npy', lattice_snapshots)
        np.save(f'{directory}/M_Tvals{T_values.shape[0]}_L{lattice.shape[0]}_Nsweeps{n_sweeps}.npy', M_lst)
        np.save(f'{directory}/E_final_Tvals{T_values.shape[0]}_L{lattice.shape[0]}_Nsweeps{n_sweeps}.npy', E_final_lst)
        np.save(f'{directory}/E_var_Tvals{T_values.shape[0]}_L{lattice.shape[0]}_Nsweeps{n_sweeps}.npy', E_var_lst)
        print(f"Saved data for {T_values.shape[0]} temperatures, L = {lattice.shape[0]}, n_sweeps = {n_sweeps}")  
    
    return np.abs(M_lst), E_final_lst, E_var_lst, lattice_snapshots

def calculate_specific_heat(E_var_lst, T_values):
    kb = 1
    return E_var_lst / (kb * T_values**2)

def find_Tc(T_values, C):
    mask = T_values > 2.12

    C_above_2 = C[mask]
    T_above_2 = T_values[mask]

    max_index = np.argmax(C_above_2)
    
    Tc = T_above_2[max_index]
    
    return Tc





if __name__ == "__main__":
    L = 40
    n_sweeps = 10000

    non_impurity_directory = 'simulation_data_wo_impurities' # Directory for non-impurity data

    lattice = create_lattice(L)
    
    energy = calculate_total_energy(lattice) # start energy


    T_values = np.linspace(1, 4, 300)
    snapshotvals = [1.5, 2.3, 2.5] # Used for all snapshots

    # M_lst, E_final_lst, E_var_lst, snaps = get_M_E_temp_evolution(lattice, T_values, n_sweeps, snapshotvals, save = True, directory=non_impurity_directory)
    M_lst = np.load(f'{non_impurity_directory}/M_Tvals{T_values.shape[0]}_L{lattice.shape[0]}_Nsweeps{n_sweeps}.npy')
    E_final_lst = np.load(f'{non_impurity_directory}/E_Tvals{T_values.shape[0]}_L{lattice.shape[0]}_Nsweeps{n_sweeps}.npy')
    E_var_lst = np.load(f'{non_impurity_directory}/E_var_Tvals{T_values.shape[0]}_L{lattice.shape[0]}_Nsweeps{n_sweeps}.npy')
    snaps = np.load(f'{non_impurity_directory}/snapshots{snapshotvals}_Tvals{T_values.shape[0]}_L{lattice.shape[0]}_Nsweeps{n_sweeps}.npy')
    
    get_plot_task_3_1_3(T_values, n_sweeps)
    
    
    

    
    

    



    
