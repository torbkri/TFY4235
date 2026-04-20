import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from numba import jit
from tqdm import tqdm

from plotting import *
from ex_3_1 import calculate_total_energy, calculate_energy_change
from ex_3_1 import calculate_specific_heat, find_Tc

def create_lattice_with_impurities(L, p):
    lattice = np.random.choice([-1, 1], size=(L, L))
    impurity_mask = np.random.rand(L, L) < p  # frozen sites

    random_frozen_spins = np.random.choice([-1, 1], size=(L, L))
    lattice[impurity_mask] = random_frozen_spins[impurity_mask]
    
    return lattice, impurity_mask

@jit(nopython = True)
def sweep_lattice_w_impurities(lattice, T, energy, impurity_mask):
    for n in range(lattice.size):
        # Pick random site
        i = np.random.randint(0, lattice.shape[0])
        j = np.random.randint(0, lattice.shape[1])

        if impurity_mask[i, j]: 
            continue

        # Get number for metropolis algorithm
        r = np.random.uniform(0, 1)

        energy_change = calculate_energy_change(i, j, lattice)

        P_star = np.exp(-energy_change / T)

        if P_star > r:
            lattice[i, j] = -lattice[i, j]
            energy += energy_change

    return lattice, energy

def run_sweeps_w_impurities(lattice, T, n_sweeps, energy, impurity_mask):
    new_lattice = lattice.copy()
    for n in range(n_sweeps):
        new_lattice, energy = sweep_lattice_w_impurities(new_lattice, T, energy, impurity_mask)
    
    return new_lattice, energy

def run_sweeps_calculate_M_E_time_evolution_w_impurities(lattice, T, n_sweeps, energy, impurity_mask):
    new_lattice = lattice.copy()
    M = np.zeros(n_sweeps)
    E = np.zeros(n_sweeps)

    for n in range(n_sweeps):
        new_lattice, energy = sweep_lattice_w_impurities(new_lattice, T, energy, impurity_mask)

        M[n] = np.mean(new_lattice)
        E[n] = energy

    return M, E, new_lattice, energy

def get_M_E_temp_evolution_w_impurities(lattice, T_values, n_sweeps, snapshotvals, impurity_mask, p, directory=None, save=False):
    M_lst = np.zeros_like(T_values)
    E_final_lst = np.zeros_like(T_values)
    E_var_lst = np.zeros_like(T_values)

    lattice_snapshots = np.zeros((len(snapshotvals), lattice.shape[0], lattice.shape[1]))

    snapshot_indices = [np.argmin(np.abs(T_values - T_snap)) for T_snap in snapshotvals]

    snapshot_counter = 0 

    for i, T in tqdm(enumerate(T_values), total=len(T_values)):
        energy = calculate_total_energy(lattice)

        M, E, flipped_lattice, energy = run_sweeps_calculate_M_E_time_evolution_w_impurities(lattice, T, n_sweeps, energy, impurity_mask)

        M_lst[i] = np.mean(flipped_lattice)
        E_final_lst[i] = energy
        E_var_lst[i] = np.var(E[-2000:]) 

        if i in snapshot_indices:
            lattice_snapshots[snapshot_counter] = flipped_lattice.copy()
            snapshot_counter += 1
  
    if save:
        np.save(f'{directory}/snapshots{snapshotvals}_Tvals{T_values.shape[0]}_L{lattice.shape[0]}_Nsweeps{n_sweeps}_p{p}.npy', lattice_snapshots)
        np.save(f'{directory}/M_Tvals{T_values.shape[0]}_L{lattice.shape[0]}_Nsweeps{n_sweeps}_p{p}.npy', M_lst)
        np.save(f'{directory}/E_final_Tvals{T_values.shape[0]}_L{lattice.shape[0]}_Nsweeps{n_sweeps}_p{p}.npy', E_final_lst)
        np.save(f'{directory}/E_var_Tvals{T_values.shape[0]}_L{lattice.shape[0]}_Nsweeps{n_sweeps}_p{p}.npy', E_var_lst)
        print(f"Saved data for {T_values.shape[0]} temperatures, L = {lattice.shape[0]}, n_sweeps = {n_sweeps}, p = {p} to {directory}")  
    
    return np.abs(M_lst), E_final_lst, E_var_lst, lattice_snapshots

def simulated_annealing(T_start, T_end, delta_T, lattice, impurity_mask):
    energy = calculate_total_energy(lattice)
    lowest_energy = energy
    sweeps_per_T = 5

    new_lattice = lattice.copy()

    T = T_start

    while T > T_end:
        
        new_lattice, energy = run_sweeps_w_impurities(new_lattice, T, sweeps_per_T, energy, impurity_mask)
        T -= delta_T

        if energy < lowest_energy:
            lowest_energy = energy
            lowest_energy_lattice = new_lattice

    M = np.mean(new_lattice) # Magnetization of last T. 
    # energy is the energy of the last step.

    return M, energy, lowest_energy_lattice, lowest_energy

def get_SA_results(T_start, T_end, delta_T, L, p_values):
    M_results = []
    E_results = []
    lowest_energy_lattices = []
    lowest_energy_results = []

    impurity_mask_lattices = [] 

    for p in p_values:
        M_temp = []
        E_temp = []

        best_lowest_energy = np.inf  # very large to start
        best_lowest_energy_lattice = None
        best_impurity_mask = None

        for i in range(3):
            lattice, impurity_mask = create_lattice_with_impurities(L, p)
            M, E, lowest_energy_lattice, lowest_energy = simulated_annealing(T_start, T_end, delta_T, lattice, impurity_mask)

            M_temp.append(M)
            E_temp.append(E)
            
            if lowest_energy < best_lowest_energy:
                best_lowest_energy = lowest_energy
                best_lowest_energy_lattice = lowest_energy_lattice
                best_impurity_mask = impurity_mask

        lowest_energy_lattices.append(best_lowest_energy_lattice)
        lowest_energy_results.append(best_lowest_energy)
        impurity_mask_lattices.append(best_impurity_mask)

        M_results.append(M_temp)
        E_results.append(E_temp)
    
    M_results = np.array(M_results)
    E_results = np.array(E_results) / L**2
    lowest_energy_lattices = np.array(lowest_energy_lattices)
    lowest_energy_results = np.array(lowest_energy_results) / L**2
    
    # Write results to a file
    with open("simulation_data_w_impurities/simulated_annealing_results.txt", "w") as file:
        file.write("Simulated Annealing Results\n")
        file.write(f"Lattice size: {L}\n")
        file.write(f"Temperature range: {T_start} to {T_end}\n")
        file.write(f"Delta T: {delta_T}\n")
        file.write(f"Impurity probabilities: {p_values}\n\n")

        for idx, p in enumerate(p_values):
            file.write(f"Impurity probability: {p}\n")
            file.write(f"Magnetization results (3 runs): {M_results[idx]}\n")
            file.write(f"Energy results (3 runs): {E_results[idx]}\n")
            file.write(f"Lowest energy: {lowest_energy_results[idx]}\n\n")
    
    np.save(f'simulation_data_w_impurities/simulated_annealing_snapshots.npy', [lowest_energy_lattices, impurity_mask_lattices])
    
    return lowest_energy_lattices, impurity_mask_lattices, lowest_energy_results


if __name__ == "__main__":
    # Example usage
    L = 40
    n_sweeps = 10000
    p_values = [0.01, 0.1, 0.3]

    directory = 'simulation_data_w_impurities'

    lattice, impurity_mask = create_lattice_with_impurities(L, p_values[1])
    T_values = np.linspace(1, 4, 300)
    
    snapshotvals = [1.5, 2.3, 2.5]
    
    # M_lst, E_final_lst, E_var_lst, lattice_snapshots = get_M_E_temp_evolution_w_impurities(lattice, T_values, n_sweeps, 
    #                                                                                        snapshotvals, impurity_mask, p
    #                                                                                        , directory=directory, save=True)
    
    # Load the results
    # snapshots = np.load(f'{directory}/snapshots{snapshotvals}_Tvals{T_values.shape[0]}_L{lattice.shape[0]}_Nsweeps{n_sweeps}_p{p}.npy')
    # M_lst = np.load(f'{directory}/M_Tvals{T_values.shape[0]}_L{lattice.shape[0]}_Nsweeps{n_sweeps}_p{p}.npy')
    # E_final_lst = np.load(f'{directory}/E_final_Tvals{T_values.shape[0]}_L{lattice.shape[0]}_Nsweeps{n_sweeps}_p{p}.npy')
    # E_var_lst = np.load(f'{directory}/E_var_Tvals{T_values.shape[0]}_L{lattice.shape[0]}_Nsweeps{n_sweeps}_p{p}.npy')

    
    # Simulated annealing
    T_start = 5
    T_end = 0.1
    delta_T = (T_start - T_end) / 10000

    # M, E, lowest_energy_lattice, lowest_energy = simulated_annealing(T_start, T_end, delta_T, lattice, impurity_mask)

    # print(f'M = {M}, E = {E/ L**2}')

    

    # latt, imp = np.load(f'simulation_data_w_impurities/simulated_annealing_snapshots.npy', allow_pickle=True)

    # print(latt.shape)
    # print(imp.shape)





    

