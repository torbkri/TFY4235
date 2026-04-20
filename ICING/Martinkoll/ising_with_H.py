import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import njit
from ising import create_lattice, neighbor_sum
mu = 1.0   
J  = 1.0   

@njit()
def total_energy_H(spins, H):
    """E = -J sum_<ij> s_i s_j  - mu*H sum_i s_i"""
    E = 0.0
    L = spins.shape[0]
    for i in range(L):
        for j in range(L):
            # nearest‐neighbors
            nb = neighbor_sum(spins, i, j)
            E -= 0.5 * J * spins[i,j]*nb
            E -= mu * H * spins[i,j]
    return E

@njit()
def metropolis_sweep_H(spins, T, H, seed=None):
    if seed is not None:
        np.random.seed(seed)

    L = spins.shape[0]
    for _ in range(L*L):
        i = np.random.randint(0, L)
        j = np.random.randint(0, L)
        s = spins[i,j]
        # local field from neighbors + external
        nb = neighbor_sum(spins, i, j)
        dE = 2 * s * (J*nb + mu*H)
        if dE <= 0 or np.random.rand() < np.exp(-dE/T):
            spins[i,j] = -s

def simulate_H(L, T, H, n_eq, n_sim, seed=None):
    spins, _ = create_lattice(L, seed)
    spins_save = []
    M_time = np.empty(n_sim)
    for _ in range(n_eq):
        metropolis_sweep_H(spins, T, H, seed)
    for k in range(n_sim):
        metropolis_sweep_H(spins, T, H, seed)
        M_time[k] = np.sum(spins)
        if k % (n_sim // 9) == 0:
            spins_save.append(spins.copy())
    return spins_save, M_time

def task_3_2_1():
    n_eq, n_sim = 2000, 20000
    '''
    L, T, H = 40, 2.3, 0.01
    spins, M_time = simulate_H(L, T, H, n_eq, n_sim)
    M_time /= (L*L)  # normalize
    #save data
    np.savez("./exam/task_3_2_1.npz", spins=spins, M_time=M_time, T=T, H=H)
    '''
    #load data
    data = np.load("./exam/task_3_2_1.npz")
    spins = data["spins"]
    M_time = data["M_time"]
    T = data["T"]
    H = data["H"]

    mosaic = '''
    aaa
    bcd
    '''
    fig, ax = plt.subplot_mosaic(mosaic=mosaic, figsize=(10, 7), layout='constrained')
    
    # Plot M(t)
    ax['a'].plot(np.arange(n_sim), M_time, lw=1)
    ax['a'].set_xlabel("Sweep", fontsize=18)
    ax['a'].set_ylabel("Magnetization", fontsize=18)
    
    # Snapshot plots
    axes = ['b', 'c', 'd']
    snaps = [2, 5, 8]
    for letter, k in zip(axes, snaps):
        ax[letter].imshow(spins[k], cmap='bwr', vmin=-1, vmax=1)
        ax[letter].axis('off')
    plt.savefig("./exam/figs/time_evo_H.png")
    plt.show()

def task_3_2_2():
    L = 40
    '''
    H_list = [0.01, 0.03, 0.05]
    temps = np.linspace(1.8, 4, 300)
    n_eq, n_sim = 2000, 10000

    # collect all data
    results = {H: {"M_avg": [], "chi": []} for H in H_list}
    for H in H_list:
        M_avg = []
        chi   = []
        for T in tqdm(temps, desc=f"H={H}", leave=False):
            _, M_time = simulate_H(L, T, H, n_eq, n_sim)
            M_arr = M_time / (L*L)
            M_avg.append(np.mean(np.abs(M_arr)))
            chi.append(np.var(M_arr, ddof=0)/T) # χ per spin
        results[H]["M_avg"] = M_avg
        results[H]["chi"]   = chi
        
    #save results
    np.savez("./exam/task_3_2_2.npz", results=results, temps=temps)
    '''
    #load data
    data = np.load("./exam/task_3_2_2.npz", allow_pickle=True)
    results = data["results"]
    temps = data["temps"]

    # now plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
    for H, item in results.item().items():
        M_avg = item["M_avg"]
        chi   = item["chi"]
        # magnetization
        ax1.scatter(temps, M_avg, marker='o', label=f"H={H}", s=2)
        # susceptibility
        ax2.scatter(temps, chi, marker='o', label=f"H={H}", s=2)
    
    #ax1.set_xlabel("T", fontsize=22)
    ax1.set_ylabel("Magnetization", fontsize=18)
    ax1.legend(fontsize=16)
    
    ax2.set_xlabel("Temperature", fontsize=18)
    ax2.set_ylabel("Suspectability", fontsize=18)
    ax2.legend(fontsize=16)
    
    plt.tight_layout()
    plt.savefig("./exam/figs/M_Chi_temp.png")
    plt.show()

def task_3_2_3():
    '''
    L = 40
    H_list = [0.01, 0.03, 0.05]
    temps = np.linspace(1.8, 3, 300)
    n_eq, n_sim = 2000, 5000
    Tc = 2.27
    beta, gamma, delta = 1/8, 7/4, 15

    X_all = []
    Y_all = []
    t_vals_all = []
    H_all = []

    for H in H_list:
        chi_vals = []
        t_vals   = []
        for T in tqdm(temps, desc=f"H={H}", leave=False):
            _, M_time = simulate_H(L, T, H, n_eq, n_sim)
            M_arr = M_time / (L*L)
            chi = np.var(M_arr) / T
            t = (T - Tc)/Tc
            chi_vals.append(chi)
            t_vals.append(t)
        t_vals = np.array(t_vals)
        chi_vals = np.array(chi_vals)

        # collapse axes:
        X = H / np.abs(t_vals)**(beta*delta)
        Y = chi_vals * np.abs(t_vals)**gamma

        X_all.append(X)
        Y_all.append(Y)
        t_vals_all.append(t_vals)
        H_all.append(H)

    # save data
    np.savez("./exam/task_3_2_3.npz", X_all=X_all, Y_all=Y_all, t_vals_all=t_vals_all, H_all=H_all)
    '''
    # Load data
    data = np.load("./exam/task_3_2_3.npz", allow_pickle=True)

    X_all = data["X_all"]
    Y_all = data["Y_all"]
    t_vals_all = data["t_vals_all"]
    H_all = data["H_all"]

    # Plotting
    plt.figure(figsize=(12, 6))
    for X, Y, t_vals, H in zip(X_all, Y_all, t_vals_all, H_all):
        mask_plus  = t_vals > 0
        mask_minus = t_vals < 0
        plt.plot(X[mask_minus], Y[mask_minus], 'o', ms=3, label=f"H={H} (T<Tc)")
        plt.plot(X[mask_plus],  Y[mask_plus],  'o', ms=3, label=f"H={H} (T>Tc)")

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r"$h/|t|^{\beta\delta}$", fontsize=22)
    plt.ylabel(r"$\chi\,|t|^{\gamma}$", fontsize=22)
    plt.legend(fontsize=18)
    plt.tight_layout()
    plt.savefig("./exam/figs/critical_scaling.png")
    plt.show()

if __name__=="__main__":
    task_3_2_1() #finished
    task_3_2_2() #finished
    task_3_2_3() #finished
    pass