import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from numba import njit

J = 1.0 

def create_lattice(L, seed=None, impurities=None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    if impurities is None:
        return np.random.choice([-1, 1], size=(L, L)), None
    else:
        lattice = np.random.choice([-1, 1], size=(L, L))
        mask = np.random.rand(L, L) < impurities
        return lattice, mask

@njit()
def neighbor_sum(spins, i, j):
    '''Sum over neighbours, works with periodic boundaries'''
    L = spins.shape[0]
    return spins[i-1, j] + spins[(i+1) % L, j] + spins[i, j-1] + spins[i, (j+1) % L]

@njit()
def total_energy(spins):
    E = 0.0
    L = spins.shape[0]
    for i in range(L):
        for j in range(L):
            # Each pair counted twice, so factor 0.5
            E -= 0.5 * J * spins[i, j] * neighbor_sum(spins, i, j)
    return E

@njit()
def magnetization(spins):
    M = 0
    L = spins.shape[0]
    for i in range(L):
        for j in range(L):
            M += spins[i, j]
    return M

@njit()
def metropolis_sweep(spins, T, seed=None, mask=None):
    L = spins.shape[0]
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    for _ in range(L*L):
        i = np.random.randint(0, L)
        j = np.random.randint(0, L)
        s = spins[i, j]

        if mask is not None and mask[i, j]: #can't flip if mask is True
            continue

        nb = neighbor_sum(spins, i, j)
        dE = 2 * J * s * nb

        #metropolis critera 
        if np.random.random() < np.exp(-dE / T): 
            spins[i, j] = -s
        #else: do nothing, spin stays the same

def simulate(L, T, n_eq, n_sim, seed=None, mask=None):
    spins, _ = create_lattice(L, seed)
    M_time = np.empty(n_sim)
    E_time = np.empty(n_sim)
    spins_save = []

    for _ in range(n_eq):
        metropolis_sweep(spins, T, seed, mask)
    
    for k in range(n_sim):
        metropolis_sweep(spins, T, seed, mask)
        M_time[k] = magnetization(spins)
        E_time[k] = total_energy(spins)
        if k % (n_sim // 9) == 0:  # Save every 1/9 of the simulation steps
            spins_save.append(spins.copy())

    return spins_save, E_time, M_time

def simulate_annealing(L, T_series, n_eq, seed=None, impurities=None):
    spins, mask = create_lattice(L, seed, impurities)
    M_time = np.empty(len(T_series))
    E_time = np.empty(len(T_series))
    spins_save = []
    min_energy = float('inf')  

    for _ in range(n_eq):
        metropolis_sweep(spins, T_series[0], seed, mask)

    for k, T in tqdm(enumerate(T_series)):
        metropolis_sweep(spins, T, seed, mask)
        M_time[k] = magnetization(spins)
        E_time[k] = total_energy(spins)

        if E_time[k] < min_energy: #save only lowest energy
            min_energy = E_time[k]
            spins_save = spins.copy() 

    return spins_save, E_time, M_time, mask



def task_3_1_1():
    '''
    L = 40
    T = [2.1, 2.3, 2.5]
    n_eq = 1000
    n_sim = 10000
    spins, E_time, M_time = [], [], []
    for t in range(len(T)):
        sim = simulate(L, T[t], n_eq, n_sim)
        spins.append(sim[0])
        E_time.append(sim[1])
        M_time.append(sim[2])
    E_time = np.array(E_time)/(L**2)
    M_time = np.array(M_time)/(L**2)
    '''
    #save data to file
    #np.savez('./exam/task_3_1_1.npz', spins=spins, E_time=E_time, M_time=M_time, T=T)
    #load data from file
    data = np.load('./exam/task_3_1_1.npz')
    spins = data['spins']
    E_time = data['E_time']
    M_time = data['M_time']
    T = data['T']
    sweep = np.arange(len(E_time[0]))
    fig, ax = plt.subplots(2, 1, figsize=(8, 10))
    for t in range(len(T)):
        ax[0].plot(E_time[t], label=f'T={T[t]}', lw=0.5)
        ax[1].plot(M_time[t], label=f'T={T[t]}', lw=0.5)
    #ax[0].set_xlabel('Sweep', fontsize=20)
    ax[0].set_ylabel('Energy', fontsize=20)
    ax[0].legend(fontsize=16)
    ax[1].set_xlabel('Sweep', fontsize=20)
    ax[1].set_ylabel('Magnetization', fontsize=20)
    ax[1].legend(fontsize=16, loc='upper left')
    plt.tight_layout()
    plt.savefig('./exam/figs/M_E_against_sweep.png')
    plt.show()

def task_3_1_2():
    L = 40
    n_eq = 1500
    n_sim = 10000
    '''
    T = np.linspace(1.5, 3.5, 300)

    E_avg = []
    M_abs_avg = []
    spins_save = []

    for t in tqdm(range(len(T)), desc="Simulating", leave=False):
        sim = simulate(L, T[t], n_eq, n_sim)
        spins_save.append(sim[0])
        E_avg.append(sim[1].mean())
        M_abs_avg.append(np.abs(sim[2]).mean())
    
    E_avg = np.array(E_avg)
    M_abs_avg = np.array(M_abs_avg)
    
    #save data to file
    np.savez('./exam/task_3_1_2.npz', spins=spins_save, E_avg=E_avg, M_abs_avg=M_abs_avg, T=T)
    '''
    #load data from file
    data = np.load('./exam/task_3_1_2.npz')
    spins_save = data['spins']
    E_avg = data['E_avg']
    M_abs_avg = data['M_abs_avg']
    T = data['T']

    mosaic = '''
    aaa
    bcd
    '''

    fig, ax = plt.subplot_mosaic(mosaic=mosaic, figsize=(10, 8), layout='constrained')

    ax['a'].scatter(T, E_avg/L**2, label='E', s=3, color='blue')
    ax['a'].set_xlabel('Temperature [T]', fontsize=20)
    ax['a'].set_ylabel('Energy', fontsize=20)
    ax['a'].tick_params(axis='y')
    ax['a'].set_ylim(-2, 0)

    ax_a_twin = ax['a'].twinx() 
    ax_a_twin.set_ylabel('Magnetization', fontsize=20)
    ax_a_twin.tick_params(axis='y')
    ax_a_twin.scatter(T, M_abs_avg/L**2, label='|M|', color='red', s=3)

    # Combine legends
    handles_a, labels_a = ax['a'].get_legend_handles_labels()
    handles_twin, labels_twin = ax_a_twin.get_legend_handles_labels()
    ax['a'].legend(handles_a + handles_twin, labels_a + labels_twin, loc='upper center', fontsize=17)

    snapshot_T = [1.7, 2.3, 3.0]

    axes = [ax['b'], ax['c'], ax['d']]
    for ax, Tsnap in zip(axes, snapshot_T):
        spins_save, _, _ = simulate(L, Tsnap, n_eq, n_sim)
        final_spins = spins_save[-1]
        ax.imshow(final_spins, cmap='bwr', vmin=-1, vmax=1)
        ax.set_title(f'T={Tsnap}', fontsize=18)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('./exam/figs/E_M_against_temps_w_snaps.png')
    plt.show()

def task_3_1_3():
    '''
    L_list = [10, 20, 30, 40]
    runs = 2
    n_eq, n_mc = 2000, 5000
    temps = np.concatenate([
        np.linspace(1.5, 2.1, 70, endpoint=False),
        np.linspace(2.1, 2.4, 100, endpoint=False),
        np.linspace(2.4, 3.2, 70)
    ])
    C_avg_all = []

    for L_val in L_list:
        C_avg = []
        for T in tqdm(temps, desc=f'L={L_val}', leave=False):
            C_runs = np.empty(runs)
            for i in range(runs):
                _, E_time, _ = simulate(L_val, T, n_eq, n_mc)
                # compute variance-based C per spin
                C_runs[i] = np.var(E_time, ddof=0) / (T**2 * (L_val**2))
            C_val = C_runs.mean()
            C_avg.append(C_val)
        C_avg_all.append(C_avg)

    C_avg_all = np.array(C_avg_all)

    # save data to file
    np.savez('./exam/task_3_1_3.npz', temps=temps, C_avg_all=C_avg_all, L_list=L_list)
    '''
    
    # load data from file
    data = np.load('./exam/task_3_1_3.npz')
    temps = data['temps']
    C_avg_all = data['C_avg_all']
    L_list = data['L_list']

    #find t_c
    C_avg = C_avg_all.mean(axis=0)
    C_avg = C_avg / np.max(C_avg)  # Normalize for better comparison
    t_c = temps[np.argmax(C_avg[temps > 1.5])]
    print(f"Critical temperature (T_c): {t_c:.2f}")

    plt.figure(figsize=(12, 6))
    for i, L_val in enumerate(L_list):
        plt.scatter(temps, C_avg_all[i], label=f'L={L_val}', s=2)
    plt.axvline(t_c, color='k', linestyle='--', label=f'Tc≈{t_c:.2f}')
    plt.xlabel('Temperature', fontsize=22)
    plt.ylabel('Specific heat', fontsize=22)
    plt.legend(fontsize=16)
    plt.savefig('./exam/figs/C_vs_T.png')
    plt.tight_layout()
    plt.show()

    #plot the loglog of the specific heat
    plt.figure(figsize=(12, 6))

    for i, L_val in enumerate(L_list):
        t = temps - t_c
        plt.scatter(t, C_avg_all[i], s=2)

        mask = (t > 0) & (t < 0.065)  #choose range of t fitting
        log_dt = np.log(t[mask])
        log_C = np.log(C_avg_all[i][mask])
        m, b = np.polyfit(log_dt, log_C, 1)
        alpha = -m
        plt.plot(t[mask], np.exp(log_dt * m + b), label=f'L={L_val}, α≈{alpha:.2f}', linestyle='--')
        print(f"Extracted α ≈ {alpha:.2f}")
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'|Temperature - $T_c$|', fontsize=22)
    plt.ylabel('Specific heat', fontsize=22)
    plt.legend(fontsize=16)
    plt.savefig('./exam/figs/C_vs_T_loglog.png')
    plt.tight_layout()
    plt.show()

def task_3_1_4():
    L = 40
    
    n_eq = 3000
    n_sim = 7000
    runs = 3
    p_list = [0.02, 0.1, 0.25]
    temps = np.linspace(1, 3.5, 200)
    '''
    results = {p: {"E_avg": [], "M_abs_avg": [], "C_avg": []} for p in p_list}

    for p in p_list:
        mask = create_lattice(L, seed=None, impurities=p)[1]
        for T in tqdm(temps, desc=f'p={p}', leave=False):
            C_runs = np.empty(runs)
            E_runs = np.empty(runs)
            M_runs = np.empty(runs)
            for i in range(runs):
                _, E_time, M_time = simulate(L, T, n_eq, n_sim, mask=mask.copy())
                C_runs[i] = np.var(E_time, ddof=0) / (T**2 * (L**2))
                E_runs[i] = E_time.mean()
                M_runs[i] = np.abs(M_time).mean()

            results[p]["E_avg"].append(E_runs.mean())
            results[p]["M_abs_avg"].append(M_runs.mean())
            results[p]["C_avg"].append(C_runs.mean())

        results[p]["E_avg"] = np.array(results[p]["E_avg"])
        results[p]["M_abs_avg"] = np.array(results[p]["M_abs_avg"])
        results[p]["C_avg"] = np.array(results[p]["C_avg"])

    # Save data to file
    np.savez('./exam/task_3_1_4.npz', temps=temps, results=results, p_list=p_list)
    '''
    # Load data from file
    data = np.load('./exam/task_3_1_4.npz', allow_pickle=True)
    temps = data['temps']
    results = data['results'].item()
    p_list = data['p_list']

    fig, ax = plt.subplots(3, 1, figsize=(8, 8))
    for p in p_list:
        ax[0].scatter(temps, results[p]["E_avg"] / L**2, label=f'p={p}', s=2)
        ax[1].scatter(temps, results[p]["M_abs_avg"] / L**2, label=f'p={p}', s=2)
        ax[2].scatter(temps, results[p]["C_avg"], label=f'p={p}', s=2)

    #ax[0].set_xlabel('T', fontsize=22)
    ax[0].set_ylabel('Energy', fontsize=20)
    #ax[1].set_xlabel('T', fontsize=22)
    ax[1].set_ylabel('Magnetization', fontsize=20)
    ax[2].set_xlabel('Temperature', fontsize=20)
    ax[2].set_ylabel('Specific Heat', fontsize=20)

    ax[0].legend(fontsize=16)
    ax[1].legend(fontsize=16)
    ax[2].legend(fontsize=16)
    fig.tight_layout()
    plt.savefig('./exam/figs/E_M_C_vs_T_with_p.png')
    plt.show()

def task_3_1_5():
    L = 40
    '''
    n_eq = 2000
    runs = 3
    p_list = [0.02, 0.1, 0.25]
    T_series = np.linspace(5.0, 0.1, 30000)

    results = []
    for p in p_list:
        for run_i in range(1, runs+1):
            best_spins, _, _, mask = simulate_annealing(L, T_series, n_eq, impurities=p)
            E_best = total_energy(best_spins) / (L*L)
            M_best = abs(np.sum(best_spins))   / (L*L)
            results.append((p, E_best, M_best, best_spins, mask))
    results = np.array(results, dtype=object)
    # Save data to file
    np.savez('./exam/task_3_1_5.npz', results=results, T_series=T_series)
    '''
    # Load data from file
    data = np.load('./exam/task_3_1_5.npz', allow_pickle=True)
    results = data['results']
    T_series = data['T_series']

    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    axes = axes.flatten()
    for i, (p, E_best, M_best, spins, mask) in enumerate(results):
        ax = axes[i]
        # Create a combined image where mask is slightly lighter
        combined_image = spins * np.where(mask, 0.5, 1.0)
        ax.imshow(combined_image, cmap='bwr', vmin=-1, vmax=1)
        ax.set_title(f'p={p}, E={E_best:.2f}, |M|={M_best:.2f}', fontsize=15)
        ax.axis('off')
        ax.set_aspect('equal')

    # Hide any unused subplots
    for j in range(len(results), len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig('./exam/figs/SA_results.png')
    plt.show()

if __name__ == "__main__":
    #task_3_1_1() #finished
    #task_3_1_2() #finished
    task_3_1_3() #finished
    #task_3_1_4() #finished
    #task_3_1_5() #finished
    pass