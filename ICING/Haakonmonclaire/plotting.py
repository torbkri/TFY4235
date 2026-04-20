import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

def plot_lattice(lattice):
    plt.imshow(lattice, cmap=cm.coolwarm, vmin = -1, vmax = 1)
    plt.colorbar(None)
    plt.title('Lattice Configuration')
    plt.show()

def plot_lattice_w_impurities(lattice, impurity_mask):
    plt.imshow(lattice, cmap=cm.bwr, vmin = -1, vmax = 1)
    plt.colorbar(None)
    plt.imshow(np.where(impurity_mask, 1, np.nan), cmap='autumn', vmin=0, vmax=1, alpha=0.8)
    plt.title('Lattice Configuration')
    plt.show()

def plot_evolution_few_temps(M_lst, temps):
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, M in enumerate(M_lst):
        ax.plot(M, label=f'T = {temps[i]}')
    ax.set_xlabel('Sweep Number', fontsize = 20)
    ax.set_ylabel('Magnetization', fontsize = 20)
    ax.legend(fontsize = 18)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.set_ylim(-1, 1)


    # fig.savefig(f'figures/magnetization_evolution.png', dpi=300)
    plt.show()
    return 0


def plot_M_E(L, T_vals, M_lst, E_final_lst):
    # mosaic = '''
    # aaa
    # bcd
    # '''

    fig, axs = plt.subplots(figsize=(10, 6))
    
    axs.scatter(T_vals, np.abs(M_lst), label = r'|M|', s = 5)
    axs.scatter(T_vals, E_final_lst / L**2, label = r'E / $L^2$', s = 5)
    axs.set_xlabel('Temperature', fontsize = 20)
    axs.set_ylabel('Magnetization / Energy', fontsize = 20)
    axs.legend(fontsize = 18, markerscale = 5)
    axs.tick_params(axis='both', which='major', labelsize=15)

    # for i, key in enumerate(['b', 'c', 'd']):
    #     snap = snaps[i]
    #     axs[key].imshow(snap, cmap=cm.bwr, vmin=-1, vmax=1)
    #     axs[key].axis('off')
    #     axs[key].set_title(f'T = {snapshotvals[i]}', fontsize = 30)


    fig.tight_layout()
    # fig.savefig(f'figures/M_E_vs_Temp.png', dpi=300)
    plt.show()

def plot_M_E_snapshots_w_impurities(T_vals, M_lst, E_lst, snaps, L, snapshotvals, impurity_mask):
    mosaic = '''
    aaa
    bcd
    '''

    fig, axs = plt.subplot_mosaic(mosaic, figsize=(10, 10))
    
    axs['a'].scatter(T_vals, np.abs(M_lst), label = r'|M|', s = 5)
    axs['a'].scatter(T_vals, E_lst / L**2, label = r'E / $L^2$', s = 5)
    axs['a'].set_xlabel('Temperature', fontsize = 20)
    axs['a'].set_ylabel('Magnetization / Energy', fontsize = 20)
    axs['a'].legend(fontsize = 18, markerscale = 5)
    axs['a'].tick_params(axis='both', which='major', labelsize=15)

    for i, key in enumerate(['b', 'c', 'd']):
        snap = snaps[i]
        axs[key].imshow(snap, cmap=cm.bwr, vmin=-1, vmax=1)
        axs[key].axis('off')
        axs[key].set_title(f'T = {snapshotvals[i]}', fontsize = 30)
        
        axs[key].imshow(
            np.where(impurity_mask, 1, np.nan),   
            cmap='autumn',                        
            vmin=0, vmax=1,
            alpha=0.8                             
        )
        

    fig.tight_layout()
    plt.show()

def get_plot_task_3_1_1(L, n_sweeps, temps):
    from ex_3_1 import create_lattice, run_sweeps_calculate_M_E_time_evolution
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, T in enumerate(temps):
        lattice = create_lattice(L)
        M, E, lattice_temp, energy = run_sweeps_calculate_M_E_time_evolution(lattice, T, 10000, 0)
        #ax.scatter(np.linspace(0, n_sweeps, n_sweeps), M, label=f'T = {T}', s = 1)
        ax.plot(M, label=f'T = {T}', alpha=0.9)
    
    ax.set_xlabel('Number of Sweeps', fontsize = 20)
    ax.set_ylabel('Magnetization', fontsize = 20)
    ax.tick_params(axis='both', which='major', labelsize=15)
    legend = ax.legend(fontsize = 18, loc = 'upper right', framealpha = 1.0)
    for line in legend.get_lines():
        line.set_linewidth(3)
    
    ax.set_ylim(-1, 1)
    
    #fig.savefig('figures/magnetization_time_evolution.png', dpi=300)

    plt.show()

def plot_snapshots_3_1_2(snaps, snapshotvals):
    fig, axs = plt.subplots(1, 3, figsize=(10, 4.3))
    for i, ax in enumerate(axs.flatten()):
        ax.imshow(snaps[i], cmap=cm.bwr, vmin=-1, vmax=1)
        ax.axis('off')
        ax.set_title(f'T = {snapshotvals[i]}', fontsize = 25)

    fig.tight_layout()
    # fig.savefig(f'figures/snapshots_3_1_2.png', dpi=300)  
    plt.show()
    return 0

def get_plot_task_3_1_3(T_values, n_sweeps):
    from ex_3_1 import calculate_specific_heat, find_Tc
    L_values = [10, 20, 30, 40]

    fig, axs = plt.subplots(2,2, figsize=(12,8))

    for i, ax in enumerate(axs.flatten()):
        E_var_lst = np.load(f'simulation_data_wo_impurities/E_var_Tvals{T_values.shape[0]}_L{L_values[i]}_Nsweeps{n_sweeps}.npy')
        C = calculate_specific_heat(E_var_lst, T_values)
        
        Tc = find_Tc(T_values, C)

        ax.scatter(T_values, C, label = f'L = {L_values[i]}', s = 10)
        ax.axvline(x = Tc, color = 'red', linestyle = '--', label = r'$T_c = {:.3f}$'.format(Tc))
        ax.legend(fontsize = 18, loc = 'upper right', markerscale = 3)

        ax.tick_params(axis='both', which='major', labelsize=15)

    fig.supxlabel('Temperature (T)', fontsize = 28)
    fig.supylabel('Specific Heat (C)', fontsize = 28)

    fig.tight_layout()
    # fig.savefig('figures/specific_heat_all_lattices.png', dpi=300)
    plt.show()

    return 0

def get_plot_3_1_4(p_values, T_values, L, n_sweeps):
    from ex_3_1 import calculate_specific_heat
    from matplotlib.patheffects import withStroke
    
    fig, axs = plt.subplots(2, 1, figsize=(8, 9))

    colors = ['blue', 'red', 'green']
    
    for i, ax in enumerate(axs.flatten()):
        for j, p in enumerate(p_values):
            if i == 0:
                M_lst = np.load(f'simulation_data_w_impurities/M_Tvals{T_values.shape[0]}_L{L}_Nsweeps{n_sweeps}_p{p}.npy')
                E_final_lst = np.load(f'simulation_data_w_impurities/E_final_Tvals{T_values.shape[0]}_L{L}_Nsweeps{n_sweeps}_p{p}.npy')

                ax.scatter(T_values, np.abs(M_lst), label=f'p = {p}', s=5, c=colors[j], alpha=0.7)
                ax.scatter(T_values, E_final_lst / (L**2), s=5, c = colors[j], alpha=0.7)

            elif i == 1:
                E_var_lst = np.load(f'simulation_data_w_impurities/E_var_Tvals{T_values.shape[0]}_L{L}_Nsweeps{n_sweeps}_p{p}.npy')
                C = calculate_specific_heat(E_var_lst, T_values)
                ax.scatter(T_values, C, label=f'p = {p}', s=5, c=colors[j], alpha=0.7)
    
    axs[0].legend(fontsize=15, loc='upper right', markerscale = 3, framealpha = 0.9)
    axs[1].legend(fontsize=15, loc='upper right', markerscale = 3)

    
    axs[0].set_ylabel('Energy / Magnetization', fontsize=20)
    axs[1].set_ylabel('Specific Heat (C)', fontsize=20)
    axs[0].tick_params(axis='both', which='major', labelsize=15)
    axs[1].tick_params(axis='both', which='major', labelsize=15)
    
    # Define path effects for annotations
    path_effects = [withStroke(linewidth=3, foreground="white")]

    axs[0].annotate('a)', xy=(0.01, 0.85), xycoords='axes fraction', fontsize=35, c='black', 
                    path_effects=path_effects)
    axs[1].annotate('b)', xy=(0.01, 0.85), xycoords='axes fraction', fontsize=35, c='black', 
                    path_effects=path_effects)

    fig.supxlabel('Temperature (T)', fontsize=20)

    fig.tight_layout()
    fig.savefig('figures/plot_3_1_4.png', dpi=300)

    plt.show()
    return 0

def get_plot_and_data_3_1_5(T_start, T_end, delta_T, L, p_values):
    from impurity_functions import get_SA_results
    # Get data
    lowest_energy_lattices, impurity_masks, lowest_energy_results = get_SA_results(T_start, T_end, delta_T, L, p_values)

    fig, axs = plt.subplots(1, 3, figsize=(10, 4.3))

    for i, p in enumerate(p_values):
        axs[i].imshow(lowest_energy_lattices[i], cmap=cm.bwr, vmin=-1, vmax=1, alpha = 0.5)
        axs[i].axis('off')
        axs[i].set_title(f'p = {p}', fontsize=25, pad = 10)

        impurity_spins = np.where(impurity_masks[i], lowest_energy_lattices[i], np.nan)

        axs[i].imshow(
            impurity_spins,
            cmap=cm.bwr,
            vmin=-1, vmax=1,
            alpha=0.8
        )

        axs[i].text(
            0.5, -0.13,  # (x, y) position in axes coordinates
            f'E = {lowest_energy_results[i]:.3f}',  # format energy nicely
            fontsize=25,
            ha='center',  # center horizontally
            transform=axs[i].transAxes  # important: coordinates relative to the subplot
        )

    fig.tight_layout()

    #fig.savefig(f'figures/SA_results.png', dpi=300)
    plt.show()

    return 0