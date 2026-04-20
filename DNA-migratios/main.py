
from Params import Params
import time
import numpy as np
from task3 import plot_potential_and_force
from task4 import gaussian_random, test_gaussian
from task5 import plot_compare, plot_trajectory, plot_energies
from task6 import run_simulation
from task7 import plot_energy_distribution, boltzmann_density_reduced
from task9 import mean_drift_velocity, plot_drift_velocity_vs_flashing_time



def main() -> None:
    print(time.ctime())
    params = Params(alpha=0.2, D= 3.25E-3, tau=7, dt=1E-4, periods=100, particles = 1)

    # params = Params(alpha=0.2, D= 10, tau=200, dt=1E-5)
    
    # run_data_10 = run_simulation(x0_hat=0.0, t_end_hat=params.tau, params=params, flashing_on=True)
    # plot_trajectory(run_data_10)
    # run_data_01 = run_simulation(x0_hat=0.0, t_end_hat=params.tau, params=params, flashing_on=False)
    # print(time.ctime())
    # plot_energy_distribution(run_data_01, params.D, bins= 100, burn_in_fraction=0.4)


    flashing = np.linspace(0.3, 15, 100)
    plot_drift_velocity_vs_flashing_time(flashing)


    # plot_trajectory(run_data_01[0])
    # print(run_data_01[1])
    # print(mean_drift_velocity(run_data_01[1]))
    # plot_energy_distribution(run_data_01, params.D, bins = 100)

    # print(mean_drift_velocity(6.7))
main()