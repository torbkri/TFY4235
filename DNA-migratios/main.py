
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
    params = Params(alpha=0.2, D= 3.25E-3, tau=5, dt=1E-4, periods=20, particles = 10)

    # params = Params(alpha=0.2, D= 10, tau=200, dt=1E-5)
    
    # run_data_10 = run_simulation(x0_hat=0.0, t_end_hat=params.tau, params=params, flashing_on=False)
    # plot_trajectory(run_data_10)
    # run_data_01 = run_simulation(x0_hat=0.0, t_end_hat=params.tau, params=params, flashing_on=False)
    # print(time.ctime())
    # plot_energy_distribution(run_data_01, params.D, bins= 100, burn_in_fraction=0.4)
    flashing = np.linspace(0.3, 20, 100)
    plot_drift_velocity_vs_flashing_time(flashing)
    # plot_trajectory(run_data_01[0])
    # print(run_data_01[1])
    # print(mean_drift_velocity(run_data_01[1]))
    # plot_energy_distribution(run_data_01, params.D, bins = 100)


    # plot_compare(run_data_10=run_data_10, run_data_01=run_data_01)

    # plot_potential_and_force(alpha=params.alpha, tau_hat=params.tau, t_hat=0.9*params.tau)
    # test_gaussian(gaussian_random)
main()