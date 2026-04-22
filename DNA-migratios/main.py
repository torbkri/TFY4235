
from Params import Params
import time
import numpy as np
from task3 import plot_potential_and_force
from task4 import gaussian_random, plot_gaussian_histogram, test_gaussian
from task5 import plot_compare, plot_trajectory, plot_energies
from task6 import run_simulation
from task7 import plot_energy_distribution, boltzmann_density_reduced
from task9 import mean_drift_velocity, plot_drift_velocity_vs_flashing_time


def main() -> None:
    print(time.ctime())

    one_cycle_params = Params(tau=10, periods=65, particles=1)

    run_one = run_simulation(0, one_cycle_params, flashing_on=True)
    plot_trajectory(run_data=run_one)

    test_params = Params(tau=7)

    # plot_gaussian_histogram(gaussian_random)
    # run_data = run_simulation(0,test_params.t_end, test_params)
    # plot_trajectory(run_data)

    # flashing_times = np.linspace(0.5, 18, 15) # Generating an array of flashing times from 0.5 to 18 seconds, with 60 points in total
    # plot_drift_velocity_vs_flashing_time(flashing_times)

    print(time.ctime())

    
main()