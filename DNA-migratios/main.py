
from Params import Params
import time
# from task3 import plot_potential_and_force
# from task4 import gaussian_random, test_gaussian
from task5 import plot_compare, plot_trajectory, plot_energies
from task6 import run_simulation
from task7 import plot_energy_distribution, boltzmann_density_reduced



def main() -> None:
    print(time.ctime())
    # params = Params(alpha=0.2, D= 10, tau=200, dt=1E-5)
    # run_data_10 = run_simulation(x0_hat=0.0, t_end_hat=params.tau, params=params, flashing_on=True)
    # plot_trajectory(run_data_10)

    params = Params(alpha=0.2, D= 10, tau=200, dt=1E-5)
    run_data_01 = run_simulation(x0_hat=0.0, t_end_hat=params.tau, params=params, flashing_on=False)
    plot_trajectory(run_data_01)

    plot_energy_distribution(run_data_01, params.D)

    # plot_compare(run_data_10=run_data_10, run_data_01=run_data_01)

    # plot_potential_and_force(alpha=params.alpha, tau_hat=params.tau, t_hat=0.9*params.tau)
    # test_gaussian(gaussian_random)
    print(time.ctime())

main()