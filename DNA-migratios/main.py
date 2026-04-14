from Params import Params
from task3 import plot_potential_and_force
from task4 import gaussian_random, test_gaussian
from task6 import run_simulation


def main() -> None:
    params = Params(alpha=0.2, D=26E-3 / 80, tau=3.0, dt=0.01)
    run_simulation(x0_hat=0.0, t_end_hat=10.0, params=params)
    plot_potential_and_force(alpha=params.alpha, tau_hat=params.tau, t_hat=0.9*params.tau)
    # test_gaussian(gaussian_random)

main()