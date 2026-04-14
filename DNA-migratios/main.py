
from task4 import gaussian_random, test_gaussian
from task6 import run_simulation


def main() -> None:
    params = Params(alpha=0.5, D=1.0, tau=1.0, dt=0.01)
    run_simulation(x0_hat=0.0, t_end_hat=10.0, params=params)

    test_gaussian(gaussian_random)

main()