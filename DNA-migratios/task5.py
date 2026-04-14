import math
import matplotlib.pyplot as plt

from Params import Params
from task3 import dU_dx
from task4 import gaussian_random


def euler_step(x_hat: float, t_hat: float, params: Params, rng: callable = gaussian_random) -> tuple[float, float]:
    drift = -dU_dx(x_hat, t_hat, params.alpha, params.tau) * params.dt
    noise = math.sqrt(2.0 * params.D * params.dt) * rng()
    x_next = x_hat + drift + noise
    t_next = t_hat + params.dt
    return x_next, t_next

def plot_trajectory(run_data):
    plt.figure()
    plt.plot(run_data["t"], run_data["x"])
    plt.xlabel("t̂")
    plt.ylabel("x̂(t̂)")
    plt.tight_layout()
    plt.show()