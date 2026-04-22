
import numpy as np
from numba import njit
from task3 import dU_dx, flashing,potential
from task4 import gaussian_random
from task5 import euler_step

@njit
def max_abs_dU_dx(alpha: float) -> float:
    return max(1.0 / alpha, 1.0 / (1.0 - alpha))

@njit
def timestep_indicator(dt_hat: float, D_hat: float, alpha: float) -> float:
    deterministic = max_abs_dU_dx(alpha) * dt_hat
    stochastic = 4.0 * (2.0 * D_hat * dt_hat) ** 0.5
    return deterministic + stochastic

@njit
def check_timestep(dt_hat: float, D_hat: float, alpha: float, safety_factor: float=0.1) -> tuple[bool, float, float]:
    lhs = timestep_indicator(dt_hat, D_hat, alpha)
    rhs = safety_factor * alpha
    ok = lhs < rhs
    return ok, lhs, rhs


def run_simulation(x0_hat: float, params, rng:callable =gaussian_random, flashing_on: bool=True) -> tuple([dict[str, list[float]], np.ndarray]):
    # if not check_timestep(params.dt, params.D, params.alpha)[0]:
    #     raise ValueError("Timestep too large for stability. Reduce dt or increase safety_factor.")

    particles = np.zeros(params.particles)

    n_steps =  int(params.t_end / params.dt)

    t_values = [0.0]
    x_values = [x0_hat]
    u_values = [potential(x0_hat, 0.0, params.alpha, params.tau, flashing_on)]

    x_hat = x0_hat
    t_hat = 0.0

    for _ in range(n_steps):
        x_hat, t_hat = euler_step(x_hat, t_hat, params, rng, flashing_on)
        t_values.append(t_hat)
        x_values.append(x_hat)
        u_values.append(potential(x_hat, t_hat, params.alpha, params.tau, flashing_on))

    for i in range(params.particles):
        x_hat = x0_hat
        t_hat = 0.0
        for _ in range(n_steps):
            x_hat, t_hat = euler_step(x_hat, t_hat, params, rng, flashing_on)
        particles[i] = x_hat

    return {"t": t_values, "x": x_values, "u": u_values}, particles
