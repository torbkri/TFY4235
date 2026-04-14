from task3 import dU_dx,potential
from task4 import gaussian_random
from task5 import euler_step

def max_abs_dU_dx(alpha):
    return max(1.0 / alpha, 1.0 / (1.0 - alpha))

def timestep_indicator(dt_hat, D_hat, alpha):
    deterministic = max_abs_dU_dx(alpha) * dt_hat
    stochastic = 4.0 * (2.0 * D_hat * dt_hat) ** 0.5
    return deterministic + stochastic

def check_timestep(dt_hat, D_hat, alpha, safety_factor=0.1):
    lhs = timestep_indicator(dt_hat, D_hat, alpha)
    rhs = safety_factor * alpha
    ok = lhs < rhs
    return ok, lhs, rhs


def run_simulation(x0_hat, t_end_hat, params, rng=gaussian_random):
    check_timestep(params.dt, params.D, params.alpha)

    n_steps = int(t_end_hat / params.dt)

    t_values = [0.0]
    x_values = [x0_hat]
    u_values = [potential(x0_hat, 0.0, params.alpha, params.tau)]

    x_hat = x0_hat
    t_hat = 0.0

    for _ in range(n_steps):
        x_hat, t_hat = euler_step(x_hat, t_hat, params, rng)
        t_values.append(t_hat)
        x_values.append(x_hat)
        u_values.append(potential(x_hat, t_hat, params.alpha, params.tau))

    return t_values, x_values, u_values