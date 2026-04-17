from numba import njit
import matplotlib.pyplot as plt
import numpy as np

@njit
def flashing(t_hat: float, tau_hat: float) -> float:
    phase = t_hat % tau_hat
    if phase < 0.75 * tau_hat:
        return 0.0
    return 1.0

@njit
def ratchet_potential(x_hat: float, alpha: float) -> float:
    x_cell = x_hat % 1.0

    if 0.0 <= x_cell < alpha:
        return x_cell / alpha
    else:
        return (1.0 - x_cell) / (1.0 - alpha)

@njit
def potential(x_hat: float, t_hat: float, alpha: float, tau_hat: float, flashing_on:bool = True) -> float:
    if flashing_on:
        return ratchet_potential(x_hat, alpha) * flashing(t_hat, tau_hat)
    return ratchet_potential(x_hat, alpha)

@njit
def dU_dx(x_hat: float, t_hat: float, alpha: float, tau_hat: float, flashing_on: bool = True) -> float:
    if flashing_on and not flashing(t_hat, tau_hat):
        return 0.0

    x_cell = x_hat % 1.0

    if 0.0 <= x_cell < alpha:
        return 1.0 / alpha
    else:
        return -1.0 / (1.0 - alpha)
    
@njit
def force(x_hat: float, t_hat: float, alpha: float, tau_hat: float) -> float:
    return -dU_dx(x_hat, t_hat, alpha, tau_hat)


def plot_potential_and_force(alpha: float, tau_hat: float, t_hat: float)-> None:
    x = np.linspace(0, 1, 1000)
    U = [potential(xi, t_hat, alpha, tau_hat) for xi in x]
    dU = [dU_dx(xi, t_hat, alpha, tau_hat) for xi in x]

    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(x, U, label="Û(x̂,t̂)")
    ax[0].set_ylabel("Û")
    ax[0].legend()
    ax[1].plot(x, dU, label="∂Û/∂x̂", color="orange")
    ax[1].set_xlabel("x̂")
    ax[1].set_ylabel("∂Û/∂x̂")
    ax[1].legend()
    fig.tight_layout()
    plt.show()

