from numba import njit
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

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
    return -dU_dx(x_hat, t_hat, alpha, tau_hat, flashing_on = False)


def plot_potential_and_force(alpha: float, tau_hat: float, t_hat: float)-> None:
    x = np.linspace(0, 3, 1000)
    U = [ratchet_potential(xi, alpha) for xi in x]
    Force = [force(xi, tau_hat=tau_hat, alpha=alpha, t_hat=t_hat) for xi in x]

    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(x, U, label="$\hat{U}(\hat{x},\hat{t})$")
    ax[0].set_ylabel("$\hat{U}$")
    ax[0].legend()
    ax[1].plot(x, Force, label="$\partial\hat{U}\partial \hat{x}$", color="orange")
    ax[1].set_xlabel("$\hat{x}$")
    ax[1].set_ylabel("$\partial\hat{U}\partial \hat{x}$")
    ax[1].legend(loc=1)
    fig.tight_layout()
    fig.savefig("/Users/torbjornkringeland/Desktop/NTNU/6/TFY4235/DNA-migratios/DNA-report/Images/potential_and_force.pgf")
