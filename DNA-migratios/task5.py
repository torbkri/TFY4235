import math
import matplotlib.pyplot as plt
from numba import njit

from Params import Params
from task3 import dU_dx
from task4 import gaussian_random


def euler_step(x_hat: float, t_hat: float, params: Params, rng: callable = gaussian_random, flashing_on: bool = True) -> tuple[float, float]:
    drift = -dU_dx(x_hat, t_hat, params.alpha, params.tau, flashing_on) * params.dt #reduced units, so no gamma necessary
    noise = math.sqrt(2.0 * params.D * params.dt) * rng()
    x_next = x_hat + drift + noise
    t_next = t_hat + params.dt
    return x_next, t_next

def plot_trajectory(run_data):
    plt.figure()
    plt.plot(run_data[0]["t"], run_data[0]["x"])
    plt.xlabel("t̂")
    plt.ylabel("x̂(t̂)")
    plt.tight_layout()
    plt.show()

def plot_energies(run_data):
    plt.figure()
    plt.plot(run_data[0]["t"], run_data[0]["u"])
    plt.xlabel("t̂")
    plt.ylabel("Û(t̂)")
    plt.tight_layout()
    plt.show()

def plot_compare(run_data_10, run_data_01, figsize=(12, 5)):
    fig, ax = plt.subplots(1,2, figsize=figsize)
    ax[0].plot(run_data_10["t"], run_data_10["x"], label="D̂=10")
    ax[0].set_xlabel("t̂")
    ax[0].set_ylabel("x̂(t̂)")
    ax[0].legend()
    ax[1].plot(run_data_01["t"], run_data_01["x"], label="D̂=0.1")
    ax[1].set_xlabel("t̂")
    ax[1].set_ylabel("x̂(t̂)")
    ax[1].legend()
    plt.tight_layout()
    plt.show()