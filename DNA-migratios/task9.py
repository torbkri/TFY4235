import numpy as np
from matplotlib import pyplot as plt
from Params import Params
from task6 import run_simulation
from numba import njit

@njit
def _drift_velocity_inner(velocities: np.ndarray, tn: float) -> np.ndarray:
    for i in range(len(velocities)):
        velocities[i] /= tn
    return velocities

def drift_velocity(run_data: tuple([dict[str, list[float]], np.ndarray])) -> float:
    tn = run_data[0]["t"][-1]
    velocities = run_data[1]
    
    return _drift_velocity_inner(velocities, tn)

def mean_drift_velocity(flashing_time:float) -> tuple([float,float]):
    params = Params(tau=flashing_time, particles=100)
    run_data = run_simulation(x0_hat=0.0, t_end_hat=params.tau, params=params, flashing_on=True)
    velocity = drift_velocity(run_data=run_data)
    return np.mean(velocity), np.std(velocity)


def plot_drift_velocity_vs_flashing_time(flashing_times: np.ndarray) -> None:
    velocities = np.zeros(len(flashing_times))
    deviations = np.zeros(len(flashing_times))
    for i, t in enumerate(flashing_times):
        print(t)
        # drift_values = np.zeros(runs)
        # for j in range(runs):
            
        velocities[i], deviations[i] = mean_drift_velocity(t)
    plt.figure()
    plt.errorbar(flashing_times, velocities, yerr = deviations, marker="o")
    plt.xlabel("Flashing time (t̂)")
    plt.ylabel("Mean drift velocity (x̂/t̂)")
    plt.tight_layout()
    plt.savefig(fname="./DNA-migratios/drift_velocity.jpg")
    plt.show()