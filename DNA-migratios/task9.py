import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from Params import Params
from task6 import run_simulation
from numba import njit
from scipy.signal import savgol_filter

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

@njit
def _drift_velocity_inner(velocities: np.ndarray, tn: float) -> np.ndarray:
    for i in range(len(velocities)):
        velocities[i] /= tn
    return velocities

def drift_velocity(run_data: tuple([dict[str, list[float]], np.ndarray])) -> float:
    tn = run_data[0]["t"][-1]
    velocities = run_data[1]
    
    return _drift_velocity_inner(velocities, tn)

def mean_drift_velocity(flashing_time:float, run_data= None) -> tuple([float,float]):
    if run_data is None:
        params = Params(tau=flashing_time, particles=1)
        run_data = run_simulation(x0_hat=0.0, t_end_hat=params.tau, params=params, flashing_on=True)
    velocity = drift_velocity(run_data=run_data)
    mean = np.mean(velocity)
    # print(velocity)
    return mean, np.std(velocity,ddof=1) / np.sqrt(len(velocity))

def plot_drift_velocity_vs_flashing_time(flashing_times: np.ndarray) -> None:
    velocities = np.zeros(len(flashing_times))
    deviations = np.zeros(len(flashing_times))
    max = 0
    for i, t in enumerate(flashing_times):
        print(t)
        # drift_values = np.zeros(runs)
        # for j in range(runs):
            
        velocities[i], deviations[i] = mean_drift_velocity(t)

    v_smooth = savgol_filter(velocities, window_length=11, polyorder=3)
    idx = np.argmax(v_smooth)
    print(f"Optimal flashing time: {flashing_times[idx]}, Drift velocity: {velocities[idx]}")
    plt.figure()
    plt.axvline(x= flashing_times[idx], color="red", linestyle="--", label=f"Optimal flashing time")
    plt.scatter(flashing_times, velocities, marker=".", label="Simulation 12 nm")
    plt.xlabel(r"Flashing time $\tau$", fontsize= 24)
    plt.ylabel(r"Mean drift velocity ($\hat{x} / \hat{t}$)", fontsize=24)
    plt.yticks(fontsize= 24)
    plt.xticks(fontsize= 24)
    plt.legend(fontsize= 24)
    plt.tight_layout()
    plt.savefig(fname=f"/Users/torbjornkringeland/Desktop/NTNU/6/TFY4235/DNA-migratios/DNA-report/Images/drift_velocity_{len(flashing_times)}.pgf")
    plt.show()