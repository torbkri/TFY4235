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

# @njit
# def _drift_velocity_inner(velocities: np.ndarray, tn: float) -> np.ndarray:
#     for i in range(len(velocities)):
#         velocities[i] /= tn
#     return velocities

# def drift_velocity(run_datAa: tuple([dict[str, list[float]], np.ndarray])) -> float:
#     tn = run_data[0]["t"][-1]
#     velocities = run_data[1]
    
#     return _drift_velocity_inner(velocities, tn)

def drift_velocities_reduced(run_data):
    t_final = run_data[0]["t"][-1]
    x_final = run_data[1]
    return x_final / t_final

def drift_velocities_physical(run_data, params):
    v_hat = drift_velocities_reduced(run_data)
    return params.L * params.omega * v_hat

# def mean_drift_velocity(flashing_time:float, run_data= None, dt:float = 1e-4) -> tuple([float,float]):
#     if run_data is None:
#         params = Params(tau=flashing_time, particles=100, dt=dt)
#         run_data = run_simulation(x0_hat=0.0, t_end_hat=params.tau, params=params, flashing_on=True)
#     velocity = drift_velocity(run_data=run_data)
#     mean = np.mean(velocity)
#         print(velocity)
#     return mean, np.std(velocity,ddof=1) / np.sqrt(len(velocity))

def mean_drift_velocity(params: Params):
    run_data = run_simulation(
        x0_hat=0.0,
        t_end_hat=params.tau,
        params=params,
        flashing_on=True
    )
    vel = drift_velocities_physical(run_data, params)
    mean = np.mean(vel)
    sem = np.std(vel, ddof=1) / np.sqrt(len(vel))
    return mean, sem

#Function for running and plotting the drift velocity for a range of flashing times
def plot_drift_velocity_vs_flashing_time(flashing_times: np.ndarray) -> None:

    #defining arrays for storing the results of the simulations
    velocities = np.zeros(len(flashing_times))
    deviations = np.zeros(len(flashing_times))
    estimated_larger_particle = np.zeros(len(flashing_times))

    #Initializing parameters for the simulations, starting with the smallest particle size
    params12 = Params(radius=12e-9, tau=flashing_times[0])
    params24 = Params(radius=24e-9, tau=flashing_times[0])
    
    for i, t in enumerate(flashing_times):
        # drift_values = np.zeros(runs)
        # for j in range(runs):
        print(i)
        velocities[i], deviations[i] = mean_drift_velocity(params12) #running the simulation for each timestep
        estimated_larger_particle[i] = velocities[i] / 2
        params12.tau = t

    v_smooth = savgol_filter(velocities, window_length=11, polyorder=5)
    idx = np.argmax(v_smooth)

    velocities_24 = np.zeros(len(flashing_times))
    for i, t in enumerate(flashing_times):
        if(i%10 == 0):
            print(f"Running simulation for flashing time {t:.2f} s, particle radius 24 nm")
        velocities_24[i], _ = mean_drift_velocity(params24) #Running the simulation for each timestep with twice the particle size
        params24.tau = t

    print(f"Optimal flashing time: {flashing_times[idx]}, Drift velocity: {velocities[idx]}")
    plt.figure()
    plt.axvline(x= flashing_times[idx], color="red", linestyle="--", label=f"Optimal flashing time")
    plt.scatter(flashing_times, velocities, marker=".", label="Simulation 12 nm")
    m = matplotlib.markers.MarkerStyle("s", fillstyle="none", transform=None, capstyle=None, joinstyle=None)
    plt.scatter(flashing_times, estimated_larger_particle, marker=m, label="Prediction 24 nm",color="coral")
    plt.scatter(flashing_times, velocities_24, marker=".", label="Simulation 24 nm", color="coral")
    plt.xlabel(r"Flashing time $\tau$", fontsize= 24)
    plt.ylabel(r"Mean drift velocity ($\hat{x} / \hat{t}$)", fontsize=24)
    plt.yticks(fontsize= 16)
    plt.xticks(fontsize= 16)
    plt.legend(fontsize= 16)
    plt.tight_layout()
    plt.savefig(fname=f"/Users/torbjornkringeland/Desktop/NTNU/6/TFY4235/DNA-migratios/DNA-report/Images/drift_velocity_{len(flashing_times)}.pgf")