from Params import Params
from task6 import run_simulation

def estimated_velocity(params: Params = None,run_data = None, scale: int = 1):
    if params is None:
        params = Params(tau= 7, particles=100, R = 3)
    if run_data is not None:
        return run_data['velocity'] * scale
    else:
        run_data = run_simulation(params)
        return run_data['velocity'] * scale