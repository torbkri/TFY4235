from dataclasses import dataclass
import numpy as np

@dataclass
class Params:
    tau: float
    alpha: float = 0.2
    D:float = 3.25E-3
    dt: float = 1E-4
    periods: int = 20
    particles: int = 1000
    R:float = 12E-6
    Omega: float = 80 / (6 *np.pi * 3*R * 1E-3)
    
