from dataclasses import dataclass
import numpy as np

@dataclass
class Params:
    tau: float
    alpha: float = 0.2
    D:float = 3.25E-3
    dt: float = 1E-4
    periods: int = 10
    particles: int = 1000
