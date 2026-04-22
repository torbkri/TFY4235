# from dataclasses import dataclass
# import numpy as np

# @dataclass
# class Params:
#     tau: float
#     alpha: float = 0.2
#     D:float = 3.25E-3
#     dt: float = 1E-4
#     periods: int = 20
#     particles: int = 1000
#     R:float = 12E-6
#     Omega: float = 80 / (6 *np.pi * 3*R * 1E-3)
    
from dataclasses import dataclass
import numpy as np

@dataclass
class Params:
    tau: float
    alpha: float = 0.2
    D: float = 3.25e-4
    dt: float = 1e-4
    periods: int = 20
    particles: int = 100
    radius: float = 12e-9
    eta: float = 1e-3
    L: float = 20e-6
    deltaU: float = 80 * 1.602176634e-19  # J

    @property
    def gamma(self):
        return 6 * np.pi * self.eta * self.radius

    @property
    def omega(self):
        return self.deltaU / (self.gamma * self.L**2)
    
    @property
    def t_end(self):
        return self.periods * self.tau

    @classmethod
    def from_physical_tau(cls, tau_phys, dt_phys, radius):
        temp = cls(tau=1.0, dt=1.0, radius=radius)
        tau_hat = tau_phys * temp.omega
        dt_hat = dt_phys * temp.omega
        return cls(tau=tau_hat, dt=dt_hat, radius=radius)
