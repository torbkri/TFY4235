from dataclasses import dataclass

@dataclass
class Params:
    alpha: float
    D: float
    tau: float
    dt: float

def main():
    params = Params(alpha=0.5, D=1.0, tau=1.0, dt=0.01)
    print(params)

main()