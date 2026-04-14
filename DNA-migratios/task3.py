
def flashing(t_hat: float, tau_hat: float) -> float:
    phase = t_hat % tau_hat
    if phase < 0.75 * tau_hat:
        return 0.0
    return 1.0

def ratchet_potential(x_hat: float, alpha: float) -> float:
    x_cell = x_hat % 1.0

    if 0.0 <= x_cell < alpha:
        return x_cell / alpha
    else:
        return (1.0 - x_cell) / (1.0 - alpha)

def potential(x_hat: float, t_hat: float, alpha: float, tau_hat: float) -> float:
    return ratchet_potential(x_hat, alpha) * flashing(t_hat, tau_hat)

def dU_dx(x_hat: float, t_hat: float, alpha: float, tau_hat: float) -> float:
    if flashing(t_hat, tau_hat) == 0.0:
        return 0.0

    x_cell = x_hat % 1.0

    if 0.0 <= x_cell < alpha:
        return 1.0 / alpha
    else:
        return -1.0 / (1.0 - alpha)
    
def force(x_hat: float, t_hat: float, alpha: float, tau_hat: float) -> float:
    return -dU_dx(x_hat, t_hat, alpha, tau_hat)