import numpy as np
import matplotlib.pyplot as plt
from config import get_path, load_system
from scipy.stats import linregress
from tqdm import tqdm


def find_d(order, dec=6, plot=False,  system='normal', n = 1000, ax = None):

    eigenvalues_squared = load_system(order, system)[0][:n]

    omega = np.sqrt(eigenvalues_squared)
    
    unique_omega, omega_counts = np.unique(omega.round(decimals=dec), return_counts=True)
    n_omega = np.cumsum(omega_counts)

    delta_n = unique_omega**2 / (4*np.pi) - n_omega

    log_omega = np.log(unique_omega)
    log_delta_n = np.log(delta_n)

    linear_fit = linregress(log_omega, log_delta_n)

    d_l = linear_fit.slope

    if ax is not None:    
        ax.loglog(unique_omega, delta_n, 'k', label=r'$\mathrm{\Delta N(\omega)}$', lw = 2)
        ax.loglog(np.exp(log_omega), np.exp(linear_fit.intercept +
                 linear_fit.slope * log_omega), 'r--', label = r'Linear fit', lw = 2)
    
    return d_l


def find_muliple_d(order, dec = 6, system = 'normal', n = 1000):

    eigenvalues_squared = load_system(order, system)[0][:n]

    length = len(eigenvalues_squared)
    if length < n:
        print(f"Warning: {n} is larger than the number of eigenvalues. Using {length} instead.")
        n = length
        

    all_omega = np.sqrt(eigenvalues_squared)

    dl = []

    for i in tqdm(range(10,n)):
        omega = all_omega[:i]
        unique_omega, omega_counts = np.unique(omega.round(decimals=dec), return_counts=True)
        n_omega = np.cumsum(omega_counts)
        delta_n = unique_omega**2 / (4*np.pi) - n_omega

        log_omega = np.log(unique_omega)
        log_delta_n = np.log(delta_n)

        linear_fit = linregress(log_omega, log_delta_n)

        d_l = linear_fit.slope

        dl.append(d_l)
    
    return dl

def compare_methods(order, n=11):

    try:
        path1 = get_path(order, 'eigenvalues')
        path2 = get_path(order, 'eigenvalues_9point')
    except ValueError:
        raise ValueError("Invalid path specified.")

    eigenvalues1 = np.sqrt(np.load(path1)[:n])
    eigenvalues2 = np.sqrt(np.load(path2)[:n])
    print(f"Comparing eigenvalues of order {order} fractal,")
    print(f"5 vs 9 point stencil:")

    for i in range(0, n):
        print(f"{i}: {eigenvalues1[i]:.4f}, {eigenvalues2[i]:.4f}")


def get_biharmonic_eigenvalues(order, n = 11):
    try:
        path = get_path(order, 'eigenvalues_biharmonic')
    except ValueError:
        raise ValueError("Invalid path specified.")

    eigenvalues = np.load(path)[:n]
    print(f"Eigenvalues of order {order} biharmonic fractal:")
    for i in range(0, n):
        print(f"{i}: {eigenvalues[i]:.4f}")


def scale_to_normal(order, n = 11):
    try:
        path = get_path(order, 'eigenvalues_biharmonic')
    except ValueError:
        raise ValueError("Invalid path specified.")

    eigenvalues_bi = np.sqrt(np.load(path)[:n])

    try:
        path = get_path(order, 'eigenvalues')
    except ValueError:
        raise ValueError("Invalid path specified.")
    
    eigenvalues = np.load(path)[:n]

    
    plt.plot(eigenvalues, eigenvalues_bi, 'k')
    plt.show()


if __name__ == "__main__":

    pass