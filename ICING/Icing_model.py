import numpy as np
from numba import njit

@njit
def energy(i:int,j:int,spins:np.ndarray,J:float = 1)->float:
    w,h = spins.shape
    spin = spins[i,j]
    return -J* 2 * spin * (spins[(i-1)%w,j]+spins[(i+1)%w,j]+spins[i,(j-1)%h]+spins[i,(j+1)%h])



