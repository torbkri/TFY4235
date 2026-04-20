import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from koch_grid import koch_curve
from config import get_path

def _plot_koch_curve(order, ax):
    # 0.5, 0.25 and 0.15 look nice with tight_layout, dpi = 300
    lw = 0.5
    if order == 4:
        lw = 0.25
    elif order == 5:
        lw = 0.15
    
    curve = koch_curve(order, 4**order)
    min_x = np.min(curve[:, 0])
    min_y = np.min(curve[:, 1])
    curve[:, 0] -= min_x
    curve[:, 1] -= min_y


    ax.plot(curve[:, 0], curve[:, 1], 'k', alpha=1, lw = lw)


# this is horribly inefficient if called repeatedly, fix if needed
def visualise_eigenvector(order, ax, eig, system = 'normal', pad = 0):

    if system == 'normal':
        eigenvectors = np.load(get_path(order, 'eigenvectors'))
    elif system == 'higher_order':
        eigenvectors = np.load(get_path(order, 'eigenvectors_9point'))
    elif system == 'biharmonic':
        eigenvectors = np.load(get_path(order, 'eigenvectors_biharmonic'))
    else:
        raise ValueError("Invalid system specified.")

    eigenvector = eigenvectors[:, eig]
    
    grid = np.load(get_path(order, 'classified'))
    M_l = np.load(get_path(order, 'M_l'))[order]
    
    grid2 = np.zeros(grid.shape)
    indices = np.argwhere(grid > 0)
    del grid

    for index, (i, j) in enumerate(indices):
        if index < M_l:
            grid2[i, j] = eigenvector[index]
    
    x = np.arange(grid2.shape[0])
    y = np.arange(grid2.shape[1])
    
    vmax = np.max(np.abs(grid2))
    vmin = -vmax
    divnorm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    ax.pcolormesh(x, y, grid2, shading='gouraud', cmap = 'seismic', norm = divnorm)
    
    _plot_koch_curve(order, ax)
    ax.set_aspect('equal')

    pad_x = pad / 100 * grid2.shape[0]
    pad_y = pad / 100 * grid2.shape[1]

    ax.set_xlim(-pad_x, grid2.shape[0] + pad_x)
    ax.set_ylim(-pad_y, grid2.shape[1] + pad_y)
    ax.set_xticks([])
    ax.set_yticks([])


if __name__ == '__main__':

    pass