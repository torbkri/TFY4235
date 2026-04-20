import numpy as np
from scipy.ndimage import binary_fill_holes
from numba import njit
from time import time
from config import get_path, load_system
import pickle

def rotation_matrix(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])


def grid_dim(order):
    if order == 0:
        return 2
    else:
        return grid_dim(order - 1) * 4 - 1


def koch_curve(order, L=1):
    koch_vectors = np.array(
        # traces out the koch curve
        [[1, 0], [0, 1], [1, 0], [0, -1], [0, -1], [1, 0], [0, 1]])

    if order == 0:
        return np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])

    else:
        points = koch_curve(order - 1)
        new_points = []

        for i in range(len(points) - 1):

            # idea: use rotation matrix to rotate the koch wave vector increments

            p1, p2 = points[i], points[i + 1]
            dist = np.linalg.norm(p2 - p1)
            point_vector = (p2 - p1)
            theta = np.arctan2(point_vector[1], point_vector[0])
            rotation_matrix_theta = rotation_matrix(theta)
            new_points.append(p1)  # include start point of segment

            for j in range(len(koch_vectors)):

                p = new_points[-1] + dist * \
                    rotation_matrix_theta @ (koch_vectors[j]) / 4
                new_points.append(p)

            new_points.append(p2)  # include end point of segment

        return np.array(new_points) * L


# use binary fill for speeeeeeeed
def koch_lattice(order, binary_fill=True):
    scale = 4**order
    curve = koch_curve(order, scale)
    curve = np.rint(curve).astype(int)

    dim = grid_dim(order)
    grid = np.zeros((dim, dim))
    min_x = np.min(curve[:, 0])
    min_y = np.min(curve[:, 1])
    curve[:, 0] -= min_x
    curve[:, 1] -= min_y

    if binary_fill:
        for p in curve:
            grid[p[1], p[0]] = 1

        binary_fill_holes(grid, output=grid)
        for p in curve:
            grid[p[1], p[0]] = -1

    else:
        for p in curve:
            grid[p[1], p[0]] = -1

        for i in range(dim):
            for j in range(dim):
                p = (i, j)
                if grid[j, i] == -1:
                    continue
                inside = ray_casting(curve, p)
                if inside:
                    grid[p[1], p[0]] = 1

    return grid


@njit
def ray_casting(curve, point, delta=0.5):
    x, y = point
    y += delta  # to avoid edge cases
    count = 0

    for i in range(len(curve) - 1):
        x1, y1 = curve[i]
        y2 = curve[i + 1][1]
        if x1 <= x or y1 == y2:
            continue

        if min(y1, y2) < y < max(y1, y2):   
            count += 1

    return count % 2 == 1


def classify_interior_points(grid):
    point_dict = {}
    counter = 1
    for i in range(len(grid)):
        for j in range(len(grid)):
            if grid[i, j] == 1:
                grid[i, j] = counter
                point_dict[counter] = (i, j)
                counter += 1

    return grid, counter - 1, point_dict


def execution_time(binary = True, average = 10, order = 3):
    start = time()
    for i in range(average):
        koch_lattice(order, binary_fill=binary)
    end = time()
    return (end - start) / average

def save_lattice():
    Ml_values = [0]
    for i in range(1,6):
        koch_grid = koch_lattice(i, binary_fill=True)
        grid, M_l, dict = classify_interior_points(koch_grid)
        np.save(get_path(i, 'classified'), grid.astype(int))
        #np.save(get_path(i, 'M_l'), M_l)
        dictpath = get_path(i, 'dict')
        with open(dictpath, 'wb') as f:
            pickle.dump(dict, f)
        Ml_values.append(int(M_l))

    np.save(get_path(i, 'M_l'), np.array(Ml_values, dtype=int))

if __name__ == '__main__':
    pass
