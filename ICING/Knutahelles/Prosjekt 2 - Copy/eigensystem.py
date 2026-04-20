import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import eigsh
from tqdm import tqdm
from config import get_path
import pickle
import time
# bruk LIL matrix, konverter til CSR før solving


def construct_eigensystem(order, path=''):

    M_l = np.load(get_path(order, 'M_l'))[order]

    if path == '':
        dictpath = get_path(order, 'dict')
        gridpath = get_path(order, 'classified')

    classification_dict = pickle.load(open(dictpath, 'rb'))
    classification_array = np.load(gridpath)

    eigensystem = lil_matrix((M_l, M_l))

    stencil = [[0, 1], [1, 0], [0, -1], [-1, 0]]

    diagonal_value = 4**(2*order+1)
    off_diag_value = -4**(2*order)

    eigensystem.setdiag(diagonal_value)

    for i in tqdm(range(1, M_l + 1), desc="Constructing eigensystem"):
        # finds the index of the point in classification_array
        index = np.array(classification_dict[i])

        for j in stencil:
            try:
                neigbour_index = tuple(index + j)
                neigbour_value = classification_array[neigbour_index]
                if neigbour_value > 0:
                    eigensystem[i-1, neigbour_value - 1] = off_diag_value

            except IndexError:
                pass

    eigensystem = csr_matrix(eigensystem)

    return eigensystem


def construct_higher_order_eigensystem(order, path=''):

    M_l = np.load(get_path(order, 'M_l'))[order]

    if path == '':
        dictpath = get_path(order, 'dict')
        gridpath = get_path(order, 'classified')

    classification_dict = pickle.load(open(dictpath, 'rb'))
    classification_array = np.load(gridpath)

    eigensystem = lil_matrix((M_l, M_l))

    stencil1 = [[0, 1], [1, 0], [0, -1], [-1, 0]]
    stencil2 = [[1, 1], [-1, 1], [-1, -1], [1, -1]]

    diagonal_value = 3*4**(2*order)
    stencil1_value = -4**(2*order-1/2)
    stencil2_value = -4**(2*order-1)

    eigensystem.setdiag(diagonal_value)

    for i in tqdm(range(1, M_l + 1), desc="Constructing eigensystem"):
        # finds the index of the point in classification_array
        index = np.array(classification_dict[i])

        for j in stencil1:
            try:
                neigbour_index = tuple(index + j)
                neigbour_value = classification_array[neigbour_index]
                if neigbour_value > 0:
                    eigensystem[i-1, neigbour_value - 1] = stencil1_value

            except IndexError:
                pass

        for j in stencil2:
            try:
                neigbour_index = tuple(index + j)
                neigbour_value = classification_array[neigbour_index]
                if neigbour_value > 0:
                    eigensystem[i-1, neigbour_value - 1] = stencil2_value

            except IndexError:
                pass

    eigensystem = csr_matrix(eigensystem)

    return eigensystem


def biharmonic_eigensystem(order, path=''):
    M_l = np.load(get_path(order, 'M_l'))[order]
    if path == '':
        dictpath = get_path(order, 'dict')
        gridpath = get_path(order, 'classified')

    classification_dict = pickle.load(open(dictpath, 'rb'))
    classification_array = np.load(gridpath)
    eigensystem = lil_matrix((M_l, M_l))

    stencil1 = [[0, 1], [1, 0], [0, -1], [-1, 0]]
    stencil2 = [[1, 1], [-1, 1], [-1, -1], [1, -1]]
    stencil3 = [[0, 2], [2, 0], [-2, 0], [0, -2]]

    diagonal_value = 20*4**(4*order)
    stencil1_value = -8*4**(4*order)
    stencil2_value = 2*4**(4*order)
    stencil3_value = 4**(4*order)

    eigensystem.setdiag(diagonal_value)

    for i in tqdm(range(1, M_l + 1), desc="Constructing eigensystem"):
        # finds the index of the point in classification_array
        index = np.array(classification_dict[i])

        for j in stencil1:
            try:
                neigbour_index = tuple(index + j)
                neigbour_value = classification_array[neigbour_index]
                if neigbour_value > 0:
                    eigensystem[i-1, neigbour_value - 1] = stencil1_value
                elif neigbour_value == -1:
                    eigensystem[i-1, i-1] += diagonal_value 
                    
            except IndexError:
                pass

        for j in stencil2:
            try:
                neigbour_index = tuple(index + j)
                neigbour_value = classification_array[neigbour_index]
                if neigbour_value > 0:
                    eigensystem[i-1, neigbour_value - 1] = stencil2_value
                # the below is for diagonal terms, but excluded after discussions with vitass
                #elif neigbour_value == -1:
                #    eigensystem[i-1, i-1] += 4*diagonal_value 

            except IndexError:
                pass

        for j in stencil3:
            try:
                neigbour_index = tuple(index + j)
                neigbour_value = classification_array[neigbour_index]
                if neigbour_value > 0:
                    eigensystem[i-1, neigbour_value - 1] = stencil3_value

            except IndexError:
                pass
    
    eigensystem = csr_matrix(eigensystem)

    return eigensystem


def solve_eigensystem(order, k, path='', system = 'normal', eigvecs = True):
    if system ==  'normal':
        eigensystem = construct_eigensystem(order, path)
    elif system == 'higher_order':
        eigensystem = construct_higher_order_eigensystem(order, path)
    elif system == 'biharmonic':
        eigensystem = biharmonic_eigensystem(order, path)
    else:
        raise ValueError("Invalid eigensystem specified.")
    
    print("Solving eigensystem. This may take a while for high k and/or large systems")
    if eigvecs:
        eigenvalues, eigenvectors = eigsh(eigensystem, k=k, sigma=0, return_eigenvectors=True)
        return eigenvalues, eigenvectors
    else:
        eigenvalues = eigsh(eigensystem, k=k, sigma=0, return_eigenvectors=False)
        return eigenvalues


def save_solution(order, k=20, path='', system = 'normal'):
    eigenvalues, eigenvectors = solve_eigensystem(order, k, path, system)
    if system == 'normal':
        np.save(get_path(order, 'eigenvectors'), eigenvectors)
        np.save(get_path(order, 'eigenvalues'), eigenvalues)
    elif system == 'higher_order':
        np.save(get_path(order, 'eigenvectors_9point'), eigenvectors)
        np.save(get_path(order, 'eigenvalues_9point'), eigenvalues)
    elif system == 'biharmonic':
        np.save(get_path(order, 'eigenvectors_biharmonic'), eigenvectors)
        np.save(get_path(order, 'eigenvalues_biharmonic'), eigenvalues)
    else:
        raise ValueError("Invalid eigensystem specified.")
    
    print("Solution saved successfully")

def save_eigenvalues(order, k = 500, system = 'normal', path = ''):
    eigenvalues = solve_eigensystem(order, k, path, system, eigvecs = False)
    if system == 'normal':
        np.save(get_path(order, 'eigenvalues'), eigenvalues)
    elif system == 'higher_order':
        np.save(get_path(order, 'eigenvalues_9point'), eigenvalues)
    else:    
        raise ValueError("Invalid eigensystem specified.")
    
    print("Solution saved successfully")

if __name__ == "__main__":
    save_solution(4, k=10, system = 'normal')
    
    pass