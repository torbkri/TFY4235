import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_path(order, operation):
    if operation == 'grid':
        return os.path.join(BASE_DIR, 'grids', f'l_{order}.npy')
    elif operation == 'classified':
        return os.path.join(BASE_DIR, 'grids', f'l_{order}_classified.npy')
    elif operation == 'dict':
        return os.path.join(BASE_DIR, 'grids', f'l_{order}_dict.pkl')
    elif operation == 'M_l':
        return os.path.join(BASE_DIR, 'grids', 'M_l.npy')
    elif operation == 'eigenvectors':
        return os.path.join(BASE_DIR, 'grids', f'eigenvectors_{order}.npy')
    elif operation == 'eigenvalues':
        return os.path.join(BASE_DIR, 'grids', f'eigenvalues_{order}.npy')
    elif operation == 'eigenvalues_9point':
        return os.path.join(BASE_DIR, 'grids', f'eigenvalues_{order}_9point.npy')
    elif operation == 'eigenvectors_9point':
        return os.path.join(BASE_DIR, 'grids', f'eigenvectors_{order}_9point.npy')
    elif operation == 'eigenvalues_biharmonic':
        return os.path.join(BASE_DIR, 'grids', f'eigenvalues_biharmonic_{order}.npy')
    elif operation == 'eigenvectors_biharmonic':
        return os.path.join(BASE_DIR, 'grids', f'eigenvectors_biharmonic_{order}.npy')
    elif operation == 'eigenvalues_square':
        return os.path.join(BASE_DIR, 'grids', f'eigenvalues_square_{order}.npy')
    elif operation == 'eigenvectors_square':
        return os.path.join(BASE_DIR, 'grids', f'eigenvectors_square_{order}.npy')
    elif operation == 'eigenvalues_LM':
        return os.path.join(BASE_DIR, 'grids', f'eigenvalues_LM_{order}.npy')
    elif operation == 'eigenvalues_9point_LM': 
        return os.path.join(BASE_DIR, 'grids', f'eigenvalues_9point_LM_{order}.npy')
    else:
        raise ValueError("Invalid system specified. Check config.py for valid paths") 


def load_system(order, system='normal'):
    if system == 'normal':
        eigenvalues = np.load(get_path(order, 'eigenvalues'))
        eigenvectors = np.load(get_path(order, 'eigenvectors'))
    elif system == 'higher_order':
        eigenvalues = np.load(get_path(order, 'eigenvalues_9point'))
        eigenvectors = np.load(get_path(order, 'eigenvectors_9point'))
    elif system == 'biharmonic':
        eigenvalues = np.load(get_path(order, 'eigenvalues_biharmonic'))
        eigenvectors = np.load(get_path(order, 'eigenvectors_biharmonic'))
    elif system == 'square':
        eigenvalues = np.load(get_path(order, 'eigenvalues_square'))
        eigenvectors = np.load(get_path(order, 'eigenvectors_square'))
    else:
        raise ValueError("Invalid system specified.")
    
    return eigenvalues, eigenvectors