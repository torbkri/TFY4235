from koch_grid import *
from eigensystem import *

save_lattice()

save_solution(5, k=12, system = 'biharmonic')
save_solution(5, k=12, system = 'normal')
save_solution(3, k=1, system = 'normal')
save_solution(4, k=1, system = 'normal')
save_eigenvalues(3, k= 1000, system = 'normal')
save_eigenvalues(4, k= 1000, system = 'normal')
save_eigenvalues(4, k= 12, system = 'higher_order')
save_solution(4, k=12, system = 'biharmonic')