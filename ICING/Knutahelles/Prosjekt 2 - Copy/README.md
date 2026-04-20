# config.py
Used for file management. 
The folder'grids' should be created in the same directory as the code.

# koch_grid.py
Creates the koch grid, places it on a lattice and classifies all lattice points. 
Performance comparison functions for the two classification methods are also included.

# eigensystem.py
Creates and solves the eigensystems for the fractals. 
Need to run koch_grid.py first.

# finding_parameters.py
Does regression analysis to estimate the fractal dimension.
Also includes convenience functions for printing calculated eigenvalues.

# visualise.py
Maps the solutions of the eigensystem back to the lattice for visualisation purposes.

# report_figs.ipynb
Creates the final report figures.

### How to run this code
1. Run the main.py script
2. Run report_figs.ipynb
3. Run the compare_methods() and get_biharmonic() functions in finding_parameters.py for l = 4 and l = 5 if wanting to check the eigenvalues.

# the scripts provided do not include any hard to find packages. Should work with standard versions of every dependency downloaded through pip