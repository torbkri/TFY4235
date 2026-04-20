# task2.ipynb
Notebook used to generate the pdf for Problem 2.

# task_3_1.py
Implemets functions used for part 1 of problem 3.
Not run directly to produce results.
Note: the variance of E is erroneously calculated using the normalised energy. This is fixed by scaling during plotting.

# task_3_2.py
Implements functions used for part 2 of problem 3.
Not run directly.
Note: the variance of M is erroneously calculated using normalised magnetisation. Fixed during plotting.

# task3_1_saving.ipynb
Used to run and save the algorithms for part 1 of problem 3. 
Code for each subtask is indicated in the notebook.
"Run all" would run all simulations and save the results in this directory.
NOTE: the simulations were not seeded, so the results may not be exactly the same between runs, especially for T_c estimates. This is discussed in the report. You should, however, be able to see the same trends. 

# task_3_2_saving.ipynb
Does the same as above, but for part 2 of problem 3.

# task_3_vis.ipynb
Creates all plots used in the report and prints data used in tables.
"Run all" visualises all plots, provided the simulation files are in the directory.
The npz. files are provided in this zip, so if wanting to check if the plots work without running the simulations, run only this notebook.

# task3_convergence.ipynb
Checks convergence behaviour of observables. Might take a while to run.


# dependecies
All scripts should work with up to date(ish) versions of Python, NumPy, Matplotlib and Jupyter Notebook.
