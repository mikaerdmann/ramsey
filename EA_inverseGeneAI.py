from geneal.genetic_algorithms import ContinuousGenAlgSolver

import EA_inverse

solver = ContinuousGenAlgSolver(
    n_genes=18,
    fitness_function=EA_inverse.ramsey_inverse,
    pop_size=10,
    max_gen=200,
    mutation_rate=0.1,
    selection_rate=0.6,
    selection_strategy="roulette_wheel",
    problem_type=float, # Defines the possible values as float numbers
    variables_limits=(0, 50) # Defines the limits of all variables between -10 and 10.
                               # Alternatively one can pass an array of tuples defining the limits
                               # for each variable: [(-10, 10), (0, 5), (0, 5), (-20, 20)]
)

solver.solve()