'''
Author: Mika Erdmann
- not used anymore - 
This script contains the inverse search (using GeneAI library) that implements the optimization of the parameter pm_welf (p_t)by minimizing the difference to the EA_inverse.get_vm_opt() result.

'''

from geneal.genetic_algorithms import ContinuousGenAlgSolver
import matplotlib.pyplot as plt
import EA_inverse
import numpy as np
import inverse_functions
import model1_functions
def ramsey_inverse(x):
    #x = np.reshape(x, (3, 18))
    #results = inverse_functions.run_model1_inverse(timeswitch=2, vm_weight=x[0], vm_cumdepr_new_inverse=x[1], vm_cumdepr_old_inverse=x[2])
    pm_welf = [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 7.5, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
    pm_welf[8 - int(n / 2):8 + int(n / 2)] = x
    results = inverse_functions.run_model1_inverse(timeswitch=2, vm_weight=pm_welf)
    vm_run = [inverse_functions.get_val(results.vm_cons), inverse_functions.get_val(results.vm_cesIO),inverse_functions.get_val(results.vm_invMacro)]
    residuals = np.sqrt(sum((vm_opt[0][8 - int(n / 2):8 + int(n / 2)] - vm_run[0][8 - int(n / 2):8 + int(n / 2)]) ** 2, 1) + sum((vm_opt[1][8 - int(n / 2):8 + int(n / 2)]- vm_run[1][8 - int(n / 2):8 + int(n / 2)]) ** 2, 1) + sum((vm_opt[2][8 - int(n / 2):8 + int(n / 2)]- vm_run[2][8 - int(n / 2):8 + int(n / 2)]) ** 2,1))
    return -residuals # TODO: -netagive?

if __name__ == "__main__":
    vm_opt = EA_inverse.get_vm_opt()
    n = 6
    solver = ContinuousGenAlgSolver(
        n_genes=n,
        fitness_function=ramsey_inverse,
        pop_size=50,
        max_gen=50,
        mutation_rate=0.1,
        selection_rate=0.7,
        selection_strategy="roulette_wheel",
        problem_type=float, # Defines the possible values as float numbers
        variables_limits=(0, 50) # Defines the limits of all variables between -10 and 10.
                                   # Alternatively one can pass an array of tuples defining the limits
                                   # for each variable: [(-10, 10), (0, 5), (0, 5), (-20, 20)]
    )

    solver.solve()

    print(f" The best fitness is: {solver.best_fitness_}")
    print(f"The best individual is: {solver.best_individual_}")
    x = solver.best_individual_
    pm_welf = [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 7.5, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
    pm_welf[8 - int(n / 2):8 + int(n / 2)] = x
    print(f"Values for best result: welf is {pm_welf}")
    results = inverse_functions.run_model1_inverse(timeswitch=2, vm_weight=pm_welf)
    vm_run = [inverse_functions.get_val(results.vm_cons), inverse_functions.get_val(results.vm_cesIO),
              inverse_functions.get_val(results.vm_invMacro)]
    tall_string = model1_functions.f_tall_string_b()
    tall_int = [int(i) for i in tall_string]
    # plot
    # Axis creation
    plt.subplot(2, 3, 1)
    plt.plot(tall_int, vm_opt[0], 'b')
    plt.legend("Consumption optimal")
    plt.subplot(2, 3, 2)
    plt.plot(tall_int, vm_run[0], 'b')
    plt.legend("Consumption run")
    plt.subplot(2, 3, 3)
    plt.title(f"Welf_weight optimized by GA", loc="center")
    plt.plot(tall_int, vm_opt[1], 'k')
    plt.legend("Kapital optimal")
    plt.subplot(2, 3, 4)
    plt.plot(tall_int, vm_run[1], 'k')
    plt.legend("Kapital run")
    plt.subplot(2, 3, 5)
    plt.plot(tall_int, vm_opt[2], "r")
    plt.legend(("Investment Optimal"), loc='upper right')
    plt.subplot(2, 3, 6)
    plt.plot(tall_int, vm_run[2], "r")
    plt.legend(("Investment run"), loc='upper right')
    plt.show()