'''
Author: Mika Erdmann
Project: Ramsey-Model and time steps
Self Adaptation EA for inverse search
'''
import numpy as np

import inverse_functions


def self_adapt(fit_func, tau=1 / 3, lamb=100, sigma=0.1, N=54, gen=100, print_results=False):
    """
    Solves the optimization problem of fit_func with the (mu, λ)-Self Adaptation
    evolution strategy

    Parameters:
    fit_func (func): evaluation function for fitness of an individual - gets
      optimized
    tau (float): scaler for mutation rate - approximately = 1/sqrt(N)
    lamb (int): number of generated individuals per generation
    sigma (float): scaler of mutation - step length
    N (int): dimensions each indiviual has
    gen (int): number of generations that are computed
    print_results (bool): print the fitness of the best individual in each gen

    Return:
    tuple:
        ((array): the overall best individum,
        (float): corresponding sigma,
        (float): fitness)
    """
    b = 15
    a = 0.001
    parent = (b - a) * np.random.Generator.random(1) + a
    best = (parent, sigma, fit_func(parent))
    for g in range(gen):
        pop = []
        for _ in range(lamb):
            # mutation
            xi = tau * np.random.randn(1)
            z = np.random.randn(N)
            sig_i = parent[1] * np.exp(xi)
            offspring = parent[0] + sig_i * z
            # safe the offspring and its fitness
            pop.append((offspring, sig_i, fit_func(offspring)))
            # evaluation & selection
        parent = sorted(pop, reverse=False, key=lambda x: x[2])[0]
        if parent[2] < best[2]:
            best = parent
        # Print results of current gen for user
        if print_results:
            print(f"The top result of Gen {g} is", parent[2])
    return best


def ramsey_inverse(x):
    x = np.reshape(x, (3, 18))
    results = inverse_functions.run_model1_inverse(timeswitch=2, vm_weight=x[0], vm_cumdepr_new_inverse=x[1],
                                                   vm_cumdepr_old_inverse=x[2])
    vm_run = [inverse_functions.get_val(results.vm_cons), inverse_functions.get_val(results.vm_cap),
              inverse_functions.get_val(results.vm_inv)]
    vm_opt_all = np.asarray(inverse_functions.get_optimal())
    pm_dt = [5, 5, 5, 5, 5, 5, 5, 5, 5, 10, 10, 10, 10, 10, 10, 10, 10, 10]
    y = [np.append(1, np.zeros((pm_dt[i] - 1))) for i in range(1, len(pm_dt))]
    index = np.asarray(y[0])
    for i in range(1, len(y)):
        index = np.append(index, y[i])
    index = np.append(np.asarray(index, dtype=bool), True)
    cons_opt = vm_opt_all[0][index]
    cap_opt = vm_opt_all[1][index]
    inv_opt = vm_opt_all[2][index]
    vm_opt = [cons_opt, cap_opt, inv_opt]
    residuals = sum((vm_opt[0] - vm_run[0]) ** 2) + sum((vm_opt[1] - vm_run) ** 2) + sum((vm_opt[2] - vm_run[2]) ** 2)
    return residuals


if __name__ == "__main__":
    for func in [ramsey_inverse]:  # add other functions
        res = self_adapt(func, sigma=0.1)
        x = np.reshape(res[0], (3, 18))
        print(f"Top result of {func.__name__} has {res[2]} for residuals")
        print(f"Values for best result: welf is {x[0]}, pm_cumdepr_new is {x[1]}, pm_cumdepr_old is {x[2]} ")
