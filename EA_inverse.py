'''
Author: Mika Erdmann
Project: Ramsey-Model and time steps

Self written Self Adaptation EA for inverse search. (not used anymore, but still contains a function get_vm_opt() which is used in all inverse models.
'''
import numpy as np
import matplotlib.pyplot as plt
import inverse_functions
import model1_functions

def get_vm_opt(timeswitch):
    vm_opt_all = np.asarray(inverse_functions.get_optimal(m=1))
    # pm_dt = [5, 5, 5, 5, 5, 5, 5, 5, 5, 10, 10, 10, 10, 10, 10, 10, 10, 10]
    # y = [np.append(1, np.zeros((pm_dt[i] - 1))) for i in range(1, len(pm_dt))]
    # index = np.asarray(y[0])
    # for i in range(1, len(y)):
    #     index = np.append(index, y[i])
    # index = np.append(np.asarray(index, dtype=bool), True)
    # cons_opt = vm_opt_all[0][index]
    # cap_opt = vm_opt_all[1][index]
    # inv_opt = vm_opt_all[2][index]
    index = inverse_functions.get_indices(timeswitch)  # 3 or 1
    cons_opt = np.asarray([vm_opt_all[0][i] for i in index])
    cap_opt = np.asarray([vm_opt_all[1][i] for i in index])
    inv_opt = np.asarray([vm_opt_all[2][i] for i in index])
    vm_opt = [cons_opt, cap_opt, inv_opt]
    return vm_opt


def self_adapt(fit_func, tau=1 / 3, lamb=10, sigma=0.1, N=6, gen=10, print_results=False):
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
    parent = abs(10 * np.random.randn(N) + 10)
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

        def take_third(elem):
            return elem[2]

        parent = sorted(pop, reverse=False, key= take_third)[0]
        if parent[2] < best[2]:
            best = parent
        # Print results of current gen for user
        if print_results:
            print(f"The top result of Gen {g} is", parent[2])
    return best


def ramsey_inverse(x):
    #x = np.reshape(x, (3, 18))
    #results = inverse_functions.run_model1_inverse(timeswitch=2, vm_weight=x[0], vm_cumdepr_new_inverse=x[1], vm_cumdepr_old_inverse=x[2])
    pm_welf = [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 7.5, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
    pm_welf[8 - int(n / 2):8 + int(n / 2)] = x
    results = inverse_functions.run_model1_inverse(timeswitch=2, vm_weight=pm_welf)
    vm_run = [inverse_functions.get_val(results.vm_cons), inverse_functions.get_val(results.vm_cesIO),inverse_functions.get_val(results.vm_invMacro)]
    # vm_opt_all = np.asarray(inverse_functions.get_optimal(m=1))
    # pm_dt = [5, 5, 5, 5, 5, 5, 5, 5, 5, 10, 10, 10, 10, 10, 10, 10, 10, 10]
    # y = [np.append(1, np.zeros((pm_dt[i] - 1))) for i in range(1, len(pm_dt))]
    # index = np.asarray(y[0])
    # for i in range(1, len(y)):
    #     index = np.append(index, y[i])
    # index = np.append(np.asarray(index, dtype=bool), True)
    # cons_opt = vm_opt_all[0][index]
    # cap_opt = vm_opt_all[1][index]
    # inv_opt = vm_opt_all[2][index]
    # vm_opt = [cons_opt, cap_opt, inv_opt]
    residuals = np.sqrt(sum((vm_opt[0][8 - int(n / 2):8 + int(n / 2)] - vm_run[0][8 - int(n / 2):8 + int(n / 2)]) ** 2, 1) + sum((vm_opt[1][8 - int(n / 2):8 + int(n / 2)]- vm_run[1][8 - int(n / 2):8 + int(n / 2)]) ** 2, 1) + sum((vm_opt[2][8 - int(n / 2):8 + int(n / 2)]- vm_run[2][8 - int(n / 2):8 + int(n / 2)]) ** 2,1))
    return residuals


if __name__ == "__main__":
    for func in [ramsey_inverse]:  # add other functions
        vm_opt = get_vm_opt()
        n = 6
        res = self_adapt(func,tau=1, sigma=0.3, gen= 100, lamb = 100, N = n)
        #x = np.reshape(res[0], (3, 18))
        x = res[0]
        pm_welf = [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 7.5, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
        pm_welf[8 - int(n / 2):8 + int(n / 2)] = x

        print(f"Top result of {func.__name__} has {res[2]} for residuals")
        #print(f"Values for best result: welf is {x[0]}, pm_cumdepr_new is {x[1]}, pm_cumdepr_old is {x[2]} ")
        print(f"Values for best result: welf is {pm_welf}")
        results = inverse_functions.run_model1_inverse(timeswitch=2, vm_weight=pm_welf)
        vm_run = [inverse_functions.get_val(results.vm_cons), inverse_functions.get_val(results.vm_cesIO),
                  inverse_functions.get_val(results.vm_invMacro)]
        tall_string = model1_functions.f_tall_string_b()
        tall_int = [int(i) for i in tall_string]
        # plot
        # Axis creation
        plt.subplot(1, 3, 1)
        plt.plot(tall_int, vm_opt[0], 'b')
        plt.legend("Consumption optimal")
        plt.subplot(2, 3, 1)
        plt.plot(tall_int, vm_run[0], 'b')
        plt.legend("Consumption run")
        plt.subplot(1, 3, 2)
        plt.title(f"Welf_weight optimized by GA", loc="center")
        plt.plot(tall_int, vm_opt[1], 'k')
        plt.legend("Kapital optimal")
        plt.subplot(2, 3, 2)
        plt.plot(tall_int, vm_run[1], 'k')
        plt.legend("Kapital run")
        plt.subplot(1, 3, 3)
        plt.plot(tall_int, vm_opt[2], "r")
        plt.legend(("Investment Optimal"), loc='upper right')
        plt.subplot(2, 3, 3)
        plt.plot(tall_int, vm_run[2], "r")
        plt.legend(("Investment run"), loc='upper right')
        plt.show()