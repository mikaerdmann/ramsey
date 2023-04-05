from geneal.genetic_algorithms import ContinuousGenAlgSolver
import matplotlib.pyplot as plt
from pymoo.termination import get_termination
from pymoo.termination.default import DefaultSingleObjectiveTermination
from pymoo.util.normalization import denormalize

import EA_inverse
import numpy as np
import inverse_functions
import model1_functions
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.algorithms.moo.nsga2 import NSGA2


class Ramsey(ElementwiseProblem):
    def __init__(self):
        xl = np.zeros(n)
        xu = np.ones(n) * 30
        super().__init__(n_var=n, n_obj=1, n_ieq_constr=0, xl=xl, xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = ramsey_inverse(x)


import pymoo

t = 0
if t == 1:
    vm_opt = EA_inverse.get_vm_opt()
    results = inverse_functions.run_model1_inverse(timeswitch=2,
                                                   vm_weight=[5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 7.5, 10.0, 10.0,
                                                              10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
    vm_run = [inverse_functions.get_val(results.vm_cons), inverse_functions.get_val(results.vm_cesIO),
              inverse_functions.get_val(results.vm_invMacro)]
    tall_string = model1_functions.f_tall_string_b()
    tall_int = [int(i) for i in tall_string]
    n = 2
    residuals1 = np.sqrt(
        sum((vm_opt[0][8 - int(n / 2):8 + int(n / 2)] - vm_run[0][8 - int(n / 2):8 + int(n / 2)]) ** 2, 1) + sum(
            (vm_opt[1][8 - int(n / 2):8 + int(n / 2)] - vm_run[1][8 - int(n / 2):8 + int(n / 2)]) ** 2, 1) + sum(
            (vm_opt[2][8 - int(n / 2):8 + int(n / 2)] - vm_run[2][8 - int(n / 2):8 + int(n / 2)]) ** 2, 1))
    print(f" The residuals are: {residuals1}")
    residuals2 = np.sqrt(
        sum((vm_opt[0] - vm_run[0]) ** 2, 1) + sum((vm_opt[1] - vm_run[1]) ** 2, 1) + sum((vm_opt[2] - vm_run[2]) ** 2,
                                                                                          1))
    print(f"The residuals over the whole time are {residuals2}")
    # Axis creation
    plt.subplot(2, 3, 1)
    plt.plot(tall_int, vm_opt[0], 'b')
    plt.legend("Consumption optimal")
    plt.subplot(2, 3, 2)
    plt.plot(tall_int, vm_run[0], 'b')
    plt.legend("Consumption run")
    plt.subplot(2, 3, 3)
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


def ramsey_inverse(x):
    # x = np.reshape(x, (3, 18))
    # results = inverse_functions.run_model1_inverse(timeswitch=2, vm_weight=x[0], vm_cumdepr_new_inverse=x[1], vm_cumdepr_old_inverse=x[2])
    pm_welf = [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 7.5, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
    pm_welf[8 - int(n / 2):8 + int(n / 2)] = x
    results = inverse_functions.run_model1_inverse(timeswitch=2, vm_weight=pm_welf)
    vm_run = [inverse_functions.get_val(results.vm_cons), inverse_functions.get_val(results.vm_cesIO),
              inverse_functions.get_val(results.vm_invMacro)]
    residuals1 = np.sqrt(
        sum((vm_opt[0][8 - int(n / 2):8 + int(n / 2)] - vm_run[0][8 - int(n / 2):8 + int(n / 2)]) ** 2, 1) + sum(
            (vm_opt[1][8 - int(n / 2):8 + int(n / 2)] - vm_run[1][8 - int(n / 2):8 + int(n / 2)]) ** 2, 1) + sum(
            (vm_opt[2][8 - int(n / 2):8 + int(n / 2)] - vm_run[2][8 - int(n / 2):8 + int(n / 2)]) ** 2, 1))
    residuals2 = np.sqrt(
        sum((vm_opt[0] - vm_run[0]) ** 2, 1) + sum((vm_opt[1] - vm_run[1]) ** 2, 1) + sum((vm_opt[2] - vm_run[2]) ** 2,
                                                                                          1))
    return residuals2


if __name__ == "__main__" and t != 1:
    vm_opt = EA_inverse.get_vm_opt()
    n = 6
    np.random.seed(1)
    Problem = Ramsey()
    x0 = denormalize(np.random.random(Problem.n_var), Problem.xl, Problem.xu)
    algorithm = CMAES(x0=x0, sigma=0.1, restarts=11, restart_from_best=True,bipop=True)
    #termination = DefaultSingleObjectiveTermination(xtol=1e-8,cvtol=1e-6,ftol=1e-6,period=20,n_max_gen=1000,n_max_evals=100000)
    termination = get_termination("time", "00:05:00")

    # algorithm = NSGA2()
    # res = minimize(Problem,algorithm,seed=1, x0=np.random.random(Problem.n_var), verbose = True)
    res = minimize(Problem, algorithm, termination, verbose=True, pf = Problem.pareto_front)
    x = list(res.X)
    pm_welf = [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 7.5, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
    pm_welf[8 - int(n / 2):8 + int(n / 2)] = x
    results = inverse_functions.run_model1_inverse(timeswitch=2, vm_weight=pm_welf)
    vm_run = [inverse_functions.get_val(results.vm_cons), inverse_functions.get_val(results.vm_cesIO),
              inverse_functions.get_val(results.vm_invMacro)]
    tall_string = model1_functions.f_tall_string_b()
    tall_int = [int(i) for i in tall_string]
    residuals1 = np.sqrt(
        sum((vm_opt[0][8 - int(n / 2):8 + int(n / 2)] - vm_run[0][8 - int(n / 2):8 + int(n / 2)]) ** 2, 1) + sum(
            (vm_opt[1][8 - int(n / 2):8 + int(n / 2)] - vm_run[1][8 - int(n / 2):8 + int(n / 2)]) ** 2, 1) + sum(
            (vm_opt[2][8 - int(n / 2):8 + int(n / 2)] - vm_run[2][8 - int(n / 2):8 + int(n / 2)]) ** 2, 1))
    print(f" The residuals over the {n} time steps around 2060 are: {residuals1}")
    residuals2 = np.sqrt(
        sum((vm_opt[0] - vm_run[0]) ** 2, 1) + sum((vm_opt[1] - vm_run[1]) ** 2, 1) + sum((vm_opt[2] - vm_run[2]) ** 2,
                                                                                          1))
    residuals3 = np.sqrt((vm_opt[0] - vm_run[0]) ** 2 + (vm_opt[1] - vm_run[1]) ** 2 + (vm_opt[2] - vm_run[2]) ** 2)

    print(f"The residuals over the whole time are {residuals2}")

    # Axis creation
    plt.subplot(2, 3, 1)
    plt.plot(tall_int, vm_opt[0], 'b')
    plt.legend("Consumption optimal")
    plt.subplot(2, 3, 2)
    plt.plot(tall_int, vm_run[0], 'b')
    plt.legend("Consumption run")
    plt.subplot(2, 3, 3)
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
    plt.subplot(1,2,1)
    plt.plot(tall_int, residuals3)
    plt.title("Residuals over K, C, I per time step")
    plt.show()

