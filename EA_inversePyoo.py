from geneal.genetic_algorithms import ContinuousGenAlgSolver
import matplotlib.pyplot as plt
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
    results = inverse_functions.run_model1_inverse(timeswitch=2, vm_weight=[5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 7.5, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
    vm_run = [inverse_functions.get_val(results.vm_cons), inverse_functions.get_val(results.vm_cesIO),
              inverse_functions.get_val(results.vm_invMacro)]
    tall_string = model1_functions.f_tall_string_b()
    tall_int = [int(i) for i in tall_string]
    n = 2
    residuals = np.sqrt(sum((vm_opt[0][8 - int(n / 2):8 + int(n / 2)] - vm_run[0][8 - int(n / 2):8 + int(n / 2)]) ** 2, 1) + sum((vm_opt[1][8 - int(n / 2):8 + int(n / 2)]- vm_run[1][8 - int(n / 2):8 + int(n / 2)]) ** 2, 1) + sum((vm_opt[2][8 - int(n / 2):8 + int(n / 2)]- vm_run[2][8 - int(n / 2):8 + int(n / 2)]) ** 2,1))
    print(f" The residuals are: {residuals}")

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
    #x = np.reshape(x, (3, 18))
    #results = inverse_functions.run_model1_inverse(timeswitch=2, vm_weight=x[0], vm_cumdepr_new_inverse=x[1], vm_cumdepr_old_inverse=x[2])
    pm_welf = [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 7.5, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
    pm_welf[8 - int(n / 2):8 + int(n / 2)] = x
    results = inverse_functions.run_model1_inverse(timeswitch=2, vm_weight=pm_welf)
    vm_run = [inverse_functions.get_val(results.vm_cons), inverse_functions.get_val(results.vm_cesIO),inverse_functions.get_val(results.vm_invMacro)]
    residuals = np.sqrt(sum((vm_opt[0][8 - int(n / 2):8 + int(n / 2)] - vm_run[0][8 - int(n / 2):8 + int(n / 2)]) ** 2, 1) + sum((vm_opt[1][8 - int(n / 2):8 + int(n / 2)]- vm_run[1][8 - int(n / 2):8 + int(n / 2)]) ** 2, 1) + sum((vm_opt[2][8 - int(n / 2):8 + int(n / 2)]- vm_run[2][8 - int(n / 2):8 + int(n / 2)]) ** 2,1))
    return residuals


if __name__ == "__main__" and t != 1:
    vm_opt = EA_inverse.get_vm_opt()
    n = 4

    Problem = Ramsey()
    #algorithm = CMAES()
    algorithm = NSGA2()
    #res = minimize(Problem,algorithm,seed=1, x0=np.random.random(Problem.n_var), verbose = True)
    res = minimize(Problem, algorithm, verbose = True)
    x = list(res.X)
    pm_welf = [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 7.5, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
    pm_welf[8 - int(n / 2):8 + int(n / 2)] = x
    results = inverse_functions.run_model1_inverse(timeswitch=2, vm_weight=pm_welf)
    vm_run = [inverse_functions.get_val(results.vm_cons), inverse_functions.get_val(results.vm_cesIO),
              inverse_functions.get_val(results.vm_invMacro)]
    tall_string = model1_functions.f_tall_string_b()
    tall_int = [int(i) for i in tall_string]
    residuals = np.sqrt(
        sum((vm_opt[0][8 - int(n / 2):8 + int(n / 2)] - vm_run[0][8 - int(n / 2):8 + int(n / 2)]) ** 2, 1) + sum(
            (vm_opt[1][8 - int(n / 2):8 + int(n / 2)] - vm_run[1][8 - int(n / 2):8 + int(n / 2)]) ** 2, 1) + sum(
            (vm_opt[2][8 - int(n / 2):8 + int(n / 2)] - vm_run[2][8 - int(n / 2):8 + int(n / 2)]) ** 2, 1))
    print(f" The residuals are: {residuals}")

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