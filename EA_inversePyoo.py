from geneal.genetic_algorithms import ContinuousGenAlgSolver
import matplotlib.pyplot as plt
import EA_inverse
import numpy as np
import inverse_functions
import model1_functions
import pymoo
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
    n = 10
    from pymoo.core.problem import Problem

    class Ramsey(Problem):
        def __init__(self):
            super().__init__(n_var=n, n_obj=1, n_ieq_constr=0, xl=0.0, xu=30.0)

        def _evaluate(self, x, out, *args, **kwargs):
            out["F"] = ramsey_inverse(x)

    from pymoo.optimize import minimize
    from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
    Problem = Ramsey()
    algorithm = CMAES()

    res = minimize(Problem,algorithm,seed=1, x0=np.random.random(Problem.n_var))