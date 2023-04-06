from geneal.genetic_algorithms import ContinuousGenAlgSolver
import matplotlib.pyplot as plt
from pymoo.termination import get_termination
from pymoo.termination.default import DefaultSingleObjectiveTermination
from pymoo.util.normalization import denormalize
import pymoo
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


def ramsey_inverse(x):
    # x = np.reshape(x, (3, 18))
    # results = inverse_functions.run_model1_inverse(timeswitch=2, vm_weight=x[0], vm_cumdepr_new_inverse=x[1], vm_cumdepr_old_inverse=x[2])
    pm_welf = [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 7.5, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
    # pm_welf[9 - int(n / 2):8 + int(n / 2)] = x
    pm_welf[8 - a:8 + b] = x
    results = inverse_functions.run_model1_inverse(timeswitch=2, vm_weight=pm_welf)
    vm_run = [inverse_functions.get_val(results.vm_cons), inverse_functions.get_val(results.vm_cesIO),
              inverse_functions.get_val(results.vm_invMacro)]
    # residuals1 = np.sqrt(
    #     sum((vm_opt[0][9 - int(n / 2):8 + int(n / 2)] - vm_run[0][9 - int(n / 2):8 + int(n / 2)]) ** 2, 1) + sum(
    #         (vm_opt[1][9- int(n / 2):8 + int(n / 2)] - vm_run[1][9- int(n / 2):8 + int(n / 2)]) ** 2, 1) + sum(
    #         (vm_opt[2][9 - int(n / 2):8 + int(n / 2)] - vm_run[2][9 - int(n / 2):8 + int(n / 2)]) ** 2, 1))
    residuals2 = np.sqrt(
        sum((vm_opt[0] - vm_run[0]) ** 2, 1) + sum((vm_opt[1] - vm_run[1]) ** 2, 1) + sum(
            (vm_opt[2] - vm_run[2]) ** 2,
            1))
    return residuals2


Ti = [0,1]
As = [8,1,0,0]
Bs = [10,3,2,1]
ABs = np.asarray([As,Bs])
Residuals_all = np.empty((2,4))
Residuals_t = np.empty((1,18))
for i in range(ABs.shape[1]):
    a = ABs[0][i]
    b = ABs[1][i]
    n = np.arange(8 - a, 8 + b).size
    for t in Ti:
        if __name__ == "__main__" and t == 1:
            vm_opt = EA_inverse.get_vm_opt(1)
            results = inverse_functions.run_model1_inverse(timeswitch=2,
                                                           vm_weight=[5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 7.5, 10.0,
                                                                      10.0,
                                                                      10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
            vm_run = [inverse_functions.get_val(results.vm_cons), inverse_functions.get_val(results.vm_cesIO),
                      inverse_functions.get_val(results.vm_invMacro)]
            tall_string = model1_functions.f_tall_string_b()
            tall_int = [int(i) for i in tall_string]

            residuals1 = np.sqrt(
                sum((vm_opt[0][8-a:8 + b] - vm_run[0][8 - a:8 + b]) ** 2, 1) + sum(
                    (vm_opt[1][8-a:8 + b] - vm_run[1][8-a:8 + b]) ** 2, 1) + sum(
                    (vm_opt[2][8-a:8 + b] - vm_run[2][8-a:8 + b]) ** 2, 1))
            # residuals1 = np.sqrt(
            #     sum((vm_opt[0][9 - int(n / 2):8 + int(n / 2)] - vm_run[0][9 - int(n / 2):8 + int(n / 2)]) ** 2, 1) + sum(
            #         (vm_opt[1][9 - int(n / 2):8 + int(n / 2)] - vm_run[1][9 - int(n / 2):8 + int(n / 2)]) ** 2, 1) + sum(
            #         (vm_opt[2][9 - int(n / 2):8 + int(n / 2)] - vm_run[2][9 - int(n / 2):8 + int(n / 2)]) ** 2, 1))
            print(f" The residuals are: {residuals1}")
            residuals2 = np.sqrt(
                sum((vm_opt[0] - vm_run[0]) ** 2, 1) + sum((vm_opt[1] - vm_run[1]) ** 2, 1) + sum(
                    (vm_opt[2] - vm_run[2]) ** 2,
                    1))
            print(f"The residuals over the whole time are {residuals2}")

            # Axis creation
            fig, axs = plt.subplots(3, 2)
            axs[0,0].plot(tall_int, vm_opt[0], 'b')
            axs[0,0] .legend("Consumption optimal")
            axs[0,1].plot(tall_int, vm_run[0], 'b')
            axs[0,1].legend("Consumption run")
            axs[1,0].plot(tall_int, vm_opt[1], 'k')
            axs[1,0].legend("Kapital optimal")
            axs[1,1].plot(tall_int, vm_run[1], 'k')
            axs[1,1].legend("Kapital run")
            axs[2,0].plot(tall_int, vm_opt[2], "r")
            axs[2,0].legend(("Investment Optimal"), loc='upper right')
            axs[2,1].plot(tall_int, vm_run[2], "r")
            axs[2,1].legend(("Investment run"), loc='upper right')

        if __name__ == "__main__" and t != 1:
            vm_opt = EA_inverse.get_vm_opt(1)
            np.random.seed(1)
            Problem = Ramsey()
            x0 = denormalize(np.random.random(Problem.n_var), Problem.xl, Problem.xu)
            algorithm = CMAES(x0=x0, sigma=0.1, restarts=11, restart_from_best=True, bipop=True)
            # termination = DefaultSingleObjectiveTermination(xtol=1e-8,cvtol=1e-6,ftol=1e-6,period=20,n_max_gen=1000,n_max_evals=100000)
            termination = get_termination("time", "00:10:00")

            # algorithm = NSGA2()
            # res = minimize(Problem,algorithm,seed=1, x0=np.random.random(Problem.n_var), verbose = True)
            res = minimize(Problem, algorithm, termination, verbose=True)
            x = list(res.X)
            pm_welf = [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 7.5, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
            pm_welf[9 - int(n / 2):8 + int(n / 2)] = x
            results = inverse_functions.run_model1_inverse(timeswitch=2, vm_weight=pm_welf)
            vm_run = [inverse_functions.get_val(results.vm_cons), inverse_functions.get_val(results.vm_cesIO),
                      inverse_functions.get_val(results.vm_invMacro)]
            tall_string = model1_functions.f_tall_string_b()
            tall_int = [int(i) for i in tall_string]
            residuals1 = np.sqrt(
                sum((vm_opt[0][8 - a:8 + b] - vm_run[0][8 - a:8 + b]) ** 2, 1) + sum(
                    (vm_opt[1][8 - a:8 + b] - vm_run[1][8 - a:8 + b]) ** 2, 1) + sum(
                    (vm_opt[2][8 - a:8 + b] - vm_run[2][8 - a:8 + b]) ** 2, 1))
            # residuals1 = np.sqrt(
            #     sum((vm_opt[0][9 - int(n / 2):8 + int(n / 2)] - vm_run[0][9 - int(n / 2):8 + int(n / 2)]) ** 2, 1) + sum(
            #         (vm_opt[1][9 - int(n / 2):8 + int(n / 2)] - vm_run[1][9 - int(n / 2):8 + int(n / 2)]) ** 2, 1) + sum(
            #         (vm_opt[2][9 - int(n / 2):8 + int(n / 2)] - vm_run[2][9 - int(n / 2):8 + int(n / 2)]) ** 2, 1))
            print(f" The residuals over the {n} time steps around 2060 are: {residuals1}")
            residuals2 = np.sqrt(sum((vm_opt[0] - vm_run[0]) ** 2, 1) + sum((vm_opt[1] - vm_run[1]) ** 2, 1) + sum(
                (vm_opt[2] - vm_run[2]) ** 2, 1))
            residuals3 = np.sqrt((vm_opt[0] - vm_run[0]) ** 2 + (vm_opt[1] - vm_run[1]) ** 2 + (vm_opt[2] - vm_run[2]) ** 2)

            print(f"The residuals over the whole time are {residuals2}")
            print(f"The residuals per time step are {residuals3}")
            Residuals_all[0][i] = residuals1
            Residuals_all[1][i] = residuals2
            Residuals_t[i] = residuals3
            # Axis creation
            fig3, axs = plt.subplots(3, 2)
            axs[0, 0].plot(tall_int, vm_opt[0], 'b')
            axs[0, 0].legend("Consumption optimal")
            axs[0, 1].plot(tall_int, vm_run[0], 'b')
            axs[0, 1].legend("Consumption run")
            axs[1, 0].plot(tall_int, vm_opt[1], 'k')
            axs[1, 0].legend("Kapital optimal")
            axs[1, 1].plot(tall_int, vm_run[1], 'k')
            axs[1, 1].legend("Kapital run")
            axs[2, 0].plot(tall_int, vm_opt[2], "r")
            axs[2, 0].legend(("Investment Optimal"), loc='upper right')
            axs[2, 1].plot(tall_int, vm_run[2], "r")
            axs[2, 1].legend(("Investment run"), loc='upper right')
            fig3.suptitle(f"Pyomo model using optimized pm_welf with n = {ni} and optimizing over all residuals.")

            fig2, axs = plt.subplots(1,2)
            axs[0].plot(tall_int, residuals3)
            fig2.suptitle("Residuals over K, C, I per time step")
        fig.savefig(f"C:\\Users\\mikae\\Documents\\Uni\\Project 1\\Results\\06042023\\Benchmark_n{n}.png")
        fig2.savefig(f"C:\\Users\\mikae\\Documents\\Uni\\Project 1\\Results\\06042023\\Opt_n{n}.png")
        fig3.savefig(f"C:\\Users\\mikae\\Documents\\Uni\\Project 1\\Results\\06042023\\Residuals_opt_n{n}.png")
        print(Residuals_all)
        print(Residuals_t)