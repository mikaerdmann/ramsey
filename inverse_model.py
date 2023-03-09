'''
Author: Mika Erdmann
Project: Ramsey-Model and time steps
This script implements a pyomo model for inverse search
'''

import pyomo.environ as pyo
import model1
import model1_functions
import model1_inverse
from pyomo.opt import SolverFactory
from pyomo.opt import SolverStatus, TerminationCondition
import numpy as np


# Optimal model, get optimal variables
def get_optimal(m):
    opt_model = model1.run_model(timeswitch=3, weight=1, depr=2)
    cons_opt_dict = opt_model.vm_cons.get_values()
    cap_opt_dict = opt_model.vm_cesIO.get_values()
    inv_opt_dict = opt_model.vm_invMacro.get_values()

    result_cons = cons_opt_dict.values()
    cons_opt = list(result_cons)
    results_cap = cap_opt_dict.values()
    cap_opt = list(results_cap)
    results_inv = inv_opt_dict.values()
    inv_opt = list(results_inv)
    vm_opt = [cons_opt, cap_opt, inv_opt]
    return vm_opt  # for every year

def get_optimal_t(m):  # get only the needed values of vm_opt
    vm_opt = np.asarray(get_optimal(m))
    pm_dt = model1_functions.f_tall_diff(m)
    x = [np.append(np.zeros((pm_dt[i]-1)), 1) for i in range(1,len(pm_dt))]
    index = np.asarray(x[0])
    for i in range(1, len(x)):
        index = np.append(index, x[i])
    index = np.asarray(index, dtype= bool)
    cons_opt = vm_opt[0][index]
    cap_opt = vm_opt[1][index]
    inv_opt = vm_opt[2][index]
    vm_opt = [cons_opt, cap_opt, inv_opt]
    return vm_opt

# optimization of parameters
def run_model(t):
    model = pyo.ConcreteModel()

    model.time = t  # the time representaton used in the model for which the parameters should be otpimitzed
    # Time Parameters
    tall_string = model1_functions.f_tall_string(model)
    tall_int = [int(i) for i in tall_string]
    vm_opt = get_optimal_t(model)
    model.cons_opt = vm_opt[0]
    model.cap_opt = vm_opt[1]
    model.inv_opt = vm_opt[2]
    # Model time parameters
    model.N = pyo.Param(initialize=len(tall_int))
    model.Tall = pyo.RangeSet(0, model.N - 1)
    # Variables to optimize
    model.vm_modeloutput = pyo.Var(domain=pyo.NonNegativeReals, initialize= get_optimal_t)
    model.vm_cumdepr_new = pyo.Var(model.Tall, domain=pyo.NonNegativeReals, bounds=(1e-5, None),
                                   initialize=model1_functions.f_cumdepr_new_2)
    model.vm_cumdepr_old = pyo.Var(model.Tall, domain=pyo.NonNegativeReals, bounds=(1e-5, None),
                                   initialize=model1_functions.f_cumdepr_old_2)
    model.vm_weight = pyo.Var(model.Tall, domain=pyo.NonNegativeReals, bounds = (1,11), initialize=model1_functions.f_pm_ts)

    # Objective rule:
    # Small deviation from optimal model output
    # TODO: How to weigh the difference between the three different opt_vms
    def objective_rule(m):
        difference = m.vm_opt-m.vm_modeloutput
        return difference

    model.OBJ = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

    # Constraints:

    # Get model values (in every constraint)
    def constraint_rule_1(m):

        runmodel = model1_inverse.run_model1_inverse(timeswitch=m.time, vm_weight= m.vm_weight, vm_cumdepr_new_inverse=m.vm_cumdepr_new, vm_cumdepr_old_inverse=m.vm_cumdepr_old)
        cons_dict = runmodel.vm_cons.get_values()
        cap_dict = runmodel.vm_cesIO.get_values()
        inv_dict = runmodel.vm_invMacro.get_values()

        res_cons = cons_dict.values()
        model_cons_opt = list(res_cons)
        res_cap = cap_dict.values()
        model_cap_opt = list(res_cap)
        res_inv = inv_dict.values()
        model_inv_opt = list(res_inv)
        vm_run = [model_cons_opt, model_cap_opt, model_inv_opt]
        return model.vm_modeloutput == vm_run # TODO Achtung, muss das hier wieder delokalisiert werden

    model.Constraint1 = pyo.Constraint(rule=constraint_rule_1)

    opt = SolverFactory('ipopt', executable="C:\\Ipopt-3.14.11-win64-msvs2019-md\\bin\\ipopt.exe")
    # opt.set_options("halt_on_ampl_error=yes")
    # opt.options['print_level'] = 5
    # opt.options['output_file'] = "C:\\Users\\mikae\\PycharmProjects\\Ramseyvenv\\my_ipopt_log.txt"
    results = opt.solve(model, tee=True)
    # Solver result analysis
    if (results.solver.status == SolverStatus.ok) and (
            results.solver.termination_condition == TerminationCondition.optimal):
        print("this is feasible and optimal")
    elif results.solver.termination_condition == TerminationCondition.infeasible:
        print("do something about it? or exit?")
    else:
        # something else is wrong
        print(str(results.solver))
    return model

run_model(2)
