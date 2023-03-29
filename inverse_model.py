'''
Author: Mika Erdmann
Project: Ramsey-Model and time steps
This script implements a pyomo model for inverse search.
'''

import pyomo.environ as pyo
import model1_functions
import inverse_functions as inf
from pyomo.opt import SolverFactory
from pyomo.opt import SolverStatus, TerminationCondition
import numpy as np


# optimization of parameters
def run_model(t):
    model = pyo.ConcreteModel()

    model.time = t  # the time representaton used in the model for which the parameters should be otpimitzed
    # Time Parameters
    tall_string = model1_functions.f_tall_string(model)
    tall_int = [int(i) for i in tall_string]
    # Model time parameters
    model.N = pyo.Param(initialize=len(tall_int))
    model.Tall = pyo.RangeSet(0, model.N - 1)
    # parameters necessary to initialize cumdepr_factors and pm_welf
    model.pm_dt = pyo.Param(model.Tall, initialize=model1_functions.f_dt)
    model.pm_delta_kap = pyo.Param(initialize=0.05)  # default = 0.05
    # Optimal variables
    model.cons_opt = pyo.Param(model.Tall, domain=pyo.NonNegativeReals,
                               initialize=inf.get_optimal_cons)
    model.cap_opt = pyo.Param(model.Tall, domain=pyo.NonNegativeReals, initialize=inf.get_optimal_cap)
    model.inv_opt = pyo.Param(model.Tall, domain=pyo.NonNegativeReals, initialize=inf.get_optimal_inv)
    # Variables to optimize
    model.vm_modeloutput = pyo.Var(domain=pyo.NonNegativeReals, initialize=1)  # TODO write initialize function
    model.vm_cumdepr_new = pyo.Var(model.Tall, domain=pyo.NonNegativeReals, bounds=(1e-5, None),
                                   initialize=model1_functions.f_cumdepr_new_2)
    model.vm_cumdepr_old = pyo.Var(model.Tall, domain=pyo.NonNegativeReals, bounds=(1e-5, None),
                                   initialize=model1_functions.f_cumdepr_old_2)
    model.vm_weight = pyo.Var(model.Tall, domain=pyo.NonNegativeReals, bounds=(1, 11),
                              initialize=model1_functions.f_pm_ts)
    model.vm_cons_run = pyo.Var(model.Tall, domain=pyo.NonNegativeReals, bounds=(1e-3, None),
                                initialize=1)  # TODO write initialize function
    model.vm_cap_run = pyo.Var(model.Tall, domain=pyo.NonNegativeReals, bounds=(1e-3, None),
                               initialize=1)  # TODO write initialize function
    model.vm_inv_run = pyo.Var(model.Tall, domain=pyo.NonNegativeReals, bounds=(1e-3, None),
                               initialize=1)  # TODO write initialize function

    # Objective rule:
    # Small deviation from optimal model output
    # TODO: How to weigh the difference between the three different opt_vms

    def objective_rule(m):

        return sum(((np.asarray(inf.get_par(m.cons_opt)) - np.asarray(inf.get_val(m.vm_cons_run))) ** 2 + (
                    np.asarray(inf.get_par(m.cap_opt)) - np.asarray(inf.get_val(m.vm_cap_run))) ** 2 + (
                            np.asarray(inf.get_par(m.inv_opt)) - np.asarray(
                                inf.get_val(m.vm_inv_run))) ** 2) ** 0.5)

    model.OBJ = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

    # Constraints:
    # Get model values (in every constraint)
    def constraint_rule_cons(m, t):
        return model.vm_cons_run == inf.cons_constraint(m, t)

    def constraint_rule_cap(m, t):
        return model.vm_cap_run == inf.cap_constraint(m, t)

    def constraint_rule_inv(m, t):
        return model.vm_inv_run == inf.inv_constraint(m, t)

    # construct constraints
    model.Constraint_cons = pyo.Constraint(model.Tall, rule=constraint_rule_cons)
    model.Constraint_cap = pyo.Constraint(model.Tall, rule=constraint_rule_cap)
    model.Constraint_inv = pyo.Constraint(model.Tall, rule=constraint_rule_inv)

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


if __name__ == "__main__":
    run_model(2)
