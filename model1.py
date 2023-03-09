'''
Author: Mika Erdmann
Project: Ramsey-Model and time steps
'''

import pyomo.environ as pyo
import model1_functions as func
from pyomo.opt import SolverFactory
from pyomo.opt import SolverStatus, TerminationCondition
import matplotlib.pyplot as plt
import numpy as np

# Different time definitions
# 1: 5 year time steps
# 2: 5 year time steps from 2020-2060, then 10 year time steps
# 3: 1 year time steps
# 4: 1 year time steps from 2020-2060, then 5 year time steps

# Different welf weightings:
# 0: no weighting
# 1: weighting ~ timestep pm_ts
# 2: weighting ~ average timesteps pm_4ts
# 3: weighting ~ 0.9 * pm_ts

# Different deprec. factors:
# 1: old is zero and new is ricardas script
# 2: uneven + even old and new
# 3: regression 1
# 4: regression 2

def run_model(timeswitch, weight, depr, reg =0):  # Do not change reg = 0 here!
    model = pyo.ConcreteModel()
    # Tall switch
    model.time = timeswitch
    tall_string = func.f_tall_string(model)
    model.welf_weight = weight
    model.depr = depr
    model.invreg = reg

    tall_int = [int(i) for i in tall_string]

    # Model time parameters
    model.N = pyo.Param(initialize=len(tall_int))
    model.Tall = pyo.RangeSet(0, model.N - 1)

    # Parameters
    model.pm_tall_val = pyo.Param(model.Tall, initialize=func.f_tall_val)
    model.pm_firstyear = pyo.Param(initialize=model.pm_tall_val[0])

    model.pm_dt = pyo.Param(model.Tall, initialize=func.f_dt)
    model.pm_ts = pyo.Param(model.Tall, initialize=func.f_pm_ts)
    model.pm_4ts = pyo.Param(model.Tall, initialize=func.f_pm_4ts)  # average over 4 ts
    model.pm_welf = pyo.Param(model.Tall, initialize=func.f_pm_welf)  # model.pm_ts or 1 or model.pm_4ts
    model.pm_delta_kap = pyo.Param(initialize=0.05)  # default = 0.05
    model.pm_cap_expo = pyo.Param(initialize=0.5)  # default = 0.5
    model.pm_ies = pyo.Param(initialize=0.9)  # default = 1
    model.pm_pop = pyo.Param(initialize=1)  # default = 1
    model.pm_prtp = pyo.Param(initialize=0.03)  # default = 0.03
    model.sm_cesIO = pyo.Param(initialize=25)  # default = 25

    # deprec factors
    model.pm_cumdepr_new = pyo.Param(model.Tall, initialize=func.f_cumdepr_new)  # 0 or func.f_cumdepr_new
    model.pm_cumdepr_old = pyo.Param(model.Tall, initialize=func.f_cumdepr_old)  # 0 or func.f_cumdepr_old)

    # Variables
    model.vm_cesIO = pyo.Var(model.Tall, within=pyo.NonNegativeReals, bounds=func.f_ces_bound,
                             initialize=model.sm_cesIO)
    model.vm_cons = pyo.Var(model.Tall, domain=pyo.NonNegativeReals, bounds=func.f_cons_bound,
                            initialize=model.sm_cesIO / 3)
    model.vm_invMacro = pyo.Var(model.Tall, domain=pyo.NonNegativeReals, bounds=(1e-3, None),
                                initialize= 0)
    model.vm_utility = pyo.Var(model.Tall, domain= pyo.NonNegativeReals, bounds=(1e-3, None), initialize= func.f_vm_utilitylog)

    model.vm_welfare_t = pyo.Var(model.Tall, domain=pyo.NonNegativeReals, bounds=(1e-3, 500000), initialize= func.f_initialize_welf)
    model.v_welfare = pyo.Var(domain=pyo.NonNegativeReals, bounds = (1e-3, 500000), initialize=0)

    # Objective
    def obj_expression(m):
        # cons = [m.vm_cons[i].value for i in range(0, len(m.Tall))]
        # utility = func.f_utility(cons, m.pm_ies.value)
        # welfare_t = [((1 / (1 + m.pm_prtp)) ** (m.pm_tall_val[i] - 2005) * m.pm_pop * utility[i] * m.pm_welf[i]) for i in range(0, len(utility))]  # stattdessen t??
        # welfare = sum(welfare_t)
        return pyo.summation(m.vm_welfare_t) # has as input a seperate variable that saves the welfare of every timestep

    # Constraints

    def welfare_t_rule(m,t):  # computes the welfare of every time step based on the seperate variable vm_utility, that is computed for every timestep
        return m.vm_welfare_t[t] == 1 / (1 + m.pm_prtp) ** (m.pm_tall_val[t] - 2005) * m.pm_pop * m.vm_utility[t] * m.pm_welf[t]


    def welfare_rule(m):
        return m.v_welfare == pyo.summation(m.vm_welfare_t)  # cpmputes the objective value in a variable (not used in OBJ, only for easier retrieving of Objective Value

    def utility_rule(m,t):  # computes the utility of every time step based on the consumption level inn that period
        if m.pm_ies == 1:
            return m.vm_utility[t] == pyo.log(m.vm_cons[t])
        else:
            return m.vm_utility[t] == (m.vm_cons[t] ** (1 - 1 / m.pm_ies) - 1) / (1 - 1 / m.pm_ies)


    def production_constraint_rule(m, t):
        # return the expression for the production constraint for each t
        return m.vm_cons[t] + m.vm_invMacro[t] == m.vm_cesIO[t] ** m.pm_cap_expo

    def capital_constraint_rule(m,t):
        if m.pm_tall_val[t] > m.pm_firstyear:
            return m.vm_cesIO[t] == (1 - m.pm_delta_kap) ** m.pm_dt[t] * m.vm_cesIO[t - 1] + m.pm_cumdepr_old[t] * m.vm_invMacro[t - 1] + m.pm_cumdepr_new[t] * m.vm_invMacro[t]
        else:
            return m.vm_cesIO[t] == m.sm_cesIO + m.pm_cumdepr_new[t] * m.vm_invMacro[t]

    # the next lines implements the objective in the model
    model.OBJ = pyo.Objective(rule=obj_expression, sense=pyo.maximize)

    # the next line creates one constraint for each member of the set model.Tall
    model.welf_constraint = pyo.Constraint(model.Tall, rule= welfare_t_rule)
    model.welf_constraint2 = pyo.Constraint(rule=welfare_rule)
    model.utility_constraint = pyo.Constraint(model.Tall, rule= utility_rule)
    model.prod_Constraint = pyo.Constraint(model.Tall,
                                           rule=production_constraint_rule)
    model.cap_Constraint = pyo.Constraint(model.Tall,
                                          rule=capital_constraint_rule)
    # The next lines solve the model
    opt = SolverFactory('ipopt', executable="C:\\Ipopt-3.14.11-win64-msvs2019-md\\bin\\ipopt.exe")
    # opt.set_options("halt_on_ampl_error=yes")
    #opt.options['print_level'] = 6
    # opt.options['output_file'] = "C:\\Users\\mikae\\PycharmProjects\\Ramseyvenv\\my_ipopt_log.txt"
    results = opt.solve(model, tee=True)
    # Solver result analisis
    if (results.solver.status == SolverStatus.ok) and (
            results.solver.termination_condition == TerminationCondition.optimal):
        print("this is feasible and optimal")
    elif results.solver.termination_condition == TerminationCondition.infeasible:
        print("infeasible: termination")
    else:
        # something else is wrong
        print(str(results.solver))
    return model

model = run_model(timeswitch=2, weight=1, depr= 3)
print(pyo.summation(model.vm_welfare_t))
model.pm_welf.pprint()
