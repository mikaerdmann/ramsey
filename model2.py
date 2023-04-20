'''
Author: Mika Erdmann
Project: Ramsey-Model and time steps
This script implements the model using additionally the cumulative investment as a variable. 
- not used anymore - 
'''

import pyomo.environ as pyo
import model2_functions as func
from pyomo.opt import SolverFactory
from pyomo.opt import SolverStatus, TerminationCondition
import numpy as np


def run_model(timeswitch, weight, depr, reg = 1):  # Do not change reg = 1 here!
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
    model.pm_ies = pyo.Param(initialize=1)  # default = 1
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
                                initialize= 1)
    # cumulative investment as a variable depending on vm_invmacro?
    model.vm_cuminv = pyo.Var(model.Tall, domain=pyo.NonNegativeReals, bounds = (1e-3, None), initialize=func.f_cum_inv)
    model.vm_utility = pyo.Var(model.Tall, domain=pyo.NonNegativeReals, bounds=(1e-3, None),
                               initialize=func.f_vm_utilitylog)

    model.vm_welfare_t = pyo.Var(model.Tall, domain=pyo.NonNegativeReals, bounds=(1e-3, 500000),
                                 initialize=func.f_initialize_welf)
    model.v_welfare = pyo.Var(domain=pyo.NonNegativeReals, bounds=(1e-3, 500000), initialize=0)

    # Objective
    def obj_expression(m):  # as input this function  has the model
        # cons = [m.vm_cons[i].value for i in range(0, len(m.Tall))]
        # utility = func.f_utility(cons, m.pm_ies.value)
        # welfare_t = [((1 / (1 + m.pm_prtp)) ** (m.pm_tall_val[i] - 2005) * m.pm_pop * utility[i] * m.pm_welf[i]) for i in range(0, len(utility))]  # stattdessen t??
        # welfare = sum(welfare_t)
        return pyo.summation(m.vm_welfare_t)

    # Welfare constraints

    def welfare_t_rule(m, t):
        return m.vm_welfare_t[t] == (1 / (1 + m.pm_prtp)) ** (m.pm_tall_val[t] - 2005) * m.pm_pop * m.vm_utility[t] * \
            m.pm_welf[t]

    def welfare_rule(m):
        return m.v_welfare == pyo.summation(m.vm_welfare_t)

    def utility_rule(m, t):
        if m.pm_ies == 1:
            return m.vm_utility[t] == pyo.log(m.vm_cons[t])
        else:
            return m.vm_utility[t] == (m.vm_cons[t] ** (1 - 1 / m.pm_ies) - 1) / (1 - 1 / m.pm_ies)

    # Constraints

    def production_constraint_rule(m, t):
        # return the expression for the production constraint for each t
        return m.vm_cons[t] + m.vm_invMacro[t] == func.f_prod(m.vm_cesIO[t], m.pm_cap_expo)


    def cuminv_constraint_rule(m,t):
        if t == 0:
            dts = range(1, m.pm_dt[t] + 1)
            return m.vm_cuminv[t] == sum(np.asarray([(1 - m.pm_delta_kap) ** dts[i] * (m.vm_invMacro[t] + (
                        m.vm_invMacro[t] - m.vm_invMacro[t]) / m.pm_dt[
                                t] * dts[i]) for i in range(0, len(dts))]))
            #  return m.vm_cuminv[t] == m.pm_cumdepr_new[t] * m.vm_invMacro[t]
        else:
            dts = range(1, m.pm_dt[t] + 1)
            return m.vm_cuminv[t] == sum(np.asarray([(1-m.pm_delta_kap) ** (m.pm_dt[t] - dts[i]) * (m.vm_invMacro[t - 1] + (m.vm_invMacro[t] - m.vm_invMacro[t - 1]) / m.pm_dt[
                t] * dts[i]) for i in range(0, len(dts))]))

    def capital_constraint_rule(m,t):
        if m.pm_tall_val[t] > m.pm_firstyear:
            # cum_inv: how much investment is left from the other periods
            return m.vm_cesIO[t] == (1 - m.pm_delta_kap) ** (m.pm_tall_val[t] - m.pm_tall_val[t - 1]) * m.vm_cesIO[t - 1] + m.vm_cuminv[t]
        else:
            return m.vm_cesIO[t] == m.sm_cesIO + m.vm_cuminv[t]

    # the next lines implements the objective in the model
    model.OBJ = pyo.Objective(rule=obj_expression, sense=pyo.maximize)

    # the next line creates one constraint for each member of the set model.Tall
    model.welf_constraint = pyo.Constraint(model.Tall, rule=welfare_t_rule)
    model.welf_constraint2 = pyo.Constraint(rule=welfare_rule)
    model.utility_constraint = pyo.Constraint(model.Tall, rule=utility_rule)
    model.prod_Constraint = pyo.Constraint(model.Tall,
                                           rule=production_constraint_rule)  # here the time is used as a range for the constraints
    model.cuminv_Constraint = pyo.Constraint(model.Tall,rule = cuminv_constraint_rule )
    model.cap_Constraint = pyo.Constraint(range(0, model.N.value),
                                          rule=capital_constraint_rule)  # here a range
    # The next lines solve the model
    opt = SolverFactory('ipopt', executable="C:\\Ipopt-3.14.11-win64-msvs2019-md\\bin\\ipopt.exe")
    # opt.set_options("halt_on_ampl_error=yes")
    # opt.options['print_level'] = 5
    # opt.options['output_file'] = "C:\\Users\\mikae\\PycharmProjects\\Ramseyvenv\\my_ipopt_log.txt"
    results = opt.solve(model, tee=True)
    # Solver result analisis
    if (results.solver.status == SolverStatus.ok) and (
            results.solver.termination_condition == TerminationCondition.optimal):
        print("this is feasible and optimal")
    elif results.solver.termination_condition == TerminationCondition.infeasible:
        print("do something about it? or exit?")
    else:
        # something else is wrong
        print(str(results.solver))
    return model

model = run_model(timeswitch=2, weight=1, depr= 2)

