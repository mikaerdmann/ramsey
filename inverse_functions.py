'''
Author: Mika Erdmann
Project: Ramsey-Model and time steps
This script contains the necessary functions for the inverse search.
1. a slightly adapted verson of model 1, but has slightly different inputs and outputs
2. constraint functions

'''

import pyomo.environ as pyo
import model1_functions as func
from pyomo.opt import SolverFactory
from pyomo.opt import SolverStatus, TerminationCondition
import numpy as np
import model1
import model1_functions
import matplotlib.pyplot as plt


# 1. Adapted model 1 with different inputs
def run_model1_inverse(timeswitch, vm_weight,c_n = 0, c_o = 0, depr = 2, reg=0):  # Do not change reg = 0 here!
    model = pyo.ConcreteModel()
    # Tall switch
    model.time = timeswitch
    tall_string = func.f_tall_string(model)
    model.invreg = reg

    tall_int = [int(i) for i in tall_string]

    # Model time parameters
    model.N = pyo.Param(initialize=len(tall_int))
    model.Tall = pyo.RangeSet(0, model.N - 1)
    model.depr = depr
    # Parameters
    model.pm_tall_val = pyo.Param(model.Tall, initialize=func.f_tall_val)
    model.pm_firstyear = pyo.Param(initialize=model.pm_tall_val[0])

    model.pm_dt = pyo.Param(model.Tall, initialize=func.f_dt)
    model.pm_ts = pyo.Param(model.Tall, initialize=func.f_pm_ts)
    model.pm_4ts = pyo.Param(model.Tall, initialize=func.f_pm_4ts)  # average over 4 ts
    model.pm_delta_kap = pyo.Param(initialize=0.05)  # default = 0.05
    model.pm_cap_expo = pyo.Param(initialize=0.5)  # default = 0.5
    model.pm_ies = pyo.Param(initialize=0.9)  # default = 1
    model.pm_pop = pyo.Param(initialize=1)  # default = 1
    model.pm_prtp = pyo.Param(initialize=0.03)  # default = 0.03
    model.sm_cesIO = pyo.Param(initialize=25)  # default = 25
    # deprec factors
    if model.depr != 0:
        model.pm_cumdepr_new = pyo.Param(model.Tall, initialize=model1_functions.f_cumdepr_new)
        model.pm_cumdepr_old = pyo.Param(model.Tall, initialize=model1_functions.f_cumdepr_old)
    if model.depr == 0:
        model.pm_cumdepr_new = c_n
        model.pm_cumdepr_old = c_o

    # welf weight

    def vm_weight_rule(m, t):
        return vm_weight[t]

    model.pm_welf = pyo.Param(model.Tall, initialize=vm_weight_rule)

    # Variables
    model.vm_cesIO = pyo.Var(model.Tall, within=pyo.NonNegativeReals, bounds=func.f_ces_bound,
                             initialize=model.sm_cesIO)
    model.vm_cons = pyo.Var(model.Tall, domain=pyo.NonNegativeReals, bounds=func.f_cons_bound,
                            initialize=model.sm_cesIO / 3)
    model.vm_invMacro = pyo.Var(model.Tall, domain=pyo.Reals, bounds=(None, None),
                                initialize=0)
    model.vm_utility = pyo.Var(model.Tall, domain=pyo.NonNegativeReals, bounds=(1e-3, None),
                               initialize=func.f_vm_utilitylog)

    model.vm_welfare_t = pyo.Var(model.Tall, domain=pyo.NonNegativeReals, bounds=(1e-3, 500000),
                                 initialize=func.f_initialize_welf)
    model.v_welfare = pyo.Var(domain=pyo.NonNegativeReals, bounds=(1e-3, 500000), initialize=0)

    # Objective
    def obj_expression(m):
        # cons = [m.vm_cons[i].value for i in range(0, len(m.Tall))]
        # utility = func.f_utility(cons, m.pm_ies.value)
        # welfare_t = [((1 / (1 + m.pm_prtp)) ** (m.pm_tall_val[i] - 2005) * m.pm_pop * utility[i] * m.pm_welf[i]) for i in range(0, len(utility))]  # stattdessen t??
        # welfare = sum(welfare_t)
        return pyo.summation(
            m.vm_welfare_t)  # has as input a seperate variable that saves the welfare of every timestep

    # Constraints

    def welfare_t_rule(m,
                       t):  # computes the welfare of every time step based on the seperate variable vm_utility, that is computed for every timestep
        return m.vm_welfare_t[t] == 1 / (1 + m.pm_prtp) ** (m.pm_tall_val[t] - 2005) * m.pm_pop * m.vm_utility[t] * \
            m.pm_welf[t]

    def welfare_rule(m):
        return m.v_welfare == pyo.summation(
            m.vm_welfare_t)  # cpmputes the objective value in a variable (not used in OBJ, only for easier retrieving of Objective Value

    def utility_rule(m, t):  # computes the utility of every time step based on the consumption level inn that period
        if m.pm_ies == 1:
            return m.vm_utility[t] == pyo.log(m.vm_cons[t])
        else:
            return m.vm_utility[t] == (m.vm_cons[t] ** (1 - 1 / m.pm_ies) - 1) / (1 - 1 / m.pm_ies)

    def production_constraint_rule(m, t):
        # return the expression for the production constraint for each t
        return m.vm_cons[t] + m.vm_invMacro[t] == m.vm_cesIO[t] ** m.pm_cap_expo

    def capital_constraint_rule(m, t):
        if m.pm_tall_val[t] > m.pm_firstyear:
            return m.vm_cesIO[t] == (1 - m.pm_delta_kap) ** m.pm_dt[t] * m.vm_cesIO[t - 1] + m.pm_cumdepr_old[t] * \
                m.vm_invMacro[t - 1] + m.pm_cumdepr_new[t] * m.vm_invMacro[t]
        else:
            return m.vm_cesIO[t] == m.sm_cesIO + m.pm_cumdepr_new[t] * m.vm_invMacro[t]

    # the next lines implements the objective in the model
    model.OBJ = pyo.Objective(rule=obj_expression, sense=pyo.maximize)

    # the next line creates one constraint for each member of the set model.Tall
    model.welf_constraint = pyo.Constraint(model.Tall, rule=welfare_t_rule)
    model.welf_constraint2 = pyo.Constraint(rule=welfare_rule)
    model.utility_constraint = pyo.Constraint(model.Tall, rule=utility_rule)
    model.prod_Constraint = pyo.Constraint(model.Tall,
                                           rule=production_constraint_rule)
    model.cap_Constraint = pyo.Constraint(model.Tall,
                                          rule=capital_constraint_rule)
    # The next lines solve the model
    opt = SolverFactory('ipopt', executable="C:\\Ipopt-3.14.11-win64-msvs2019-md\\bin\\ipopt.exe")
    # opt.set_options("halt_on_ampl_error=yes")
    # opt.options['print_level'] = 6
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


# 2. constraint functions (only used in inverse_model.py, which is not used anymore)
def cons_constraint(m, t):
    w = get_val(m.vm_weight)
    cum_new = get_val(m.vm_cumdepr_new)
    cum_old = get_val(m.vm_cumdepr_old)
    runmodel = run_model1_inverse(timeswitch=m.time, vm_weight=w,
                                  vm_cumdepr_new_inverse=cum_new,
                                  vm_cumdepr_old_inverse=cum_old)
    cons_dict = runmodel.vm_cons.get_values()
    res_cons = cons_dict.values()
    return list(res_cons)[t]


def cap_constraint(m, t):
    runmodel = run_model1_inverse(timeswitch=m.time, vm_weight=m.vm_weight,
                                  vm_cumdepr_new_inverse=m.vm_cumdepr_new,
                                  vm_cumdepr_old_inverse=m.vm_cumdepr_old)
    cap_dict = runmodel.vm_cesIO.get_values()
    res_cap = cap_dict.values()
    return list(res_cap)[t]


def inv_constraint(m, t):
    runmodel = run_model1_inverse(timeswitch=m.time, vm_weight=m.vm_weight,
                                  vm_cumdepr_new_inverse=m.vm_cumdepr_new,
                                  vm_cumdepr_old_inverse=m.vm_cumdepr_old)
    inv_dict = runmodel.vm_invMacro.get_values()
    res_inv = inv_dict.values()
    return list(res_inv)[t]


# 3. Retrieving optimal values

# Optimal model, get optimal variables
def get_optimal(m):
    opt_model = model1.run_model(timeswitch=1, weight=1, depr=2) # timeswitch 3 or 1
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
    # pm_dt = model1_functions.f_tall_diff(m)
    # x = [np.append(1,np.zeros((pm_dt[i]-1))) for i in range(1,len(pm_dt))]
    # index = np.asarray(x[0])
    # for i in range(1, len(x)):
    #     index = np.append(index, x[i])
    # index = np.append(np.asarray(index, dtype= bool), True)
    index = get_indices(1) # 3 or 1
    cons_opt = np.asarray([vm_opt[0][i] for i in index])
    cap_opt = np.asarray([vm_opt[1][i] for i in index])
    inv_opt = np.asarray([vm_opt[2][i] for i in index])
    vm_opt = [cons_opt, cap_opt, inv_opt]
    return vm_opt


def get_optimal_cons(m, t):
    vm_opt = get_optimal_t(m)  # TODO write function for this
    return vm_opt[0][t]


def get_optimal_cap(m, t):
    vm_opt = get_optimal_t(m)
    return vm_opt[1][t]


def get_optimal_inv(m, t):
    vm_opt = get_optimal_t(m)
    return vm_opt[2][t]


def get_val(vm):
    vm_dict = vm.get_values()
    result_vm = vm_dict.values()
    vm_opt = list(result_vm)
    return vm_opt


def get_par(pm):
    pm_dict = pm.extract_values()
    result_pm = pm_dict.values()
    pm_opt = list(result_pm)
    return pm_opt


def f_cumdepr_new(m, t):
    return m.pm_cumdepr_new[t]


def f_cumdepr_old(m, t):
    return m.pm_cumdepr_old[t]


def f_weight(m, t):
    return m.weight[t]


def get_indices(timeswitch):
    """
    :param timeswitch: 3 for indices for tall_c, 1 for indices for tall_a as benchmark
    """
    tall_c = [int(i) for i in model1_functions.f_tall_string_c()]
    tall_b = [int(i) for i in model1_functions.f_tall_string_b()]
    tall_a = [int(i) for i in model1_functions.f_tall_string_a()]
    if timeswitch == 3:
        index_bc = [tall_c.index(i) for i in tall_b]
        return index_bc
    if timeswitch == 1:
        index_ba = [tall_a.index(i) for i in tall_b]
        return index_ba
