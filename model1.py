'''
Author: Mika Erdmann
Project: Ramsey-Model and time steps
'''

import pyomo.environ as pyo
from matplotlib import rcParams

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
    model.vm_invMacro = pyo.Var(model.Tall, domain= pyo.Reals, bounds=(None, None), # Achtung, hier wird invMacro nicht gebounded
                                initialize= 0)
    # model.vm_invMacro = pyo.Var(model.Tall, domain= pyo.Reals, bounds=(1e-3, None),
    #                                 initialize= 0)
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
    #opt.options['print_level'] = 0
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

if __name__ == "__main__":
    rcParams['figure.figsize'] = 7, 7
    Ti = [1,2,3,4,5,6,7,8]
    timecase = ["Equal timesteps \n with "r"$\Delta_t = 5$", "Unequal timesteps \n with "r"$\Delta_t = 5$ for $t<2060$", "Equal timesteps \n with "r"$\Delta_t = 1$","Unqual timesteps \n with "r"$\Delta_t = 1$ for $t<2060$", "One time timestep change", "One time timestep change", "One time timestep change", "One time timestep change"]
    for t in Ti:
        model = run_model(timeswitch=t, weight=1, depr= 2)
        print(pyo.summation(model.vm_welfare_t))
        print(model.pm_cumdepr_new.extract_values())
        print(model.pm_cumdepr_old.extract_values())
        cons_opt_dict = model.vm_cons.get_values()
        cap_opt_dict = model.vm_cesIO.get_values()
        inv_opt_dict = model.vm_invMacro.get_values()

        # Convert object to a list
        result_cons = cons_opt_dict.values()
        cons_opt = list(result_cons)
        results_cap = cap_opt_dict.values()
        cap_opt = list(results_cap)
        results_inv = inv_opt_dict.values()
        inv_opt = list(results_inv)
        pm_tall_val = model.pm_tall_val.extract_values()
        tall_int = list(pm_tall_val.values())
        res = [cons_opt, cap_opt, inv_opt]
        tickstep = [[1, 2], [5, 20], [1, 5]]
        # Axis creation
        # Visualisation of model runs in a loop
        fig3, axs = plt.subplots(3, 1)
        for a in range(0, 3):
            # Major ticks every 20, minor ticks every 5
            major_ticks = np.arange(-10, max(res[a]), tickstep[a][1])
            minor_ticks = np.arange(-10, max(res[a]), tickstep[a][0])
            major_ticksx = np.arange(0, 2160, 25)
            minor_ticksx = np.arange(0, 2160, 5)
            axs[a].set_xticks(major_ticksx)
            axs[a].set_xticks(minor_ticksx, minor=True)
            axs[a].set_yticks(major_ticks)
            axs[a].set_yticks(minor_ticks, minor=True)
            # Or if you want different settings for the grids:
            axs[a].grid(which='minor', alpha=0.2)
            axs[a].grid(which='major', alpha=0.5)

        axs[0].set_ylim(2, 10)
        axs[1].set_ylim(0, 50)
        axs[2].set_ylim(-8, 5)
        # plot
        axs[0].plot(tall_int, cons_opt, 'b')
        axs[0].scatter(tall_int, cons_opt, marker=".", linewidths=1, color="b")
        axs[0].set_ylabel("Consumption")
        axs[0].set_title(f"Optimal paths for time case: \n {timecase[t-1]}")
        #axs[0].set_xlabel("Time")
        axs[1].plot(tall_int, cap_opt, 'k')
        axs[1].scatter(tall_int, cap_opt, marker=".", linewidths=1, color="k")
        axs[1].set_ylabel("Capital")
        #axs[1].set_xlabel("Time")
        axs[2].plot(tall_int, inv_opt, 'r')
        axs[2].scatter(tall_int, inv_opt, marker=".", linewidths=1, color="r")
        axs[2].set_ylabel("Investment")
        axs[2].set_xlabel("Time")

        plt.subplots_adjust(hspace=0.9, top=0.9)
        plt.tight_layout()
        plt.savefig(f"C:\\Users\\mikae\\Documents\\Uni\Project 1\\report\\ramseyreport\\Outcomes_new\\Results_t{t}.png", dpi=500)

        # eta = np.asarray(list(model.pm_welf.extract_values().values())) / ((1 + 0.03) ** (np.asarray(tall_int) - 2005))
        # theta = eta/np.asarray(list(model.pm_dt.extract_values().values()))
        # fig4, axs = plt.subplots(1)
        # axs.plot(tall_int, eta)
        # axs.set_ylabel(f""r"$\eta_t$")
        # axs.set_xlabel("Period n")
        # #fig4.savefig(f"C:\\Users\\mikae\\Documents\\Uni\\Project 1\\report\\ramseyreport\\eta_model1_t{t}")
        #
        # fig5, axs = plt.subplots(1)
        # axs.plot(tall_int, theta)
        # axs.set_ylabel(f""r"$\theta_t$")
        # axs.set_xlabel("Period n")
        # #fig5.savefig(f"C:\\Users\\mikae\\Documents\\Uni\\Project 1\\report\\ramseyreport\\theta_model1_t{t}")
