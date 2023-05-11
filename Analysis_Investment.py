'''
Author: Mika Erdmann
Project: Ramsey model and time steps

This script analyses the different investment representations
'''
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

import main_run as main
import pyomo.environ as pyo

# Comparison between the different investment representations

# Set Parameters:
t: int = 2  # Time representation
weights: list = [1]  # Welfare weighting parameter # List with len(weights) = 1 !
investment: list = [2,3,4]  # investment representation


# Visualisation of model runs in loop
case = [""r"$pm_{cumdepr} =$ " + "\n 1. standard", ""r"$pm_{cumdepr}$ = " + "\n 2.linear regression a", ""r"$pm_{cumdepr}$ = " + "\n 3.linear regression b" ]

# Visualisation of model runs in a loop
rcParams['figure.figsize'] = 10,10
fig3, axs = plt.subplots(3, len(investment))

for i in range(0, len(investment)):
    model = main.run_experiment(t, weights[0],investment[i])
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
    tickstep = [[1,2],[5,20],[1,5]]
    # Axis creation

    for a in range(0,3):
        # Major ticks every 20, minor ticks every 5
        major_ticks = np.arange(-10, max(res[a]), tickstep[a][1])
        minor_ticks = np.arange(-10, max(res[a]), tickstep[a][0])
        major_ticksx = np.arange(0, 2160, 25)
        minor_ticksx = np.arange(0, 2160, 5)
        axs[a,i].set_xticks(major_ticksx)
        axs[a,i].set_xticks(minor_ticksx, minor=True)
        axs[a,i].set_yticks(major_ticks)
        axs[a,i].set_yticks(minor_ticks, minor=True)
        # Or if you want different settings for the grids:
        axs[a,i].grid(which='minor', alpha=0.2)
        axs[a,i].grid(which='major', alpha=0.5)

    axs[0,i].set_ylim(2,8)
    axs[1,i].set_ylim(0,50)
    axs[2,i].set_ylim(-7,5)
    # plot
    axs[0,i].plot(tall_int, cons_opt, 'b')
    axs[0,i].scatter(tall_int, cons_opt, marker= ".", linewidths= 1, color = "b")
    if i == 0:
        axs[0,i].set_ylabel("Consumption")
    axs[0,i].set_title(f"Optimal paths with {case[i]}")
    #axs[0,i].set_xlabel("Time")
    axs[1,i].plot(tall_int, cap_opt, 'k')
    axs[1,i].scatter(tall_int, cap_opt, marker=".", linewidths=1, color="k")
    if i == 0:
        axs[1,i].set_ylabel("Capital")
    #axs[1,i].set_xlabel("Time")
    axs[2,i].plot(tall_int, inv_opt, 'r')
    axs[2,i].scatter(tall_int, inv_opt, marker=".", linewidths=1, color="r")
    if i == 0:
        axs[2,i].set_ylabel("Investment")
    axs[2,i].set_xlabel("Time")

plt.subplots_adjust(hspace=0.9, top=0.9)
plt.tight_layout()
#plt.show()
plt.savefig(f"C:\\Users\\mikae\\Documents\\Uni\Project 1\\report\\ramseyreport\\Results_comp_t2_inv.png", dpi = 500)

#
# plt.figure(2)
# plt.subplots_adjust(hspace=0.8, top=0.8)
# plt.suptitle(f"Comparison cumdeprec old and new, weight = {weights[0]} time= {t}", fontsize=18, y=0.95)
# counter = 1
# for i in range(0, len(investment)):
#     model = models_inv[i]
#     pm_tall_val = model.pm_tall_val.extract_values()
#     tall_int = list(pm_tall_val.values())
#     # Axis creation
#     plt.subplot(3, 3, counter)
#     plt.plot(tall_int, models_inv[i].pm_cumdepr_new, 'b', label = "cumdepr_new")
#     plt.plot(tall_int,models_inv[i].pm_cumdepr_old, 'r', label = "cumdepr_old")
#     plt.legend()
#     counter = counter + 1
#     x,y,modelversion = main.f_decide(investment[i])
#     if modelversion == 2:
#         plt.title(f" Welf_weight: {weights[0]}. Inv: {investment[i]}.\nFactors only used for initialisation!", loc="center")
#     else:
#         plt.title(f" Welf_weight: {weights[0]}. Inv: {investment[i]}.", loc="left")
#
