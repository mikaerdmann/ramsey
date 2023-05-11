'''
Author: Mika Erdmann
Project: Ramsey model and time steps

This script analyses the different welfare weight representations
'''
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

import main_run as main
import pyomo.environ as pyo

# Comparison between the different welfare weight representations

# Set Parameters
t: int = 2
weights: list = [1,3]
investment: int = 2

# Visualisation of model runs in a loop
rcParams['figure.figsize'] = 8,3
rcParams['grid.linewidth'] = 0.5
rcParams["axes.grid"] = True
plt.subplots_adjust(hspace=0.8, top=0.8)
counter = 1
for i in range(0, len(weights)):
    model = main.run_experiment(t, weights[i],investment)
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
    tickstep = [[1,5],[5,10],[1,5]]
    # Axis creation
    fig3, axs = plt.subplots(3, 1)

    for a in range(0,3):
        # Major ticks every 20, minor ticks every 5
        major_ticks = np.arange(-10, max(res[a]), tickstep[a][1])
        minor_ticks = np.arange(-10, max(res[a]), tickstep[a][0])
        major_ticksx = np.arange(0, 2160, 10)
        minor_ticksx = np.arange(0, 2160, 5)
        axs[a].set_xticks(major_ticksx)
        axs[a].set_xticks(minor_ticksx, minor=True)
        axs[a].set_yticks(major_ticks)
        axs[a].set_yticks(minor_ticks, minor=True)
        # Or if you want different settings for the grids:
        axs[a].grid(which='minor', alpha=0.2)
        axs[a].grid(which='major', alpha=0.5)
        axs[a].set_ylim(((abs(min(res[a])) // 5)+1)*(-5), ((max(res[a])//5) + 1) * 5)


    # plot

    axs[0].plot(tall_int, cons_opt, 'b')
    axs[0].scatter(tall_int, cons_opt, marker= ".", linewidths= 1, color = "b")
    axs[0].set_ylabel("Consumption")
    axs[0].set_xlabel("Time")
    axs[1].plot(tall_int, cap_opt, 'k')
    axs[1].scatter(tall_int, cap_opt, marker=".", linewidths=1, color="k")
    axs[1].set_ylabel("Capital")
    axs[1].set_xlabel("Time")
    axs[2].plot(tall_int, inv_opt, 'r')
    axs[2].scatter(tall_int, inv_opt, marker=".", linewidths=1, color="r")
    axs[2].set_ylabel("Investment")
    axs[2].set_xlabel("Time")

    counter = counter +3
plt.tight_layout()
plt.show()
#plt.savefig(f"C:\\Users\\mikae\\Documents\\Uni\Project 1\\report\\ramseyreport\\Results_comp_t2_welf")
