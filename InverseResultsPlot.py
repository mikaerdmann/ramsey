import numpy as np
from matplotlib import pyplot as plt

import EA_inverse
import inverse_functions
import model1_functions

# optimal paths
vm_opt = EA_inverse.get_vm_opt(1)

# parameter data
# A: Only pm_welf
pt_a =[2.778249,2.795309,2.809293,2.822271,2.827965,2.835979,2.831489,2.863690,3.715012,5.101043,5.707896,6.145588,6.687873,7.383467,7.821238,9.153110,9.025051,10.0]
cumdepr_new_a = [2.8525, 2.8525, 2.8525, 2.8525, 2.8525, 2.8525, 2.8525, 2.8525, 2.8525, 4.1379, 4.1379, 4.1379, 4.1379, 4.1379, 4.1379, 4.1379, 4.1379, 4.1379]
cumdepr_old_a = [1.671881, 1.671881, 1.671881, 1.671881, 1.671881, 1.671881, 1.671881, 1.671881, 1.671881, 3.113989, 3.113989, 3.113989, 3.113989, 3.113989, 3.113989, 3.113989, 3.113989, 3.113989]

# B:
pt_b = [5.09701415, 3.54594728, 4.97798292, 3.97722519, 5.48997567, 4.65332096, 5.90115678, 3.56755949, 6.51352029, 8.31932407, 9.07812366, 10.90054316, 10.21542658, 8.84269898, 9.88067977, 9.40822885, 8.34884158, 10]
cumdepr_new_b = [2.84704373, 2.06744216, 2.94303539, 2.64522215, 3.39748425, 2.90877267, 3.02466147, 1.53892547, 2.26719811, 3.52756712, 4.10771244,4.80332963, 4.23696131, 3.49748968, 3.86237131, 3.54338718, 3.80335217, 2.84130592]
cumdepr_old_b = [2.31865745, 2.44623049, 1.57824887, 1.87288757, 1.12809904,1.6181046, 1.50625036, 2.99442391, 2.25565755, 4.48193629, 3.93258263, 3.25949003, 3.83007666, 4.53909301, 4.14754318, 4.34987811, 3.97964424, 1.62816191]

# C:
pt_c = [4.45800757, 4.91900817, 4.70475974, 4.69812889, 4.56972016,4.64619008, 4.12737432, 6.70234829, 5.60575463, 5.3134377 , 6.23085977, 6.59331251, 7.31204892, 7.96002862, 8.74520911,9.77117822, 11. ,10.]
cumdepr_new_c = [2.55421465, 2.55421465, 2.55421465, 2.55421465, 2.55421465, 2.55421465, 2.55421465, 2.55421465, 3.08424348, 3.08424348, 3.08424348, 3.08424348, 3.08424348, 3.08424348, 3.08424348, 3.08424348, 3.08424348, 3.08424348]
cumdepr_old_c = [2.22741717, 2.22741717, 2.22741717, 2.22741717, 2.22741717, 2.22741717, 2.22741717, 2.22741717, 4.37906119, 4.37906119, 4.37906119, 4.37906119, 4.37906119, 4.37906119, 4.37906119, 4.37906119, 4.37906119, 4.37906119]

# D:
pt_d = [8.290055103491845, 8.290055103491845, 8.290055103491845, 8.290055103491845, 8.290055103491845, 8.290055103491845, 8.290055103491845, 8.290055103491845, 10.9547446072895, 9.82272725573696, 9.82272725573696, 9.82272725573696, 9.82272725573696, 9.82272725573696, 9.82272725573696, 9.82272725573696, 9.82272725573696, 9.82272725573696]
cumdepr_new_d = [1.96248022, 1.96248022, 1.96248022, 1.96248022, 1.96248022,1.96248022, 1.96248022, 1.96248022, 3.84850821, 3.84850821,3.84850821, 3.84850821, 3.84850821, 3.84850821, 3.84850821,3.84850821, 3.84850821, 3.84850821]
cumdepr_old_d = [3.1653694 , 3.1653694 , 3.1653694 , 3.1653694 , 3.1653694 ,3.1653694 , 3.1653694 , 3.1653694 , 4.31974226, 4.31974226,4.31974226, 4.31974226, 4.31974226, 4.31974226, 4.31974226,4.31974226, 4.31974226, 4.31974226]

# E:
pt_e = [5,5,5,5,5,5,5,5,7.5,10,10,10,10,10,10,10,10,10]
cumdepr_new_e = [1.114,1.114,1.114,1.114,1.114,1.114,1.114,1.114,5.124,5.124,5.124,5.124,5.124,5.124,5.124,5.124,5.124,5.124]
cumdepr_old_e = [3.880,3.880,3.880,3.880,3.880,3.880,3.880,3.880,2.414,2.414,2.414,2.414,2.414,2.414,2.414,2.414,2.414,2.414]
letters = ["A","B","C","D","E"]

# parameter lists:
pt_list = [pt_a, pt_b, pt_c, pt_d, pt_e]
cumdepr_new_list = [cumdepr_new_a, cumdepr_new_b, cumdepr_new_c, cumdepr_new_d, cumdepr_new_e]
cumdepr_old_list = [cumdepr_old_a, cumdepr_old_b, cumdepr_old_c, cumdepr_old_d, cumdepr_old_e]

# loop thorugh different parameters
for pt_i in range(0,len(pt_list)): # loop over pt specifications
    print(f"We are at p_t = {letters[pt_i]}")
    pm_welf = pt_list[pt_i]
    cumdepr_old2 = cumdepr_old_list[pt_i]
    cumdepr_new2 = cumdepr_new_list[pt_i]
    # plot results
    results = inverse_functions.run_model1_inverse(timeswitch=2, vm_weight=pm_welf, depr=0, c_o=cumdepr_old2, c_n=cumdepr_new2)
    vm_run = [inverse_functions.get_val(results.vm_cons), inverse_functions.get_val(results.vm_cesIO),
              inverse_functions.get_val(results.vm_invMacro)]
    tall_string = model1_functions.f_tall_string_b()
    tall_int = [int(i) for i in tall_string]

    # Axis creation
    fig, axs = plt.subplots(3, 2)

    tickstep = [[1, 2], [5, 20], [1, 5]]
    # Axis creation
    for i in range(0,2): # loop over test case and standard case
        if i == 0:
            res = vm_opt
        if i == 1:
            res = vm_run

        cons_opt = res[0]
        cap_opt = res[1]
        inv_opt = res[2]

        case = ["Equally spaced "r"$p_t$", "Modified \n "r"$p_t$-case:" + f" {letters[pt_i]}"]

        for a in range(0, 3): # loop over cons, cap, inv paths
            # Major ticks every 20, minor ticks every 5
            major_ticks = np.arange(-10, max(res[a]), tickstep[a][1])
            minor_ticks = np.arange(-10, max(res[a]), tickstep[a][0])
            major_ticksx = np.arange(0, 2160, 25)
            minor_ticksx = np.arange(0, 2160, 5)
            axs[a, i].set_xticks(major_ticksx)
            axs[a, i].set_xticks(minor_ticksx, minor=True)
            axs[a, i].set_yticks(major_ticks)
            axs[a, i].set_yticks(minor_ticks, minor=True)
            # Or if you want different settings for the grids:
            axs[a, i].grid(which='minor', alpha=0.2)
            axs[a, i].grid(which='major', alpha=0.5)

        axs[0, i].set_ylim(2, 10)
        axs[1, i].set_ylim(0, 50)
        axs[2, i].set_ylim(-7, 5)
        # plot
        axs[0, i].plot(tall_int, cons_opt, 'b')
        axs[0, i].scatter(tall_int, cons_opt, marker=".", linewidths=1, color="b")
        if i == 0:
            axs[0, i].set_ylabel("Consumption")
        axs[0, i].set_title(f"Optimal paths for {case[i]}")
        # axs[0,i].set_xlabel("Time")
        axs[1, i].plot(tall_int, cap_opt, 'k')
        axs[1, i].scatter(tall_int, cap_opt, marker=".", linewidths=1, color="k")
        if i == 0:
            axs[1, i].set_ylabel("Capital")
        # axs[1,i].set_xlabel("Time")
        axs[2, i].plot(tall_int, inv_opt, 'r')
        axs[2, i].scatter(tall_int, inv_opt, marker=".", linewidths=1, color="r")
        if i == 0:
            axs[2, i].set_ylabel("Investment")
        axs[2, i].set_xlabel("Time")

    plt.subplots_adjust(hspace=0.9, top=0.9)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"C:\\Users\\mikae\\Documents\\Uni\Project 1\\report\\ramseyreport\\Outcomes_new\\PyooOutcome_{letters[pt_i]}.png", dpi=500)

