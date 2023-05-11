import numpy as np

import EA_inverse
import inverse_functions
import model1_functions

# optimal paths
vm_opt = EA_inverse.get_vm_opt(1)

# parameter data
# A: Only pm_welf
pt_a =[2.778249,2.795309,2.809293,2.822271,2.827965,2.835979,2.831489,2.863690,3.715012,5.101043,5.707896,6.145588,6.687873,7.383467,7.821238,9.153110,9.025051,10.0]
cumdepr_new_a = [2.8525, 2.8525, 2.8525, 2.8525, 2.8525, 2.8525, 2.8525, 2.8525, 4.1379, 4.1379, 4.1379, 4.1379, 4.1379, 4.1379, 4.1379, 4.1379, 4.1379, 4.1379]
cumdepr_old_a = [1.671881, 1.671881, 1.671881, 1.671881, 1.671881, 1.671881, 1.671881, 1.671881, 3.113989, 3.113989, 3.113989, 3.113989, 3.113989, 3.113989, 3.113989, 3.113989, 3.113989, 3.113989]

# B:
pt_b = [3.54594728, 4.97798292, 3.97722519, 5.48997567, 4.65332096, 5.90115678, 3.56755949, 6.51352029, 8.31932407, 9.07812366,, 10.90054316, 10.21542658, 8.84269898, 9.88067977, 9.40822885, 8.34884158, 10]
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
pt_e = [5,5,5,5,5,5,5,7.5,10,10,10,10,10,10,10,10,10,10]
cumdepr_new_e = [1.114,1.114,1.114,1.114,1.114,1.114,1.114,1.114,5.124,5.124,5.124,5.124,5.124,5.124,5.124,5.124,5.124,5.124]
cumdepr_old_e = [3.880,3.880,3.880,3.880,3.880,3.880,3.880,3.880,2.414,2.414,2.414,2.414,2.414,2.414,2.414,2.414,2.414,2.414]

# loop thorugh different parameters

# plot results
results = inverse_functions.run_model1_inverse(timeswitch=2, vm_weight=pm_welf, depr=0, c_o=cumdepr_old2, c_n=cumdepr_new2)
vm_run = [inverse_functions.get_val(results.vm_cons), inverse_functions.get_val(results.vm_cesIO),
          inverse_functions.get_val(results.vm_invMacro)]
tall_string = model1_functions.f_tall_string_b()
tall_int = [int(i) for i in tall_string]

residuals_all = np.sqrt(sum((vm_opt[0] - vm_run[0]) ** 2, 1) + sum((vm_opt[1] - vm_run[1]) ** 2, 1) + sum(
    (vm_opt[2] - vm_run[2]) ** 2, 1))
residuals_t = np.sqrt((vm_opt[0] - vm_run[0]) ** 2 + (vm_opt[1] - vm_run[1]) ** 2 + (vm_opt[2] - vm_run[2]) ** 2)

print(f"The residuals over the whole time are {residuals_all}")
print(f"The residuals per time step are {residuals_t}")
#Residuals_all[i] = residuals2
#Residuals_t[i] = residuals3
# Axis creation
fig3, axs = plt.subplots(3, 2)
axs[0,0].set_ylim([0, 10])
axs[0,1].set_ylim([0, 10])
axs[1,0].set_ylim([0, 55])
axs[1,1].set_ylim([0, 55])
axs[2,0].set_ylim([-10, 10])
axs[2,1].set_ylim([-10, 10])
axs[0, 0].plot(tall_int, vm_opt[0], 'b')
axs[0, 0].set_ylabel("Consumption")
axs[0, 0].title.set_text("Optimal paths for equal timesteps")
axs[0, 0].set_xlabel("Time")
axs[0, 1].plot(tall_int, vm_run[0], 'b')
axs[0, 1].set_ylabel("Consumption")
axs[0, 1].set_xlabel("Time")
axs[0,1].set_title(
    f"Optimized parameters \n and "r"$\Delta_t = 5  \ for \  t < 2060$" + "\n and "r"$\Delta_t = 10 \ for \ t>2060$")
axs[1, 0].plot(tall_int, vm_opt[1], 'k')
axs[1, 0].set_ylabel("Capital")
axs[1, 0].set_xlabel("Time")
axs[1, 1].plot(tall_int, vm_run[1], 'k')
axs[1, 1].set_ylabel("Capital")
axs[1, 1].set_xlabel("Time")
axs[2, 0].plot(tall_int, vm_opt[2], "r")
axs[2, 0].set_ylabel("Investment")
axs[2, 0].set_xlabel("Time")
axs[2, 1].plot(tall_int, vm_run[2], "r")
axs[2, 1].set_ylabel("Investment")
axs[2, 1].set_xlabel("Time")

#fig3.suptitle(f"Pyomo model using optimized pm_welf with n = {N} and optimizing over all residuals.")
