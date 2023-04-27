'''
Author: Mika Erdmann
Project: Ramsey model and time steps

This script analyses the different welfare weight representations
'''
import matplotlib.pyplot as plt
import main_run as main
import pyomo.environ as pyo

# Comparison between the different welfare weight representations

# Set Parameters
t: int = 5
weights: list = [1,3]
investment: int = 2

# Visualisation of model runs in a loop
plt.figure()
plt.subplots_adjust(hspace=0.8, top=0.8)
plt.suptitle(f"Comparison welfare weightings.\ntime= {t}, Investement = {investment}", fontsize=18, y=0.95)
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
    print(cons_opt)
    # Axis creation
    plt.subplot(2, 3, counter)
    plt.plot(tall_int, cons_opt, 'b')
    plt.ylabel("Consumption")
    plt.xlabel("Time")
    plt.subplot(2,3,counter+1)
    plt.title(f"Time: {t}. Welf_weight: {weights[i]}. Welfare: {pyo.value(model.v_welfare)}", loc= "center")
    plt.plot(tall_int, cap_opt, 'k')
    plt.ylabel("Kapital")
    plt.xlabel("Time")
    plt.subplot(2,3,counter+2)
    plt.plot(tall_int, inv_opt, "r")
    plt.ylabel("Investment")
    plt.xlabel("Time")
    counter = counter +3
plt.show()

