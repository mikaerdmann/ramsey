'''
Author: Mika Erdmann
Project: Ramsey model and time steps

This script analyses the different investment representations
'''
import matplotlib.pyplot as plt

import main_run as main
import pyomo.environ as pyo

# Comparison between the different investment representations

# Set Parameters:
t: int = 2  # Time representation
weights: list = [3]  # Welfare weighting parameter # List with len(weights) = 1 !
investment: list = [1,2,3,4,5]  # investment representation


# Visualisation of model runs in loop
plt.figure(1)
plt.subplots_adjust(hspace=0.8, top=0.8)
plt.suptitle(f"Comparison inv weightings, time= {t}", fontsize=18, y=0.95)
counter = 1
models_inv = list([])
for i in range(0, len(weights)):
    for j in range(0, len(investment)):
        model = main.run_experiment(t, weights[i], investment[j])
        cons_opt_dict = model.vm_cons.get_values()
        cap_opt_dict = model.vm_cesIO.get_values()
        inv_opt_dict = model.vm_invMacro.get_values()

        print("Welfare for this solution: ", pyo.value(model.OBJ))
        # Convert object to a list
        result_cons = cons_opt_dict.values()
        cons_opt = list(result_cons)
        results_cap = cap_opt_dict.values()
        cap_opt = list(results_cap)
        results_inv = inv_opt_dict.values()
        inv_opt = list(results_inv)
        pm_tall_val = model.pm_tall_val.extract_values()
        tall_int = list(pm_tall_val.values())
        # Axis creation
        plt.subplot(5, 3, counter)
        plt.plot(tall_int, cons_opt, 'b')
        plt.legend("Consumption")
        plt.subplot(5,3,counter+1)
        plt.title(f"Time:, {t}. Welf_weight: {weights[i]}. Inv: {investment[j]}", loc= "center")
        plt.plot(tall_int, cap_opt, 'k')
        plt.legend("Kapital")
        plt.subplot(5,3,counter+2)
        plt.plot(tall_int, inv_opt, "r")
        plt.legend(("Investment"), loc='upper right')
        models_inv.append(model)
        counter = counter +3


plt.figure(2)
plt.subplots_adjust(hspace=0.8, top=0.8)
plt.suptitle(f"Comparison cumdeprec old and new, weight = {weights[0]} time= {t}", fontsize=18, y=0.95)
counter = 1
for i in range(0, len(investment)):
    model = models_inv[i]
    pm_tall_val = model.pm_tall_val.extract_values()
    tall_int = list(pm_tall_val.values())
    # Axis creation
    plt.subplot(4, 3, counter)
    plt.plot(tall_int, models_inv[i].pm_cumdepr_new, 'b', label = "cumdepr_new")
    plt.plot(tall_int,models_inv[i].pm_cumdepr_old, 'r', label = "cumdepr_old")
    plt.legend()
    counter = counter + 1
    x,y,modelversion = main.f_decide(investment[i])
    if modelversion == 2:
        plt.title(f" Welf_weight: {weights[0]}. Inv: {investment[i]}.\nFactors only used for initialisation!", loc="center")
    else:
        plt.title(f" Welf_weight: {weights[0]}. Inv: {investment[i]}.", loc="left")

plt.show()