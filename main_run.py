'''
Author: Mika Erdmann
Project: Ramsey-Model and time steps
This script is the main script that runs the model using different specifications
'''
from matplotlib import pyplot as plt

import inverse_functions
import model1_functions


def run_experiment(time_rep, welf_weight, cum_inv_rep):
    # Varying Parameters
    time_representation = time_rep
    welfare_parameter = welf_weight
    cum_investment_representation = cum_inv_rep


    # Importing based on parameters
    cumdepr_factor, investment_regression, modelversion = f_decide(cum_investment_representation)

    if modelversion == 1:
        import model1 as pyomo_model
    if modelversion == 2:
        import model2 as pyomo_model


    model = pyomo_model.run_model(timeswitch=time_representation, weight= welfare_parameter, depr= cumdepr_factor, reg = investment_regression)
    return model


def f_decide(invrep):  # Combines the different modelversions needed for the different time representations
    if invrep == 1:
        cumdepr_factor = 1
        investment_regression = 0
        modelversion = 1
    if invrep == 2:
        cumdepr_factor = 2
        investment_regression = 0
        modelversion = 1
    if invrep == 3:
        cumdepr_factor = 3
        investment_regression = 0
        modelversion = 1
    if invrep == 4:
        cumdepr_factor = 4
        investment_regression = 0
        modelversion = 1
    if invrep == 5:
        cumdepr_factor = 2  # Achtung, ist aber nur bei Initialisierung benutzt
        investment_regression = 1
        modelversion = 2
    return cumdepr_factor, investment_regression, modelversion

if __name__ == '__main__':
    welf = 1
    inv = 2
    time = 4
    model = run_experiment(time,welf,inv)
    print(model.pm_welf.extract_values().values())
    vm_run = [inverse_functions.get_val(model.vm_cons), inverse_functions.get_val(model.vm_cesIO),
              inverse_functions.get_val(model.vm_invMacro)]
    tall_string = model1_functions.f_tall_string_b()
    tall_int = [int(i) for i in tall_string]
    fig3, axs = plt.subplots(3, 1)
    axs[0].plot(tall_int, vm_run[0], 'b')
    axs[0].legend("Consumption run")
    axs[1].plot(tall_int, vm_run[1], 'k')
    axs[1].legend("Kapital run")
    axs[2].plot(tall_int, vm_run[2], "r")
    axs[2].legend(("Investment run"), loc='upper right')
    fig3.suptitle(f"Pyomo model with t {time}, welf {welf} and inv {inv}")
    plt.show()
