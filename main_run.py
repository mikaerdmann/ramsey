'''
Author: Mika Erdmann
Project: Ramsey-Model and time steps
This script is the main script that runs the model using different specifications
'''


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


model = run_experiment(2,1,2)
