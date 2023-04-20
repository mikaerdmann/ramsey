'''
Author: Mika Erdmann
Project: Ramsey model and time steps

This script includes the functions that the model 2 script calls.
'''

import numpy as np

def f_tall_string(m):
    """
    Uses the time representation switch and gives the tall string function for
    initializing the time paramaters of the model
    :param m: model
    :return: the string with time steps
    """
    if m.time == 1:
        return f_tall_string_a()
    if m.time == 2:
        return f_tall_string_b()
    if m.time == 3:
        return f_tall_string_c()
    if m.time == 4:
        return f_tall_string_d()


def f_tall_string_a():
    return ["2020", "2025", "2030", "2035", "2040", "2045", "2050", "2055", "2060", "2065", "2070", "2075",
                   "2080", "2085", "2090", "2095", "2100", "2105", "2110", "2115", "2120", "2125", "2130", "2135",
                   "2140", "2145", "2150"]


def f_tall_string_b():
    return ["2020", "2025", "2030", "2035", "2040", "2045", "2050", "2055", "2060", "2070",
                   "2080", "2090", "2100", "2110", "2120", "2130",
                   "2140", "2150"]


def f_tall_string_c():
    return ['2020', '2021', '2022', '2023', '2024', '2025', '2026', '2027',
       '2028', '2029', '2030', '2031', '2032', '2033', '2034', '2035',
       '2036', '2037', '2038', '2039', '2040', '2041', '2042', '2043',
       '2044', '2045', '2046', '2047', '2048', '2049', '2050', '2051',
       '2052', '2053', '2054', '2055', '2056', '2057', '2058', '2059',
       '2060', '2061', '2062', '2063', '2064', '2065', '2066', '2067',
       '2068', '2069', '2070', '2071', '2072', '2073', '2074', '2075',
       '2076', '2077', '2078', '2079', '2080', '2081', '2082', '2083',
       '2084', '2085', '2086', '2087', '2088', '2089', '2090', '2091',
       '2092', '2093', '2094', '2095', '2096', '2097', '2098', '2099',
       '2100', '2101', '2102', '2103', '2104', '2105', '2106', '2107',
       '2108', '2109', '2110', '2111', '2112', '2113', '2114', '2115',
       '2116', '2117', '2118', '2119', '2120', '2121', '2122', '2123',
       '2124', '2125', '2126', '2127', '2128', '2129', '2130', '2131',
       '2132', '2133', '2134', '2135', '2136', '2137', '2138', '2139',
       '2140', '2141', '2142', '2143', '2144', '2145', '2146', '2147',
       '2148', '2149']


def f_tall_string_d():
    return ['2020', '2021', '2022', '2023', '2024', '2025', '2026', '2027',
       '2028', '2029', '2030', '2031', '2032', '2033', '2034', '2035',
       '2036', '2037', '2038', '2039', '2040', '2041', '2042', '2043',
       '2044', '2045', '2046', '2047', '2048', '2049', '2050', '2051',
       '2052', '2053', '2054', '2055', '2056', '2057', '2058', '2059',
       '2060','2065', '2070', '2075', '2080', '2085', '2090', '2095',
        '2100', '2105', '2110', '2115', '2120', '2125', '2130', '2135',
        '2140', '2145', '2150']


def f_tall_int(m): # converts strings to integers
    tall_string = f_tall_string(m)
    return [int(i) for i in tall_string]


def f_tall_val(m,t): # saves tall as integers
    tall_string = f_tall_string(m)
    tall_int = f_tall_int(m)
    return tall_int[t]


def f_tall_diff(m): # computes the difference between the years in tall.
    tall = f_tall_int(m)
    tall_diff = np.diff(tall)
    tall_diff= np.insert(tall_diff, 0, tall_diff[0]) # Insert at position zero the difference between 1st and 2nd year
    return tall_diff


def f_dt(m,t):  # returns for every t the difference between t and t-1
        diff = f_tall_diff(m)
        return diff[t]  # timestep from t to t+1


def f_pm_ts(m, t):   # returns the average time difference (dt) between the years (neighbouring 2 steps)
    if 0 < t < m.N.value-1:
        return (m.pm_dt[t]+m.pm_dt[t+1])/2
    if t == 0:
        return m.pm_dt[t]
    else:
        return m.pm_dt[t]


def f_pm_4ts(m, t):  # dt average over 4 neighbouring time steps
    if 1 < t < (m.N.value-1):
        return (m.pm_dt[t-2] + m.pm_dt[t-1]+m.pm_dt[t] + m.pm_dt[t+1])/4
    if t == 1:
        return (m.pm_dt[t-1]+m.pm_dt[t])/2
    if t == m.N.value-1:
        return (m.pm_dt[t-1]+m.pm_dt[t])/2
    else:
        return m.pm_dt[t]

def f_pm_ts09(m,t):
    if t>1:
        if m.pm_dt[t]-m.pm_dt[t-1] == 0:
            return f_pm_ts(m,t)
        else:
            return 0.9 * f_pm_ts(m,t)
    else:
        return f_pm_ts(m,t)

def f_pm_welf_experiment(m, t):
    if t < m.N / 2:
        return 100
    else:
        return 1
def f_pm_welf(m,t):  # uses the welf.weight switch and returns the adequate function for initializing model.pm_welf
    if m.welf_weight == 0:
        return 1
    if m.welf_weight == 1:
        return f_pm_ts(m, t)
    if m.welf_weight == 2:
        return f_pm_4ts(m, t)
    if m.welf_weight == 3:
        return f_pm_ts09(m,t)  # 0.9 nur für zeitsprung
    if m.welf_weight == 4:
        return f_pm_welf_experiment(m, t)


def f_cumdepr_old(m,t):  # switch function for old
    if m.depr == 1:
        #return 0
        return f_cumdepr_old_1(m,t)
    if m.depr == 2:
        #return 0
        return f_cumdepr_old_2(m,t)
    if m.depr == 3:
        return f_cumdepr_old_3(m,t)
    if m.depr == 4:
        return f_cumdepr_old_4(m,t)


def f_cumdepr_new(m,t):  # switch function for new
    if m.depr == 1:
        return f_cumdepr_new_1(m,t)
    if m.depr == 2:
        return f_cumdepr_new_2(m,t)
    if m.depr == 3:
        return f_cumdepr_new_3(m,t)
    if m.depr == 4:
        return f_cumdepr_new_4(m,t)

# This is the one from remind-simplified-Ricarda
def f_cumdepr_old_1(m, t):  # TODO: Does this work for dt = 10? Why + 0.5??
    pm_cumDeprecFactor_old = ((1 - m.pm_delta_kap) ** (m.pm_dt[t]/2 + 0.5) - (1 - m.pm_delta_kap) ** m.pm_dt[t])/m.pm_delta_kap
    return pm_cumDeprecFactor_old

def f_cumdepr_new_1(m, t):
    pm_cumDeprecFactor_new = ( 1 - (1 - m.pm_delta_kap) ** (m.pm_dt[t]/2 + 0.5) )/m.pm_delta_kap
    return pm_cumDeprecFactor_new

# Version 2: This is the corrected one from remind simplified, Differentiating uneven and even dts corrected

def f_cumdepr_old_2(m,t):
    # if dt[t] is even:
    dt = m.pm_dt[t]
    if (dt % 2) == 0:
        pm_cumDeprecFactor_old  = ((1 - m.pm_delta_kap) ** (m.pm_dt[t] / 2) - (1 - m.pm_delta_kap) ** (m.pm_dt[t]))/ m.pm_delta_kap - 1 / 2 * (1 - m.pm_delta_kap) ** (m.pm_dt[t] / 2)
        return pm_cumDeprecFactor_old
    # if dt[t] is uneven:
    else:
        pm_cumDeprecFactor_old = ((1 - m.pm_delta_kap) ** (m.pm_dt[t] / 2 + 0.5) - (1 - m.pm_delta_kap) ** (m.pm_dt[t])) / m.pm_delta_kap
    return pm_cumDeprecFactor_old

def f_cumdepr_new_2(m,t):
    dt = m.pm_dt[t]
    if (dt % 2) == 0:
        pm_cumDeprecFactor_new = (1 - (1-m.pm_delta_kap) ** (m.pm_dt[t] / 2 ) ) / m.pm_delta_kap - 1 / 2 * (1 - m.pm_delta_kap) ** (m.pm_dt[t] / 2)
        return pm_cumDeprecFactor_new
    else:
        pm_cumDeprecFactor_new = (1 - (1-m.pm_delta_kap) ** (m.pm_dt[t] / 2 + 0.5 ) ) / m.pm_delta_kap
        return pm_cumDeprecFactor_new

# 3rd Version: Linear regression of Investemtn between time steps
# Can be used in 2 ways: either use only 0*cumdepr_old + 1*cumdepr_new

def f_cumdepr_old_3(m,t):
    pm_cumdeprFactor_old = (1 - m.pm_delta_kap - (1 + m.pm_delta_kap * (m.pm_dt[t] - 1)) * (1 - m.pm_delta_kap)**m.pm_dt[t]) / (m.pm_delta_kap ** 2 * m.pm_dt[t])
    return pm_cumdeprFactor_old


def f_cumdepr_new_3(m,t):  #
    pm_cumDeprecFactor_new = ((1 - m.pm_delta_kap) ** (m.pm_dt[t] + 1) + m.pm_delta_kap * m.pm_dt[t] + m.pm_delta_kap - 1) / (m.pm_delta_kap**2 * m.pm_dt[t])
    return pm_cumDeprecFactor_new

# Second version of linear regression (0.5*(1-d)* cumdepr_old * 0.5*cumdepr_new)
def f_cumdepr_old_4(m,t):
    pm_cumDeprecFactor_old = ((m.pm_delta_kap * ((m.pm_delta_kap - 2) * m.pm_dt[t] + 2) - 2) *( 1 - m.pm_delta_kap) ** m.pm_dt[t] - 2 * m.pm_delta_kap + 2) / (2 * m.pm_delta_kap**2 * m.pm_dt[t])
    return pm_cumDeprecFactor_old


def f_cumdepr_new_4(m,t):
    pm_cumDeprecFactor_new = (2 * (1 - m.pm_delta_kap) ** (m.pm_dt[t] + 1) + 2 * m.pm_delta_kap*(1 + m.pm_dt[t]) - (m.pm_delta_kap)**2 * m.pm_dt[t] - 2) / (2 * m.pm_delta_kap**2 * m.pm_dt[t])
    return pm_cumDeprecFactor_new


def f_inv_linreg(m,t):
    dts = range(1, m.pm_dt[t]+1)
    inv_lin = np.zeros(len(dts))
    for i in range(0, len(dts)):
        inv_lin[i] = m.vm_invMacro[t - 1].value + (m.vm_invMacro[t].value - m.vm_invMacro[t - 1].value) / m.pm_dt[t] * dts[i]
    return inv_lin

# Switch function for the whole cumulative investment function (inlcuding regressions)
def f_cum_inv(m,t):
    if t == 0:
        return m.pm_cumdepr_new[t] * m.vm_invMacro[t]
    else:
        if m.invreg == 1:
            return f_cum_inv_reg(m,t)
        if m.invreg == 0:
            raise AttributeError("This modelversion should implement a regression!")


def f_cum_inv_reg(m,t):
    inv_lin = f_inv_linreg(m, t)
    cum_investment = sum((1 - m.pm_delta_kap) * inv_lin)
    return cum_investment

def f_cum_inv_oldnew(m,t):
    investment = m.pm_cumdepr_old[t] * m.vm_invMacro[t - 1] + m.pm_cumdepr_new[t] * m.vm_invMacro[t]
    return investment

    # bounds
def f_ces_bound(m):
    vm_cesIO_lo = 1e-2
    vm_cesIO_up = 50000
    return (vm_cesIO_lo, vm_cesIO_up)


def f_ces_initialize(m,t):
    tall = m.Tall
    vm_cesIO_0 = np.ones(len(tall))*m.sm_cesIO
    return vm_cesIO_0[t]


def f_cons_bound(m):
    vm_cons_lo = 1e-3
    vm_cons_up = np.inf
    return (vm_cons_lo, vm_cons_up)


def f_cons_initialize(m,t):
    tall = m.Tall
    vm_cons_0 = np.ones(len(tall))*10
    return vm_cons_0[t]


def f_utility(cons, ies):  # falschh wegen np.log
    cons = np.asarray(cons)
    if ies != 1:
        utility = (cons ** (1 - 1 / ies) - 1) / (1 - 1 / ies)
    if ies == 1:
        utility = np.log(cons)
    return utility

def f_vm_utility(m,t):
    return (m.vm_cons[t]**(1-1/m.pm_ies)-1)/(1-1/m.pm_ies)

def f_vm_utilitylog(m,t):
    return np.log(m.vm_cons[t].value)

def f_initialize_welf(m,t):
    return (1 / (1 + m.pm_prtp)) ** (m.pm_tall_val[t] - 2005) * m.pm_pop * m.vm_utility[t] * m.pm_welf[t]


def f_prod(cesIO, delta_kap):
    return cesIO ** delta_kap