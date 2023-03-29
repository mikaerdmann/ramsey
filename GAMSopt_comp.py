'''
Author: Mika Erdmann
Project: Ramsey-Model and time steps
This script tests reliability of the model. It compares the outputs from the model 1 with the GAMS output
'''

import matplotlib.pyplot as plt
import model1_functions as func


tall_string = func.f_tall_string_b()
tall_int = [int(i) for i in tall_string]
plt.figure()
plt.subplots_adjust(hspace=0.8, top=0.8)
plt.suptitle(f"GAMS optimal. time = 2. Inv = . Welf = 1", fontsize=18, y=0.95)


vm_cesIOGams = [31.1489, 33.9129, 36.1448, 37.9764, 39.52,40.8695,42.1034,43.2873,44.4824,44.1973,43.6561,42.979,42.2098,41.3109,40.1309,37.9198,34.4819,4.27971]
vm_consGams = [3.45078,3.67215,3.83804,3.96046,4.04814,4.10691,4.13998,4.14806,4.64823,4.34337,4.16656,4.06946,4.01767,3.99384,3.99255,4.02174,4.11717,6.5933]
vm_invMacroGams = [2.15563,2.1758,2.19666,2.22112,2.25106,2.28788,2.33277,2.38683,2.45294,2.39894,2.35004,2.30152,2.25011,2.18285,2.07834,1.79336,1.49691,-5.08213]
plt.subplot(3,1,1)
plot1 = plt.plot(tall_int, vm_cesIOGams)
plt.subplot(3,1,2)
plot2=plt.plot(tall_int, vm_consGams)
plt.subplot(3,1,3)
plot2=plt.plot(tall_int, vm_invMacroGams)

plt.show()
