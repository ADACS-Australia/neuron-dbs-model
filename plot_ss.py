from scipy.io import loadmat
from matplotlib import pyplot as plt

data = loadmat('Cortex_BasalGanglia_DBS_model/Simulation_Output_Results/Steady_State_Simulation/STN_LFP.mat',simplify_cells=True)

signal = data['block']['segments']['analogsignals']['signal']

plt.plot(signal)

plt.xlabel('Time')
plt.ylabel('Signal')

plt.show()
