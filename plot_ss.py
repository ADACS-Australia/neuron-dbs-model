#!/usr/bin/env python

from matplotlib import pyplot as plt
from neo.io import NeoMatlabIO
import sys

filename = "Cortex_BasalGanglia_DBS_model/Simulation_Output_Results/Steady_State_Simulation/STN_LFP.mat"

block = NeoMatlabIO(filename).read_block()
signal = block.segments[0].analogsignals[0]

plt.plot(signal)

plt.xlabel("Time")
plt.ylabel("Signal")

# plt.show()

if len(sys.argv) > 1:
    n = sys.argv[1]
else:
    n = 1
plt.savefig(f"plot_{n}.pdf")
