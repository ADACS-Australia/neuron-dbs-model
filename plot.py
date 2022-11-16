#!/usr/bin/env python

from matplotlib import pyplot as plt
from neo.io import NeoMatlabIO
import argparse

parser = argparse.ArgumentParser(description="Plot STN_LFP.mat files")
parser.add_argument('filename', help='file to plot')
parser.add_argument('-s','--save',action='store_true',default=False)
args = parser.parse_args()

block = NeoMatlabIO(args.filename).read_block()
signal = block.segments[0].analogsignals[0]

plt.plot(signal)

plt.xlabel("Time")
plt.ylabel("Signal")

if args.save:
    plt.savefig("plot.pdf")
else:
    plt.show()
