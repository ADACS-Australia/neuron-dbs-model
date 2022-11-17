#!/usr/bin/env python

from matplotlib import pyplot as plt
from neo.io import NeoMatlabIO
import argparse

parser = argparse.ArgumentParser(description="Plot *.mat files")
parser.add_argument("filename", help="file to plot")
parser.add_argument(
    "-s", "--save", action="store_true", default=False, help="save the plot"
)
parser.add_argument(
    "-o",
    "--output",
    default="plot.pdf",
    help="output file name (ignored unless -s is specified. Default: plot.pdf)",
)
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
