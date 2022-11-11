#!/usr/bin/env bash
set -e

# Default to 1 MPI task
NP="${1:-1}"

# Default output dir
export PYNN_OUTPUT_DIRNAME="${PYNN_OUTPUT_DIRNAME:-Simulation_Output_Results}"
cd Cortex_BasalGanglia_DBS_model
mkdir -p ${PYNN_OUTPUT_DIRNAME}

mpirun -np $NP time nrniv -nogui -python -mpi run_CBG_Model_to_SS.py
