#!/usr/bin/env bash
set -e

NP="${1:-1}"

cd Cortex_BasalGanglia_DBS_model
mpirun -np $NP time nrniv -nogui -python -mpi run_CBG_Model_to_SS.py
cd $OLDPWD
./plot_ss.py $NP
