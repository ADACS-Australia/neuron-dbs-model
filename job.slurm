#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --job-name=neuron
#SBATCH --output=neuron.qout
#SBATCH --time=0-02:00:00
#SBATCH --mem=4G
#SBATCH --account=<account>

source ozstar_modules.sh
source <path/to/venv/>

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYNN_OUTPUT_DIRNAME="results_${SLURM_NTASKS_PER_NODE}"

cd Cortex_BasalGanglia_DBS_model/
mkdir -p ${PYNN_OUTPUT_DIRNAME}

srun time -f "real %e  user %U  sys %S" nrniv -nogui -python -mpi run_CBG_Model_to_SS.py

cd $OLDPWD
