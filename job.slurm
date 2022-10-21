#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --job-name=neuron
#SBATCH --time=0-01:00:00
#SBATCH --mem=4G

. ozstar_modules.sh
. ../neuron/bin/activate

cd Cortex_BasalGanglia_DBS_model/
python run_CBG_Model_to_SS.py neuron
