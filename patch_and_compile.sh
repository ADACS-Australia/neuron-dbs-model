#!/usr/bin/env bash
set -e

MODEL="Cortex_BasalGanglia_DBS_model"

echo "Changing directory to $MODEL"
cd $MODEL
# Compile
nrnivmodl
cd $OLDPWD

PYNN=$(python -c 'import pyNN; print(pyNN.__path__[0])')
echo "Changing directory to $PYNN"
cd $PYNN

# Apply patches
diff -ru . $OLDPWD/$MODEL/Updated_PyNN_Files/ | patch -p0

# Compile
cd neuron/nmodl
nrnivmodl
