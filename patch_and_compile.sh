#!/usr/bin/env bash
set -e

echo "--> Building MOD files"
nrnivmodl hoc/

PYNN=$(python -c 'import pyNN; print(pyNN.__path__[0])')
echo "--> Changing directory to $PYNN"
cd $PYNN

# Apply patches
diff -ru . $OLDPWD/Updated_PyNN_Files/ | patch -p0 || echo "WARNING: Assuming pyNN is already patched."

# Compile
cd neuron/nmodl
echo "--> Changing directory to $(pwd)"
nrnivmodl
