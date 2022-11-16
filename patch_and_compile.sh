#!/usr/bin/env bash
set -e

echo "--> Building MOD files"
nrnivmodl hoc/

PYNN=$(python -c 'import pyNN; print(pyNN.__path__[0])')
echo "--> Applying PyNN patches"
cd $PYNN
diff -ru . $OLDPWD/Cortex_BasalGanglia_DBS_model/Updated_PyNN_Files/ | patch -p0 || echo "WARNING: Assuming pyNN is already patched."
