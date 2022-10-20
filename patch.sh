#!/usr/bin/env bash
set -e

PYNN=$(python -c 'import pyNN; print(pyNN.__path__[0])')
echo "Changing directory to $PYNN"
cd $PYNN
diff -ru . $OLDPWD/Cortex_BasalGanglia_DBS_model/Updated_PyNN_Files/ | patch -p0
