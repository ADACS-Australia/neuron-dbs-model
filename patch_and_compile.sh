#!/usr/bin/env bash
set -e

echo "--> Building MOD files"
nrnivmodl hoc/

PYNN=$(python -c 'import pyNN; print(pyNN.__path__[0])')
echo "--> Applying PyNN patches"
echo "Found PyNN install dir: $PYNN/"
cd $PYNN
patch -p2 -Ni $OLDPWD/pynn.patch -r $OLDPWD/.rej || { echo "WARNING: Assuming pyNN is already patched. Removing .rej file"; rm $OLDPWD/.rej; }
