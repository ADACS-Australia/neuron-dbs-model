# Setup
Steps:

1) Install dependencies
```
$ pip install -r requirements.txt
$ pip install nrnutils>=0.2.0 # requires NEURON to be installed first
```

2) Patch your pyNN installation using the given script
```
$ ./patch_and_compile.sh
Changing directory to /Users/<user>/anaconda/envs/neuron/lib/python3.10/site-packages/pyNN
patching file ./common/control.py
patching file ./neuron/__init__.py
patching file ./neuron/simulator.py
patching file ./neuron/standardmodels/electrodes.py
```

3) Run the model
```
$ cd Cortex_BasalGanglia_DBS_model/
$ python run_CBG_Model_to_SS.py neuron
```

# Installing on OzSTAR
Steps:

1) Load OzSTAR modules
```
$ . ozstar_modules.sh
```

2) Create Python venv
```
$ python -m venv neuron
```

3) Follow installation instructions above
