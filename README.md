# Setup
Steps:

1) Install dependencies
```
$ pip install -r requirements.txt
$ pip install nrnutils>=0.2.0 # requires NEURON to be installed first
```

2) Compile the NEURON model mod files using `nrnivmodl`
```
$ cd Cortex_BasalGanglia_DBS_model/
$ nrnivmodl
```

3) Patch your pyNN installation using the given script
```
$ ./patch_and_compile.sh
Changing directory to /Users/<user>/anaconda/envs/neuron/lib/python3.10/site-packages/pyNN
patching file ./common/control.py
patching file ./neuron/__init__.py
patching file ./neuron/simulator.py
patching file ./neuron/standardmodels/electrodes.py
```

5) Run the model
```
$ cd Cortex_BasalGanglia_DBS_model/
$ python run_CBG_Model_to_SS.py neuron
```
