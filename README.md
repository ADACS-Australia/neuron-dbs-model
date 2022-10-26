# Setup
Steps:

1) Install dependencies
```shell
$ pip install -r requirements.txt
$ pip install nrnutils>=0.2.0 # requires NEURON to be installed first
```

2) Patch your pyNN installation using the given script
```shell
$ ./patch_and_compile.sh
Changing directory to /Users/<user>/anaconda/envs/neuron/lib/python3.10/site-packages/pyNN
patching file ./common/control.py
patching file ./neuron/__init__.py
patching file ./neuron/simulator.py
patching file ./neuron/standardmodels/electrodes.py
```

3) Run the model
```shell
$ cd Cortex_BasalGanglia_DBS_model/
$ python run_CBG_Model_to_SS.py neuron
```

To run with MPI
```shell
$ cd Cortex_BasalGanglia_DBS_model/
$ mpirun -np 4 nrniv -nogui -python -mpi run_CBG_Model_to_SS.py
```

You can change the output directory for results using the environment variable `PYNN_OUTPUT_DIRNAME`
```shell
$ cd Cortex_BasalGanglia_DBS_model/
$ export PYNN_OUTPUT_DIRNAME="my_results_dir"
$ mpirun -np 4 nrniv -nogui -python -mpi run_CBG_Model_to_SS.py
```

For convenience, you can also use the provided run script instead
```shell
$ PYNN_OUTPUT_DIRNAME="my_results_dir" ./run_ss.sh 4
```

# Installing on OzSTAR
Steps:

1) Load OzSTAR modules
```shell
$ . ozstar_modules.sh
```

2) Create Python venv
```shell
$ python -m venv neuron
```

3) Follow installation instructions above
