# Setup
Steps:

0) (optional) Create a virtual environment
```shell
$ python -m venv neuron-env
$ source ./neuron-env/bin/activate
```

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

3.1) Run a steady state sim
```shell
$ cd Cortex_BasalGanglia_DBS_model/
$ ./run_steadystate.py
```

3.2) Run the model with a config file
```shell
$ cd Cortex_BasalGanglia_DBS_model/
$ ./run_model.py conf_amp.yml
```

To run with MPI
```shell
$ cd Cortex_BasalGanglia_DBS_model/
$ mpirun -n 4 ./run_model.py conf_amp.yml
```

You can change the output directory for results using the environment variable `PYNN_OUTPUT_DIRNAME`
```shell
$ cd Cortex_BasalGanglia_DBS_model/
$ export PYNN_OUTPUT_DIRNAME="my_results_dir"
$ mpirun -n 4 ./run_model.py conf_amp.yml
```

# Plotting results
To plot the DBS signal
```shel
$ ./plot_ss.py Cortex_BasalGanglia_DBS_model/Simulation_Output_Results/Steady_State_Simulation/STN_LFP.mat
```

To save your plot, us the command line option `-s`
```shell
$ ./plot_ss.py --help
usage: plot.py [-h] [-s] filename

Plot STN_LFP.mat files

positional arguments:
  filename    file to plot

options:
  -h, --help  show this help message and exit
  -s, --save
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
