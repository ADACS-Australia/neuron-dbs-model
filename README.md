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

3) Run the model with a config file
```shell
$ ./run_model.py example_configs/conf_amp.yml
```

To run with MPI
```shell
$ mpirun -n 4 ./run_model.py example_configs/conf_amp.yml
```

You can change the output directory for results
```shell
$ mpirun -n 4 ./run_model.py -o my_results example_configs/conf_amp.yml
```

```shell
$ ./run_model.py -h
numprocs=1
usage: run_model.py [-h] [-o OUTPUT_DIR] [config_file]

CBG Model

positional arguments:
  config_file           yaml configuration file

options:
  -h, --help            show this help message and exit
  -o OUTPUT_DIR, --output-dir OUTPUT_DIR
                        output directory name
```

# Plotting results
To plot the DBS signal and show it interactively
```shell
$ ./plot.py my_results/STN_LFP.mat
```

You can also save your plot directly
```shell
$ ./plot.py -h
usage: plot.py [-h] [-s] [-o OUTPUT] filename

Plot *.mat files

positional arguments:
  filename              file to plot

options:
  -h, --help            show this help message and exit
  -s, --save            save the plot
  -o OUTPUT, --output OUTPUT
                        output file name (ignored unless -s is specified. Default: plot.pdf)
```

# Installing on OzSTAR
Steps:

1) Load OzSTAR modules
```shell
$ source ozstar_modules.sh
```

2) Create Python venv and activate
```shell
$ python -m venv neuron
$ source ./neuron/bin/activate
```

3) Then follow installation instructions above
