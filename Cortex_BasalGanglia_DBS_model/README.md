Network model of the cortico-basal ganglia network with closed-loop DBS to test closed-loop DBS control strategies.

This is the readme for the models associated with the paper:

Fleming JE, Dunn E, Lowery MM (2020) Simulation of Closed-Loop Deep Brain Stimulation Control Schemes for Suppression of
Pathological Beta Oscillations in Parkinson's Disease. Frontiers in Neuroscience 14:166
[http://dx.doi.org/10.3389/fnins.2020.00166](http://dx.doi.org/10.3389/fnins.2020.00166)

The model files were contributed by JE Fleming

Model Requirements: Model is simulated using PyNN with NEURON as it's backend simulator, thus follow their
installation instructions at:

1. Neuron - [https://www.neuron.yale.edu/neuron/download](https://www.neuron.yale.edu/neuron/download)
2. PyNN - [https://pypi.org/project/PyNN/](https://pypi.org/project/PyNN/) -
[http://neuralensemble.org/docs/PyNN/](http://neuralensemble.org/docs/PyNN/)

Model Setup:

1. Copy the included PyNN files from the downloaded model folder to their corresponding location on your computer (i.e.
the directory of your PyNN instatllation - Updated PyNN files are needed for correct simulation of the
multicompartmental cortical neurons and for loading model simulations from a presimulated steady state.
2. Compile the NEURON model mod files using either mknrndll or nrnivmodl, for windows or Linux, respectively.
3. Run `run_CBG_Model_to_SS.py`

Example

4. From the command line/terminal navigate to the folder containing the model.
5. Execute `python run_CBG_Model_to_SS.py neuron`

Explanation


There is an initial transient period in the model (~6 seconds). This model simulation runs the model for the transient
period and creates a binary file (`steady_state.bin`) at the end of the simulation. This binary file captures the state
of the model at the end of this transient simulation (i.e. after the model has reasched the steady state)

Subsequent runs of the model can use either

`run_CBG_Model_Amplitude_Modulation_Controller.py` or `run_CBG_Model_Frequency_Modulation_Controller.py` to load
the previously saved model steady state and run a model simulation from this point simulating either amplitude or
frequency modeulation, respectively.

Running the Model: Once the steady state of the model has been saved you can run the model by navigating to the model
directory in the command line and typing:

`python run_CBG_Model_Amplitude_Modulation_Controller.py neuron`

Output files of the simulation are then written to a `Simulation_Output_Results` folder when the simulation is
finished. Model outputs are structured using the neo file format as detailed in
[https://neo.readthedocs.io/en/stable/.](https://neo.readthedocs.io/en/stable/.)
