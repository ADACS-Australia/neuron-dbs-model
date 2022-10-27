# -*- coding: utf-8 -*-
"""
Created on Wed April 03 14:27:26 2019

Description: Cortico-Basal Ganglia Network Model implemented in PyNN using the simulator Neuron.
            This version of the model loads the model steady state and implements DBS frequency
            modulation controllers where the beta ARV from the STN LFP is calculated at each
            controller call and used to update the frequency of the DBS waveform that is applied
            to the network. Full documentation of the model and controllers used is given in:

            https://www.frontiersin.org/articles/10.3389/fnins.2020.00166/

@author: John Fleming, john.fleming@ucdconnect.ie
"""

import neuron
from pyNN.neuron import (
    setup,
    run_until,
    end,
    simulator,
    Population,
    Projection,
    FromFileConnector,
    StaticSynapse,
    SpikeSourceArray,
    NoisyCurrentSource,
    StepCurrentSource,
)
from pyNN.random import RandomDistribution, NumpyRNG
from pyNN import space
from Cortical_Basal_Ganglia_Cell_Classes import (
    Cortical_Neuron_Type,
    Interneuron_Type,
    STN_Neuron_Type,
    GP_Neuron_Type,
    Thalamic_Neuron_Type,
)
from Electrode_Distances import (
    distances_to_electrode,
    collateral_distances_to_electrode,
)
from pyNN.parameters import Sequence
from Controllers import StandardPIDController
import neo.io
import quantities as pq
import numpy as np
import math
from utils import make_beta_cheby1_filter, calculate_avg_beta_power
import datetime

# Import global variables for GPe DBS
import Global_Variables as gv

h = neuron.h


if __name__ == "__main__":

    # Setup simulation
    setup(timestep=0.01, rngseed=3695)
    steady_state_duration = 6000.0  # Duration of simulation steady state
    # TODO: Fix the steady_state restore error when simulation_runtime < steady_state_duration - 1
    simulation_runtime = 32000.0  # Duration of simulation from steady state
    simulation_duration = (
        steady_state_duration + simulation_runtime + simulator.state.dt
    )  # Total simulation time
    rec_sampling_interval = 0.5  # Signals are sampled every 0.5 ms
    Pop_size = 100

    # Make beta band filter centred on 25Hz (cutoff frequencies are 21-29 Hz) for biomarker estimation
    beta_b, beta_a = make_beta_cheby1_filter(
        fs=(1.0 / rec_sampling_interval) * 1000, n=4, rp=0.5, low=21, high=29
    )

    # Use CVode to calculate i_membrane_ for fast LFP calculation
    cvode = h.CVode()
    cvode.active(0)

    # Get the second spatial derivative (the segment current) for the collateral
    cvode.use_fast_imem(1)

    # Set initial values for cell membrane voltages
    v_init = -68

    # Create random distribution for cell membrane noise current
    r_init = RandomDistribution("uniform", (0, Pop_size))

    # Create Spaces for STN Population
    STN_Electrode_space = space.Space(axes="xy")
    STN_space = space.RandomStructure(
        boundary=space.Sphere(2000)
    )  # Sphere with radius 2000um

    # Generate Poisson-distributed Striatal Spike trains
    # striatal_spike_times = generate_poisson_spike_times(Pop_size, steady_state_duration, simulation_runtime,
    #                                                     20, 1.0, 3695)

    # Save/load Striatal Spike times
    # np.save('Striatal_Spike_Times.npy', striatal_spike_times)		# Save spike times, so they can be reloaded
    striatal_spike_times = np.load(
        "Striatal_Spike_Times.npy", allow_pickle=True
    )  # Load spike times from file
    for i in range(0, Pop_size):
        spike_times = striatal_spike_times[i][0].value
        spike_times = spike_times[spike_times > steady_state_duration]
        striatal_spike_times[i][0] = Sequence(spike_times)

    # Generate teh neuron populations
    Cortical_Pop = Population(
        Pop_size,
        Cortical_Neuron_Type(soma_bias_current_amp=0.245),
        structure=STN_space,
        label="Cortical Neurons",
    )  # Better than above (ibias=0.2575)
    Interneuron_Pop = Population(
        Pop_size,
        Interneuron_Type(bias_current_amp=0.070),
        initial_values={"v": v_init},
        label="Interneurons",
    )
    STN_Pop = Population(
        Pop_size,
        STN_Neuron_Type(bias_current=-0.125),
        structure=STN_space,
        initial_values={"v": v_init},
        label="STN Neurons",
    )
    GPe_Pop = Population(
        Pop_size,
        GP_Neuron_Type(bias_current=-0.009),
        initial_values={"v": v_init},
        label="GPe Neurons",
    )  # GPe/i have the same parameters, but different bias currents
    GPi_Pop = Population(
        Pop_size,
        GP_Neuron_Type(bias_current=0.006),
        initial_values={"v": v_init},
        label="GPi Neurons",
    )  # GPe/i have the same parameters, but different bias currents
    Striatal_Pop = Population(
        Pop_size,
        SpikeSourceArray(spike_times=striatal_spike_times[0][0]),
        label="Striatal Neuron Spike Source",
    )
    Thalamic_Pop = Population(
        Pop_size,
        Thalamic_Neuron_Type(),
        initial_values={"v": v_init},
        label="Thalamic Neurons",
    )

    # Update the spike times for the striatal populations
    for i in range(0, Pop_size):
        Striatal_Pop[i].spike_times = striatal_spike_times[i][0]

    # Load Cortical Bias currents for beta burst modulation
    burst_times_script = "burst_times_1.txt"
    burst_level_script = "burst_level_1.txt"
    modulation_times = np.loadtxt(burst_times_script, delimiter=",")
    modulation_signal = np.loadtxt(burst_level_script, delimiter=",")
    modulation_signal = 0.02 * modulation_signal  # Scale the modulation signal
    cortical_modulation_current = StepCurrentSource(
        times=modulation_times, amplitudes=modulation_signal
    )
    Cortical_Pop.inject(cortical_modulation_current)

    # Generate Noisy current sources
    Cortical_Pop_Membrane_Noise = [
        NoisyCurrentSource(
            mean=0,
            stdev=0.005,
            start=steady_state_duration,
            stop=simulation_duration,
            dt=1.0,
        )
        for count in range(Pop_size)
    ]
    Interneuron_Pop_Membrane_Noise = [
        NoisyCurrentSource(
            mean=0,
            stdev=0.005,
            start=steady_state_duration,
            stop=simulation_duration,
            dt=1.0,
        )
        for count in range(Pop_size)
    ]

    # Inject each membrane noise current into each cortical and interneuron in network
    for Cortical_Neuron, Cortical_Neuron_Membrane_Noise in zip(
        Cortical_Pop, Cortical_Pop_Membrane_Noise
    ):
        Cortical_Neuron.inject(Cortical_Neuron_Membrane_Noise)

    for Interneuron, Interneuron_Membrane_Noise in zip(
        Interneuron_Pop, Interneuron_Pop_Membrane_Noise
    ):
        Interneuron.inject(Interneuron_Membrane_Noise)

    # Load cortical positions - Comment/Remove to generate new positions
    Cortical_Neuron_xy_Positions = np.loadtxt("cortical_xy_pos.txt", delimiter=",")
    Cortical_Neuron_x_Positions = Cortical_Neuron_xy_Positions[0, :]
    Cortical_Neuron_y_Positions = Cortical_Neuron_xy_Positions[1, :]

    # Set cortical xy positions to those loaded in
    for cell_id, Cortical_cell in enumerate(Cortical_Pop):
        Cortical_cell.position[0] = Cortical_Neuron_x_Positions[cell_id]
        Cortical_cell.position[1] = Cortical_Neuron_y_Positions[cell_id]

    # Load STN positions - Comment/Remove to generate new positions
    STN_Neuron_xy_Positions = np.loadtxt("STN_xy_pos.txt", delimiter=",")
    STN_Neuron_x_Positions = STN_Neuron_xy_Positions[0, :]
    STN_Neuron_y_Positions = STN_Neuron_xy_Positions[1, :]

    # Set STN xy positions to those loaded in
    for cell_id, STN_cell in enumerate(STN_Pop):
        STN_cell.position[0] = STN_Neuron_x_Positions[cell_id]
        STN_cell.position[1] = STN_Neuron_y_Positions[cell_id]
        STN_cell.position[2] = 500

    # # Position Check -
    # # 1) Make sure cells are bounded in 4mm space in x, y coordinates
    # # 2) Make sure no cells are placed inside the stimulating/recording electrode -0.5mm<x<0.5mm, -1.5mm<y<2mm
    # for Cortical_cell in Cortical_Pop:
    #     while(((np.abs(Cortical_cell.position[0]) > 2000) or (np.abs(Cortical_cell.position[1]) > 2000)) or
    #           ((np.abs(Cortical_cell.position[0]) < 500) and (-1500 < Cortical_cell.position[1] < 2000))):
    #         Cortical_cell.position = STN_space.generate_positions(1).flatten()
    # # Save the generated cortical xy positions to a textfile
    # np.savetxt('cortical_xy_pos.txt', Cortical_Axon_Pop.positions, delimiter=',')

    for STN_cell in STN_Pop:
        while (
            (np.abs(STN_cell.position[0]) > 2000)
            or (np.abs(STN_cell.position[1]) > 2000)
        ) or (
            (np.abs(STN_cell.position[0]) < 500)
            and (-1500 < STN_cell.position[1] < 2000)
        ):
            STN_cell.position = STN_space.generate_positions(1).flatten()
    # # Save the generated STN xy positions to a textfile
    # np.savetxt('STN_xy_pos.txt', STN_Pop.positions, delimiter=',')

    # Assign Positions for recording and stimulating electrode point sources
    recording_electrode_1_position = np.array([0, -1500, 250])
    recording_electrode_2_position = np.array([0, 1500, 250])
    stimulating_electrode_position = np.array([0, 0, 250])

    # Calculate STN cell distances to each recording electrode - only using xy coordinates for distance calculations
    STN_recording_electrode_1_distances = distances_to_electrode(
        recording_electrode_1_position, STN_Pop
    )
    STN_recording_electrode_2_distances = distances_to_electrode(
        recording_electrode_2_position, STN_Pop
    )

    # Calculate Cortical Collateral distances from the stimulating electrode
    # uses xyz coordinates for distance calculation - these distances need to be in um for xtra
    Cortical_Collateral_stimulating_electrode_distances = (
        collateral_distances_to_electrode(
            stimulating_electrode_position, Cortical_Pop, L=500, nseg=11
        )
    )
    # # Save the generated cortical collateral stimulation electrode distances to a textfile
    # np.savetxt('cortical_collateral_electrode_distances.txt', Cortical_Collateral_stimulating_electrode_distances,
    #            delimiter=',')

    # Synaptic Connections
    # Add variability to Cortical connections - cortical interneuron connection weights are uniformly random
    gCtxInt_max_weight = 2.5e-3  # Ctx -> Int max coupling value
    gIntCtx_max_weight = 6.0e-3  # Int -> Ctx max coupling value
    gCtxInt = RandomDistribution(
        "uniform", (0, gCtxInt_max_weight), rng=NumpyRNG(seed=3695)
    )
    gIntCtx = RandomDistribution(
        "uniform", (0, gIntCtx_max_weight), rng=NumpyRNG(seed=3695)
    )

    # Define other synaptic connection weights and delays
    syn_CorticalAxon_Interneuron = StaticSynapse(weight=gCtxInt, delay=2)
    syn_Interneuron_CorticalSoma = StaticSynapse(weight=gIntCtx, delay=2)
    syn_CorticalSpikeSourceCorticalAxon = StaticSynapse(weight=0.25, delay=0)
    syn_CorticalCollateralSTN = StaticSynapse(weight=0.12, delay=1)
    syn_STNGPe = StaticSynapse(weight=0.111111, delay=4)
    syn_GPeGPe = StaticSynapse(weight=0.015, delay=4)
    syn_GPeSTN = StaticSynapse(weight=0.111111, delay=3)
    syn_StriatalGPe = StaticSynapse(weight=0.01, delay=1)
    syn_STNGPi = StaticSynapse(weight=0.111111, delay=2)
    syn_GPeGPi = StaticSynapse(weight=0.111111, delay=2)
    syn_GPiThalamic = StaticSynapse(weight=3.0, delay=2)
    syn_ThalamicCortical = StaticSynapse(weight=5, delay=2)
    syn_CorticalThalamic = StaticSynapse(weight=0.0, delay=2)

    # # Create new network topology Connections
    # prj_CorticalAxon_Interneuron = Projection(Cortical_Pop, Interneuron_Pop,
    #                                           FixedNumberPreConnector(n=10, allow_self_connections=False),
    #                                           syn_CorticalAxon_Interneuron, source='middle_axon_node',
    #                                           receptor_type='AMPA')
    # prj_Interneuron_CorticalSoma = Projection(Interneuron_Pop, Cortical_Pop,
    #                                           FixedNumberPreConnector(n=10, allow_self_connections=False),
    #                                           syn_Interneuron_CorticalSoma, receptor_type='GABAa')
    # prj_CorticalSTN = Projection(Cortical_Pop, STN_Pop, FixedNumberPreConnector(n=5, allow_self_connections=False),
    #                              syn_CorticalCollateralSTN, source='collateral(0.5)', receptor_type='AMPA')
    # prj_STNGPe = Projection(STN_Pop, GPe_Pop, FixedNumberPreConnector(n=1, allow_self_connections=False), syn_STNGPe,
    #                         source='soma(0.5)', receptor_type='AMPA')
    # prj_GPeGPe = Projection(GPe_Pop, GPe_Pop, FixedNumberPreConnector(n=1, allow_self_connections=False), syn_GPeGPe,
    #                         source='soma(0.5)', receptor_type='GABAa')
    # prj_GPeSTN = Projection(GPe_Pop, STN_Pop, FixedNumberPreConnector(n=2, allow_self_connections=False), syn_GPeSTN,
    #                         source='soma(0.5)', receptor_type='GABAa')
    # prj_StriatalGPe = Projection(Striatal_Pop, GPe_Pop, FixedNumberPreConnector(n=1, allow_self_connections=False),
    #                              syn_StriatalGPe, source='soma(0.5)', receptor_type='GABAa')
    # prj_STNGPi = Projection(STN_Pop, GPi_Pop, FixedNumberPreConnector(n=1,allow_self_connections=False), syn_STNGPi,
    #                         source='soma(0.5)', receptor_type='AMPA')
    # prj_GPeGPi = Projection(GPe_Pop, GPi_Pop, FixedNumberPreConnector(n=1,allow_self_connections=False), syn_GPeGPi,
    #                         source='soma(0.5)', receptor_type='GABAa')
    # prj_GPiThalamic = Projection(GPi_Pop, Thalamic_Pop, FixedNumberPreConnector(n=1,allow_self_connections=False),
    #                              syn_GPiThalamic, source='soma(0.5)', receptor_type='GABAa')
    # prj_ThalamicCortical = Projection(Thalamic_Pop, Cortical_Pop,
    #                                   FixedNumberPreConnector(n=1,allow_self_connections=False), syn_ThalamicCortical,
    #                                   source='soma(0.5)', receptor_type='AMPA')
    # prj_CorticalThalamic = Projection(Cortical_Pop, Thalamic_Pop,
    #                                   FixedNumberPreConnector(n=1,allow_self_connections=False), syn_CorticalThalamic,
    #                                   source='soma(0.5)', receptor_type='AMPA')

    # Load network topology from file
    prj_CorticalAxon_Interneuron = Projection(
        Cortical_Pop,
        Interneuron_Pop,
        FromFileConnector("CorticalAxonInterneuron_Connections.txt"),
        syn_CorticalAxon_Interneuron,
        source="middle_axon_node",
        receptor_type="AMPA",
    )
    prj_Interneuron_CorticalSoma = Projection(
        Interneuron_Pop,
        Cortical_Pop,
        FromFileConnector("InterneuronCortical_Connections.txt"),
        syn_Interneuron_CorticalSoma,
        receptor_type="GABAa",
    )
    prj_CorticalSTN = Projection(
        Cortical_Pop,
        STN_Pop,
        FromFileConnector("CorticalSTN_Connections.txt"),
        syn_CorticalCollateralSTN,
        source="collateral(0.5)",
        receptor_type="AMPA",
    )
    prj_STNGPe = Projection(
        STN_Pop,
        GPe_Pop,
        FromFileConnector("STNGPe_Connections.txt"),
        syn_STNGPe,
        source="soma(0.5)",
        receptor_type="AMPA",
    )
    prj_GPeGPe = Projection(
        GPe_Pop,
        GPe_Pop,
        FromFileConnector("GPeGPe_Connections.txt"),
        syn_GPeGPe,
        source="soma(0.5)",
        receptor_type="GABAa",
    )
    prj_GPeSTN = Projection(
        GPe_Pop,
        STN_Pop,
        FromFileConnector("GPeSTN_Connections.txt"),
        syn_GPeSTN,
        source="soma(0.5)",
        receptor_type="GABAa",
    )
    prj_StriatalGPe = Projection(
        Striatal_Pop,
        GPe_Pop,
        FromFileConnector("StriatalGPe_Connections.txt"),
        syn_StriatalGPe,
        source="soma(0.5)",
        receptor_type="GABAa",
    )
    prj_STNGPi = Projection(
        STN_Pop,
        GPi_Pop,
        FromFileConnector("STNGPi_Connections.txt"),
        syn_STNGPi,
        source="soma(0.5)",
        receptor_type="AMPA",
    )
    prj_GPeGPi = Projection(
        GPe_Pop,
        GPi_Pop,
        FromFileConnector("GPeGPi_Connections.txt"),
        syn_GPeGPi,
        source="soma(0.5)",
        receptor_type="GABAa",
    )
    prj_GPiThalamic = Projection(
        GPi_Pop,
        Thalamic_Pop,
        FromFileConnector("GPiThalamic_Connections.txt"),
        syn_GPiThalamic,
        source="soma(0.5)",
        receptor_type="GABAa",
    )
    prj_ThalamicCortical = Projection(
        Thalamic_Pop,
        Cortical_Pop,
        FromFileConnector("ThalamicCorticalSoma_Connections.txt"),
        syn_ThalamicCortical,
        source="soma(0.5)",
        receptor_type="AMPA",
    )
    prj_CorticalThalamic = Projection(
        Cortical_Pop,
        Thalamic_Pop,
        FromFileConnector("CorticalSomaThalamic_Connections.txt"),
        syn_CorticalThalamic,
        source="soma(0.5)",
        receptor_type="AMPA",
    )

    """
    # Save the network topology so it can be reloaded
    #prj_CorticalSpikeSourceCorticalSoma.saveConnections(file="CorticalSpikeSourceCorticalSoma_Connections.txt")
    prj_CorticalAxon_Interneuron.saveConnections(file="CorticalAxonInterneuron_Connections.txt")
    prj_Interneuron_CorticalSoma.saveConnections(file="InterneuronCortical_Connections.txt")
    prj_CorticalSTN.saveConnections(file="CorticalSTN_Connections.txt")
    prj_STNGPe.saveConnections(file="STNGPe_Connections.txt")
    prj_GPeGPe.saveConnections(file="GPeGPe_Connections.txt")
    prj_GPeSTN.saveConnections(file="GPeSTN_Connections.txt")
    prj_StriatalGPe.saveConnections(file="StriatalGPe_Connections.txt")
    prj_STNGPi.saveConnections(file="STNGPi_Connections.txt")
    prj_GPeGPi.saveConnections(file="GPeGPi_Connections.txt")
    prj_GPiThalamic.saveConnections(file="GPiThalamic_Connections.txt")
    prj_ThalamicCortical.saveConnections(file="ThalamicCorticalSoma_Connections.txt")
    prj_CorticalThalamic.saveConnections(file="CorticalSomaThalamic_Connections.txt")
    """

    # Define state variables to record from each population
    Cortical_Pop.record("soma(0.5).v", sampling_interval=rec_sampling_interval)
    Cortical_Pop.record("collateral(0.5).v", sampling_interval=rec_sampling_interval)
    Interneuron_Pop.record("soma(0.5).v", sampling_interval=rec_sampling_interval)
    STN_Pop.record("soma(0.5).v", sampling_interval=rec_sampling_interval)
    STN_Pop.record("AMPA.i", sampling_interval=rec_sampling_interval)
    STN_Pop.record("GABAa.i", sampling_interval=rec_sampling_interval)
    Striatal_Pop.record("spikes")
    GPe_Pop.record("soma(0.5).v", sampling_interval=rec_sampling_interval)
    GPi_Pop.record("soma(0.5).v", sampling_interval=rec_sampling_interval)
    Thalamic_Pop.record("soma(0.5).v", sampling_interval=rec_sampling_interval)

    # Conductivity and resistivity values for homogenous, isotropic medium
    sigma = 0.27  # Latikka et al. 2001 - Conductivity of Brain tissue S/m
    rho = 1 / (
        sigma * 1e-2
    )  # rho needs units of ohm cm for xtra mechanism (S/m -> S/cm)

    # Calculate transfer resistances for each collateral segment for xtra - units are Mohms
    collateral_rx = (
        (rho / (4 * math.pi))
        * (1 / Cortical_Collateral_stimulating_electrode_distances)
        * 0.01
    )

    # Convert ndarray to array of Sequence objects - needed to set cortical collateral transfer resistances
    collateral_rx_seq = np.ndarray(shape=(1, Pop_size), dtype=Sequence).flatten()
    for ii in range(0, Pop_size):
        collateral_rx_seq[ii] = Sequence(collateral_rx[ii, :].flatten())

    # Assign transfer resistances values to collaterals
    for ii, cortical_cell in enumerate(Cortical_Pop):
        cortical_cell.collateral_rx = collateral_rx_seq[ii]

    # Create times for when the DBS controller will be called
    # Window length for filtering biomarker
    controller_window_length = 2000.0  # ms
    controller_window_length_no_samples = int(
        controller_window_length / rec_sampling_interval
    )

    # Window Tail length - removed post filtering, prior to biomarker calculation
    controller_window_tail_length = 100.0  # ms
    controller_window_tail_length_no_samples = int(
        controller_window_tail_length / rec_sampling_interval
    )

    controller_sampling_time = 20.0  # ms
    controller_call_times = np.arange(
        steady_state_duration + controller_window_length + controller_sampling_time,
        simulation_duration,
        controller_sampling_time,
    )

    # Initialize the Controller being used. Controller sampling period, Ts, is in sec.
    # # P Controller:
    # controller = StandardPIDController(SetPoint=1.0414e-04, Kp=416.7, Ti=0.0, Td=0, Ts=0.02, MinValue=0.0,
    #                                      MaxValue=250.0)
    # PI Controller:
    controller = StandardPIDController(
        SetPoint=1.0414e-04,
        Kp=19.3,
        Ti=0.2,
        Td=0,
        Ts=0.02,
        MinValue=0.0,
        MaxValue=250.0,
    )
    start_timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    simulation_identifier = controller.get_label() + "-" + start_timestamp
    simulation_output_dir = (
        "Simulation_Output_Results/Controller_Simulations/Freq/" + simulation_identifier
    )

    # Generate a square wave which represents the DBS signal - Needs to be initialized to zero when unused to prevent
    # open-circuit of cortical collateral extracellular mechanism
    (
        DBS_Signal,
        DBS_times,
        next_DBS_pulse_time,
        last_DBS_pulse_time,
    ) = controller.generate_dbs_signal(
        start_time=steady_state_duration + 10 + simulator.state.dt,
        stop_time=simulation_duration,
        last_pulse_time_prior=steady_state_duration,
        dt=simulator.state.dt,
        amplitude=-1.0,
        frequency=130.0,
        pulse_width=0.06,
        offset=0,
    )

    DBS_Signal = np.hstack((np.array([0, 0]), DBS_Signal))
    DBS_times = np.hstack((np.array([0, steady_state_duration + 10]), DBS_times))

    # Get DBS time indexes which corresponds to controller call times
    controller_DBS_indices = []
    for controller_call_time in controller_call_times:
        controller_DBS_indices.extend(
            [np.where(DBS_times == controller_call_time)[0][0]]
        )

    # Set first portion of DBS signal (Up to first controller call after steady state) to zero amplitude
    DBS_Signal[0:] = 0
    next_DBS_pulse_time = controller_call_times[0]

    DBS_Signal_neuron = h.Vector(DBS_Signal)
    DBS_times_neuron = h.Vector(DBS_times)

    # Play DBS signal to global variable is_xtra
    DBS_Signal_neuron.play(h._ref_is_xtra, DBS_times_neuron, 1)

    # Get DBS_Signal_neuron as a numpy array for easy updating
    updated_DBS_signal = DBS_Signal_neuron.as_numpy()

    # Initialize tracking the frequencies calculated by the controller
    last_freq_calculated = 0
    last_DBS_pulse_time = steady_state_duration

    # GPe DBS current stimulation - precalculated for % of collaterals entrained for varying DBS amplitude
    interp_DBS_amplitudes = np.array(
        [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2, 2.25, 2.50, 3, 4, 5]
    )
    interp_collaterals_entrained = np.array(
        [0, 0, 0, 1, 4, 8, 19, 30, 43, 59, 82, 100, 100, 100]
    )
    GPe_stimulation_order = np.loadtxt("GPe_Stimulation_Order.txt", delimiter=",")
    GPe_stimulation_order = [int(index) for index in GPe_stimulation_order]

    # Make new GPe DBS vector for each GPe neuron - each GPe neuron needs a pointer to its own DBS signal
    GPe_DBS_Signal_neuron = []
    GPe_DBS_times_neuron = []
    updated_GPe_DBS_signal = []
    for i in range(0, Pop_size):
        (
            GPe_DBS_Signal,
            GPe_DBS_times,
            _,
            GPe_last_DBS_pulse_time,
        ) = controller.generate_dbs_signal(
            start_time=steady_state_duration + 10 + simulator.state.dt,
            stop_time=simulation_duration,
            last_pulse_time_prior=steady_state_duration,
            dt=simulator.state.dt,
            amplitude=100.0,
            frequency=130.0,
            pulse_width=0.06,
            offset=0,
        )

        GPe_DBS_Signal = np.hstack((np.array([0, 0]), GPe_DBS_Signal))
        GPe_DBS_times = np.hstack(
            (np.array([0, steady_state_duration + 10]), GPe_DBS_times)
        )

        # Set the GPe DBS signals to zero amplitude
        GPe_DBS_Signal[0:] = 0
        GPe_next_DBS_pulse_time = controller_call_times[0]

        # Neuron vector of GPe DBS signals
        GPe_DBS_Signal_neuron.append(h.Vector(GPe_DBS_Signal))
        GPe_DBS_times_neuron.append(h.Vector(GPe_DBS_times))

        # Play the stimulation into each GPe neuron
        GPe_DBS_Signal_neuron[i].play(
            gv.GPe_stimulation_iclamps[i]._ref_amp, GPe_DBS_times_neuron[i], 1
        )

        # Hold a reference to the signal as a numpy array, and append to list of GPe stimulation signals
        updated_GPe_DBS_signal.append(GPe_DBS_Signal_neuron[i].as_numpy())

    # Initialise STN LFP list
    STN_LFP = []
    STN_LFP_AMPA = []
    STN_LFP_GABAa = []

    # Variables for writing simulation data
    last_write_time = steady_state_duration

    # Load the steady state
    run_until(steady_state_duration + simulator.state.dt, run_from_steady_state=True)

    # Reload striatal spike times after loading the steady state
    for i in range(0, Pop_size):
        Striatal_Pop[i].spike_times = striatal_spike_times[i][0]

    # For loop to integrate the model up to each controller call
    for controller_call_index, controller_call_time in enumerate(controller_call_times):

        # Integrate model to controller_call_time
        run_until(controller_call_time - simulator.state.dt)

        print(("Controller Called at t: %f" % simulator.state.t))

        # Calculate the LFP and biomarkers, etc.
        STN_AMPA_i = np.array(STN_Pop.get_data("AMPA.i").segments[0].analogsignals[0])
        STN_GABAa_i = np.array(STN_Pop.get_data("GABAa.i").segments[0].analogsignals[0])
        STN_Syn_i = STN_AMPA_i + STN_GABAa_i

        # STN LFP Calculation - Syn_i is in units of nA -> LFP units are mV
        STN_LFP_1 = (
            (1 / (4 * math.pi * sigma))
            * np.sum(
                (1 / (STN_recording_electrode_1_distances * 1e-6))
                * STN_Syn_i.transpose(),
                axis=0,
            )
            * 1e-6
        )
        STN_LFP_2 = (
            (1 / (4 * math.pi * sigma))
            * np.sum(
                (1 / (STN_recording_electrode_2_distances * 1e-6))
                * STN_Syn_i.transpose(),
                axis=0,
            )
            * 1e-6
        )
        STN_LFP = np.hstack((STN_LFP, STN_LFP_1 - STN_LFP_2))

        # STN LFP AMPA and GABAa Contributions
        STN_LFP_AMPA_1 = (
            (1 / (4 * math.pi * sigma))
            * np.sum(
                (1 / (STN_recording_electrode_1_distances * 1e-6))
                * STN_AMPA_i.transpose(),
                axis=0,
            )
            * 1e-6
        )
        STN_LFP_AMPA_2 = (
            (1 / (4 * math.pi * sigma))
            * np.sum(
                (1 / (STN_recording_electrode_2_distances * 1e-6))
                * STN_AMPA_i.transpose(),
                axis=0,
            )
            * 1e-6
        )
        STN_LFP_AMPA = np.hstack((STN_LFP_AMPA, STN_LFP_AMPA_1 - STN_LFP_AMPA_2))
        STN_LFP_GABAa_1 = (
            (1 / (4 * math.pi * sigma))
            * np.sum(
                (1 / (STN_recording_electrode_1_distances * 1e-6))
                * STN_GABAa_i.transpose(),
                axis=0,
            )
            * 1e-6
        )
        STN_LFP_GABAa_2 = (
            (1 / (4 * math.pi * sigma))
            * np.sum(
                (1 / (STN_recording_electrode_2_distances * 1e-6))
                * STN_GABAa_i.transpose(),
                axis=0,
            )
            * 1e-6
        )
        STN_LFP_GABAa = np.hstack((STN_LFP_GABAa, STN_LFP_GABAa_1 - STN_LFP_GABAa_2))

        # Biomarker Calculation:
        lfp_beta_average_value = calculate_avg_beta_power(
            lfp_signal=STN_LFP[-controller_window_length_no_samples:],
            tail_length=controller_window_tail_length_no_samples,
            beta_b=beta_b,
            beta_a=beta_a,
        )
        print("Beta Average: %f" % lfp_beta_average_value)

        # Calculate the updated DBS Frequency
        DBS_amp = 1.5
        DBS_freq = controller.update(
            state_value=lfp_beta_average_value, current_time=simulator.state.t
        )

        # Update the DBS Signal
        if controller_call_index + 1 < len(controller_call_times):

            # Check if the frequency needs to change before the last time that was calculated
            if DBS_freq != last_freq_calculated:
                if DBS_freq == 0.0:  # Check if DBS wants to turn off
                    next_DBS_pulse_time = 1e9
                else:  # Calculate new next pulse time if DBS is on
                    T = (1.0 / DBS_freq) * 1e3
                    next_DBS_pulse_time = last_DBS_pulse_time + T - 0.06

                    # Need to check for situation when new DBS time is less than the current time
                    if next_DBS_pulse_time <= simulator.state.t:
                        next_DBS_pulse_time = simulator.state.t

            # Calculate new DBS segment from the next DBS pulse time
            if next_DBS_pulse_time < controller_call_times[controller_call_index + 1]:

                GPe_next_DBS_pulse_time = next_DBS_pulse_time

                # DBS Cortical Collateral Stimulation
                (
                    new_DBS_Signal_Segment,
                    new_DBS_times_Segment,
                    next_DBS_pulse_time,
                    last_DBS_pulse_time,
                ) = controller.generate_dbs_signal(
                    start_time=next_DBS_pulse_time,
                    stop_time=controller_call_times[controller_call_index + 1],
                    last_pulse_time_prior=last_DBS_pulse_time,
                    dt=simulator.state.dt,
                    amplitude=-DBS_amp,
                    frequency=DBS_freq,
                    pulse_width=0.06,
                    offset=0,
                )

                # Update DBS segment - replace original DBS array values with updated ones
                window_start_index = np.where(DBS_times == new_DBS_times_Segment[0])[0][
                    0
                ]
                new_window_sample_length = len(new_DBS_Signal_Segment)
                updated_DBS_signal[
                    window_start_index : window_start_index + new_window_sample_length
                ] = new_DBS_Signal_Segment

                # DBS GPe neuron stimulation
                num_GPe_Neurons_entrained = int(
                    np.interp(
                        DBS_amp, interp_DBS_amplitudes, interp_collaterals_entrained
                    )
                )

                # Make copy of current DBS segment and rescale for GPe neuron stimulation
                GPe_DBS_Signal_Segment = new_DBS_Signal_Segment.copy()
                GPe_DBS_Signal_Segment *= -1
                GPe_DBS_Signal_Segment[GPe_DBS_Signal_Segment > 0] = 100

                # Stimulate the entrained GPe neurons
                for i in np.arange(0, num_GPe_Neurons_entrained):
                    updated_GPe_DBS_signal[GPe_stimulation_order[i]][
                        window_start_index : window_start_index
                        + new_window_sample_length
                    ] = GPe_DBS_Signal_Segment

                # Remember the last frequency that was calculated
                last_freq_calculated = DBS_freq

            else:
                pass

        # Write population data to file
        write_index = "{:.0f}_".format(controller_call_index)
        suffix = "_{:.0f}ms-{:.0f}ms".format(last_write_time, simulator.state.t)
        controller_label = controller.get_label()

        STN_Pop.write_data(
            simulation_output_dir
            + "/STN_Pop/"
            + write_index
            + "STN_Soma_v"
            + suffix
            + ".mat",
            "soma(0.5).v",
            clear=True,
        )

        last_write_time = simulator.state.t

    # # Write population membrane voltage data to file
    # Cortical_Pop.write_data(simulation_output_dir+"/Cortical_Pop/Cortical_Collateral_v.mat", 'collateral(0.5).v',
    #                         clear=False)
    # Cortical_Pop.write_data(simulation_output_dir+"/Cortical_Pop/Cortical_Soma_v.mat", 'soma(0.5).v', clear=True)
    # Interneuron_Pop.write_data(simulation_output_dir+"/Interneuron_Pop/Interneuron_Soma_v.mat", 'soma(0.5).v',
    #                            clear=True)
    # GPe_Pop.write_data(simulation_output_dir+"/GPe_Pop/GPe_Soma_v.mat", 'soma(0.5).v', clear=True)
    # GPi_Pop.write_data(simulation_output_dir+"/GPi_Pop/GPi_Soma_v.mat", 'soma(0.5).v', clear=True)
    # Thalamic_Pop.write_data(simulation_output_dir+"/Thalamic_Pop/Thalamic_Soma_v.mat", 'soma(0.5).v', clear=True)

    # Write controller values to csv files
    controller_measured_beta_values = np.asarray(controller.get_state_history())
    controller_measured_error_values = np.asarray(controller.get_error_history())
    controller_output_values = np.asarray(controller.get_output_history())
    controller_sample_times = np.asarray(controller.get_sample_times())
    np.savetxt(
        simulation_output_dir + "/controller_beta_values.csv",
        controller_measured_beta_values,
        delimiter=",",
    )
    np.savetxt(
        simulation_output_dir + "/controller_error_values.csv",
        controller_measured_error_values,
        delimiter=",",
    )
    np.savetxt(
        simulation_output_dir + "/controller_frequency_values.csv",
        controller_output_values,
        delimiter=",",
    )
    np.savetxt(
        simulation_output_dir + "/controller_sample_times.csv",
        controller_sample_times,
        delimiter=",",
    )

    # Write the STN LFP to .mat file
    STN_LFP_Block = neo.Block(name="STN_LFP")
    STN_LFP_seg = neo.Segment(name="segment_0")
    STN_LFP_Block.segments.append(STN_LFP_seg)
    STN_LFP_signal = neo.AnalogSignal(
        STN_LFP,
        units="mV",
        t_start=0 * pq.ms,
        sampling_rate=pq.Quantity(simulator.state.dt, "1/ms"),
    )
    STN_LFP_seg.analogsignals.append(STN_LFP_signal)

    w = neo.io.NeoMatlabIO(filename=simulation_output_dir + "/STN_LFP.mat")
    w.write_block(STN_LFP_Block)

    # # Write LFP AMPA and GABAa components to file
    # STN_LFP_AMPA_Block = neo.Block(name='STN_LFP_AMPA')
    # STN_LFP_AMPA_seg = neo.Segment(name='segment_0')
    # STN_LFP_AMPA_Block.segments.append(STN_LFP_AMPA_seg)
    # STN_LFP_AMPA_signal = neo.AnalogSignal(STN_LFP_AMPA, units='mV', t_start=0*pq.ms,
    #                                        sampling_rate=pq.Quantity(simulator.state.dt, '1/ms'))
    # STN_LFP_AMPA_seg.analogsignals.append(STN_LFP_AMPA_signal)
    # w = neo.io.NeoMatlabIO(filename=simulation_output_dir+"/STN_LFP_AMPA.mat")
    # w.write_block(STN_LFP_AMPA_Block)
    #
    # STN_LFP_GABAa_Block = neo.Block(name='STN_LFP_GABAa')
    # STN_LFP_GABAa_seg = neo.Segment(name='segment_0')
    # STN_LFP_GABAa_Block.segments.append(STN_LFP_GABAa_seg)
    # STN_LFP_GABAa_signal = neo.AnalogSignal(STN_LFP_GABAa, units='mV', t_start=0*pq.ms,
    #                                         sampling_rate=pq.Quantity(simulator.state.dt, '1/ms'))
    # STN_LFP_GABAa_seg.analogsignals.append(STN_LFP_GABAa_signal)
    # w = neo.io.NeoMatlabIO(filename=simulation_output_dir+"/STN_LFP_GABAa.mat")
    # w.write_block(STN_LFP_GABAa_Block)

    # Write the DBS Signal to .mat file
    # DBS Amplitude
    DBS_Block = neo.Block(name="DBS_Signal")
    DBS_Signal_seg = neo.Segment(name="segment_0")
    DBS_Block.segments.append(DBS_Signal_seg)
    DBS_signal = neo.AnalogSignal(
        DBS_Signal_neuron,
        units="mA",
        t_start=0 * pq.ms,
        sampling_rate=pq.Quantity(1.0 / simulator.state.dt, "1/ms"),
    )
    DBS_Signal_seg.analogsignals.append(DBS_signal)
    DBS_times = neo.AnalogSignal(
        DBS_times_neuron,
        units="ms",
        t_start=DBS_times_neuron * pq.ms,
        sampling_rate=pq.Quantity(1.0 / simulator.state.dt, "1/ms"),
    )
    DBS_Signal_seg.analogsignals.append(DBS_times)

    w = neo.io.NeoMatlabIO(filename=simulation_output_dir + "/DBS_Signal.mat")
    w.write_block(DBS_Block)

    print("Simulation Done!")

    end()
