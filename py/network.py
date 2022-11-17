from pathlib import Path

import numpy as np
from pyNN import space
from pyNN.neuron import (
    FixedNumberPreConnector,
    FromFileConnector,
    NoisyCurrentSource,
    Population,
    Projection,
    SpikeSourceArray,
    StaticSynapse,
    StepCurrentSource,
)
from pyNN.parameters import Sequence
from pyNN.random import NumpyRNG, RandomDistribution

from .Cortical_Basal_Ganglia_Cell_Classes import (
    Cortical_Neuron_Type,
    GP_Neuron_Type,
    Interneuron_Type,
    STN_Neuron_Type,
    Thalamic_Neuron_Type,
)
from .utils import generate_poisson_spike_times

DATA_DIR = Path(__file__).resolve().parent / "data"
BURSTS_DIR = DATA_DIR / "bursts"
CONNECTIONS_DIR = DATA_DIR / "connections"


def load_network(
    steady_state_duration,
    simulation_duration,
    simulation_runtime,
    v_init,
    rng_seed=3695,
    Pop_size=None,
):

    create = Pop_size is not None
    np.random.seed(rng_seed)

    # Sphere with radius 2000 um
    STN_space = space.RandomStructure(
        boundary=space.Sphere(2000), rng=NumpyRNG(seed=rng_seed)
    )

    striatal_spike_times_file = DATA_DIR / "Striatal_Spike_Times.npy"
    if create:
        # Generate Poisson-distributed Striatal Spike trains
        striatal_spike_times = generate_poisson_spike_times(
            Pop_size, steady_state_duration, simulation_runtime, 20, 1.0, rng_seed
        )
        # Save spike times so they can be reloaded
        np.save(striatal_spike_times_file, striatal_spike_times)
    else:
        # Load striatal spike times from file
        striatal_spike_times = np.load(striatal_spike_times_file, allow_pickle=True)
        Pop_size = len(striatal_spike_times[:, 0])

    for i in range(0, Pop_size):
        spike_times = striatal_spike_times[i][0].value
        spike_times = spike_times[spike_times > steady_state_duration]
        striatal_spike_times[i][0] = Sequence(spike_times)

    # Generate the cortico-basal ganglia neuron populations
    Cortical_Pop = Population(
        Pop_size,
        Cortical_Neuron_Type(soma_bias_current_amp=0.245),
        structure=STN_space,
        label="Cortical Neurons",
    )
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
    # GPe/i have the same parameters, but different bias currents
    GPe_Pop = Population(
        Pop_size,
        GP_Neuron_Type(bias_current=-0.009),
        initial_values={"v": v_init},
        label="GPe Neurons",
    )
    GPi_Pop = Population(
        Pop_size,
        GP_Neuron_Type(bias_current=0.006),
        initial_values={"v": v_init},
        label="GPi Neurons",
    )
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

    Striatal_Pop.set(spike_times=striatal_spike_times[:, 0])

    # Generate Noisy current sources for cortical pyramidal and interneuron populations
    # Inject each membrane noise current into each cortical and interneuron in network
    for cell in Cortical_Pop:
        cell.inject(
            NoisyCurrentSource(
                mean=0,
                stdev=0.005,
                start=steady_state_duration,
                stop=simulation_duration,
                dt=1.0,
            )
        )

    for cell in Interneuron_Pop:
        cell.inject(
            NoisyCurrentSource(
                mean=0,
                stdev=0.005,
                start=steady_state_duration,
                stop=simulation_duration,
                dt=1.0,
            )
        )

    # Load burst times and scale the modulation signal
    modulation_t = np.loadtxt(BURSTS_DIR / "burst_times_1.txt", delimiter=",")
    modulation_s = np.loadtxt(BURSTS_DIR / "burst_level_1.txt", delimiter=",") * 0.02
    cortical_modulation_current = StepCurrentSource(
        times=modulation_t, amplitudes=modulation_s
    )
    Cortical_Pop.inject(cortical_modulation_current)

    cortical_xy_pos_file = DATA_DIR / "cortical_xy_pos.txt"
    STN_Neuron_xy_Positions_file = DATA_DIR / "STN_xy_pos.txt"
    if create:
        # Position Check -
        # 1) Make sure cells are bounded in 4mm space in x, y coordinates
        # 2) Make sure no cells are placed inside the stimulating/recording
        # electrode -0.5mm<x<0.5mm, -1.5mm<y<2mm
        for Cortical_cell in Cortical_Pop:
            x = Cortical_cell.position[0]
            y = Cortical_cell.position[1]
            while ((np.abs(x) > 2000) or (np.abs(y) > 2000)) or (
                (np.abs(x) < 500) and (-1500 < y < 2000)
            ):
                Cortical_cell.position = STN_space.generate_positions(1).flatten()

        # Save the generated cortical xy positions to a textfile
        np.savetxt(cortical_xy_pos_file, Cortical_Pop.positions, delimiter=",")

        for STN_cell in STN_Pop:
            x = STN_cell.position[0]
            y = STN_cell.position[1]
            while ((np.abs(x) > 2000) or ((np.abs(y) > 2000))) or (
                (np.abs(x) < 500) and (-1500 < y < 2000)
            ):
                STN_cell.position = STN_space.generate_positions(1).flatten()

        # Save the generated STN xy positions to a textfile
        np.savetxt(STN_Neuron_xy_Positions_file, STN_Pop.positions, delimiter=",")
    else:
        # Load cortical positions - Comment/Remove to generate new positions
        Cortical_Neuron_xy_Positions = np.loadtxt(cortical_xy_pos_file, delimiter=",")

        # Set cortical xy positions to those loaded in
        for ii, cell in enumerate(Cortical_Pop):
            cell.position[:2] = Cortical_Neuron_xy_Positions[:2, ii]

        # Load STN positions - Comment/Remove to generate new positions
        STN_Neuron_xy_Positions = np.loadtxt(
            STN_Neuron_xy_Positions_file, delimiter=","
        )

        # Set STN xy positions to those loaded in
        for ii, cell in enumerate(STN_Pop):
            cell.position[:2] = STN_Neuron_xy_Positions[:2, ii]
            cell.position[2] = 500

    # Synaptic Connections
    # Add variability to Cortical connections - cortical interneuron
    # connection weights are random from uniform distribution
    gCtxInt_max_weight = 2.5e-3  # Ctx -> Int max coupling value
    gIntCtx_max_weight = 6.0e-3  # Int -> Ctx max coupling value
    gCtxInt = RandomDistribution(
        "uniform", (0, gCtxInt_max_weight), rng=NumpyRNG(seed=rng_seed)
    )
    gIntCtx = RandomDistribution(
        "uniform", (0, gIntCtx_max_weight), rng=NumpyRNG(seed=rng_seed)
    )

    # Define other synaptic connection weights and delays
    syn_CorticalAxon_Interneuron = StaticSynapse(weight=gCtxInt, delay=2)
    syn_Interneuron_CorticalSoma = StaticSynapse(weight=gIntCtx, delay=2)
    # syn_CorticalSpikeSourceCorticalAxon = StaticSynapse(weight=0.25, delay=0)
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

    if create:
        # Create new network topology Connections
        connections = dict(
            CorticalAxonInterneuron=FixedNumberPreConnector(
                n=10, allow_self_connections=False
            ),
            CorticalSTN=FixedNumberPreConnector(n=10, allow_self_connections=False),
            CorticalSomaThalamic=FixedNumberPreConnector(
                n=5, allow_self_connections=False
            ),
            GPeGPe=FixedNumberPreConnector(n=1, allow_self_connections=False),
            GPeGPi=FixedNumberPreConnector(n=1, allow_self_connections=False),
            GPeSTN=FixedNumberPreConnector(n=2, allow_self_connections=False),
            GPiThalamic=FixedNumberPreConnector(n=1, allow_self_connections=False),
            InterneuronCortical=FixedNumberPreConnector(
                n=1, allow_self_connections=False
            ),
            STNGPe=FixedNumberPreConnector(n=1, allow_self_connections=False),
            STNGPi=FixedNumberPreConnector(n=1, allow_self_connections=False),
            StriatalGPe=FixedNumberPreConnector(n=1, allow_self_connections=False),
            ThalamicCorticalSoma=FixedNumberPreConnector(
                n=1, allow_self_connections=False
            ),
        )
    else:
        # Load network topology from file
        keys = [
                "CorticalAxonInterneuron"
                "CorticalSTN"
                "CorticalSomaThalamic"
                "GPeGPe"
                "GPeGPi"
                "GPeSTN"
                "GPiThalamic"
                "InterneuronCortical"
                "STNGPe"
                "STNGPi"
                "StriatalGPe"
                "ThalamicCorticalSoma"
            ]
        connections = {}
        for key in keys:
            connections[key] = FromFileConnector(str(CONNECTIONS_DIR / f"{key}_Connections.txt"))

    prj_CorticalAxon_Interneuron = Projection(
        Cortical_Pop,
        Interneuron_Pop,
        connections["CorticalAxonInterneuron_Connections.txt"],
        syn_CorticalAxon_Interneuron,
        source="middle_axon_node",
        receptor_type="AMPA",
    )
    prj_Interneuron_CorticalSoma = Projection(
        Interneuron_Pop,
        Cortical_Pop,
        connections["InterneuronCortical_Connections.txt"],
        syn_Interneuron_CorticalSoma,
        receptor_type="GABAa",
    )
    prj_CorticalSTN = Projection(
        Cortical_Pop,
        STN_Pop,
        connections["CorticalSTN_Connections.txt"],
        syn_CorticalCollateralSTN,
        source="collateral(0.5)",
        receptor_type="AMPA",
    )
    prj_STNGPe = Projection(
        STN_Pop,
        GPe_Pop,
        connections["STNGPe_Connections.txt"],
        syn_STNGPe,
        source="soma(0.5)",
        receptor_type="AMPA",
    )
    prj_GPeGPe = Projection(
        GPe_Pop,
        GPe_Pop,
        connections["GPeGPe_Connections.txt"],
        syn_GPeGPe,
        source="soma(0.5)",
        receptor_type="GABAa",
    )
    prj_GPeSTN = Projection(
        GPe_Pop,
        STN_Pop,
        connections["GPeSTN_Connections.txt"],
        syn_GPeSTN,
        source="soma(0.5)",
        receptor_type="GABAa",
    )
    prj_StriatalGPe = Projection(
        Striatal_Pop,
        GPe_Pop,
        connections["StriatalGPe_Connections.txt"],
        syn_StriatalGPe,
        source="soma(0.5)",
        receptor_type="GABAa",
    )
    prj_STNGPi = Projection(
        STN_Pop,
        GPi_Pop,
        connections["STNGPi_Connections.txt"],
        syn_STNGPi,
        source="soma(0.5)",
        receptor_type="AMPA",
    )
    prj_GPeGPi = Projection(
        GPe_Pop,
        GPi_Pop,
        connections["GPeGPi_Connections.txt"],
        syn_GPeGPi,
        source="soma(0.5)",
        receptor_type="GABAa",
    )
    prj_GPiThalamic = Projection(
        GPi_Pop,
        Thalamic_Pop,
        connections["GPiThalamic_Connections.txt"],
        syn_GPiThalamic,
        source="soma(0.5)",
        receptor_type="GABAa",
    )
    prj_ThalamicCortical = Projection(
        Thalamic_Pop,
        Cortical_Pop,
        connections["ThalamicCorticalSoma_Connections.txt"],
        syn_ThalamicCortical,
        source="soma(0.5)",
        receptor_type="AMPA",
    )
    prj_CorticalThalamic = Projection(
        Cortical_Pop,
        Thalamic_Pop,
        connections["CorticalSomaThalamic_Connections.txt"],
        syn_CorticalThalamic,
        source="soma(0.5)",
        receptor_type="AMPA",
    )

    if create:
        # Save the network topology so it can be reloaded
        # prj_CorticalSpikeSourceCorticalSoma.saveConnections(file="CorticalSpikeSourceCorticalSoma_Connections.txt")
        prj_CorticalAxon_Interneuron.saveConnections(
            file=str(CONNECTIONS_DIR / "CorticalAxonInterneuron_Connections.txt")
        )
        prj_Interneuron_CorticalSoma.saveConnections(
            file=str(CONNECTIONS_DIR / "InterneuronCortical_Connections.txt")
        )
        prj_CorticalSTN.saveConnections(
            file=str(CONNECTIONS_DIR / "CorticalSTN_Connections.txt")
        )
        prj_STNGPe.saveConnections(file=str(CONNECTIONS_DIR / "STNGPe_Connections.txt"))
        prj_GPeGPe.saveConnections(file=str(CONNECTIONS_DIR / "GPeGPe_Connections.txt"))
        prj_GPeSTN.saveConnections(file=str(CONNECTIONS_DIR / "GPeSTN_Connections.txt"))
        prj_StriatalGPe.saveConnections(
            file=str(CONNECTIONS_DIR / "StriatalGPe_Connections.txt")
        )
        prj_STNGPi.saveConnections(file=str(CONNECTIONS_DIR / "STNGPi_Connections.txt"))
        prj_GPeGPi.saveConnections(file=str(CONNECTIONS_DIR / "GPeGPi_Connections.txt"))
        prj_GPiThalamic.saveConnections(
            file=str(CONNECTIONS_DIR / "GPiThalamic_Connections.txt")
        )
        prj_ThalamicCortical.saveConnections(
            file=str(CONNECTIONS_DIR / "ThalamicCorticalSoma_Connections.txt")
        )
        prj_CorticalThalamic.saveConnections(
            file=str(CONNECTIONS_DIR / "CorticalSomaThalamic_Connections.txt")
        )

    # Load GPe stimulation order
    GPe_stimulation_order = np.loadtxt(
        DATA_DIR / "GPe_Stimulation_Order.txt", delimiter=","
    )
    GPe_stimulation_order = [int(index) for index in GPe_stimulation_order]

    return (
        Pop_size,
        striatal_spike_times,
        Cortical_Pop,
        Interneuron_Pop,
        STN_Pop,
        GPe_Pop,
        GPi_Pop,
        Striatal_Pop,
        Thalamic_Pop,
        prj_CorticalAxon_Interneuron,
        prj_Interneuron_CorticalSoma,
        prj_CorticalSTN,
        prj_STNGPe,
        prj_GPeGPe,
        prj_GPeSTN,
        prj_StriatalGPe,
        prj_STNGPi,
        prj_GPeGPi,
        prj_GPiThalamic,
        prj_ThalamicCortical,
        prj_CorticalThalamic,
        GPe_stimulation_order,
    )
