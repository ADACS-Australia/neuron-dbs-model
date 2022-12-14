import numpy as np
import scipy.signal as signal
from pyNN.parameters import Sequence


def generate_dbs_signal(
    start_time,
    stop_time,
    dt,
    amplitude,
    frequency,
    pulse_width,
    offset,
    last_pulse_time_prior=0,
):
    """Generate monophasic square-wave DBS signal

    Example inputs:
        start_time = 0                # ms
        stop_time = 12000            # ms
        dt = 0.01                    # ms
        amplitude = -1.0            # mA (<0 = cathodic, >0 = anodic)
        frequency = 130.0            # Hz
        pulse_width    = 0.06            # ms
        offset = 0                    # mA
    """

    _times = np.arange(start_time, stop_time, dt)
    if len(_times) == 0:
        _times = np.array([start_time])
    times = np.round(_times, 2)

    tmp = np.arange(0, stop_time - start_time, dt) / 1000.0
    if len(tmp) == 0:
        tmp = np.array([0.0])

    if frequency == 0:
        dbs_signal = np.zeros(len(tmp))
        last_pulse_time = last_pulse_time_prior
        next_pulse_time = 1e9
    else:
        # Calculate the duty cycle of the DBS signal
        isi = 1000.0 / frequency  # time is in ms
        duty_cycle = pulse_width / isi
        tt = 2.0 * np.pi * frequency * tmp
        dbs_signal = offset + 0.5 * (1.0 + signal.square(tt, duty=duty_cycle))
        dbs_signal[-1] = 0.0

        # Calculate the time for the first pulse of the next segment
        try:
            last_pulse_index = np.where(np.diff(dbs_signal) < 0)[0][-1]
        except IndexError:
            # Catch times when signal may be flat
            last_pulse_index = len(dbs_signal) - 1

        # Track when the last pulse was
        last_pulse_time = times[last_pulse_index]
        next_pulse_time = last_pulse_time + isi - pulse_width

        # Rescale amplitude
        dbs_signal *= amplitude

    return dbs_signal, times, next_pulse_time, last_pulse_time


def generate_poisson_spike_times(
    pop_size, start_time, duration, fr, timestep, random_seed
):
    """generate_population_spike_times generates (N = pop_size) Poisson
    distributed spiketrains with firing rate fr.

    Example inputs:
        pop_size = 10
        start_time = 0.0		# ms
        end_time = 6000.0		# ms
        timestep = 1  			# ms
        fr = 1					# Hz
    """

    # Convert to sec for calculating the spikes matrix
    dt = float(timestep) / 1000.0  # sec
    sim_time = float(((start_time + duration) - start_time) / 1000.0)  # sec
    n_bins = int(np.floor(sim_time / dt))

    spike_matrix = np.where(np.random.uniform(0, 1, (pop_size, n_bins)) < fr * dt)

    # Create time vector - ms
    t_vec = np.arange(start_time, start_time + duration, timestep)

    # Make array of spike times
    for neuron_index in np.arange(pop_size):
        neuron_spike_times = t_vec[
            spike_matrix[1][np.where(spike_matrix[0][:] == neuron_index)]
        ]
        if neuron_index == 0:
            spike_times = Sequence(neuron_spike_times)
        else:
            spike_times = np.vstack((spike_times, Sequence(neuron_spike_times)))

    return spike_times


def make_beta_cheby1_filter(fs, n, rp, low, high):
    """Calculate bandpass filter coefficients (1st Order Chebyshev Filter)"""
    nyq = 0.5 * fs
    lowcut = low / nyq
    highcut = high / nyq

    b, a = signal.cheby1(n, rp, [lowcut, highcut], "band")

    return b, a


def calculate_avg_beta_power(lfp_signal, tail_length, beta_b, beta_a):
    """Calculate the average power in the beta-band for the current LFP signal
    window, i.e. beta Average Rectified Value (ARV)

    Inputs:
        lfp_signal          - window of LFP signal (samples)

        tail_length         - tail length which will be discarded due to
                              filtering artifact (samples)

        beta_b, beta_a      - filter coefficients for filtering the beta-band
                              from the signal
    """

    lfp_beta_signal = signal.filtfilt(beta_b, beta_a, lfp_signal)
    lfp_beta_signal_rectified = np.absolute(lfp_beta_signal)
    avg_beta_power = np.mean(lfp_beta_signal_rectified[-2 * tail_length : -tail_length])

    return avg_beta_power
