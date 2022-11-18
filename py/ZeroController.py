# -*- coding: utf-8 -*-
"""
Created on Wed April 03 14:27:26 2019

Description: Controller class implementations for:

https://www.frontiersin.org/articles/10.3389/fnins.2020.00166/

@author: John Fleming, john.fleming@ucdconnect.ie
"""

import math

import numpy as np
import scipy.signal as signal
from mpi4py import MPI

rank = MPI.COMM_WORLD.Get_rank()


class ZeroController:
    """
    Dummy controller with no stimulation
    Template for all other controllers
    """

    label = "Zero_Controller"

    def __init__(self, setpoint=0.0, ts=0.02):
        # Initial Controller Values
        self.setpoint = setpoint
        self.ts = ts  # should be in sec

        self.current_time = 0.0  # (sec)
        self.last_time = 0.0
        self.last_error = 0.0
        self.last_output_value = 0.0

        # Initialize the output value of the controller
        self.output_value = 0.0

        # Lists for tracking controller history
        self.state_history = []
        self.error_history = []
        self.output_history = []
        self.sample_times = []

    def clear(self):
        """Clears controller variables"""

        self.last_error = 0.0

        self.state_history = []
        self.error_history = []
        self.output_history = []
        self.sample_times = []

        self.output_value = 0.0

    def update(self, state_value, current_time):
        """Update controller state"""

        # Calculate Error - if setpoint > 0.0, then normalize error with
        # respect to set point
        if self.setpoint == 0.0:
            error = state_value - self.setpoint
        else:
            error = (state_value - self.setpoint) / self.setpoint

        # Converting from msec to sec
        self.current_time = current_time / 1000.0

        # Remember last time and last error for next calculation
        self.last_time = self.current_time
        self.last_error = error

        self.output_value = 0

        # Update the last output value
        self.last_output_value = self.output_value

        # Record state, error, y(t), and sample time values
        self.state_history.append(state_value)
        self.error_history.append(error)
        self.output_history.append(self.output_value)
        # Convert from msec to sec
        self.sample_times.append(current_time / 1000)

        # Return controller output
        return self.output_value

    def generate_dbs_signal(
        self,
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

        times = np.round(np.arange(start_time, stop_time, dt), 2)
        tmp = np.arange(0, stop_time - start_time, dt) / 1000.0

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
                next_pulse_time = times[last_pulse_index] + isi - pulse_width

                # Track when the last pulse was
                last_pulse_time = times[last_pulse_index]

            except IndexError:
                # Catch times when signal may be flat
                last_pulse_index = len(dbs_signal) - 1
                next_pulse_time = times[last_pulse_index] + isi - pulse_width

            # Rescale amplitude
            dbs_signal *= amplitude

        return dbs_signal, times, next_pulse_time, last_pulse_time
