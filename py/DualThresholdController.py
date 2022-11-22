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


class DualThresholdController:
    """Dual-Threshold Controller Class"""

    label = "DualThresholdController"
    units = "mA"

    def __init__(
        self,
        # setpoint=0.0,
        ts=0.02,
        minvalue=0.0,
        maxvalue=1e9,
        rampduration=0.25,
        lowerthreshold=0.0,
        upperthreshold=0.1,
    ):
        # Initial Controller Values
        # self.setpoint = setpoint
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

        self.maxvalue = maxvalue
        self.minvalue = minvalue
        # should be defined in sec, i.e. 0.25 sec
        self.rampduration = rampduration
        self.upperthreshold = upperthreshold
        self.lowerthreshold = lowerthreshold

    def clear(self):
        """Clears controller variables"""

        self.last_error = 0.0

        self.state_history = []
        self.error_history = []
        self.output_history = []
        self.sample_times = []

        self.last_output_value = 0.0
        self.output_value = 0.0

    def get_error(self, state_value):
        # Check how to update controller value and calculate error with
        # respect to upper/lower threshold
        # Increase if above upper threshold
        if state_value > self.upperthreshold:
            error = (state_value - self.upperthreshold) / self.upperthreshold
            increment = self.outputvalueincrement
        # Decrease if below lower threshold
        elif state_value < self.lowerthreshold:
            error = (state_value - self.lowerthreshold) / self.lowerthreshold
            increment = -self.outputvalueincrement
        # Do nothing when within upper and lower thresholds
        else:
            error = 0
            increment = 0
        return error, increment

    def set_output(self, state_value):
        """Calculates updated controller output value for given reference feedback
        if state_value > upper_threshold:
            y(t) = y(t-1) + u(t)
        elif state_value < lower_threshold:
            y(t) = y(t-1) + u(t)
        else:
            y(t) = y(t-1)
        where:

        u(t) = maxvalue / (rampduration / ts)
        if state_value(t) > upperthreshold
        or
        u(t) = -maxvalue / (rampduration / ts)
        if state_value(t) < lowerthreshold

        """

        error, increment = self.get_error(state_value)

        # Bound the controller output (between minvalue - maxvalue)
        if self.last_output_value + increment > self.maxvalue:
            self.output_value = self.maxvalue
        elif self.last_output_value + increment < self.minvalue:
            self.output_value = self.minvalue
        else:
            self.output_value = self.last_output_value + increment

        return error

    def update(self, state_value, current_time):
        """Update controller state"""
        self.current_time = current_time

        error = self.set_output(state_value)

        # Remember last time and last error for next calculation
        self.last_time = self.current_time
        self.last_error = error

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

    # Calculate how much controller output value will change each controller call
    @staticmethod
    def _outputvalueincrement(maxvalue, minvalue, rampduration, ts):
        return (maxvalue - minvalue) / math.ceil(rampduration / ts)

    @property
    def maxvalue(self):
        return self._maxvalue

    @maxvalue.setter
    def maxvalue(self, maxvalue):
        self._maxvalue = maxvalue
        self.outputvalueincrement = self._outputvalueincrement(
            maxvalue, self.minvalue, self.rampduration, self.ts
        )

    @property
    def minvalue(self):
        return self._minvalue

    @minvalue.setter
    def minvalue(self, minvalue):
        """Sets the lower bound for the controller output"""
        self._minvalue = minvalue
        self.outputvalueincrement = self._outputvalueincrement(
            self.maxvalue, minvalue, self.rampduration, self.ts
        )

    @property
    def rampduration(self):
        return self._rampduration

    @rampduration.setter
    def rampduration(self, ramp_duration):
        """Sets how long the controller output takes to reach its max value"""
        self._rampduration = ramp_duration
        self.outputvalueincrement = self._outputvalueincrement(
            self.maxvalue, self.minvalue, ramp_duration, self.ts
        )

    @property
    def ts(self):
        return self._ts

    @ts.setter
    def ts(self, ts):
        """Sets the sampling rate of the controller"""
        self._ts = ts
        self.outputvalueincrement = self._outputvalueincrement(
            self.maxvalue, self.minvalue, self.rampduration, ts
        )
