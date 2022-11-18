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

    def __init__(
        self,
        lowerthreshold=0.0,
        upperthreshold=0.1,
        minvalue=0.0,
        maxvalue=1e9,
        rampduration=0.25,
        ts=0.02,
    ):
        # Initial Controller Values
        self.upperthreshold = upperthreshold
        self.lowerthreshold = lowerthreshold
        self.maxvalue = maxvalue
        self.minvalue = minvalue
        # should be defined in sec, i.e. 0.25 sec
        self.rampduration = rampduration
        # should be in sec as per above
        self.ts = ts

        # Calculate how much controller output value will change
        # each controller call
        self.outputvalueincrement = (self.maxvalue - self.minvalue) / math.ceil(
            self.rampduration / self.ts
        )

        # Initialize the output value of the controller
        self.last_output_value = 0.0
        self.output_value = 0.0

        # Lists for tracking controller history
        self.state_history = []
        self.error_history = []
        self.output_history = []
        self.sample_times = []

    def clear(self):
        """Clears current dual-threshold controller output value and history"""

        self.state_history = []
        self.error_history = []
        self.output_history = []
        self.sample_times = []

        self.last_output_value = 0.0
        self.output_value = 0.0

    def update(self, state_value, current_time):
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

        # Bound the controller output (between minvalue - maxvalue)
        if self.last_output_value + increment > self.maxvalue:
            self.output_value = self.maxvalue
        elif self.last_output_value + increment < self.minvalue:
            self.output_value = self.minvalue
        else:
            self.output_value = self.last_output_value + increment

        # Record state, error and sample time values
        self.state_history.append(state_value)
        self.error_history.append(error)
        self.output_history.append(self.output_value)
        # Convert from msec to sec
        self.sample_times.append(current_time / 1000)

        self.last_output_value = self.output_value

        return self.output_value

    def setMaxValue(self, max_value):
        """Sets the upper bound for the controller output"""
        self.maxvalue = max_value
        self.outputvalueincrement = (self.maxvalue - self.minvalue) / (
            self.rampduration / self.ts
        )

    def setMinValue(self, min_value):
        """Sets the lower bound for the controller output"""
        self.minvalue = min_value
        self.outputvalueincrement = (self.maxvalue - self.minvalue) / (
            self.rampduration / self.ts
        )

    def setRampDuration(self, ramp_duration):
        """Sets how long the controller output takes to reach its max value"""
        self.rampduration = ramp_duration
        self.outputvalueincrement = (self.maxvalue - self.minvalue) / (
            self.rampduration / self.ts
        )

    def setTs(self, ts):
        """Sets the sampling rate of the controller"""
        self.ts = ts
        self.outputvalueincrement = (self.maxvalue - self.minvalue) / (
            self.rampduration / self.ts
        )
