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


class ConstantController:
    """Constant DBS Parameter Controller Class"""

    label = "Constant_Controller"

    def __init__(
        self,
        setpoint=0.0,
        minvalue=0.0,
        maxvalue=1e9,
        constantvalue=0.0,
        ts=0.0,
        units="mA",
    ):
        # Initial Controller Values
        self.setpoint = setpoint
        self.maxvalue = maxvalue
        self.minvalue = minvalue
        self.constantvalue = constantvalue
        self.ts = ts  # should be in sec as per above
        self.units = units

        # Set output value
        self.output_value = 0

        # Lists for tracking controller history
        self.state_history = []
        self.error_history = []
        self.output_history = []
        self.sample_times = []

    def clear(self):
        """Clears current On-Off controller output value and history"""

        self.state_history = []
        self.error_history = []
        self.output_history = []
        self.sample_times = []

        self.output_value = 0.0

    def update(self, state_value, current_time):
        """Calculates biomarker for constant DBS value

        u = self.constantvalue

        """

        # Calculate Error - if setpoint > 0.0
        # normalize error with respect to set point
        if self.setpoint == 0.0:
            error = state_value - self.setpoint
        else:
            error = (state_value - self.setpoint) / self.setpoint

        # Bound the controller output (between minvalue - maxvalue)
        if self.constantvalue > self.maxvalue:
            self.output_value = self.maxvalue
        elif self.constantvalue < self.minvalue:
            self.output_value = self.minvalue
        else:
            self.output_value = self.constantvalue

        # Record state, error and sample time values
        self.state_history.append(state_value)
        self.error_history.append(error)
        self.output_history.append(self.output_value)
        # Convert from msec to sec
        self.sample_times.append(current_time / 1000)

        return self.output_value
