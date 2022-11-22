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


class StandardPIDController:
    """Standard PID Controller Class"""

    label = "Standard_PID_Controller"
    units = "mA"

    def __init__(
        self,
        setpoint=0.0,
        ts=0.02,
        kp=0.0,
        ti=0.0,
        td=0.0,
        minvalue=0.0,
        maxvalue=1e9,
    ):
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

        self.kp = kp
        self.ti = ti
        self.td = td

        # Set output value bounds
        self.minvalue = minvalue
        self.maxvalue = maxvalue

        # Initialize controller terms
        self.ITerm = 0.0
        self.DTerm = 0.0

    def clear(self):
        """Clears controller variables"""

        self.last_error = 0.0

        self.state_history = []
        self.error_history = []
        self.output_history = []
        self.sample_times = []

        self.last_output_value = 0.0
        self.output_value = 0.0

        self.ITerm = 0.0
        self.DTerm = 0.0

    def get_error(self, state_value):
        # Calculate Error - if setpoint > 0.0, then normalize error with
        # respect to set point
        if self.setpoint == 0.0:
            error = state_value - self.setpoint
        else:
            error = (state_value - self.setpoint) / self.setpoint
        return error

    def set_output(self, state_value):
        """Calculates controller output signal for given reference feedback

        where:

        u(t) = K_p (e(t) + (1/T_i)* \\int_{0}^{t} e(t)dt + T_d {de}/{dt})

        where the error calculated is the tracking error (r(t) - y(t))

        """

        error = self.get_error(state_value)

        delta_time = self.ts
        delta_error = error - self.last_error

        self.ITerm += error * delta_time

        self.DTerm = 0.0
        if delta_time > 0:
            self.DTerm = delta_error / delta_time

        # Calculate u(t) - catch potential division by zero error
        try:
            u = self.kp * (
                error + ((1.0 / self.ti) * self.ITerm) + (self.td * self.DTerm)
            )
        except ZeroDivisionError:
            u = self.kp * (error + (0.0 * self.ITerm) + (self.td * self.DTerm))

        # Bound the controller output if necessary
        # (between minvalue - maxvalue)
        if u > self.maxvalue:
            self.output_value = self.maxvalue
            # Back-calculate the integral error
            self.ITerm -= error * delta_time
        elif u < self.minvalue:
            self.output_value = self.minvalue
            # Back-calculate the integral error
            self.ITerm -= error * delta_time
        else:
            self.output_value = u

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
