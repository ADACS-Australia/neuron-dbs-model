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
        super().__init__(setpoint,ts)

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

        super().clear()

        self.ITerm = 0.0
        self.DTerm = 0.0


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
