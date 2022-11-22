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
        ts=0.0,
        minvalue=0.0,
        maxvalue=1e9,
        constantvalue=0.0,

    ):
        super().__init__(setpoint,ts)

        self.maxvalue = maxvalue
        self.minvalue = minvalue
        self.constantvalue = constantvalue

    def set_output(self, state_value):
        """Always sets controller output value to constant"""

        error = self.get_error(state_value)

        # Bound the controller output (between minvalue - maxvalue)
        if self.constantvalue > self.maxvalue:
            self.output_value = self.maxvalue
        elif self.constantvalue < self.minvalue:
            self.output_value = self.minvalue
        else:
            self.output_value = self.constantvalue

        return error
