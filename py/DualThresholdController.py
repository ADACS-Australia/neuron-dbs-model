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
from .OnOffController import OnOffController

rank = MPI.COMM_WORLD.Get_rank()


class DualThresholdController(OnOffController):
    """Dual-Threshold Controller Class"""

    label = "DualThresholdController"

    def __init__(
        self,
        ts=0.02,
        minvalue=0.0,
        maxvalue=1e9,
        rampduration=0.25,
        lowerthreshold=0.0,
        upperthreshold=0.1,
    ):
        super().__init__(None,ts,minvalue,maxvalue,rampduration)

        del self.setpoint
        self.upperthreshold = upperthreshold
        self.lowerthreshold = lowerthreshold

    def get_error(self, state_value):
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
        return error, increment
