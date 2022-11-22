# -*- coding: utf-8 -*-
"""
Created on Wed April 03 14:27:26 2019

Description: Controller class implementations for:

https://www.frontiersin.org/articles/10.3389/fnins.2020.00166/

@author: John Fleming, john.fleming@ucdconnect.ie
"""

import math

from mpi4py import MPI

from .ConstantController import ConstantController

rank = MPI.COMM_WORLD.Get_rank()


class OnOffController(ConstantController):
    """On-Off Controller Class"""

    label = "On_Off_Controller"

    def __init__(
        self,
        setpoint=0.0,
        ts=0.02,
        minvalue=0.0,
        maxvalue=1e9,
        rampduration=0.25,
    ):
        super().__init__(setpoint, ts, minvalue, maxvalue, constantvalue=None)
        del self.constantvalue

        # should be defined in sec, i.e. 0.25 sec
        self.rampduration = rampduration

    def get_error(self, state_value):
        """Calculates updated controller output value for given reference feedback

        y(t) = y(t-1) + u(t)

        where:

        u(t) = maxvalue / (rampduration/ts) if e(t) > setpoint
        or -maxvalue / (rampduration/ts) if e(t) < setpoint

        """
        # Calculate Error - if setpoint > 0.0, then normalize error with
        # respect to set point
        if self.setpoint == 0.0:
            error = state_value - self.setpoint
            increment = 0.0
        else:
            error = (state_value - self.setpoint) / self.setpoint
            if error > 0.0:
                increment = self.outputvalueincrement
            else:
                increment = -self.outputvalueincrement
        return error, increment

    def set_output(self, state_value):

        error, increment = self.get_error(state_value)

        # Bound the controller output (between minvalue - maxvalue)
        if self.last_output_value + increment > self.maxvalue:
            self.output_value = self.maxvalue
        elif self.last_output_value + increment < self.minvalue:
            self.output_value = self.minvalue
        else:
            self.output_value = self.last_output_value + increment

        return error

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
