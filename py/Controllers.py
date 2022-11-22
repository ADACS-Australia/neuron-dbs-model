# -*- coding: utf-8 -*-
"""
Created on Wed April 03 14:27:26 2019

Description: Controller class implementations for:

https://www.frontiersin.org/articles/10.3389/fnins.2020.00166/

@author: John Fleming, john.fleming@ucdconnect.ie
"""

import math

import numpy as np
from mpi4py import MPI

rank = MPI.COMM_WORLD.Get_rank()


class ZeroController:
    """
    Dummy controller with no stimulation
    Template for all other controllers
    """

    label = "Zero_Controller"
    units = "mA"

    def __init__(
        self,
        setpoint=0.0,
        ts=0.02,
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
        # Calculate Error - if setpoint > 0.0, then normalize error with
        # respect to set point
        if self.setpoint == 0.0:
            error = state_value - self.setpoint
        else:
            error = (state_value - self.setpoint) / self.setpoint
        return error

    def set_output(self, state_value):
        """Always sets controller output value to zero"""

        error = self.get_error(state_value)
        self.output_value = 0.0

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


class ConstantController(ZeroController):
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
        super().__init__(setpoint, ts)

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
        super().__init__(None, ts, minvalue, maxvalue, rampduration)

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


class StandardPIDController(ZeroController):
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
        super().__init__(setpoint, ts)

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


class IterativeFeedbackTuningPIController(ZeroController):

    label = "Iterative_Feedback_Tuning_PI_Controller"

    def __init__(
        self,
        setpoint=0.0,
        ts=0.02,
        stage_length=0.0,
        kp=0.0,
        ti=0.0,
        minvalue=0.0,
        maxvalue=1e9,
        gamma=0.005,
        lam=1e-8,
        min_kp=0,
        min_ti=0,
    ):
        super().__init__(setpoint, ts)

        self.stage_length = stage_length
        self.stage_length_samples = int(np.ceil(stage_length / ts)) + 1
        self.kp = kp
        self.ti = ti
        self.gamma = gamma
        self.lam = lam
        self.iteration_stage = -1
        self.min_kp = min_kp
        self.min_ti = min_ti

        # Set output value bounds
        self.min_value = minvalue
        self.max_value = maxvalue

        self.stage_start_time = 0.0
        self.integral_term = 0.0
        self.integral_term_history = []
        self.iteration_history = []
        self.reference_history = []
        self.parameter_history = []
        self.recorded_output = np.zeros(self.stage_length_samples)

    def clear(self):
        """Clears controller variables"""

        super().clear()

        self.integral_term = 0.0
        self.integral_term_history = []
        self.iteration_history = []
        self.parameter_history = []
        self.recorded_output = np.zeros(self.stage_length_samples)

    def get_error(self, state_value):
        if self.setpoint != 0:
            error = (state_value - self.setpoint) / self.setpoint
        else:
            error = state_value
        return error

    def set_output(self, state_value):
        """"""
        elapsed_time = (self.current_time - self.stage_start_time) / 1000
        self.setpoint = self.reference_signal(elapsed_time)
        if rank == 0:
            print(
                "Stage: %d, Elapsed time: %.3f s, Reference: %.5f"
                % (self.iteration_stage, elapsed_time, self.setpoint)
            )

        if elapsed_time >= self.stage_length:
            if (
                self.iteration_stage == 0
                and len(self.error_history) < self.stage_length_samples
            ):
                if rank == 0:
                    print("Extending stage 0 to gather more samples")
            elif (
                self.iteration_stage == 1
                and len(self.error_history) < 2 * self.stage_length_samples
            ):
                if rank == 0:
                    print("Extending stage 1 to gather more samples")
            else:
                if self.iteration_stage == 1:
                    self.kp, self.ti = self.new_controller_parameters()
                    if rank == 0:
                        print(f"New params: kp={self.kp}, ti={self.ti}")
                self.stage_start_time = self.current_time
                elapsed_time = 0
                self.iteration_stage = (self.iteration_stage + 1) % 2
                if rank == 0:
                    print("Stage change, now at stage %d" % self.iteration_stage)

        if self.iteration_stage == 0:
            sample = int(elapsed_time / self.ts)
            self.recorded_output[sample] = state_value

        error = self.get_error(state_value)

        self.integral_term += error * self.ts

        # Calculate u(t)
        try:
            u = self.kp * (error + ((1.0 / self.ti) * self.integral_term))
        except ZeroDivisionError:
            u = self.kp * (error + (0.0 * self.integral_term))

        # Bound the controller output
        if u > self.max_value:
            self.output_value = self.max_value
            # Back-calculate the integral error
            self.integral_term -= error * self.ts
        elif u < self.min_value:
            self.output_value = self.min_value
            # Back-calculate the integral error
            self.integral_term -= error * self.ts
        else:
            self.output_value = u

    def update(self, state_value, current_time):
        super().update(state_value, current_time)

        self.parameter_history.append([self.kp, self.ti])
        self.integral_term_history.append(self.integral_term)
        self.iteration_history.append(self.iteration_stage)
        self.reference_history.append(self.setpoint)

        # Return controller output
        return self.output_value

    def dc_drho(self, s):
        kp = self.kp
        ti = self.ti
        ts = self.ts
        yout_kp = np.zeros(len(s) + 1)
        yout_ti = np.zeros(len(s) + 1)
        for i, u in enumerate(s):
            yout_kp[i + 1] = u / kp
            yout_ti[i + 1] = (ti**2 * yout_ti[i] - ts * u) / (ti**2 + ti * ts)

        return yout_kp[1:], yout_ti[1:]

    def compute_fitness_gradient(self):
        y1 = self.error_history[
            -2 * self.stage_length_samples : -self.stage_length_samples
        ]
        y2 = self.error_history[-self.stage_length_samples :]
        u1 = self.output_history[
            -2 * self.stage_length_samples : -self.stage_length_samples
        ]
        u2 = self.output_history[-self.stage_length_samples :]
        y_tilde = np.array(y1)
        u_rho = u1
        dy_dkp, dy_dti = self.dc_drho(y2)
        du_dkp, du_dti = self.dc_drho(u2)

        dy_drho = np.vstack((dy_dkp, dy_dti))
        du_drho = np.vstack((du_dkp, du_dti))

        lam = self.lam
        y_part = y_tilde * dy_drho
        u_part = u_rho * du_drho

        grad = np.sum((y_part + lam * u_part), axis=1) / self.stage_length_samples
        return grad

    def new_controller_parameters(self):
        rho = np.array([self.kp, self.ti])
        gamma = self.gamma
        # TODO: Make r, min_kp, min_ti parameters
        r = np.identity(2)
        if len(self.error_history) >= 2 * self.stage_length_samples:
            grad = self.compute_fitness_gradient()
            if rank == 0:
                print(f"Gradient: ({grad[0]}, {grad[1]})'")
        else:
            grad = [0, 0]
            if rank == 0:
                print("Too few samples, skipping update")
        new_rho = rho - gamma * np.dot(r, grad)
        if new_rho[0] < self.min_kp:
            new_rho[0] = self.min_kp
        if new_rho[1] < self.min_ti:
            new_rho[1] = self.min_ti
        return new_rho[0], new_rho[1]

    def reference_signal(self, elapsed_time):
        sample = int(elapsed_time / self.ts)
        if self.iteration_stage == 0:
            return self.setpoint
        if sample > self.stage_length_samples:
            sample = self.stage_length_samples - 1
        return self.recorded_output[sample]
