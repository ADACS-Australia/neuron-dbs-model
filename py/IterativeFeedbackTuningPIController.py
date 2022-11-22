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


class IterativeFeedbackTuningPIController:

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
        super().__init__(setpoint,ts)

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
