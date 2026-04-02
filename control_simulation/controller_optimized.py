#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PID tracking controllers (ASCII-only).
"""

from typing import Dict, Tuple
import numpy as np
from scipy.interpolate import CubicSpline


class OptimizedPIDController:
    def __init__(
        self,
        Kp: float = 2.0,
        Ki: float = 0.1,
        Kd: float = 1.0,
        integral_limit: float = 50.0,
        output_limit: float = 20.0,
    ):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral_limit = integral_limit
        self.output_limit = output_limit
        self.integral_error = np.zeros(3)

    def reset(self) -> None:
        self.integral_error = np.zeros(3)

    def compute_control(
        self,
        reference_position: np.ndarray,
        reference_velocity: np.ndarray,
        current_position: np.ndarray,
        current_velocity: np.ndarray,
        dt: float,
        feedforward_accel: np.ndarray = None,
    ) -> np.ndarray:
        pos_err = reference_position - current_position
        vel_err = reference_velocity - current_velocity

        self.integral_error += pos_err * dt
        norm_int = np.linalg.norm(self.integral_error)
        if norm_int > self.integral_limit:
            self.integral_error = self.integral_error / norm_int * self.integral_limit

        control = self.Kp * pos_err + self.Ki * self.integral_error + self.Kd * vel_err
        # If provided, treat feedforward_accel as additional desired accel.
        if feedforward_accel is not None:
            control = control + feedforward_accel

        norm_u = np.linalg.norm(control)
        if norm_u > self.output_limit:
            control = control / norm_u * self.output_limit

        return control


class FeedforwardController:
    def __init__(self, spacecraft, asteroid):
        self.sc = spacecraft
        self.ast = asteroid

    def compute_feedforward(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        mass: float,
        reference_accel: np.ndarray,
    ) -> np.ndarray:
        if hasattr(self.ast, "compute_gravity"):
            g = self.ast.compute_gravity(position)
        else:
            r_norm = np.linalg.norm(position)
            g = -self.ast.mu * position / (r_norm ** 3) if r_norm > 1e-10 else np.zeros(3)

        omega = getattr(self.ast, "omega", np.zeros(3))
        coriolis = -2 * np.cross(omega, velocity)
        centrifugal = -np.cross(omega, np.cross(omega, position))

        a_thrust = reference_accel - g - coriolis - centrifugal
        thrust = a_thrust * mass

        norm_t = np.linalg.norm(thrust)
        if norm_t > self.sc.T_max:
            thrust = thrust / norm_t * self.sc.T_max
        return thrust


class TrajectoryTracker:
    def __init__(
        self,
        spacecraft,
        asteroid,
        Kp: float = 2.0,
        Ki: float = 0.1,
        Kd: float = 1.0,
        use_feedforward: bool = True,
    ):
        self.spacecraft = spacecraft
        self.asteroid = asteroid
        self.use_feedforward = use_feedforward
        # PID outputs desired acceleration; limit by max accel at initial mass.
        max_accel = spacecraft.T_max / max(spacecraft.m0, 1.0)
        self.pid = OptimizedPIDController(Kp=Kp, Ki=Ki, Kd=Kd, output_limit=max_accel)
        self.feedforward = FeedforwardController(spacecraft, asteroid) if use_feedforward else None
        self.pos_splines = None
        self.vel_splines = None
        self.accel_splines = None

    def set_reference_trajectory(
        self,
        t: np.ndarray,
        r: np.ndarray,
        v: np.ndarray = None,
        a: np.ndarray = None,
    ) -> None:
        self.pos_splines = [CubicSpline(t, r[:, i]) for i in range(3)]
        if v is not None:
            self.vel_splines = [CubicSpline(t, v[:, i]) for i in range(3)]
        else:
            self.vel_splines = [s.derivative() for s in self.pos_splines]

        if a is not None:
            self.accel_splines = [CubicSpline(t, a[:, i]) for i in range(3)]
        else:
            self.accel_splines = [s.derivative() for s in self.vel_splines]

    def get_reference_at_time(self, t: float) -> Dict:
        pos = np.array([s(t) for s in self.pos_splines])
        vel = np.array([s(t) for s in self.vel_splines])
        acc = np.array([s(t) for s in self.accel_splines])
        return {"position": pos, "velocity": vel, "acceleration": acc}

    def compute_control(
        self,
        current_time: float,
        current_position: np.ndarray,
        current_velocity: np.ndarray,
        current_mass: float,
        dt: float,
    ) -> Tuple[np.ndarray, Dict]:
        ref = self.get_reference_at_time(current_time)
        # PID outputs desired acceleration in inertial frame
        acc_cmd = self.pid.compute_control(
            ref["position"],
            ref["velocity"],
            current_position,
            current_velocity,
            dt,
            ref["acceleration"],
        )

        # Dynamics compensation at current state
        if hasattr(self.asteroid, "compute_gravity"):
            g = self.asteroid.compute_gravity(current_position)
        else:
            r_norm = np.linalg.norm(current_position)
            g = -self.asteroid.mu * current_position / (r_norm ** 3) if r_norm > 1e-10 else np.zeros(3)
        omega = getattr(self.asteroid, "omega", np.zeros(3))
        coriolis = -2 * np.cross(omega, current_velocity)
        centrifugal = -np.cross(omega, np.cross(omega, current_position))

        control = (acc_cmd - g - coriolis - centrifugal) * current_mass

        norm_u = np.linalg.norm(control)
        if norm_u > self.spacecraft.T_max:
            control = control / norm_u * self.spacecraft.T_max

        info = {
            "pos_error": float(np.linalg.norm(current_position - ref["position"])),
            "vel_error": float(np.linalg.norm(current_velocity - ref["velocity"])),
        }
        return control, info

    def reset(self) -> None:
        self.pid.reset()


class AdaptiveController:
    def __init__(self, spacecraft, asteroid, gamma: float = 0.1, lambda_gain: float = 1.0):
        self.sc = spacecraft
        self.ast = asteroid
        self.gamma = gamma
        self.lambda_gain = lambda_gain
        self.mu_hat = asteroid.mu
        self.mu_hat_dot = 0.0
        self.pid = OptimizedPIDController(Kp=2.0, Ki=0.1, Kd=1.0)

    def reset(self) -> None:
        self.pid.reset()
        self.mu_hat = self.ast.mu
        self.mu_hat_dot = 0.0
