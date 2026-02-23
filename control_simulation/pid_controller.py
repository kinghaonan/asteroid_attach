#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PID控制器实现

用于轨迹跟踪控制。
"""

import numpy as np


class PIDController:
    """
    PID控制器

    用于航天器轨迹跟踪控制。
    """

    def __init__(self, Kp=1.0, Ki=0.1, Kd=0.5):
        """
        初始化PID控制器

        Parameters:
            Kp: 比例增益
            Ki: 积分增益
            Kd: 微分增益
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

        # 误差积分
        self.integral_error = np.zeros(3)
        # 上一次的误差
        self.prev_error = np.zeros(3)
        # 积分限幅
        self.integral_limit = 100.0

    def reset(self):
        """重置控制器状态"""
        self.integral_error = np.zeros(3)
        self.prev_error = np.zeros(3)

    def compute_control(
        self,
        reference_position,
        reference_velocity,
        current_position,
        current_velocity,
        dt,
    ):
        """
        计算控制指令

        Parameters:
            reference_position: 参考位置 (3,)
            reference_velocity: 参考速度 (3,)
            current_position: 当前位置 (3,)
            current_velocity: 当前速度 (3,)
            dt: 时间步长

        Returns:
            control_force: 控制力 (3,)
        """
        # 位置误差
        position_error = reference_position - current_position

        # 速度误差
        velocity_error = reference_velocity - current_velocity

        # 积分误差
        self.integral_error += position_error * dt

        # 积分限幅
        self.integral_error = np.clip(
            self.integral_error, -self.integral_limit, self.integral_limit
        )

        # 微分误差（使用速度误差）
        derivative_error = velocity_error

        # PID控制律
        control_force = (
            self.Kp * position_error
            + self.Ki * self.integral_error
            + self.Kd * derivative_error
        )

        # 更新上一次误差
        self.prev_error = position_error.copy()

        return control_force

    def track_trajectory(self, reference_trajectory, current_state, dt):
        """
        跟踪参考轨迹

        Parameters:
            reference_trajectory: 参考轨迹字典，包含 'position' 和 'velocity'
            current_state: 当前状态字典，包含 'position' 和 'velocity'
            dt: 时间步长

        Returns:
            control_force: 控制力 (3,)
        """
        ref_pos = reference_trajectory["position"]
        ref_vel = reference_trajectory["velocity"]
        curr_pos = current_state["position"]
        curr_vel = current_state["velocity"]

        return self.compute_control(ref_pos, ref_vel, curr_pos, curr_vel, dt)


class AdaptivePIDController(PIDController):
    """
    自适应PID控制器

    根据跟踪误差自适应调整增益。
    """

    def __init__(
        self, Kp=1.0, Ki=0.1, Kd=0.5, adaptation_rate=0.01, error_threshold=1.0
    ):
        """
        初始化自适应PID控制器

        Parameters:
            Kp: 初始比例增益
            Ki: 初始积分增益
            Kd: 初始微分增益
            adaptation_rate: 自适应调整速率
            error_threshold: 误差阈值，超过此值开始调整
        """
        super().__init__(Kp, Ki, Kd)
        self.adaptation_rate = adaptation_rate
        self.error_threshold = error_threshold
        self.base_Kp = Kp
        self.base_Ki = Ki
        self.base_Kd = Kd

    def compute_control(
        self,
        reference_position,
        reference_velocity,
        current_position,
        current_velocity,
        dt,
    ):
        """
        计算控制指令（带自适应调整）
        """
        # 计算误差
        position_error = reference_position - current_position
        error_norm = np.linalg.norm(position_error)

        # 自适应调整增益
        if error_norm > self.error_threshold:
            # 误差大时增大比例增益，减小积分增益
            factor = 1.0 + self.adaptation_rate * (error_norm - self.error_threshold)
            self.Kp = self.base_Kp * factor
            self.Ki = self.base_Ki / factor
        else:
            # 误差小时恢复基础增益
            self.Kp = self.base_Kp
            self.Ki = self.base_Ki

        # 调用父类方法
        return super().compute_control(
            reference_position,
            reference_velocity,
            current_position,
            current_velocity,
            dt,
        )
