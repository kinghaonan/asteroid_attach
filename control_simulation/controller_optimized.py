#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小行星附着最优轨迹设计 - 优化版控制器

核心改进：
1. 前馈+反馈组合控制
2. 改进的自适应PID
3. 抗积分饱和
4. 轨迹跟踪优化
5. 推力约束处理
6. 扰动观测器

目标性能：
- 轨迹跟踪误差 < 3m（位置）
- 速度跟踪误差 < 1m/s
"""

import numpy as np
from typing import Dict, Tuple, List, Optional, Callable
from scipy.interpolate import CubicSpline


class OptimizedPIDController:
    """
    优化版PID控制器
    
    改进点：
    1. 抗积分饱和（条件积分）
    2. 前馈控制（基于参考轨迹）
    3. 自适应增益
    4. 推力限制和分配
    
    Attributes:
        Kp: 比例增益
        Ki: 积分增益
        Kd: 微分增益
    """
    
    def __init__(
        self,
        Kp: float = 2.0,
        Ki: float = 0.1,
        Kd: float = 1.0,
        integral_limit: float = 50.0,
        output_limit: float = 20.0,
        anti_windup: bool = True,
    ):
        """
        初始化优化版PID控制器
        
        Parameters:
            Kp: 比例增益
            Ki: 积分增益
            Kd: 微分增益
            integral_limit: 积分限幅
            output_limit: 输出限幅
            anti_windup: 是否启用抗积分饱和
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral_limit = integral_limit
        self.output_limit = output_limit
        self.anti_windup = anti_windup
        
        # 状态变量
        self.integral_error = np.zeros(3)
        self.prev_error = np.zeros(3)
        self.prev_time = None
        
        # 增益自适应参数
        self.adaptive_enabled = False
        self.Kp_base = Kp
        self.Ki_base = Ki
        self.Kd_base = Kd
    
    def reset(self):
        """重置控制器状态"""
        self.integral_error = np.zeros(3)
        self.prev_error = np.zeros(3)
        self.prev_time = None
    
    def enable_adaptive(self, enabled: bool = True):
        """启用/禁用自适应增益"""
        self.adaptive_enabled = enabled
    
    def _update_gains(self, error_norm: float):
        """
        自适应更新增益
        
        根据误差大小调整增益：
        - 大误差：增大P，减小I，增大D
        - 小误差：恢复基础增益
        """
        if not self.adaptive_enabled:
            return
        
        # 误差阈值
        error_threshold = 5.0
        
        if error_norm > error_threshold:
            # 大误差时调整
            factor = 1.0 + 0.01 * (error_norm - error_threshold)
            factor = min(factor, 3.0)  # 最大3倍
            
            self.Kp = self.Kp_base * factor
            self.Ki = self.Ki_base / factor
            self.Kd = self.Kd_base * factor
        else:
            # 小误差时恢复
            self.Kp = self.Kp_base
            self.Ki = self.Ki_base
            self.Kd = self.Kd_base
    
    def compute_control(
        self,
        reference_position: np.ndarray,
        reference_velocity: np.ndarray,
        current_position: np.ndarray,
        current_velocity: np.ndarray,
        dt: float,
        feedforward_accel: np.ndarray = None,
    ) -> np.ndarray:
        """
        计算控制指令
        
        Parameters:
            reference_position: 参考位置
            reference_velocity: 参考速度
            current_position: 当前位置
            current_velocity: 当前速度
            dt: 时间步长
            feedforward_accel: 前馈加速度（可选）
            
        Returns:
            控制力
        """
        # 位置误差
        position_error = reference_position - current_position
        error_norm = np.linalg.norm(position_error)
        
        # 自适应增益
        self._update_gains(error_norm)
        
        # 速度误差
        velocity_error = reference_velocity - current_velocity
        
        # 积分误差更新
        self.integral_error += position_error * dt
        
        # 抗积分饱和
        if self.anti_windup:
            integral_norm = np.linalg.norm(self.integral_error)
            if integral_norm > self.integral_limit:
                self.integral_error = self.integral_error / integral_norm * self.integral_limit
        
        # 微分项（使用速度误差，避免位置噪声放大）
        derivative_error = velocity_error
        
        # PID控制
        control = (
            self.Kp * position_error +
            self.Ki * self.integral_error +
            self.Kd * derivative_error
        )
        
        # 添加前馈项
        if feedforward_accel is not None:
            control += feedforward_accel
        
        # 输出限幅
        control_norm = np.linalg.norm(control)
        if control_norm > self.output_limit:
            control = control / control_norm * self.output_limit
        
        # 更新状态
        self.prev_error = position_error.copy()
        
        return control
    
    def track_trajectory(
        self,
        reference_trajectory: Dict,
        current_state: Dict,
        dt: float,
    ) -> np.ndarray:
        """
        跟踪参考轨迹
        
        Parameters:
            reference_trajectory: 参考轨迹
            current_state: 当前状态
            dt: 时间步长
            
        Returns:
            控制力
        """
        ref_pos = reference_trajectory.get('position', np.zeros(3))
        ref_vel = reference_trajectory.get('velocity', np.zeros(3))
        ref_accel = reference_trajectory.get('acceleration', None)
        
        curr_pos = current_state.get('position', np.zeros(3))
        curr_vel = current_state.get('velocity', np.zeros(3))
        
        return self.compute_control(
            ref_pos, ref_vel, curr_pos, curr_vel, dt, ref_accel
        )


class FeedforwardController:
    """
    前馈控制器
    
    基于参考轨迹计算期望控制输入
    """
    
    def __init__(self, spacecraft, asteroid):
        """
        初始化前馈控制器
        
        Parameters:
            spacecraft: 航天器对象
            asteroid: 小行星对象
        """
        self.sc = spacecraft
        self.ast = asteroid
    
    def compute_feedforward(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        mass: float,
        reference_accel: np.ndarray,
    ) -> np.ndarray:
        """
        计算前馈控制
        
        Parameters:
            position: 当前位置
            velocity: 当前速度
            mass: 当前质量
            reference_accel: 参考加速度
            
        Returns:
            前馈控制力
        """
        # 引力
        r_norm = np.linalg.norm(position)
        if r_norm > 1e-10:
            g = -self.ast.mu * position / (r_norm ** 3)
        else:
            g = np.zeros(3)
        
        # 科里奥利力和离心力
        omega = self.ast.omega
        coriolis = -2 * np.cross(omega, velocity)
        centrifugal = -np.cross(omega, np.cross(omega, position))
        
        # 所需推力加速度
        a_thrust = reference_accel - g - coriolis - centrifugal
        
        # 转换为推力
        feedforward = a_thrust * mass
        
        # 推力限制
        thrust_norm = np.linalg.norm(feedforward)
        if thrust_norm > self.sc.T_max:
            feedforward = feedforward / thrust_norm * self.sc.T_max
        
        return feedforward


class TrajectoryTracker:
    """
    轨迹跟踪控制器
    
    组合前馈和反馈控制实现高精度轨迹跟踪
    """
    
    def __init__(
        self,
        spacecraft,
        asteroid,
        Kp: float = 2.0,
        Ki: float = 0.1,
        Kd: float = 1.0,
        use_feedforward: bool = True,
    ):
        """
        初始化轨迹跟踪控制器
        
        Parameters:
            spacecraft: 航天器对象
            asteroid: 小行星对象
            Kp, Ki, Kd: PID增益
            use_feedforward: 是否使用前馈
        """
        self.spacecraft = spacecraft
        self.asteroid = asteroid
        self.use_feedforward = use_feedforward
        
        # PID控制器
        self.pid = OptimizedPIDController(
            Kp=Kp, Ki=Ki, Kd=Kd,
            output_limit=spacecraft.T_max
        )
        
        # 前馈控制器
        if use_feedforward:
            self.feedforward = FeedforwardController(spacecraft, asteroid)
        else:
            self.feedforward = None
        
        # 参考轨迹插值器
        self.trajectory_spline = None
        self.time_nodes = None
    
    def set_reference_trajectory(
        self,
        t: np.ndarray,
        r: np.ndarray,
        v: np.ndarray = None,
    ):
        """
        设置参考轨迹
        
        Parameters:
            t: 时间节点
            r: 位置轨迹 (n, 3)
            v: 速度轨迹 (n, 3)，可选
        """
        self.time_nodes = t
        
        # 位置样条插值
        self.pos_splines = [CubicSpline(t, r[:, i]) for i in range(3)]
        
        # 速度样条插值（如果没有提供，则从位置导出）
        if v is not None:
            self.vel_splines = [CubicSpline(t, v[:, i]) for i in range(3)]
        else:
            self.vel_splines = [spline.derivative() for spline in self.pos_splines]
        
        # 加速度样条插值
        self.accel_splines = [spline.derivative() for spline in self.vel_splines]
    
    def get_reference_at_time(self, t: float) -> Dict:
        """
        获取指定时刻的参考状态
        
        Parameters:
            t: 时间
            
        Returns:
            参考状态字典
        """
        position = np.array([spline(t) for spline in self.pos_splines])
        velocity = np.array([spline(t) for spline in self.vel_splines])
        acceleration = np.array([spline(t) for spline in self.accel_splines])
        
        return {
            'position': position,
            'velocity': velocity,
            'acceleration': acceleration
        }
    
    def compute_control(
        self,
        current_time: float,
        current_position: np.ndarray,
        current_velocity: np.ndarray,
        current_mass: float,
        dt: float,
    ) -> Tuple[np.ndarray, Dict]:
        """
        计算控制指令
        
        Parameters:
            current_time: 当前时间
            current_position: 当前位置
            current_velocity: 当前速度
            current_mass: 当前质量
            dt: 时间步长
            
        Returns:
            control: 控制力
            info: 控制信息
        """
        # 获取参考状态
        ref = self.get_reference_at_time(current_time)
        
        # 计算前馈控制
        if self.use_feedforward and self.feedforward is not None:
            feedforward = self.feedforward.compute_feedforward(
                current_position,
                current_velocity,
                current_mass,
                ref['acceleration']
            )
        else:
            feedforward = None
        
        # 计算反馈控制
        feedback = self.pid.compute_control(
            ref['position'],
            ref['velocity'],
            current_position,
            current_velocity,
            dt,
            feedforward / current_mass if feedforward is not None else None
        )
        
        # 组合控制
        if feedforward is not None:
            control = feedback + feedforward
        else:
            control = feedback
        
        # 推力限制
        thrust_norm = np.linalg.norm(control)
        if thrust_norm > self.spacecraft.T_max:
            control = control / thrust_norm * self.spacecraft.T_max
        
        # 计算跟踪误差
        pos_error = np.linalg.norm(current_position - ref['position'])
        vel_error = np.linalg.norm(current_velocity - ref['velocity'])
        
        info = {
            'pos_error': pos_error,
            'vel_error': vel_error,
            'ref_position': ref['position'],
            'ref_velocity': ref['velocity'],
            'thrust_magnitude': thrust_norm
        }
        
        return control, info
    
    def reset(self):
        """重置控制器"""
        self.pid.reset()


class AdaptiveController:
    """
    自适应控制器
    
    基于Lyapunov方法的自适应控制
    """
    
    def __init__(
        self,
        spacecraft,
        asteroid,
        gamma: float = 0.1,
        lambda_gain: float = 1.0,
    ):
        """
        初始化自适应控制器
        
        Parameters:
            spacecraft: 航天器对象
            asteroid: 小行星对象
            gamma: 自适应增益
            lambda_gain: 滑模增益
        """
        self.sc = spacecraft
        self.ast = asteroid
        self.gamma = gamma
        self.lambda_gain = lambda_gain
        
        # 参数估计（引力参数不确定性）
        self.mu_hat = asteroid.mu
        self.mu_hat_dot = 0.0
        
        # PID控制器作为基础
        self.pid = OptimizedPIDController(Kp=2.0, Ki=0.1, Kd=1.0)
    
    def update_parameter_estimate(self, estimation_error: float, dt: float):
        """
        更新参数估计
        
        Parameters:
            estimation_error: 估计误差
            dt: 时间步长
        """
        # 自适应律
        self.mu_hat_dot = self.gamma * estimation_error
        self.mu_hat += self.mu_hat_dot * dt
    
    def compute_control(
        self,
        current_state: np.ndarray,
        reference_state: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """
        计算自适应控制
        
        Parameters:
            current_state: 当前状态 [r, v, m]
            reference_state: 参考状态 [r, v, m]
            dt: 时间步长
            
        Returns:
            控制力
        """
        r = current_state[0:3]
        v = current_state[3:6]
        m = current_state[6]
        
        r_ref = reference_state[0:3]
        v_ref = reference_state[3:6]
        
        # 误差
        e_r = r_ref - r
        e_v = v_ref - v
        
        # 滑模变量
        s = e_v + self.lambda_gain * e_r
        
        # 使用估计参数计算引力
        r_norm = np.linalg.norm(r)
        if r_norm > 1e-10:
            g_hat = -self.mu_hat * r / (r_norm ** 3)
        else:
            g_hat = np.zeros(3)
        
        # 参考加速度
        ref_accel = np.zeros(3)  # 简化处理
        
        # 控制律
        # u = m * (a_ref + g_hat + Ks * sign(s) + Kp * e_r + Kd * e_v)
        Ks = 0.5  # 滑模增益
        
        control = m * (
            ref_accel +
            g_hat +
            Ks * np.sign(s) +
            self.pid.Kp * e_r +
            self.pid.Kd * e_v
        )
        
        # 推力限制
        thrust_norm = np.linalg.norm(control)
        if thrust_norm > self.sc.T_max:
            control = control / thrust_norm * self.sc.T_max
        
        return control
    
    def reset(self):
        """重置控制器"""
        self.pid.reset()
        self.mu_hat = self.ast.mu
        self.mu_hat_dot = 0.0


# 测试
if __name__ == "__main__":
    class ExampleAsteroid:
        def __init__(self):
            self.omega = np.array([0, 0, 2 * np.pi / (5.27 * 3600)])
            self.mu = 446300.0
    
    class ExampleSpacecraft:
        def __init__(self):
            self.T_max = 20.0
            self.I_sp = 400.0
            self.g0 = 9.81
            self.m0 = 1000.0
    
    asteroid = ExampleAsteroid()
    spacecraft = ExampleSpacecraft()
    
    # 创建控制器
    pid = OptimizedPIDController(Kp=2.0, Ki=0.1, Kd=1.0)
    pid.enable_adaptive(True)
    
    tracker = TrajectoryTracker(spacecraft, asteroid)
    
    # 设置参考轨迹
    t = np.linspace(0, 770, 100)
    r = np.zeros((100, 3))
    for i in range(3):
        r[:, i] = np.linspace(10177, 676, 100) if i == 0 else np.linspace(6956, 5121, 100)
    
    tracker.set_reference_trajectory(t, r)
    
    print("优化版控制器测试完成")
