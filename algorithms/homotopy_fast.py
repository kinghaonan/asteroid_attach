#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小行星附着最优轨迹设计 - 快速同伦法

利用bang-bang控制特性简化同伦优化：
1. 从bang-bang解出发
2. 渐进优化推力大小
3. 快速收敛策略
"""

import numpy as np
from scipy.optimize import minimize, least_squares
from scipy.integrate import solve_ivp
from typing import Dict, Tuple, List, Optional
import time
import warnings


class FastHomotopyOptimizer:
    """
    快速同伦法轨迹优化器
    
    简化策略：
    1. 使用bang-bang初始解
    2. 渐进细化推力曲线
    3. 两步优化（粗调+精调）
    """
    
    def __init__(
        self,
        asteroid,
        spacecraft,
        n_nodes: int = 30,
        n_homotopy_steps: int = 3,
        verbose: bool = True,
    ):
        self.ast = asteroid
        self.sc = spacecraft
        self.n_nodes = n_nodes
        self.n_homotopy_steps = n_homotopy_steps
        self.verbose = verbose
        
        self.n_states = 7
        self.n_controls = 3
    
    def compute_gravity(self, r: np.ndarray) -> np.ndarray:
        """计算引力加速度"""
        r_norm = np.linalg.norm(r)
        if r_norm < 1e-10:
            return np.zeros(3)
        return -self.ast.mu * r / (r_norm ** 3)
    
    def state_derivative(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        """状态导数"""
        r = state[0:3]
        v = state[3:6]
        m = max(state[6], 1.0)
        
        g = self.compute_gravity(r)
        omega = self.ast.omega
        coriolis = -2 * np.cross(omega, v)
        centrifugal = -np.cross(omega, np.cross(omega, r))
        
        thrust_mag = np.linalg.norm(control)
        thrust_accel = control / m
        
        r_dot = v
        v_dot = g + coriolis + centrifugal + thrust_accel
        m_dot = -thrust_mag / (self.sc.I_sp * self.sc.g0) if thrust_mag > 0 else 0.0
        
        return np.concatenate([r_dot, v_dot, [m_dot]])
    
    def forward_integrate(
        self,
        r0: np.ndarray,
        v0: np.ndarray,
        m0: float,
        U: np.ndarray,
        t_span: List[float],
    ) -> np.ndarray:
        """前向积分得到状态轨迹"""
        t0, tf = t_span
        dt = (tf - t0) / (self.n_nodes - 1)
        
        X = np.zeros((self.n_nodes, self.n_states))
        X[0, 0:3] = r0
        X[0, 3:6] = v0
        X[0, 6] = m0
        
        for k in range(self.n_nodes - 1):
            # RK4积分
            k1 = self.state_derivative(X[k], U[k])
            k2 = self.state_derivative(X[k] + 0.5*dt*k1, U[k])
            k3 = self.state_derivative(X[k] + 0.5*dt*k2, U[k])
            k4 = self.state_derivative(X[k] + dt*k3, U[k])
            
            X[k+1] = X[k] + dt/6*(k1 + 2*k2 + 2*k3 + k4)
            X[k+1, 6] = max(X[k+1, 6], 0.3 * m0)
        
        return X
    
    def generate_bangbang_control(
        self,
        r0: np.ndarray,
        v0: np.ndarray,
        m0: float,
        rf: np.ndarray,
        vf: np.ndarray,
        t_span: List[float],
    ) -> Tuple[np.ndarray, float, float]:
        """生成bang-bang控制序列"""
        t0, tf = t_span
        
        # 使用更合理的切换时间估计
        r0_norm = np.linalg.norm(r0)
        rf_norm = np.linalg.norm(rf)
        v0_norm = np.linalg.norm(v0)
        vf_norm = np.linalg.norm(vf)
        
        # 估计Delta-V需求（简化）
        # 主要用于：消除初始速度 + 建立目标速度
        delta_v = v0_norm + vf_norm + np.linalg.norm(rf - r0) / tf * 0.5
        
        # 推力加速度
        a_thrust = self.sc.T_max / m0
        t_burn = delta_v / a_thrust
        
        # 限制燃烧时间不超过总时间的60%
        t_burn = min(t_burn, tf * 0.6)
        
        # 切换时间：均匀分配
        t_switch1 = t_burn * 0.5
        t_switch2 = tf - t_burn * 0.5
        
        # 确保滑行段存在
        if t_switch2 <= t_switch1 + 10:
            t_switch1 = tf * 0.25
            t_switch2 = tf * 0.75
        
        U = np.zeros((self.n_nodes, self.n_controls))
        
        # 使用位置插值生成合理的控制
        for k in range(self.n_nodes):
            t = t0 + k * (tf - t0) / (self.n_nodes - 1)
            
            # 插值位置
            alpha = k / (self.n_nodes - 1)
            r_interp = r0 * (1 - alpha) + rf * alpha
            v_interp = v0 * (1 - alpha) + vf * alpha
            
            if t < t_switch1:
                # 第一阶段：减速
                if v0_norm > 0.1:
                    thrust_dir = -v0 / v0_norm
                else:
                    thrust_dir = (rf - r0) / np.linalg.norm(rf - r0)
                U[k] = thrust_dir * self.sc.T_max * 0.8
            elif t >= t_switch2:
                # 第三阶段：制动
                # 使用比例制导
                t_remain = tf - t
                if t_remain > 1.0:
                    v_desired = (rf - r_interp) / t_remain * 0.3
                    thrust_dir = v_desired - v_interp
                    thrust_norm = np.linalg.norm(thrust_dir)
                    if thrust_norm > 0.1:
                        thrust_dir = thrust_dir / thrust_norm
                    U[k] = thrust_dir * self.sc.T_max * 0.8
                else:
                    if vf_norm > 0.1:
                        U[k] = -vf / vf_norm * self.sc.T_max * 0.8
        
        return U, t_switch1, t_switch2
    
    def optimize_with_relaxed_constraints(
        self,
        U0: np.ndarray,
        r0: np.ndarray,
        v0: np.ndarray,
        m0: float,
        rf: np.ndarray,
        vf: np.ndarray,
        t_span: List[float],
        constraint_weight: float = 0.5,
    ) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """
        带松弛约束的优化
        
        constraint_weight: 约束权重（0=完全松弛，1=完全约束）
        """
        t0, tf = t_span
        dt = (tf - t0) / (self.n_nodes - 1)
        
        def residuals(u_flat):
            U = u_flat.reshape((self.n_nodes, self.n_controls))
            X = self.forward_integrate(r0, v0, m0, U, t_span)
            
            pos_err = X[-1, 0:3] - rf
            vel_err = X[-1, 3:6] - vf
            
            # 加权的终端约束 - 增加速度权重
            residuals = np.concatenate([
                pos_err * 20.0 * constraint_weight,
                vel_err * 100.0 * constraint_weight,  # 更高的速度权重
                U.flatten() * 0.0001,
            ])
            
            return residuals
        
        bounds = []
        for _ in range(self.n_nodes):
            for _ in range(3):
                bounds.append((-self.sc.T_max, self.sc.T_max))
        
        result = least_squares(
            residuals,
            U0.flatten(),
            bounds=(np.array([b[0] for b in bounds]), np.array([b[1] for b in bounds])),
            method='trf',
            ftol=1e-6,
            max_nfev=50,
            verbose=0,
        )
        
        U_opt = result.x.reshape((self.n_nodes, self.n_controls))
        X_opt = self.forward_integrate(r0, v0, m0, U_opt, t_span)
        
        pos_err = np.linalg.norm(X_opt[-1, 0:3] - rf)
        vel_err = np.linalg.norm(X_opt[-1, 3:6] - vf)
        
        return U_opt, X_opt, pos_err, vel_err
    
    def optimize(
        self,
        r0: np.ndarray,
        v0: np.ndarray,
        m0: float,
        rf: np.ndarray,
        vf: np.ndarray,
        t_span: List[float],
    ) -> Dict:
        """
        执行轨迹优化
        
        渐进式优化：
        1. 生成bang-bang初始解
        2. 同伦步骤：逐步收紧约束
        3. 最终精细优化
        """
        start_time = time.time()
        t0, tf = t_span
        
        if self.verbose:
            print("\n" + "="*60)
            print("快速同伦法轨迹优化")
            print("="*60)
            print(f"节点数: {self.n_nodes}")
            print(f"同伦步数: {self.n_homotopy_steps}")
            print(f"初始位置: {r0}")
            print(f"目标位置: {rf}")
        
        # 步骤1：生成bang-bang初始解
        U, t_s1, t_s2 = self.generate_bangbang_control(r0, v0, m0, rf, vf, t_span)
        X = self.forward_integrate(r0, v0, m0, U, t_span)
        
        pos_err = np.linalg.norm(X[-1, 0:3] - rf)
        vel_err = np.linalg.norm(X[-1, 3:6] - vf)
        
        if self.verbose:
            print(f"\nBang-bang初始解:")
            print(f"  切换时间: {t_s1:.1f}s, {t_s2:.1f}s")
            print(f"  位置误差: {pos_err:.2f}m")
            print(f"  速度误差: {vel_err:.2f}m/s")
        
        # 步骤2：同伦优化
        constraint_weights = np.linspace(0.3, 1.0, self.n_homotopy_steps)
        
        for i, weight in enumerate(constraint_weights):
            if self.verbose:
                print(f"\n同伦步骤 {i+1}/{self.n_homotopy_steps} (权重={weight:.2f})...")
            
            U, X, pos_err, vel_err = self.optimize_with_relaxed_constraints(
                U, r0, v0, m0, rf, vf, t_span, weight
            )
            
            if self.verbose:
                print(f"  位置误差: {pos_err:.4f}m, 速度误差: {vel_err:.4f}m/s")
        
        # 最终结果
        elapsed_time = time.time() - start_time
        fuel = m0 - X[-1, 6]
        success = pos_err < 10.0 and vel_err < 5.0
        
        if self.verbose:
            print("\n" + "="*60)
            print("优化结果")
            print("="*60)
            print(f"成功: {success}")
            print(f"位置误差: {pos_err:.4f} m")
            print(f"速度误差: {vel_err:.4f} m/s")
            print(f"燃料消耗: {fuel:.2f} kg")
            print(f"计算时间: {elapsed_time:.2f} s")
        
        return {
            'success': success,
            't': np.linspace(t0, tf, self.n_nodes),
            'X': X,
            'U': U,
            'r': X[:, 0:3],
            'v': X[:, 3:6],
            'm': X[:, 6],
            'u': np.linalg.norm(U, axis=1) / self.sc.T_max,
            'fuel_consumption': fuel,
            'pos_error': pos_err,
            'vel_error': vel_err,
            'elapsed_time': elapsed_time,
        }
