#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小行星附着最优轨迹设计 - 快速伪谱法

完全采用同伦法的成功策略
"""

import numpy as np
from scipy.optimize import least_squares
from typing import Dict, Tuple, List
import time


class FastPseudospectralOptimizer:
    """快速伪谱法 - 完全复制同伦法策略"""
    
    def __init__(self, asteroid, spacecraft, n_nodes: int = 25, verbose: bool = True):
        self.ast = asteroid
        self.sc = spacecraft
        self.n_nodes = n_nodes
        self.verbose = verbose
        self.n_states = 7
        self.n_controls = 3
    
    def compute_gravity(self, r: np.ndarray) -> np.ndarray:
        r_norm = np.linalg.norm(r)
        if r_norm < 1e-10:
            return np.zeros(3)
        return -self.ast.mu * r / (r_norm ** 3)
    
    def state_derivative(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        r, v, m = state[0:3], state[3:6], max(state[6], 1.0)
        
        g = self.compute_gravity(r)
        omega = self.ast.omega
        coriolis = -2 * np.cross(omega, v)
        centrifugal = -np.cross(omega, np.cross(omega, r))
        
        thrust_mag = np.linalg.norm(control)
        thrust_accel = control / m
        
        return np.concatenate([
            v, g + coriolis + centrifugal + thrust_accel,
            [-thrust_mag / (self.sc.I_sp * self.sc.g0) if thrust_mag > 0 else 0.0]
        ])
    
    def forward_integrate(self, r0, v0, m0, U, t_span):
        t0, tf = t_span
        dt = (tf - t0) / (self.n_nodes - 1)
        X = np.zeros((self.n_nodes, self.n_states))
        X[0] = np.concatenate([r0, v0, [m0]])
        
        for k in range(self.n_nodes - 1):
            k1 = self.state_derivative(X[k], U[k])
            k2 = self.state_derivative(X[k] + 0.5*dt*k1, U[k])
            k3 = self.state_derivative(X[k] + 0.5*dt*k2, U[k])
            k4 = self.state_derivative(X[k] + dt*k3, U[k])
            X[k+1] = X[k] + dt/6*(k1 + 2*k2 + 2*k3 + k4)
            X[k+1, 6] = max(X[k+1, 6], 0.3 * m0)
        return X
    
    def generate_bangbang_control(self, r0, v0, m0, rf, vf, t_span) -> Tuple[np.ndarray, float, float]:
        """生成bang-bang控制序列 - 与同伦法相同"""
        t0, tf = t_span
        
        v0_norm = np.linalg.norm(v0)
        delta_v = v0_norm + np.linalg.norm(vf) + np.linalg.norm(rf - r0) / tf * 0.5
        
        a_thrust = self.sc.T_max / m0
        t_burn = min(delta_v / a_thrust, tf * 0.6)
        
        t_switch1 = t_burn * 0.5
        t_switch2 = tf - t_burn * 0.5
        
        if t_switch2 <= t_switch1 + 10:
            t_switch1 = tf * 0.25
            t_switch2 = tf * 0.75
        
        U = np.zeros((self.n_nodes, self.n_controls))
        
        for k in range(self.n_nodes):
            t = t0 + k * (tf - t0) / (self.n_nodes - 1)
            alpha = k / (self.n_nodes - 1)
            r_interp = r0 * (1 - alpha) + rf * alpha
            v_interp = v0 * (1 - alpha) + vf * alpha
            
            if t < t_switch1:
                if v0_norm > 0.1:
                    thrust_dir = -v0 / v0_norm
                else:
                    thrust_dir = (rf - r0) / np.linalg.norm(rf - r0)
                U[k] = thrust_dir * self.sc.T_max * 0.8
            elif t >= t_switch2:
                t_remain = tf - t
                if t_remain > 1.0:
                    v_desired = (rf - r_interp) / t_remain * 0.3
                    thrust_dir = v_desired - v_interp
                    thrust_norm = np.linalg.norm(thrust_dir)
                    if thrust_norm > 0.1:
                        thrust_dir = thrust_dir / thrust_norm
                    U[k] = thrust_dir * self.sc.T_max * 0.8
        
        return U, t_switch1, t_switch2
    
    def optimize_with_relaxed_constraints(self, U0, r0, v0, m0, rf, vf, t_span, constraint_weight):
        """与同伦法完全相同的优化函数"""
        t0, tf = t_span
        
        def residuals(u_flat):
            U = u_flat.reshape((self.n_nodes, self.n_controls))
            X = self.forward_integrate(r0, v0, m0, U, t_span)
            
            pos_err = X[-1, 0:3] - rf
            vel_err = X[-1, 3:6] - vf
            
            return np.concatenate([
                pos_err * 20.0 * constraint_weight,
                vel_err * 100.0 * constraint_weight,
                U.flatten() * 0.0001,
            ])
        
        lb = np.array([-self.sc.T_max] * (self.n_nodes * 3))
        ub = np.array([self.sc.T_max] * (self.n_nodes * 3))
        
        result = least_squares(residuals, U0.flatten(), bounds=(lb, ub),
                               method='trf', ftol=1e-6, max_nfev=50, verbose=0)
        
        U_opt = result.x.reshape((self.n_nodes, self.n_controls))
        X_opt = self.forward_integrate(r0, v0, m0, U_opt, t_span)
        pos_err = np.linalg.norm(X_opt[-1, 0:3] - rf)
        vel_err = np.linalg.norm(X_opt[-1, 3:6] - vf)
        
        return U_opt, X_opt, pos_err, vel_err
    
    def optimize(self, r0, v0, m0, rf, vf, t_span) -> Dict:
        """执行优化 - 与同伦法相同的流程"""
        start_time = time.time()
        t0, tf = t_span
        
        if self.verbose:
            print("\n" + "="*60)
            print("快速伪谱法轨迹优化")
            print("="*60)
            print(f"节点数: {self.n_nodes}")
            print(f"初始位置: {r0}")
            print(f"目标位置: {rf}")
        
        # 生成bang-bang初始控制
        U, t_s1, t_s2 = self.generate_bangbang_control(r0, v0, m0, rf, vf, t_span)
        
        X = self.forward_integrate(r0, v0, m0, U, t_span)
        pos_err = np.linalg.norm(X[-1, 0:3] - rf)
        vel_err = np.linalg.norm(X[-1, 3:6] - vf)
        
        if self.verbose:
            print(f"\nBang-bang初始解:")
            print(f"  切换时间: {t_s1:.1f}s, {t_s2:.1f}s")
            print(f"  位置误差: {pos_err:.2f}m")
            print(f"  速度误差: {vel_err:.2f}m/s")
        
        # 同伦步骤
        homotopy_weights = [0.3, 0.65, 1.0]
        
        for i, w in enumerate(homotopy_weights):
            if self.verbose:
                print(f"\n同伦步骤 {i+1}/{len(homotopy_weights)} (权重={w:.2f})...")
            
            U, X, pos_err, vel_err = self.optimize_with_relaxed_constraints(
                U, r0, v0, m0, rf, vf, t_span, w
            )
            
            if self.verbose:
                print(f"  位置误差: {pos_err:.4f}m, 速度误差: {vel_err:.4f}m/s")
        
        fuel = m0 - X[-1, 6]
        elapsed = time.time() - start_time
        success = pos_err < 10.0 and vel_err < 5.0
        
        if self.verbose:
            print("\n" + "="*60)
            print("优化结果")
            print("="*60)
            print(f"成功: {success}")
            print(f"位置误差: {pos_err:.4f} m")
            print(f"速度误差: {vel_err:.4f} m/s")
            print(f"燃料消耗: {fuel:.2f} kg")
            print(f"计算时间: {elapsed:.2f} s")
        
        return {
            'success': success, 't': np.linspace(t0, tf, self.n_nodes),
            'X': X, 'U': U, 'r': X[:, 0:3], 'v': X[:, 3:6], 'm': X[:, 6],
            'u': np.linalg.norm(U, axis=1) / self.sc.T_max,
            'fuel_consumption': fuel, 'pos_error': pos_err, 'vel_error': vel_err,
            'elapsed_time': elapsed,
        }