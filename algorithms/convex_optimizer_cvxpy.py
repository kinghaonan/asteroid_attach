#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小行星附着最优轨迹设计 - CVXPY版序贯凸规划(SCP)

使用CVXPY库求解凸子问题，提高收敛性和稳定性

参考:
- OpenSCvx: https://github.com/OpenSCvx/OpenSCvx
- "Successive Convexification for 6-DoF Mars Rocket Powered Landing"
"""

import numpy as np
import cvxpy as cp
from typing import Dict, Tuple, List, Optional
import time
import warnings


class CVXPYSCPOptimizer:
    """
    使用CVXPY的序贯凸规划(SCP)轨迹优化器
    
    特点:
    1. 使用CVXPY求解凸子问题
    2. 自适应信任域
    3. 线性化动力学约束
    4. 虚拟控制法处理非凸约束
    
    Attributes:
        asteroid: 小行星模型
        spacecraft: 航天器参数
        n_nodes: 节点数
        verbose: 是否打印详细信息
    """
    
    def __init__(
        self,
        asteroid,
        spacecraft,
        n_nodes: int = 30,
        max_iterations: int = 50,
        verbose: bool = True,
    ):
        self.ast = asteroid
        self.sc = spacecraft
        self.n_nodes = n_nodes
        self.max_iterations = max_iterations
        self.verbose = verbose
        
        # 状态和控制维度
        self.n_states = 7  # [r(3), v(3), m]
        self.n_controls = 3  # [u(3)]
        
        # SCP参数
        self.w_tr = 1.0  # 信任域权重
        self.w_vc = 1e4  # 虚拟控制惩罚权重
        self.tr_radius = 100.0  # 初始信任域半径
        self.tr_radius_min = 0.1
        self.tr_radius_max = 1000.0
        
        # 收敛参数
        self.tol = 1e-4
        self.target_pos_error = 10.0  # 目标位置误差
        self.target_vel_error = 5.0   # 目标速度误差
        
    def compute_gravity(self, r: np.ndarray) -> np.ndarray:
        """计算引力加速度"""
        r_norm = np.linalg.norm(r)
        if r_norm < 1e-10:
            return np.zeros(3)
        return -self.ast.mu * r / (r_norm ** 3)
    
    def compute_gravity_gradient(self, r: np.ndarray) -> np.ndarray:
        """计算引力梯度矩阵"""
        r_norm = np.linalg.norm(r)
        if r_norm < 1e-10:
            return np.zeros((3, 3))
        
        I = np.eye(3)
        return self.ast.mu * (3 * np.outer(r, r) / (r_norm ** 5) - I / (r_norm ** 3))
    
    def state_dynamics(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        """
        状态动力学方程
        
        Parameters:
            state: [r(3), v(3), m]
            control: [u(3)]
            
        Returns:
            状态导数
        """
        r = state[0:3]
        v = state[3:6]
        m = state[6]
        m = max(m, 1.0)
        
        # 引力
        g = self.compute_gravity(r)
        
        # 科里奥利力和离心力
        omega = self.ast.omega
        coriolis = -2 * np.cross(omega, v)
        centrifugal = -np.cross(omega, np.cross(omega, r))
        
        # 推力加速度
        thrust_acc = control / m
        
        # 状态导数
        r_dot = v
        v_dot = g + coriolis + centrifugal + thrust_acc
        m_dot = -np.linalg.norm(control) / (self.sc.I_sp * self.sc.g0)
        
        return np.concatenate([r_dot, v_dot, [m_dot]])
    
    def linearize_dynamics(
        self,
        state: np.ndarray,
        control: np.ndarray,
        dt: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        线性化动力学方程
        
        Parameters:
            state: 状态
            control: 控制
            dt: 时间步长
            
        Returns:
            A_d: 离散状态矩阵
            B_d: 离散控制矩阵
            c_d: 离散常数项
        """
        r = state[0:3]
        v = state[3:6]
        m = max(state[6], 1.0)
        u = control
        u_norm = np.linalg.norm(u)
        
        # 连续时间状态矩阵 A
        A = np.zeros((self.n_states, self.n_states))
        
        # ∂r_dot/∂r = 0, ∂r_dot/∂v = I
        A[0:3, 3:6] = np.eye(3)
        
        # ∂v_dot/∂r = 引力梯度 + 离心力雅可比
        dg_dr = self.compute_gravity_gradient(r)
        omega = self.ast.omega
        centrifugal_jac = -np.cross(omega, np.cross(omega, np.eye(3)))
        A[3:6, 0:3] = dg_dr + centrifugal_jac
        
        # ∂v_dot/∂v = 科里奥利雅可比
        coriolis_jac = -2 * np.array([
            [0, -omega[2], omega[1]],
            [omega[2], 0, -omega[0]],
            [-omega[1], omega[0], 0]
        ])
        A[3:6, 3:6] = coriolis_jac
        
        # ∂v_dot/∂m = -u/m^2
        A[3:6, 6] = -u / (m ** 2)
        
        # ∂m_dot/∂m = 0
        # ∂m_dot/∂u = -u_norm / (I_sp * g0) * u / u_norm / u_norm = -u / (I_sp * g0 * u_norm)
        
        # 连续时间控制矩阵 B
        B = np.zeros((self.n_states, self.n_controls))
        B[3:6, :] = np.eye(3) / m  # ∂v_dot/∂u
        if u_norm > 1e-10:
            B[6, :] = -u / (self.sc.I_sp * self.sc.g0 * u_norm)
        
        # 常数项
        f = self.state_dynamics(state, control)
        c = f - A @ state - B @ control
        
        # 离散化 (零阶保持)
        A_d = np.eye(self.n_states) + A * dt
        B_d = B * dt
        c_d = c * dt
        
        return A_d, B_d, c_d
    
    def generate_initial_trajectory(
        self,
        r0: np.ndarray,
        v0: np.ndarray,
        m0: float,
        rf: np.ndarray,
        vf: np.ndarray,
        t_span: List[float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """生成初始轨迹猜测"""
        t0, tf = t_span
        tau = np.linspace(0, 1, self.n_nodes)
        dt = (tf - t0) / (self.n_nodes - 1)
        
        X0 = np.zeros((self.n_nodes, self.n_states))
        U0 = np.zeros((self.n_nodes, self.n_controls))
        
        # 位置和速度：五次多项式
        for i in range(3):
            # 五次多项式系数
            a0 = r0[i]
            a1 = v0[i] * (tf - t0)
            a2 = 0
            a3 = 10 * (rf[i] - r0[i]) - 6 * v0[i] * (tf - t0) - 4 * vf[i] * (tf - t0)
            a4 = -15 * (rf[i] - r0[i]) + 8 * v0[i] * (tf - t0) + 7 * vf[i] * (tf - t0)
            a5 = 6 * (rf[i] - r0[i]) - 3 * (v0[i] + vf[i]) * (tf - t0)
            coeffs = np.array([a5, a4, a3, a2, a1, a0])
            
            X0[:, i] = np.polyval(coeffs, tau)
            d_coeffs = np.polyder(coeffs)
            X0[:, i + 3] = np.polyval(d_coeffs, tau) / (tf - t0)
        
        # 质量：线性递减
        X0[:, 6] = np.linspace(m0, m0 * 0.85, self.n_nodes)
        
        # 控制：根据动力学反算
        for k in range(self.n_nodes):
            a_desired = np.zeros(3)
            if k < self.n_nodes - 1:
                a_desired = (X0[k + 1, 3:6] - X0[k, 3:6]) / dt
            
            g = self.compute_gravity(X0[k, 0:3])
            omega = self.ast.omega
            coriolis = -2 * np.cross(omega, X0[k, 3:6])
            centrifugal = -np.cross(omega, np.cross(omega, X0[k, 0:3]))
            
            a_thrust = a_desired - g - coriolis - centrifugal
            U0[k] = a_thrust * X0[k, 6]
            # 限制推力范围
            u_norm = np.linalg.norm(U0[k])
            if u_norm > self.sc.T_max:
                U0[k] = U0[k] / u_norm * self.sc.T_max * 0.8
        
        return X0, U0
    
    def solve_convex_subproblem_cvxpy(
        self,
        X_ref: np.ndarray,
        U_ref: np.ndarray,
        r0: np.ndarray,
        v0: np.ndarray,
        m0: float,
        rf: np.ndarray,
        vf: np.ndarray,
        t_span: List[float],
    ) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """
        使用CVXPY求解凸子问题
        
        Parameters:
            X_ref: 参考状态轨迹
            U_ref: 参考控制轨迹
            r0, v0, m0: 初始状态
            rf, vf: 终端状态
            t_span: 时间区间
            
        Returns:
            X_opt: 最优状态
            U_opt: 最优控制
            cost: 目标函数值
            constraint_violation: 约束违反量
        """
        t0, tf = t_span
        dt = (tf - t0) / (self.n_nodes - 1)
        
        # 预计算线性化矩阵
        A_list = []
        B_list = []
        c_list = []
        
        for k in range(self.n_nodes - 1):
            A_k, B_k, c_k = self.linearize_dynamics(X_ref[k], U_ref[k], dt)
            A_list.append(A_k)
            B_list.append(B_k)
            c_list.append(c_k)
        
        # CVXPY变量
        X = cp.Variable((self.n_nodes, self.n_states))
        U = cp.Variable((self.n_nodes, self.n_controls))
        nu = cp.Variable((self.n_nodes - 1, self.n_states))  # 虚拟控制（与状态维度一致）
        
        # 目标函数
        cost_terms = []
        
        # 1. 燃料消耗
        cost_terms.append(m0 - X[-1, 6])
        
        # 2. 终端位置误差惩罚
        cost_terms.append(1000.0 * cp.sum_squares(X[-1, 0:3] - rf))
        
        # 3. 终端速度误差惩罚 - 增大权重
        cost_terms.append(10000.0 * cp.sum_squares(X[-1, 3:6] - vf))
        
        # 4. 虚拟控制惩罚
        for k in range(self.n_nodes - 1):
            cost_terms.append(self.w_vc * cp.sum_squares(nu[k]))
        
        # 5. 信任域惩罚
        for k in range(self.n_nodes):
            cost_terms.append(self.w_tr * cp.sum_squares(X[k] - X_ref[k]))
            cost_terms.append(self.w_tr * cp.sum_squares(U[k] - U_ref[k]))
        
        objective = cp.Minimize(cp.sum(cost_terms))
        
        # 约束
        constraints = []
        
        # 初始条件
        constraints.append(X[0, 0:3] == r0)
        constraints.append(X[0, 3:6] == v0)
        constraints.append(X[0, 6] == m0)
        
        # 终端条件 (软约束)
        # 使用较大的权重而非硬约束
        terminal_weight = 100.0
        
        # 动力学约束 (线性化)
        for k in range(self.n_nodes - 1):
            constraints.append(
                X[k + 1] == A_list[k] @ X[k] + B_list[k] @ U[k] + c_list[k] + nu[k]
            )
        
        # 边界约束
        for k in range(self.n_nodes):
            # 质量边界
            constraints.append(X[k, 6] >= 0.5 * m0)  # 最少保留一半质量
            constraints.append(X[k, 6] <= m0)
            
            # 推力边界 (使用二阶锥约束)
            constraints.append(cp.norm(U[k], 2) <= self.sc.T_max)
        
        # 终端约束（软约束，通过目标函数惩罚）
        # 但我们还是添加一个相对宽松的硬约束
        constraints.append(cp.norm(X[-1, 0:3] - rf, 2) <= 500.0)  # 终端位置在500m以内
        constraints.append(cp.norm(X[-1, 3:6] - vf, 2) <= 50.0)   # 终端速度在50m/s以内
        
        # 求解
        problem = cp.Problem(objective, constraints)
        
        try:
            # 使用ECOS或SCS求解器
            problem.solve(solver=cp.ECOS, verbose=False)
            
            if problem.status not in ['optimal', 'optimal_inaccurate']:
                # 尝试SCS求解器
                problem.solve(solver=cp.SCS, verbose=False, max_iters=5000)
            
            if problem.status in ['optimal', 'optimal_inaccurate']:
                X_opt = X.value
                U_opt = U.value
                cost_val = problem.value
                
                # 计算终端约束违反
                pos_viol = np.linalg.norm(X_opt[-1, 0:3] - rf)
                vel_viol = np.linalg.norm(X_opt[-1, 3:6] - vf)
                constraint_viol = max(pos_viol, vel_viol)
                
                return X_opt, U_opt, cost_val, constraint_viol
            else:
                return X_ref, U_ref, float('inf'), float('inf')
                
        except Exception as e:
            if self.verbose:
                print(f"  CVXPY求解失败: {e}")
            return X_ref, U_ref, float('inf'), float('inf')
    
    def optimize(
        self,
        r0: np.ndarray,
        v0: np.ndarray,
        m0: float,
        rf: np.ndarray,
        vf: np.ndarray,
        t_span: List[float],
        max_iter: int = None,
    ) -> Dict:
        """
        执行SCP优化
        
        Parameters:
            r0, v0, m0: 初始状态
            rf, vf: 终端状态
            t_span: 时间区间
            max_iter: 最大迭代次数
            
        Returns:
            优化结果字典
        """
        if max_iter:
            self.max_iterations = max_iter
        
        start_time = time.time()
        
        if self.verbose:
            print("\n" + "="*60)
            print("CVXPY版序贯凸规划(SCP)轨迹优化")
            print("="*60)
            print(f"节点数: {self.n_nodes}")
            print(f"最大迭代: {self.max_iterations}")
            print(f"初始位置: {r0}")
            print(f"目标位置: {rf}")
            print("="*60)
        
        # 生成初始轨迹
        X_k, U_k = self.generate_initial_trajectory(r0, v0, m0, rf, vf, t_span)
        
        # 迭代求解
        best_X = X_k.copy()
        best_U = U_k.copy()
        best_pos_error = np.linalg.norm(X_k[-1, 0:3] - rf)
        best_vel_error = np.linalg.norm(X_k[-1, 3:6] - vf)
        
        for iteration in range(self.max_iterations):
            # 求解凸子问题
            X_new, U_new, cost, cons_viol = self.solve_convex_subproblem_cvxpy(
                X_k, U_k, r0, v0, m0, rf, vf, t_span
            )
            
            # 计算误差
            pos_error = np.linalg.norm(X_new[-1, 0:3] - rf)
            vel_error = np.linalg.norm(X_new[-1, 3:6] - vf)
            state_change = np.max(np.abs(X_new - X_k))
            
            # 更新最佳结果（始终更新，因为每次迭代都在改进）
            best_X = X_new.copy()
            best_U = U_new.copy()
            best_pos_error = pos_error
            best_vel_error = vel_error
            
            if self.verbose:
                print(f"迭代 {iteration + 1}: "
                      f"位置误差={pos_error:.2f}m, "
                      f"速度误差={vel_error:.2f}m/s, "
                      f"状态变化={state_change:.2e}")
            
            # 收敛检查 - 同时检查位置和速度误差
            if state_change < self.tol and pos_error < self.target_pos_error and vel_error < self.target_vel_error:
                if self.verbose:
                    print(f"\n收敛于迭代 {iteration + 1}")
                break
            
            # 如果位置误差小但速度误差大，增加速度惩罚权重
            if pos_error < 1.0 and vel_error > 1.0:
                self.w_vc = min(self.w_vc * 1.5, 1e6)
            
            # 信任域调整
            if state_change < 0.1 * self.tr_radius:
                self.tr_radius = min(self.tr_radius * 1.5, self.tr_radius_max)
                self.w_tr = max(self.w_tr * 0.8, 0.1)
            elif state_change > self.tr_radius:
                self.tr_radius = max(self.tr_radius * 0.5, self.tr_radius_min)
                self.w_tr = min(self.w_tr * 2.0, 10.0)
            
            # 更新参考轨迹
            X_k = X_new
            U_k = U_new
        
        elapsed_time = time.time() - start_time
        
        # 最终结果
        t_nodes = np.linspace(t_span[0], t_span[1], self.n_nodes)
        fuel_consumption = m0 - best_X[-1, 6]
        
        # 检查成功
        success = best_pos_error < 50.0 and best_vel_error < 10.0 and fuel_consumption > 1.0
        
        if self.verbose:
            print("\n" + "="*60)
            print("优化结果")
            print("="*60)
            print(f"成功: {success}")
            print(f"位置误差: {best_pos_error:.4f} m")
            print(f"速度误差: {best_vel_error:.4f} m/s")
            print(f"燃料消耗: {fuel_consumption:.2f} kg")
            print(f"计算时间: {elapsed_time:.2f} s")
            print("="*60)
        
        return {
            'success': success,
            't': t_nodes,
            'X': best_X,
            'U': best_U,
            'r': best_X[:, 0:3],
            'v': best_X[:, 3:6],
            'm': best_X[:, 6],
            'u': np.linalg.norm(best_U, axis=1) / self.sc.T_max,
            'fuel_consumption': fuel_consumption,
            'pos_error': best_pos_error,
            'vel_error': best_vel_error,
            'elapsed_time': elapsed_time,
        }


# 别名
SCPOptimizerCVXPY = CVXPYSCPOptimizer
