#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小行星附着最优轨迹设计 - 伪谱法

实现基于伪谱法的全局轨迹优化。
使用Legendre-Gauss-Lobatto (LGL) 配点和微分矩阵。

核心算法：
- 切比雪夫节点生成
- 微分矩阵计算（重心公式）
- 伪谱离散化动力学
- 非线性规划求解（SLSQP/trust-constr）

优化重点：
1. 增强初始猜测可行性
2. 改进约束处理机制
3. 状态归一化，提升数值稳定性
4. 两阶段优化策略
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, NonlinearConstraint
from typing import Dict, Tuple, Optional, List
import time
import warnings


class StateScaler:
    """状态变量归一化处理，消除量纲差异"""

    def __init__(self):
        self.r_scale = 1e4  # 位置：米→万米
        self.v_scale = 1e1  # 速度：米/秒→十米/秒
        self.m_scale = 1e3  # 质量：千克→吨
        self.lam_scale = 1e-3  # 协态变量缩放

    def scale_state(self, r, v, m, lam_r, lam_v, lam_m):
        """状态编码：原始值→归一化值"""
        return (
            r / self.r_scale,
            v / self.v_scale,
            m / self.m_scale,
            lam_r * self.lam_scale,
            lam_v * self.lam_scale,
            lam_m * self.lam_scale,
        )

    def unscale_state(self, r_s, v_s, m_s, lam_r_s, lam_v_s, lam_m_s):
        """状态解码：归一化值→原始值"""
        return (
            r_s * self.r_scale,
            v_s * self.v_scale,
            m_s * self.m_scale,
            lam_r_s / self.lam_scale,
            lam_v_s / self.lam_scale,
            lam_m_s / self.lam_scale,
        )


class PseudospectralOptimizer:
    """
    伪谱法轨迹优化器

    使用Legendre伪谱法进行全局轨迹优化。
    将连续时间最优控制问题转化为非线性规划问题。

    Attributes:
        asteroid: 小行星模型
        spacecraft: 航天器参数
        n_nodes: 配点数
        scaler: 状态缩放器
        debug_mode: 调试模式
    """

    def __init__(self, asteroid, spacecraft, n_nodes: int = 30):
        """
        初始化伪谱法优化器

        Parameters:
            asteroid: 小行星对象
            spacecraft: 航天器对象
            n_nodes: 配点数（默认30）
        """
        self.ast = asteroid
        self.sc = spacecraft
        self.n_nodes = n_nodes
        self.scaler = StateScaler()
        self.debug_mode = False
        self.fun_evals = 0
        self.con_evals = 0

    def set_debug_mode(self, debug: bool):
        """设置调试模式"""
        self.debug_mode = debug

    def chebyshev_nodes(self, N: int) -> np.ndarray:
        """
        生成切比雪夫节点（LGL点）

        τ_k = -cos(kπ/N), k=0,1,...,N

        Parameters:
            N: 节点数

        Returns:
            tau: 归一化时间节点 [-1, 1]
        """
        k = np.arange(N + 1)
        tau = -np.cos(k * np.pi / N)
        return tau

    def barycentric_weights(self, x: np.ndarray) -> np.ndarray:
        """
        计算重心权重

        w_j = 1 / ∏(x_j - x_k) for k≠j

        Parameters:
            x: 节点位置

        Returns:
            w: 重心权重
        """
        n = len(x)
        w = np.ones(n)
        for j in range(n):
            for k in range(n):
                if k != j:
                    w[j] *= x[j] - x[k]
        w = 1.0 / w
        return w

    def differentiation_matrix(self, tau: np.ndarray) -> np.ndarray:
        """
        计算伪谱微分矩阵（重心公式）

        D[i,j] = (w_j/w_i)/(τ_i-τ_j)  for i≠j
        D[i,i] = -ΣD[i,j]  for i=j

        Parameters:
            tau: 归一化时间节点

        Returns:
            D: 微分矩阵
        """
        n = len(tau)
        D = np.zeros((n, n))
        w = self.barycentric_weights(tau)

        for i in range(n):
            for j in range(n):
                if i != j:
                    D[i, j] = (w[j] / w[i]) / (tau[i] - tau[j])
            D[i, i] = -np.sum(D[i, :])

        return D

    def optimal_control(
        self, lam_v: np.ndarray, lam_m: float, m: float
    ) -> Tuple[np.ndarray, float]:
        """
        计算最优控制

        Parameters:
            lam_v: 速度协态
            lam_m: 质量协态
            m: 质量

        Returns:
            chi: 推力方向
            u: 推力大小
        """
        lam_v_norm = np.linalg.norm(lam_v)
        if lam_v_norm < 1e-10:
            return np.zeros(3), 0.0

        chi = -lam_v / lam_v_norm
        S = 1 - lam_m - (self.sc.I_sp * self.sc.g0 / m) * lam_v_norm

        if S < 0:
            u = 1.0
        else:
            u = 0.0

        return chi, u

    def dynamics(
        self,
        r: np.ndarray,
        v: np.ndarray,
        m: float,
        lam_r: np.ndarray,
        lam_v: np.ndarray,
        lam_m: float,
    ) -> Tuple:
        """
        系统动力学方程

        Parameters:
            r, v, m: 状态
            lam_r, lam_v, lam_m: 协态

        Returns:
            各变量的导数
        """
        chi, u = self.optimal_control(lam_v, lam_m, m)

        # 引力与环境力
        g = self.ast.gravity_gradient(r)
        omega = self.ast.omega
        coriolis = -2 * np.cross(omega, v)
        centrifugal = -np.cross(omega, np.cross(omega, r))

        # 推力加速度
        thrust_acc = (self.sc.T_max / m) * u * chi if m > 1e-3 else np.zeros(3)

        # 状态导数
        r_dot = v
        v_dot = g + coriolis + centrifugal + thrust_acc
        m_dot = -self.sc.T_max * u / (self.sc.I_sp * self.sc.g0) if m > 1e-3 else 0

        # 协态导数
        omega_cross = np.array(
            [
                [0, -omega[2], omega[1]],
                [omega[2], 0, -omega[0]],
                [-omega[1], omega[0], 0],
            ]
        )

        # 引力梯度矩阵（简化）
        r_norm = np.linalg.norm(r)
        if r_norm < 1e-3:
            G = np.zeros((3, 3))
        else:
            G = -self.ast.mu * (3 * np.outer(r, r) / r_norm**5 - np.eye(3) / r_norm**3)

        lam_r_dot = (
            -G.T @ lam_v + omega_cross @ (omega_cross @ lam_r) - omega_cross @ lam_v
        )
        lam_v_dot = -lam_r + 2 * omega_cross @ lam_v
        lam_m_dot = (self.sc.T_max / m**2) * u * np.dot(chi, lam_v) if m > 1e-3 else 0

        return r_dot, v_dot, m_dot, lam_r_dot, lam_v_dot, lam_m_dot

    def generate_initial_guess(
        self, r0: np.ndarray, v0: np.ndarray, m0: float, rf: np.ndarray, vf: np.ndarray
    ) -> np.ndarray:
        """
        生成初始猜测（Hermite插值）

        使用三次Hermite插值确保平滑过渡。

        Parameters:
            r0, v0, m0: 初始状态
            rf, vf: 终端状态

        Returns:
            x0: 初始猜测向量
        """
        N = self.n_nodes
        alpha = np.linspace(0, 1, N + 1).reshape(-1, 1)

        # Hermite插值
        r_guess = r0 + (rf - r0) * (3 * alpha**2 - 2 * alpha**3)
        v_guess = v0 + (vf - v0) * (6 * alpha * (1 - alpha))
        m_guess = m0 - 0.05 * m0 * alpha.flatten()

        # 协态猜测
        lam_r_guess = np.ones((N + 1, 3)) * 0.01
        lam_v_guess = np.ones((N + 1, 3)) * (-0.01)
        lam_m_guess = np.ones(N + 1) * (-0.001)

        # 归一化
        r_guess_s, v_guess_s, m_guess_s, lam_r_guess_s, lam_v_guess_s, lam_m_guess_s = (
            self.scaler.scale_state(
                r_guess, v_guess, m_guess, lam_r_guess, lam_v_guess, lam_m_guess
            )
        )

        # 展平
        x0 = []
        for i in range(N + 1):
            x0.extend(r_guess_s[i])
            x0.extend(v_guess_s[i])
            x0.append(m_guess_s[i])
            x0.extend(lam_r_guess_s[i])
            x0.extend(lam_v_guess_s[i])
            x0.append(lam_m_guess_s[i])

        return np.array(x0)

    def objective(self, x: np.ndarray) -> float:
        """
        目标函数：最小化燃料消耗

        Parameters:
            x: 决策变量

        Returns:
            目标函数值
        """
        self.fun_evals += 1
        N = self.n_nodes
        states = np.reshape(x, (N + 1, 14))

        fuel = 0
        for i in range(N + 1):
            r_s = states[i, 0:3]
            v_s = states[i, 3:6]
            m_s = states[i, 6]
            lam_v_s = states[i, 10:13]
            lam_m_s = states[i, 13]

            # 反归一化
            r, v, m, _, lam_v, lam_m = self.scaler.unscale_state(
                r_s, v_s, m_s, np.zeros(3), lam_v_s, lam_m_s
            )

            _, u = self.optimal_control(lam_v, lam_m, m)
            fuel += u

        return fuel / (N + 1)

    def constraints(
        self,
        x: np.ndarray,
        r0: np.ndarray,
        v0: np.ndarray,
        m0: float,
        rf: np.ndarray,
        vf: np.ndarray,
        D_scaled: np.ndarray,
    ) -> np.ndarray:
        """
        约束函数

        包括：
        1. 初始条件约束
        2. 终端条件约束
        3. 动力学约束（伪谱离散）
        4. 横截条件

        Parameters:
            x: 决策变量
            r0, v0, m0: 初始状态
            rf, vf: 终端状态
            D_scaled: 缩放的微分矩阵

        Returns:
            约束违反向量
        """
        self.con_evals += 1
        N = self.n_nodes
        states = np.reshape(x, (N + 1, 14))

        eq_constraints = []

        # 初始条件（反归一化后比较）
        r0_s = r0 / self.scaler.r_scale
        v0_s = v0 / self.scaler.v_scale
        m0_s = m0 / self.scaler.m_scale
        eq_constraints.extend(states[0, 0:3] - r0_s)
        eq_constraints.extend(states[0, 3:6] - v0_s)
        eq_constraints.append(states[0, 6] - m0_s)

        # 终端条件
        rf_s = rf / self.scaler.r_scale
        vf_s = vf / self.scaler.v_scale
        eq_constraints.extend(states[-1, 0:3] - rf_s)
        eq_constraints.extend(states[-1, 3:6] - vf_s)

        # 动力学约束（内部节点）
        for i in range(1, N + 1):
            r_s = states[i, 0:3]
            v_s = states[i, 3:6]
            m_s = states[i, 6]
            lam_r_s = states[i, 7:10]
            lam_v_s = states[i, 10:13]
            lam_m_s = states[i, 13]

            # 反归一化
            r, v, m, lam_r, lam_v, lam_m = self.scaler.unscale_state(
                r_s, v_s, m_s, lam_r_s, lam_v_s, lam_m_s
            )

            # 计算动力学导数
            r_dot, v_dot, m_dot, lam_r_dot, lam_v_dot, lam_m_dot = self.dynamics(
                r, v, m, lam_r, lam_v, lam_m
            )

            # 归一化导数
            r_dot_s = r_dot / self.scaler.r_scale
            v_dot_s = v_dot / self.scaler.v_scale
            m_dot_s = m_dot / self.scaler.m_scale
            lam_r_dot_s = lam_r_dot * self.scaler.lam_scale
            lam_v_dot_s = lam_v_dot * self.scaler.lam_scale
            lam_m_dot_s = lam_m_dot * self.scaler.lam_scale

            # 伪谱导数
            r_dot_ps = D_scaled[i] @ states[:, 0:3]
            v_dot_ps = D_scaled[i] @ states[:, 3:6]
            m_dot_ps = D_scaled[i] @ states[:, 6]
            lam_r_dot_ps = D_scaled[i] @ states[:, 7:10]
            lam_v_dot_ps = D_scaled[i] @ states[:, 10:13]
            lam_m_dot_ps = D_scaled[i] @ states[:, 13]

            # 添加约束
            eq_constraints.extend(r_dot_ps - r_dot_s)
            eq_constraints.extend(v_dot_ps - v_dot_s)
            eq_constraints.append(m_dot_ps - m_dot_s)
            eq_constraints.extend(lam_r_dot_ps - lam_r_dot_s)
            eq_constraints.extend(lam_v_dot_ps - lam_v_dot_s)
            eq_constraints.append(lam_m_dot_ps - lam_m_dot_s)

        # 横截条件
        eq_constraints.append(states[-1, 13])

        return np.array(eq_constraints)

    def optimize(
        self,
        r0: np.ndarray,
        v0: np.ndarray,
        m0: float,
        rf: np.ndarray,
        vf: np.ndarray,
        t_span: List[float],
        max_iter: int = 500,
    ) -> Dict:
        """
        执行伪谱法优化

        Parameters:
            r0, v0, m0: 初始状态
            rf, vf: 终端状态
            t_span: 时间区间 [t0, tf]
            max_iter: 最大迭代次数

        Returns:
            result: 优化结果字典
        """
        print(f"\n=== 伪谱法轨迹优化 ===")
        print(f"节点数: {self.n_nodes + 1}")

        t0, tf = t_span
        N = self.n_nodes

        # 生成节点和微分矩阵
        tau = self.chebyshev_nodes(N)
        t_nodes = (tf - t0) / 2 * (tau + 1) + t0
        D = self.differentiation_matrix(tau)
        D_scaled = 2 / (tf - t0) * D

        # 初始猜测
        x0 = self.generate_initial_guess(r0, v0, m0, rf, vf)

        # 约束函数包装
        def con_func(x):
            return self.constraints(x, r0, v0, m0, rf, vf, D_scaled)

        # 计算约束数量
        con_count = len(con_func(x0))
        print(f"变量数: {len(x0)}, 约束数: {con_count}")

        # 非线性约束
        nlc = NonlinearConstraint(con_func, np.zeros(con_count), np.zeros(con_count))

        # 优化求解
        start_time = time.time()
        self.fun_evals = 0
        self.con_evals = 0

        try:
            # 阶段1: SLSQP初步优化
            print("\n阶段1: SLSQP初步优化...")
            result_slsqp = minimize(
                self.objective,
                x0,
                constraints=nlc,
                method="SLSQP",
                options={"maxiter": 300, "ftol": 1e-4, "disp": False},
            )

            if result_slsqp.success or result_slsqp.status in [0, 1, 2]:
                # 阶段2: trust-constr精细优化
                print("阶段2: trust-constr精细优化...")
                result = minimize(
                    self.objective,
                    result_slsqp.x,
                    constraints=nlc,
                    method="trust-constr",
                    options={
                        "maxiter": max_iter,
                        "verbose": 0,
                        "gtol": 1e-6,
                        "xtol": 1e-7,
                    },
                )
            else:
                result = result_slsqp

        except Exception as e:
            print(f"优化失败: {str(e)}")
            result = result_slsqp if "result_slsqp" in locals() else None

        end_time = time.time()
        print(f"优化完成，耗时: {end_time - start_time:.2f}s")

        if result is None or not result.success:
            return {"success": False, "message": "优化失败"}

        # 提取结果
        states_opt = np.reshape(result.x, (N + 1, 14))

        r_opt = states_opt[:, 0:3] * self.scaler.r_scale
        v_opt = states_opt[:, 3:6] * self.scaler.v_scale
        m_opt = states_opt[:, 6] * self.scaler.m_scale
        lam_r_opt = states_opt[:, 7:10] / self.scaler.lam_scale
        lam_v_opt = states_opt[:, 10:13] / self.scaler.lam_scale
        lam_m_opt = states_opt[:, 13] / self.scaler.lam_scale

        # 计算控制历史
        u_opt = np.zeros(N + 1)
        chi_opt = np.zeros((N + 1, 3))
        for i in range(N + 1):
            chi, u = self.optimal_control(lam_v_opt[i], lam_m_opt[i], m_opt[i])
            u_opt[i] = u
            chi_opt[i] = chi

        # 性能指标
        pos_err = np.linalg.norm(r_opt[-1] - rf)
        vel_err = np.linalg.norm(v_opt[-1] - vf)
        fuel_consumption = m0 - m_opt[-1]

        print(f"\n优化结果:")
        print(f"终端质量: {m_opt[-1]:.2f} kg")
        print(f"燃料消耗: {fuel_consumption:.2f} kg")
        print(f"位置误差: {pos_err:.2f} m")
        print(f"速度误差: {vel_err:.2f} m/s")

        return {
            "success": result.success,
            "t": t_nodes,
            "r": r_opt,
            "v": v_opt,
            "m": m_opt,
            "u": u_opt,
            "chi": chi_opt,
            "lam_r": lam_r_opt,
            "lam_v": lam_v_opt,
            "lam_m": lam_m_opt,
            "fuel_consumption": fuel_consumption,
            "pos_error": pos_err,
            "vel_error": vel_err,
        }

    def plot_results(
        self,
        result: Dict,
        r0: np.ndarray,
        rf: np.ndarray,
        save_path: Optional[str] = None,
    ):
        """
        绘制优化结果

        Parameters:
            result: 优化结果字典
            r0, rf: 初始和目标位置
            save_path: 保存路径（可选）
        """
        if not result["success"]:
            print("优化失败，无法绘图")
            return

        fig = plt.figure(figsize=(15, 10))
        t = result["t"]

        # 位置
        ax = fig.add_subplot(2, 3, 1)
        ax.plot(t, result["r"][:, 0], label="x")
        ax.plot(t, result["r"][:, 1], label="y")
        ax.plot(t, result["r"][:, 2], label="z")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Position (m)")
        ax.legend()
        ax.set_title("Position")
        ax.grid(True)

        # 速度
        ax = fig.add_subplot(2, 3, 2)
        ax.plot(t, result["v"][:, 0], label="vx")
        ax.plot(t, result["v"][:, 1], label="vy")
        ax.plot(t, result["v"][:, 2], label="vz")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Velocity (m/s)")
        ax.legend()
        ax.set_title("Velocity")
        ax.grid(True)

        # 质量
        ax = fig.add_subplot(2, 3, 3)
        ax.plot(t, result["m"])
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Mass (kg)")
        ax.set_title("Mass")
        ax.grid(True)

        # 3D轨迹
        ax = fig.add_subplot(2, 3, 4, projection="3d")
        ax.plot(result["r"][:, 0], result["r"][:, 1], result["r"][:, 2])
        ax.scatter(r0[0], r0[1], r0[2], c="r", marker="o", label="Start")
        ax.scatter(rf[0], rf[1], rf[2], c="g", marker="^", label="Target")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.legend()
        ax.set_title("3D Trajectory")

        # 推力
        ax = fig.add_subplot(2, 3, 5)
        ax.plot(t, result["u"] * self.sc.T_max)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Thrust (N)")
        ax.set_title("Thrust")
        ax.grid(True)

        # 协态
        ax = fig.add_subplot(2, 3, 6)
        ax.plot(t, result["lam_m"])
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("λ_m")
        ax.set_title("Mass Costate")
        ax.grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"结果已保存至: {save_path}")

        plt.show()


# 示例使用
if __name__ == "__main__":

    class ExampleAsteroid:
        def __init__(self):
            self.omega = np.array([0, 0, 2 * np.pi / (5.27 * 3600)])
            self.mu = 4.463e5

        def gravity_gradient(self, r):
            r_norm = np.linalg.norm(r)
            if r_norm < 1e-3:
                return np.zeros(3)
            return -self.mu * r / (r_norm**3)

    class ExampleSpacecraft:
        def __init__(self):
            self.T_max = 20.0
            self.I_sp = 400.0
            self.g0 = 9.81
            self.m0 = 1000.0

    asteroid = ExampleAsteroid()
    spacecraft = ExampleSpacecraft()
    optimizer = PseudospectralOptimizer(asteroid, spacecraft, n_nodes=20)

    r0 = np.array([10177, 6956, 8256])
    v0 = np.array([-25, -12, -17])
    m0 = spacecraft.m0
    rf = np.array([676, 5121, 449])
    vf = np.array([0, 0, 0])
    t_span = [0, 770]

    result = optimizer.optimize(r0, v0, m0, rf, vf, t_span)
    optimizer.plot_results(result, r0, rf)
