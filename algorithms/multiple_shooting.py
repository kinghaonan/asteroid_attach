#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小行星附着最优轨迹设计 - 多重打靶法

实现分段轨迹的多重打靶法求解。
将轨迹分为多段，每段单独积分，通过连续性约束连接各段。

核心算法：
- 分段轨迹积分
- 连续性约束（位置和速度）
- 最小二乘优化求解
- 多段初始协态优化
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Optional
import warnings


class MultipleShootingOptimizer:
    """
    多重打靶法轨迹优化器

    将轨迹分为多段，分别求解后通过连续性约束连接。
    相比单段打靶法，具有更好的收敛性和数值稳定性。

    Attributes:
        asteroid: 小行星模型对象
        spacecraft: 航天器参数对象
        n_segments: 分段数
        continuity_weight: 连续性约束权重
    """

    def __init__(self, asteroid, spacecraft, n_segments: int = 5):
        """
        初始化多重打靶法优化器

        Parameters:
            asteroid: 小行星对象
            spacecraft: 航天器对象
            n_segments: 分段数（默认5段）
        """
        self.ast = asteroid
        self.sc = spacecraft
        self.n_segments = n_segments
        self.continuity_weight = 1e3  # 连续性条件的权重

    def dynamics(self, t: float, z: np.ndarray) -> np.ndarray:
        """
        状态-协态动力学方程

        Parameters:
            t: 时间
            z: 状态和协态向量 [r(3), v(3), m, lam_r(3), lam_v(3), lam_m]

        Returns:
            状态导数 dz/dt
        """
        # 解包状态和协态变量
        r = z[0:3]
        v = z[3:6]
        m = z[6]
        lam_r = z[7:10]
        lam_v = z[10:13]
        lam_m = z[13]

        # 计算控制量
        chi, u = self.optimal_control(lam_v, lam_m, m)

        # 状态方程
        drdt = v
        dvdt = (
            -2 * np.cross(self.ast.omega, v)
            - np.cross(self.ast.omega, np.cross(self.ast.omega, r))
            + self.ast.gravity_gradient(r)
            + (self.sc.T_max / m) * u * chi
        )
        dmdt = -self.sc.T_max * u / (self.sc.I_sp * self.sc.g0)

        # 协态方程
        omega_skew = np.array(
            [
                [0, -self.ast.omega[2], self.ast.omega[1]],
                [self.ast.omega[2], 0, -self.ast.omega[0]],
                [-self.ast.omega[1], self.ast.omega[0], 0],
            ]
        )

        dlam_rdt = np.cross(
            self.ast.omega, np.cross(self.ast.omega, lam_v)
        ) + self.ast.gravity_hessian(r).dot(lam_v)
        dlam_vdt = -lam_r + 2 * np.cross(self.ast.omega, lam_v)
        dlam_mdt = -np.linalg.norm(lam_v) * self.sc.T_max * u / (m**2)

        # 确保所有数组都是一维的
        drdt = np.asarray(drdt).flatten()
        dvdt = np.asarray(dvdt).flatten()
        dlam_rdt = np.asarray(dlam_rdt).flatten()
        dlam_vdt = np.asarray(dlam_vdt).flatten()

        return np.concatenate([drdt, dvdt, [dmdt], dlam_rdt, dlam_vdt, [dlam_mdt]])

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
        if lam_v_norm > 1e-6:
            chi = -lam_v / lam_v_norm
        else:
            chi = np.zeros(3)

        H_r = 1 - lam_m - (self.sc.I_sp * self.sc.g0 / m) * lam_v_norm

        if H_r > 0:
            u = 0.0
        elif H_r < 0:
            u = 1.0
        else:
            u = 0.5

        return chi, u

    def multiple_shooting(
        self,
        X: np.ndarray,
        t_span: List[float],
        r0: np.ndarray,
        v0: np.ndarray,
        m0: float,
        rf: np.ndarray,
        vf: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        多重打靶函数

        核心算法：
        1. 将决策变量分解为每个段的初始协态
        2. 对每个段进行积分
        3. 添加连续性约束（段间连接）
        4. 添加终端边界条件

        Parameters:
            X: 决策变量（所有段的初始协态）
            t_span: 时间区间 [t0, tf]
            r0, v0, m0: 初始状态
            rf, vf: 终端状态

        Returns:
            errors: 约束违反向量
            full_t: 完整时间序列
            full_y: 完整状态序列
        """
        # 将决策变量分解为每个段的初始协态变量
        lam0_segments = X.reshape(self.n_segments, 7)

        # 初始化误差向量
        errors = []

        # 初始化轨迹
        full_t = np.array([])
        full_y = np.empty((14, 0))

        # 第一个段的初始条件
        current_r = r0
        current_v = v0
        current_m = m0

        # 时间节点
        t_nodes = np.linspace(t_span[0], t_span[1], self.n_segments + 1)

        for i in range(self.n_segments):
            # 当前段的时间区间
            segment_t_span = [t_nodes[i], t_nodes[i + 1]]

            # 当前段的初始条件
            z0 = np.concatenate([current_r, current_v, [current_m], lam0_segments[i]])

            # 积分当前段
            sol = solve_ivp(
                self.dynamics, segment_t_span, z0, method="RK45", rtol=1e-6, atol=1e-9
            )

            # 提取终端状态
            zf = sol.y[:, -1]
            r_seg = zf[0:3]
            v_seg = zf[3:6]
            m_seg = zf[6]

            # 如果不是最后一段，添加连续性条件
            if i < self.n_segments - 1:
                continuity_errors = np.concatenate(
                    [r_seg - current_r, v_seg - current_v, [m_seg - current_m]]
                )
                errors.extend(self.continuity_weight * continuity_errors)

                # 更新下一段的初始状态
                current_r = r_seg
                current_v = v_seg
                current_m = m_seg

            # 存储轨迹
            full_t = np.concatenate([full_t, sol.t])
            full_y = np.concatenate([full_y, sol.y], axis=1)

        # 终端边界条件误差
        terminal_errors = np.concatenate(
            [
                current_r - rf,
                current_v - vf,
                [0],  # 最终质量协态应为0（横截条件）
            ]
        )
        errors.extend(terminal_errors)

        return np.array(errors), full_t, full_y

    def solve(
        self,
        r0: np.ndarray,
        v0: np.ndarray,
        m0: float,
        rf: np.ndarray,
        vf: np.ndarray,
        t_span: List[float],
        lam_guess: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        求解轨迹优化问题

        Parameters:
            r0, v0, m0: 初始状态
            rf, vf: 终端状态
            t_span: 时间区间
            lam_guess: 初始猜测（可选）

        Returns:
            lam0_segments: 每段的最优初始协态
            t_opt: 最优时间序列
            y_opt: 最优状态序列
            errors: 最终误差
        """
        # 初始猜测
        expected_size = self.n_segments * 7
        if lam_guess is None or len(lam_guess) != expected_size:
            lam_guess = np.tile([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0], self.n_segments)

        # 使用最小二乘法求解
        result = least_squares(
            lambda X: self.multiple_shooting(X, t_span, r0, v0, m0, rf, vf)[0],
            lam_guess,
            method="lm",
            ftol=1e-6,
            xtol=1e-6,
            gtol=1e-6,
            max_nfev=1000,
        )

        # 获取最优解
        X_opt = result.x
        errors, t_opt, y_opt = self.multiple_shooting(X_opt, t_span, r0, v0, m0, rf, vf)

        # 计算每个段的初始协态变量
        lam0_segments = X_opt.reshape(self.n_segments, 7)

        return lam0_segments, t_opt, y_opt, errors

    def optimize_with_multiple_guesses(
        self,
        r0: np.ndarray,
        v0: np.ndarray,
        m0: float,
        rf: np.ndarray,
        vf: np.ndarray,
        t_span: List[float],
        guesses: Optional[List[np.ndarray]] = None,
    ) -> Dict:
        """
        使用多初始猜测求解

        Parameters:
            r0, v0, m0: 初始状态
            rf, vf: 终端状态
            t_span: 时间区间
            guesses: 初始猜测列表（可选）

        Returns:
            result: 包含最优解的字典
        """
        if guesses is None:
            guesses = [
                np.tile([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0], self.n_segments),
                np.tile([-0.1, -0.1, -0.1, -0.1, -0.1, -0.1, 0.0], self.n_segments),
                np.tile([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.0], self.n_segments),
                np.tile([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0], self.n_segments),
            ]

        best_lam0 = None
        best_t = None
        best_y = None
        best_error = float("inf")

        for i, lam_guess in enumerate(guesses):
            print(f"尝试初始猜测 {i + 1}/{len(guesses)}")

            try:
                lam0_segments, t_opt, y_opt, errors = self.solve(
                    r0, v0, m0, rf, vf, t_span, lam_guess
                )
                current_error = np.linalg.norm(errors)

                print(f"当前误差: {current_error:.6f}")

                if current_error < best_error:
                    best_error = current_error
                    best_lam0 = lam0_segments
                    best_t = t_opt
                    best_y = y_opt

                if best_error < 1e-3:
                    print("找到足够精确的解，提前终止")
                    break

            except Exception as e:
                print(f"初始猜测 {i + 1} 失败: {str(e)}")
                continue

        if best_t is None:
            print("所有初始猜测都失败，使用默认猜测")
            default_guess = np.tile(
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0], self.n_segments
            )
            errors, best_t, best_y = self.multiple_shooting(
                default_guess, t_span, r0, v0, m0, rf, vf
            )
            best_lam0 = default_guess.reshape(self.n_segments, 7)

        return {
            "success": best_error < 1e-3,
            "initial_costates": best_lam0,
            "t": best_t,
            "y": best_y,
            "final_error": best_error,
            "final_mass": best_y[6, -1] if best_y is not None else None,
        }

    def extract_trajectory_data(self, t: np.ndarray, y: np.ndarray) -> Dict:
        """
        提取轨迹数据

        Parameters:
            t: 时间序列
            y: 状态序列

        Returns:
            data: 轨迹数据字典
        """
        r = y[0:3, :]
        v = y[3:6, :]
        m = y[6, :]
        lam_r = y[7:10, :]
        lam_v = y[10:13, :]
        lam_m = y[13, :]

        # 计算控制历史
        u_history = np.zeros(len(t))
        chi_history = np.zeros((3, len(t)))

        for i in range(len(t)):
            z = y[:, i]
            lam_v_i = z[10:13]
            lam_m_i = z[13]
            m_i = z[6]
            chi, u = self.optimal_control(lam_v_i, lam_m_i, m_i)
            u_history[i] = u
            chi_history[:, i] = chi

        return {
            "t": t,
            "r": r,
            "v": v,
            "m": m,
            "lam_r": lam_r,
            "lam_v": lam_v,
            "lam_m": lam_m,
            "u": u_history,
            "chi": chi_history,
        }

    def plot_results(
        self,
        data: Dict,
        r0: np.ndarray,
        rf: np.ndarray,
        save_path: Optional[str] = None,
    ):
        """
        绘制优化结果

        Parameters:
            data: 轨迹数据字典
            r0, rf: 初始和目标位置
            save_path: 保存路径（可选）
        """
        fig = plt.figure(figsize=(15, 10))

        # 位置随时间变化
        ax = fig.add_subplot(2, 3, 1)
        ax.plot(data["t"], data["r"][0, :], label="x")
        ax.plot(data["t"], data["r"][1, :], label="y")
        ax.plot(data["t"], data["r"][2, :], label="z")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Position (m)")
        ax.legend()
        ax.set_title("Position vs Time")
        ax.grid(True)

        # 速度随时间变化
        ax = fig.add_subplot(2, 3, 2)
        ax.plot(data["t"], data["v"][0, :], label="vx")
        ax.plot(data["t"], data["v"][1, :], label="vy")
        ax.plot(data["t"], data["v"][2, :], label="vz")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Velocity (m/s)")
        ax.legend()
        ax.set_title("Velocity vs Time")
        ax.grid(True)

        # 质量随时间变化
        ax = fig.add_subplot(2, 3, 3)
        ax.plot(data["t"], data["m"])
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Mass (kg)")
        ax.set_title("Mass vs Time")
        ax.grid(True)

        # 三维轨迹
        ax = fig.add_subplot(2, 3, 4, projection="3d")
        ax.plot(data["r"][0, :], data["r"][1, :], data["r"][2, :])
        ax.scatter(r0[0], r0[1], r0[2], c="r", marker="o", label="Start")
        ax.scatter(rf[0], rf[1], rf[2], c="g", marker="^", label="Target")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.legend()
        ax.set_title("3D Trajectory")

        # 推力大小随时间变化
        ax = fig.add_subplot(2, 3, 5)
        ax.plot(data["t"], data["u"] * self.sc.T_max)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Thrust (N)")
        ax.set_title("Thrust vs Time")
        ax.grid(True)

        # 开关函数随时间变化
        H_r_history = np.zeros(len(data["t"]))
        for i in range(len(data["t"])):
            lam_v_i = data["lam_v"][:, i]
            lam_m_i = data["lam_m"][i]
            m_i = data["m"][i]
            H_r_history[i] = (
                1
                - lam_m_i
                - (self.sc.I_sp * self.sc.g0 / m_i) * np.linalg.norm(lam_v_i)
            )

        ax = fig.add_subplot(2, 3, 6)
        ax.plot(data["t"], H_r_history)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("H_r")
        ax.set_title("Switching Function")
        ax.axhline(0, color="r", linestyle="--")
        ax.grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"结果已保存至: {save_path}")

        plt.show()


# 示例使用
if __name__ == "__main__":
    # 创建示例类
    class ExampleAsteroid:
        def __init__(self):
            self.omega = np.array([0, 0, 2 * np.pi / (5.27 * 3600)])
            self.mu = 4.463e5

        def gravity_gradient(self, r):
            r_norm = np.linalg.norm(r)
            if r_norm < 1e-3:
                return np.zeros(3)
            return -self.mu * r / (r_norm**3)

        def gravity_hessian(self, r):
            r_norm = np.linalg.norm(r)
            if r_norm < 1e-3:
                return np.zeros((3, 3))
            I = np.eye(3)
            return -self.mu * (3 * np.outer(r, r) / (r_norm**5) - I / (r_norm**3))

    class ExampleSpacecraft:
        def __init__(self):
            self.T_max = 20.0
            self.I_sp = 400.0
            self.g0 = 9.81
            self.m0 = 1000.0

    # 初始化
    asteroid = ExampleAsteroid()
    spacecraft = ExampleSpacecraft()
    optimizer = MultipleShootingOptimizer(asteroid, spacecraft, n_segments=5)

    # 边界条件
    r0 = np.array([10177, 6956, 8256])
    v0 = np.array([-25, -12, -17])
    m0 = spacecraft.m0
    rf = np.array([676, 5121, 449])
    vf = np.array([0, 0, 0])
    t_span = [0, 770]

    # 优化
    result = optimizer.optimize_with_multiple_guesses(r0, v0, m0, rf, vf, t_span)

    if result["success"]:
        data = optimizer.extract_trajectory_data(result["t"], result["y"])
        optimizer.plot_results(data, r0, rf)

        print("\n优化结果:")
        print(f"每段初始协态:\n{result['initial_costates']}")
        print(f"终端质量: {result['final_mass']:.2f} kg")
        print(f"燃料消耗: {m0 - result['final_mass']:.2f} kg")
