#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小行星附着最优轨迹设计 - 打靶法

实现单段轨迹的打靶法求解（最优控制理论）。
使用间接法求解两点边值问题（TPBVP）。

核心算法：
- 状态-协态动力学积分
- 最优控制计算（Pontryagin极大值原理）
- 打靶法求解初始协态
- 多初始猜测策略
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Optional, Callable
import warnings


class ShootingMethodOptimizer:
    """
    打靶法轨迹优化器

    基于最优控制理论，使用打靶法求解燃料最优轨迹。
    通过求解协态方程和横截条件来确定最优控制。

    Attributes:
        asteroid: 小行星模型对象
        spacecraft: 航天器参数对象
        best_error: 最优误差（用于多初始猜测）
        best_solution: 最优解
    """

    def __init__(self, asteroid, spacecraft):
        """
        初始化打靶法优化器

        Parameters:
            asteroid: 小行星对象，需包含omega, gravity_gradient(), gravity_hessian()
            spacecraft: 航天器对象，需包含T_max, I_sp, g0, m0
        """
        self.ast = asteroid
        self.sc = spacecraft
        self.best_error = float("inf")
        self.best_solution = None

    def dynamics(self, t: float, z: np.ndarray) -> np.ndarray:
        """
        状态-协态动力学方程

        基于最优控制理论的状态方程和协态方程：
        - 位置动力学: dr/dt = v
        - 速度动力学: dv/dt = g + coriolis + centrifugal + thrust
        - 质量动力学: dm/dt = -T_max*u/(I_sp*g0)
        - 协态方程: dλ/dt = -∂H/∂x

        Parameters:
            t: 时间
            z: 状态和协态向量 [r(3), v(3), m, lam_r(3), lam_v(3), lam_m]

        Returns:
            状态导数 dz/dt
        """
        # 解包状态和协态变量
        r = z[0:3]  # 位置
        v = z[3:6]  # 速度
        m = z[6]  # 质量
        lam_r = z[7:10]  # 位置协态
        lam_v = z[10:13]  # 速度协态
        lam_m = z[13]  # 质量协态

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
        计算最优控制（Pontryagin极大值原理）

        基于开关函数确定最优推力：
        - 推力方向: χ = -λ_v/||λ_v||
        - 开关函数: H_r = 1 - λ_m - (I_sp*g0/m)*||λ_v||
        - 推力大小: u = 0 if H_r > 0, u = 1 if H_r < 0

        Parameters:
            lam_v: 速度协态
            lam_m: 质量协态
            m: 质量

        Returns:
            chi: 推力方向（单位向量）
            u: 推力大小（0或1，bang-bang控制）
        """
        # 推力方向
        lam_v_norm = np.linalg.norm(lam_v)
        if lam_v_norm > 1e-6:
            chi = -lam_v / lam_v_norm
        else:
            chi = np.zeros(3)

        # 开关函数（燃料最优）
        H_r = 1 - lam_m - (self.sc.I_sp * self.sc.g0 / m) * lam_v_norm

        # 推力大小（bang-bang控制）
        if H_r > 0:
            u = 0.0
        elif H_r < 0:
            u = 1.0
        else:  # 奇异情况
            u = 0.5

        return chi, u

    def shoot(
        self,
        lam0: np.ndarray,
        t_span: List[float],
        r0: np.ndarray,
        v0: np.ndarray,
        m0: float,
        rf: np.ndarray,
        vf: np.ndarray,
    ) -> Tuple[np.ndarray, object]:
        """
        打靶函数

        积分动力学方程并计算终端误差。

        Parameters:
            lam0: 初始协态猜测 [lam_r0(3), lam_v0(3), lam_m0]
            t_span: 时间区间 [t0, tf]
            r0, v0, m0: 初始状态
            rf, vf: 终端状态

        Returns:
            error: 终端误差向量 [r_err(3), v_err(3), lam_m_err]
            sol: 积分结果对象
        """
        # 初始状态
        z0 = np.concatenate([r0, v0, [m0], lam0])

        # 积分动力学方程
        sol = solve_ivp(self.dynamics, t_span, z0, method="RK45", rtol=1e-6, atol=1e-9)

        # 提取终端状态
        zf = sol.y[:, -1]
        r_tf = zf[0:3]
        v_tf = zf[3:6]
        lam_m_tf = zf[13]

        # 计算终端误差
        error = np.concatenate([r_tf - rf, v_tf - vf, [lam_m_tf]])

        # 更新最优解
        error_norm = np.linalg.norm(error[:6])  # 不考虑横截条件
        if error_norm < self.best_error:
            self.best_error = error_norm
            self.best_solution = sol

        return error, sol

    def solve(
        self,
        r0: np.ndarray,
        v0: np.ndarray,
        m0: float,
        rf: np.ndarray,
        vf: np.ndarray,
        t_span: List[float],
        lam_guess: np.ndarray,
    ) -> Tuple[np.ndarray, object]:
        """
        求解轨迹优化问题

        使用fsolve求解打靶方程，找到满足终端条件的初始协态。

        Parameters:
            r0, v0, m0: 初始状态
            rf, vf: 终端状态
            t_span: 时间区间
            lam_guess: 初始协态猜测

        Returns:
            lam0_sol: 最优初始协态
            sol: 轨迹解
        """

        # 定义打靶方程
        def shoot_eq(lam0):
            error, _ = self.shoot(lam0, t_span, r0, v0, m0, rf, vf)
            return error

        # 使用数值方法求解打靶方程
        lam0_sol, info, ier, msg = fsolve(
            shoot_eq, lam_guess, full_output=True, xtol=1e-6, maxfev=1000
        )

        if ier != 1:
            warnings.warn(f"fsolve did not converge: {msg}")

        # 获取最优轨迹
        _, sol = self.shoot(lam0_sol, t_span, r0, v0, m0, rf, vf)

        return lam0_sol, sol

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

        尝试多个初始猜测，选择误差最小的解。

        Parameters:
            r0, v0, m0: 初始状态
            rf, vf: 终端状态
            t_span: 时间区间
            guesses: 初始猜测列表（可选）

        Returns:
            result: 包含最优解的字典
        """
        if guesses is None:
            # 默认初始猜测集
            guesses = [
                np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0]),
                np.array([1, -0.1, -0.1, -0.1, -0.1, -0.1, 0.0]),
                np.array([100, 0.01, 0.01, 0.01, 0.01, 0.01, 0.0]),
                np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]),
            ]

        best_lam0 = None
        best_sol = None
        best_error = float("inf")

        for i, lam_guess in enumerate(guesses):
            print(f"尝试初始猜测 {i + 1}/{len(guesses)}")
            self.best_error = float("inf")
            self.best_solution = None

            try:
                lam0_sol, sol = self.solve(r0, v0, m0, rf, vf, t_span, lam_guess)

                if self.best_error < best_error:
                    best_error = self.best_error
                    best_lam0 = lam0_sol
                    best_sol = (
                        self.best_solution if self.best_solution is not None else sol
                    )

                print(f"当前最佳误差: {best_error:.6f}")

                if best_error < 1e-3:
                    print("找到足够精确的解，提前终止")
                    break
            except Exception as e:
                print(f"初始猜测 {i + 1} 失败: {str(e)}")
                continue

        if best_sol is None:
            print("所有初始猜测都失败，使用最后一个解")
            best_lam0 = guesses[-1]
            _, best_sol = self.shoot(best_lam0, t_span, r0, v0, m0, rf, vf)

        return {
            "success": best_error < 1e-3,
            "initial_costate": best_lam0,
            "trajectory": best_sol,
            "final_error": best_error,
            "final_mass": best_sol.y[6, -1] if best_sol else None,
        }

    def extract_trajectory_data(self, sol: object) -> Dict:
        """
        提取轨迹数据

        Parameters:
            sol: 积分结果对象

        Returns:
            data: 包含轨迹数据的字典
        """
        t = sol.t
        r = sol.y[0:3, :]
        v = sol.y[3:6, :]
        m = sol.y[6, :]
        lam_r = sol.y[7:10, :]
        lam_v = sol.y[10:13, :]
        lam_m = sol.y[13, :]

        # 计算控制历史
        u_history = np.zeros(len(t))
        chi_history = np.zeros((3, len(t)))

        for i in range(len(t)):
            z = sol.y[:, i]
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
            z = np.concatenate(
                [
                    data["r"][:, i],
                    data["v"][:, i],
                    [data["m"][i]],
                    data["lam_r"][:, i],
                    data["lam_v"][:, i],
                    [data["lam_m"][i]],
                ]
            )
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
    # 创建示例小行星和航天器类
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
    optimizer = ShootingMethodOptimizer(asteroid, spacecraft)

    # 边界条件（433 Eros示例）
    r0 = np.array([10177, 6956, 8256])
    v0 = np.array([-25, -12, -17])
    m0 = spacecraft.m0
    rf = np.array([676, 5121, 449])
    vf = np.array([0, 0, 0])
    t_span = [0, 770]

    # 优化
    result = optimizer.optimize_with_multiple_guesses(r0, v0, m0, rf, vf, t_span)

    if result["success"] and result["trajectory"]:
        data = optimizer.extract_trajectory_data(result["trajectory"])
        optimizer.plot_results(data, r0, rf)

        print("\n优化结果:")
        print(f"初始协态: {result['initial_costate']}")
        print(f"终端质量: {result['final_mass']:.2f} kg")
        print(f"燃料消耗: {m0 - result['final_mass']:.2f} kg")
