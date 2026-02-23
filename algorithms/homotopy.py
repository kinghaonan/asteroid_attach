#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小行星附着最优轨迹设计 - 同伦法

实现从能量最优到燃料最优的平滑过渡。
使用同伦参数 ζ ∈ [0, 1] 逐步改变目标函数。

核心算法：
- 同伦参数序列生成
- 牛顿迭代求解
- 自适应步长调整
- 路径跟踪

优化流程：
1. ζ=0: 能量最优（连续推力）
2. 0<ζ<1: 混合最优
3. ζ=1: 燃料最优（bang-bang控制）
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Optional
import warnings


class HomotopyOptimizer:
    """
    同伦法轨迹优化器

    通过同伦参数从能量最优过渡到燃料最优。
    在每个同伦步骤使用打靶法求解。

    Attributes:
        asteroid: 小行星模型
        spacecraft: 航天器参数
        n_steps: 同伦步数
        debug_mode: 调试模式
    """

    def __init__(self, asteroid, spacecraft, n_steps: int = 10):
        """
        初始化同伦法优化器

        Parameters:
            asteroid: 小行星对象
            spacecraft: 航天器对象
            n_steps: 同伦步数（默认10）
        """
        self.ast = asteroid
        self.sc = spacecraft
        self.n_steps = n_steps
        self.debug_mode = False

    def set_debug_mode(self, debug: bool):
        """设置调试模式"""
        self.debug_mode = debug

    def optimal_control(
        self, lam_v: np.ndarray, lam_m: float, m: float, zeta: float = 0.0
    ) -> Tuple[np.ndarray, float]:
        """
        计算最优控制（同伦版本）

        混合目标函数：
        J = (1-ζ) * J_energy + ζ * J_fuel

        开关函数：
        S = (1-ζ) + ζ*(1-λ_m) - (I_sp*g0/m)*||λ_v||

        Parameters:
            lam_v: 速度协态
            lam_m: 质量协态
            m: 质量
            zeta: 同伦参数

        Returns:
            chi: 推力方向
            u: 推力大小
        """
        lam_v_norm = np.linalg.norm(lam_v)
        if lam_v_norm < 1e-10:
            return np.zeros(3), 0.0

        chi = -lam_v / lam_v_norm

        # 同伦开关函数
        if zeta < 1e-6:
            # 能量最优
            S = 1 - lam_m - (self.sc.I_sp * self.sc.g0 / m) * lam_v_norm
        else:
            # 混合最优
            S = (
                (1 - zeta)
                + zeta * (1 - lam_m)
                - (self.sc.I_sp * self.sc.g0 / m) * lam_v_norm
            )

        if S < 0:
            u = 1.0
        elif S > 0:
            u = 0.0
        else:
            u = 0.5

        return chi, u

    def dynamics(self, t: float, state: np.ndarray, zeta: float) -> np.ndarray:
        """
        系统动力学方程

        Parameters:
            t: 时间
            state: 状态向量 [r(3), v(3), m, lam_r(3), lam_v(3), lam_m]
            zeta: 同伦参数

        Returns:
            状态导数
        """
        r = state[0:3]
        v = state[3:6]
        m = state[6]
        lam_r = state[7:10]
        lam_v = state[10:13]
        lam_m = state[13]

        chi, u = self.optimal_control(lam_v, lam_m, m, zeta)

        # 环境力
        g = self.ast.gravity_gradient(r)
        omega = self.ast.omega
        coriolis = -2 * np.cross(omega, v)
        centrifugal = -np.cross(omega, np.cross(omega, r))

        # 状态导数
        r_dot = v
        thrust_acc = (self.sc.T_max / m) * u * chi if m > 1e-3 else np.zeros(3)
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

        return np.concatenate(
            [r_dot, v_dot, [m_dot], lam_r_dot, lam_v_dot, [lam_m_dot]]
        )

    def propagate(
        self, zeta: float, t_span: List[float], initial_state: np.ndarray
    ) -> object:
        """
        积分轨迹

        Parameters:
            zeta: 同伦参数
            t_span: 时间区间
            initial_state: 初始状态

        Returns:
            sol: 积分结果
        """
        sol = solve_ivp(
            lambda t, y: self.dynamics(t, y, zeta),
            t_span,
            initial_state,
            method="RK45",
            rtol=1e-6,
            atol=1e-9,
            dense_output=True,
        )
        return sol

    def shooting_residual(
        self,
        x0: np.ndarray,
        t_span: List[float],
        r0: np.ndarray,
        v0: np.ndarray,
        m0: float,
        rf: np.ndarray,
        vf: np.ndarray,
        zeta: float,
    ) -> np.ndarray:
        """
        打靶法残差函数

        Parameters:
            x0: 初始协态 [lam_r0(3), lam_v0(3), lam_m0]
            t_span: 时间区间
            r0, v0, m0: 初始状态
            rf, vf: 终端状态
            zeta: 同伦参数

        Returns:
            residuals: 残差向量
        """
        initial_state = np.concatenate([r0, v0, [m0], x0[0:3], x0[3:6], [x0[6]]])

        sol = self.propagate(zeta, t_span, initial_state)

        if not sol.success:
            return np.ones(7) * 1e6

        tf = t_span[1]
        final_state = sol.sol(tf)
        r_final = final_state[0:3]
        v_final = final_state[3:6]
        lam_m_final = final_state[13]

        # 残差
        residuals = np.concatenate(
            [
                r_final - rf,
                v_final - vf,
                [lam_m_final],  # 横截条件
            ]
        )

        return residuals

    def solve_homotopy(
        self,
        r0: np.ndarray,
        v0: np.ndarray,
        m0: float,
        rf: np.ndarray,
        vf: np.ndarray,
        t_span: List[float],
        init_lam_r: np.ndarray,
        init_lam_v: np.ndarray,
        init_lam_m: float,
    ) -> Dict:
        """
        同伦法主流程

        从能量最优(ζ=0)过渡到燃料最优(ζ=1)。

        Parameters:
            r0, v0, m0: 初始状态
            rf, vf: 终端状态
            t_span: 时间区间
            init_lam_r, init_lam_v, init_lam_m: 初始协态（来自能量最优解）

        Returns:
            result: 优化结果
        """
        print(f"\n=== 同伦法优化 ===")
        print(f"步数: {self.n_steps}")

        zeta_steps = np.linspace(0, 1, self.n_steps)
        current_lam = np.concatenate([init_lam_r, init_lam_v, [init_lam_m]])
        solutions = []

        # 初始解
        t0, tf = t_span
        initial_state = np.concatenate(
            [r0, v0, [m0], init_lam_r, init_lam_v, [init_lam_m]]
        )
        sol = self.propagate(0, t_span, initial_state)

        solutions.append(
            {
                "zeta": 0.0,
                "lam": current_lam.copy(),
                "fuel": m0 - sol.sol(tf)[6] if sol.success else 0,
            }
        )

        # 同伦迭代
        for i in range(1, self.n_steps):
            zeta = zeta_steps[i]
            print(f"\n步骤 {i}/{self.n_steps - 1}, ζ={zeta:.2f}")

            # 目标函数
            def objective(x):
                residuals = self.shooting_residual(x, t_span, r0, v0, m0, rf, vf, zeta)
                return np.sum(residuals**2)

            # 优化
            result = minimize(
                objective,
                current_lam,
                method="trust-constr",
                options={"maxiter": 200, "verbose": 0, "gtol": 1e-5},
            )

            if not result.success:
                print(f"警告: 步骤 {i} 未完全收敛")
                # 尝试增加迭代
                result = minimize(
                    objective,
                    result.x,
                    method="trust-constr",
                    options={"maxiter": 400, "verbose": 0, "gtol": 1e-5},
                )

            current_lam = result.x

            # 评估
            final_sol = self.propagate(
                zeta,
                t_span,
                np.concatenate(
                    [r0, v0, [m0], current_lam[0:3], current_lam[3:6], [current_lam[6]]]
                ),
            )

            m_final = final_sol.sol(tf)[6] if final_sol.success else m0
            fuel_used = m0 - m_final

            print(f"  燃料消耗: {fuel_used:.2f} kg, 残差: {result.fun:.6f}")

            solutions.append(
                {"zeta": zeta, "lam": current_lam.copy(), "fuel": fuel_used}
            )

        # 最终结果
        final_solution = self.propagate(
            1.0,
            t_span,
            np.concatenate(
                [r0, v0, [m0], current_lam[0:3], current_lam[3:6], [current_lam[6]]]
            ),
        )

        if final_solution.success:
            final_state = final_solution.sol(tf)
            r_final = final_state[0:3]
            v_final = final_state[3:6]
            m_final = final_state[6]

            pos_err = np.linalg.norm(r_final - rf)
            vel_err = np.linalg.norm(v_final - vf)

            print(f"\n优化完成:")
            print(f"位置误差: {pos_err:.2f} m")
            print(f"速度误差: {vel_err:.2f} m/s")
            print(f"燃料消耗: {m0 - m_final:.2f} kg")

            return {
                "success": True,
                "solutions": solutions,
                "final_lam": current_lam,
                "final_mass": m_final,
                "fuel_consumption": m0 - m_final,
                "pos_error": pos_err,
                "vel_error": vel_err,
            }
        else:
            return {"success": False, "message": "最终积分失败"}

    def plot_homotopy_trajectory(self, result: Dict, save_path: Optional[str] = None):
        """
        绘制同伦过程

        Parameters:
            result: 优化结果
            save_path: 保存路径（可选）
        """
        if not result["success"]:
            print("优化失败，无法绘图")
            return

        solutions = result["solutions"]
        zetas = [s["zeta"] for s in solutions]
        fuels = [s["fuel"] for s in solutions]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(zetas, fuels, "bo-", linewidth=2, markersize=8)
        ax.set_xlabel("Homotopy Parameter ζ", fontsize=12)
        ax.set_ylabel("Fuel Consumption (kg)", fontsize=12)
        ax.set_title("Homotopy Trajectory: Energy Optimal → Fuel Optimal", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.axvline(0, color="g", linestyle="--", alpha=0.5, label="Energy Optimal")
        ax.axvline(1, color="r", linestyle="--", alpha=0.5, label="Fuel Optimal")
        ax.legend()

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
    optimizer = HomotopyOptimizer(asteroid, spacecraft, n_steps=10)

    r0 = np.array([10177, 6956, 8256])
    v0 = np.array([-25, -12, -17])
    m0 = spacecraft.m0
    rf = np.array([676, 5121, 449])
    vf = np.array([0, 0, 0])
    t_span = [0, 770]

    # 初始协态（假设来自伪谱法）
    init_lam_r = np.array([0.01, 0.01, 0.01])
    init_lam_v = np.array([-0.01, -0.01, -0.01])
    init_lam_m = -0.001

    result = optimizer.solve_homotopy(
        r0, v0, m0, rf, vf, t_span, init_lam_r, init_lam_v, init_lam_m
    )
    optimizer.plot_homotopy_trajectory(result)
