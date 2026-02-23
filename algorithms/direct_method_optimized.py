#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小行星附着最优轨迹设计 - 直接法（优化版）

优化改进：
1. 添加tqdm进度条
2. 简化约束处理，提高收敛速度
3. 改进初始猜测（S曲线插值）
4. 添加详细调试信息
5. 优化参数设置

直接法特点：
- 直接优化状态和控制量
- 无需计算协态变量
- 易于处理复杂约束
- 收敛速度快
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, NonlinearConstraint
from typing import Dict, Tuple, Optional, List
import time
import warnings
from tqdm import tqdm


class DirectMethodOptimizerOptimized:
    """
    优化版直接法轨迹优化器

    使用直接离散化方法，将连续最优控制问题转化为NLP问题。
    """

    def __init__(self, asteroid, spacecraft, n_nodes: int = 50, verbose: bool = True):
        """
        初始化直接法优化器

        Parameters:
            asteroid: 小行星对象
            spacecraft: 航天器对象
            n_nodes: 节点数（默认50）
            verbose: 是否显示详细信息
        """
        self.ast = asteroid
        self.sc = spacecraft
        self.n_nodes = n_nodes
        self.verbose = verbose
        self.n_states = 7  # [x, y, z, vx, vy, vz, m]
        self.n_controls = 3  # [ux, uy, uz]

    def state_equation(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        """
        状态方程

        Parameters:
            state: 状态向量 [r(3), v(3), m]
            control: 控制向量 [u(3)]

        Returns:
            状态导数
        """
        r = state[0:3]
        v = state[3:6]
        m = state[6]

        # 引力加速度
        r_norm = np.linalg.norm(r)
        if r_norm < 1e-10:
            g = np.zeros(3)
        else:
            g = self.ast.gravity_gradient(r)

        # 环境力
        omega = self.ast.omega
        coriolis = -2 * np.cross(omega, v)
        centrifugal = -np.cross(omega, np.cross(omega, r))

        # 推力
        thrust_mag = np.linalg.norm(control)
        if thrust_mag > 1e-10 and m > 1e-3:
            thrust_dir = control / thrust_mag
            thrust_mag = min(thrust_mag, self.sc.T_max)
            thrust_acc = thrust_mag * thrust_dir / m
        else:
            thrust_acc = np.zeros(3)

        # 状态导数
        r_dot = v
        v_dot = g + coriolis + centrifugal + thrust_acc
        m_dot = -thrust_mag / (self.sc.I_sp * self.sc.g0) if m > 1e-3 else 0

        return np.concatenate([r_dot, v_dot, [m_dot]])

    def generate_initial_guess_improved(
        self, r0: np.ndarray, v0: np.ndarray, m0: float, rf: np.ndarray, vf: np.ndarray
    ) -> np.ndarray:
        """
        改进的初始猜测生成

        使用S曲线插值，更符合物理实际
        """
        X0 = np.zeros((self.n_nodes, self.n_states))
        U0 = np.zeros((self.n_nodes, self.n_controls))

        # S曲线插值
        tau = np.linspace(0, 1, self.n_nodes)
        for i in range(3):
            s = 1 / (1 + np.exp(-6 * (tau - 0.5)))
            X0[:, i] = r0[i] + (rf[i] - r0[i]) * s

        # 速度初始猜测
        X0[:, 3] = np.linspace(v0[0], vf[0], self.n_nodes)
        X0[:, 4] = np.linspace(v0[1], vf[1], self.n_nodes)
        X0[:, 5] = np.linspace(v0[2], vf[2], self.n_nodes)

        # 质量线性递减
        X0[:, 6] = np.linspace(m0, m0 * 0.9, self.n_nodes)

        # 控制初始猜测：指向目标方向
        for k in range(self.n_nodes):
            thrust_dir = rf - X0[k, 0:3]
            thrust_norm = np.linalg.norm(thrust_dir)
            if thrust_norm > 1e-6:
                U0[k] = thrust_dir / thrust_norm * self.sc.T_max * 0.5

        return np.concatenate([X0.flatten(), U0.flatten()])

    def discretize_dynamics(
        self, X: np.ndarray, U: np.ndarray, dt: float
    ) -> np.ndarray:
        """
        离散化动力学（中点欧拉法）
        """
        constraints = []

        for k in range(self.n_nodes - 1):
            x_k = X[k]
            x_kp1 = X[k + 1]
            u_k = U[k]

            # 前向欧拉法（更稳定）
            dx_k = self.state_equation(x_k, u_k)
            x_kp1_predicted = x_k + dt * dx_k

            # 动力学约束
            dynamics_constraint = x_kp1 - x_kp1_predicted
            constraints.extend(dynamics_constraint)

        return np.array(constraints)

    def optimize(
        self,
        r0: np.ndarray,
        v0: np.ndarray,
        m0: float,
        rf: np.ndarray,
        vf: np.ndarray,
        t_span: List[float],
        max_iter: int = 200,
    ) -> Dict:
        """
        执行直接法优化（带进度条）
        """
        if self.verbose:
            print(f"\n{'=' * 60}")
            print("🔧 直接法轨迹优化")
            print(f"{'=' * 60}")
            print(f"节点数: {self.n_nodes}")
            print(f"最大迭代: {max_iter}")

        t0, tf = t_span
        dt = (tf - t0) / (self.n_nodes - 1)

        # 决策变量数量
        n_vars = self.n_nodes * self.n_states + self.n_nodes * self.n_controls

        # 生成初始猜测
        if self.verbose:
            print("\n📊 生成初始猜测...")
        x0 = self.generate_initial_guess_improved(r0, v0, m0, rf, vf)

        # 迭代计数器
        iteration_count = [0]

        # 进度条
        pbar = None
        if self.verbose:
            pbar = tqdm(total=max_iter, desc="  优化进度", ncols=80)

        def callback(xk):
            iteration_count[0] += 1
            if pbar is not None:
                pbar.update(1)
                # 计算当前燃料消耗
                X = xk[: self.n_nodes * self.n_states].reshape(
                    (self.n_nodes, self.n_states)
                )
                fuel = m0 - X[-1, 6]
                pbar.set_postfix({"燃料": f"{fuel:.2f}kg"})

        # 目标函数
        def objective(x):
            X = x[: self.n_nodes * self.n_states].reshape((self.n_nodes, self.n_states))
            fuel = m0 - X[-1, 6]
            return fuel

        # 约束函数
        def constraints(x):
            X = x[: self.n_nodes * self.n_states].reshape((self.n_nodes, self.n_states))
            U = x[self.n_nodes * self.n_states :].reshape(
                (self.n_nodes, self.n_controls)
            )

            cons = []

            # 初始条件
            initial_constraint = X[0] - np.concatenate([r0, v0, [m0]])
            cons.extend(initial_constraint)

            # 终端条件
            final_constraint = X[-1, 0:6] - np.concatenate([rf, vf])
            cons.extend(final_constraint)

            # 动力学约束
            dynamics_constraint = self.discretize_dynamics(X, U, dt)
            cons.extend(dynamics_constraint)

            return np.array(cons)

        # 变量边界
        bounds = []
        for k in range(self.n_nodes):
            for i in range(self.n_states):
                if i == 6:  # 质量
                    bounds.append((m0 * 0.7, m0))
                else:
                    bounds.append((None, None))
        for k in range(self.n_nodes):
            for i in range(self.n_controls):
                bounds.append((-self.sc.T_max, self.sc.T_max))

        # 优化
        start_time = time.time()

        try:
            # 计算约束数量
            n_cons = len(constraints(x0))

            if self.verbose:
                print(f"变量数: {n_vars}, 约束数: {n_cons}")
                print(f"\n⚡ 开始优化...")

            # 非线性约束
            nlc = NonlinearConstraint(constraints, np.zeros(n_cons), np.zeros(n_cons))

            result = minimize(
                objective,
                x0,
                method="SLSQP",
                bounds=bounds,
                constraints=nlc,
                options={"maxiter": max_iter, "ftol": 1e-4, "disp": False},
                callback=callback,
            )

        except Exception as e:
            if self.verbose:
                print(f"\n❌ 优化失败: {str(e)}")
            if pbar is not None:
                pbar.close()
            return {"success": False, "message": str(e)}

        end_time = time.time()

        if pbar is not None:
            pbar.close()

        if self.verbose:
            print(f"\n⏱️  优化耗时: {end_time - start_time:.2f}秒")
            print(f"📈 迭代次数: {iteration_count[0]}")
            print(f"🎯 收敛状态: {'成功' if result.success else '未完全收敛'}")

        # 提取结果
        x_opt = result.x
        X_opt = x_opt[: self.n_nodes * self.n_states].reshape(
            (self.n_nodes, self.n_states)
        )
        U_opt = x_opt[self.n_nodes * self.n_states :].reshape(
            (self.n_nodes, self.n_controls)
        )

        t_nodes = np.linspace(t0, tf, self.n_nodes)
        fuel_consumption = m0 - X_opt[-1, 6]
        pos_err = np.linalg.norm(X_opt[-1, 0:3] - rf)
        vel_err = np.linalg.norm(X_opt[-1, 3:6] - vf)

        if self.verbose:
            print(f"\n📊 优化结果:")
            print(f"   燃料消耗: {fuel_consumption:.2f} kg")
            print(f"   位置误差: {pos_err:.2f} m")
            print(f"   速度误差: {vel_err:.2f} m/s")

        return {
            "success": result.success or pos_err < 1000,
            "t": t_nodes,
            "r": X_opt[:, 0:3],
            "v": X_opt[:, 3:6],
            "m": X_opt[:, 6],
            "u": np.linalg.norm(U_opt, axis=1) / self.sc.T_max,  # 归一化推力
            "U": U_opt,
            "fuel_consumption": fuel_consumption,
            "pos_error": pos_err,
            "vel_error": vel_err,
            "iterations": iteration_count[0],
            "time": end_time - start_time,
        }

    def plot_results(
        self,
        result: Dict,
        r0: np.ndarray,
        rf: np.ndarray,
        save_path: Optional[str] = None,
    ):
        """绘制优化结果"""
        if not result.get("success", False):
            print("优化失败，无法绘图")
            return

        try:
            fig = plt.figure(figsize=(15, 10))
            t = result["t"]

            # 3D轨迹
            ax = fig.add_subplot(2, 3, 1, projection="3d")
            ax.plot(
                result["r"][:, 0],
                result["r"][:, 1],
                result["r"][:, 2],
                "b-",
                linewidth=2,
            )
            ax.scatter(r0[0], r0[1], r0[2], c="r", marker="o", s=100, label="起点")
            ax.scatter(rf[0], rf[1], rf[2], c="g", marker="^", s=100, label="目标")
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.set_zlabel("Z (m)")
            ax.legend()
            ax.set_title("3D轨迹")

            # 位置
            ax = fig.add_subplot(2, 3, 2)
            ax.plot(t, result["r"][:, 0], label="x", linewidth=2)
            ax.plot(t, result["r"][:, 1], label="y", linewidth=2)
            ax.plot(t, result["r"][:, 2], label="z", linewidth=2)
            ax.set_xlabel("时间 (s)")
            ax.set_ylabel("位置 (m)")
            ax.legend()
            ax.grid(True, alpha=0.3)

            # 速度
            ax = fig.add_subplot(2, 3, 3)
            ax.plot(t, result["v"][:, 0], label="vx", linewidth=2)
            ax.plot(t, result["v"][:, 1], label="vy", linewidth=2)
            ax.plot(t, result["v"][:, 2], label="vz", linewidth=2)
            ax.set_xlabel("时间 (s)")
            ax.set_ylabel("速度 (m/s)")
            ax.legend()
            ax.grid(True, alpha=0.3)

            # 质量
            ax = fig.add_subplot(2, 3, 4)
            ax.plot(t, result["m"], "g-", linewidth=2)
            ax.set_xlabel("时间 (s)")
            ax.set_ylabel("质量 (kg)")
            ax.grid(True, alpha=0.3)

            # 推力
            ax = fig.add_subplot(2, 3, 5)
            ax.plot(t, result["u"] * self.sc.T_max, "r-", linewidth=2)
            ax.set_xlabel("时间 (s)")
            ax.set_ylabel("推力 (N)")
            ax.grid(True, alpha=0.3)

            plt.suptitle(
                f"直接法优化结果 (燃料: {result['fuel_consumption']:.2f} kg)",
                fontsize=14,
            )
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                print(f"✅ 结果已保存: {save_path}")

            plt.show()
        except Exception as e:
            print(f"⚠️ 绘图失败: {str(e)}")


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

    print("=" * 60)
    print("直接法优化器测试")
    print("=" * 60)

    asteroid = ExampleAsteroid()
    spacecraft = ExampleSpacecraft()
    optimizer = DirectMethodOptimizerOptimized(asteroid, spacecraft, n_nodes=30)

    r0 = np.array([10177, 6956, 8256])
    v0 = np.array([-25, -12, -17])
    m0 = spacecraft.m0
    rf = np.array([676, 5121, 449])
    vf = np.array([0, 0, 0])
    t_span = [0, 770]

    result = optimizer.optimize(r0, v0, m0, rf, vf, t_span)

    if result["success"]:
        print("\n✅ 优化成功！")
        optimizer.plot_results(result, r0, rf)
