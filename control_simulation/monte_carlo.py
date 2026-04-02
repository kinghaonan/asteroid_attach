#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
蒙特卡洛模拟器

用于进行多次随机模拟，评估轨迹优化的鲁棒性。
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Callable
import time


class MonteCarloSimulator:
    """
    蒙特卡洛模拟器

    通过多次随机模拟评估轨迹优化算法的鲁棒性和性能。

    Attributes:
        optimizer: 轨迹优化器
        asteroid: 小行星模型
        spacecraft: 航天器参数
        n_simulations: 模拟次数
    """

    def __init__(self, optimizer, asteroid, spacecraft, n_simulations: int = 100):
        """
        初始化蒙特卡洛模拟器

        Parameters:
            optimizer: 轨迹优化器实例
            asteroid: 小行星对象
            spacecraft: 航天器对象
            n_simulations: 模拟次数（默认100）
        """
        self.optimizer = optimizer
        self.ast = asteroid
        self.sc = spacecraft
        self.n_simulations = n_simulations
        self.results = []

    def perturb_initial_state(
        self,
        r0: np.ndarray,
        v0: np.ndarray,
        position_noise: float = 100.0,
        velocity_noise: float = 5.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        扰动初始状态

        Parameters:
            r0: 初始位置
            v0: 初始速度
            position_noise: 位置扰动标准差（m）
            velocity_noise: 速度扰动标准差（m/s）

        Returns:
            r_perturbed, v_perturbed: 扰动后的状态
        """
        r_perturbed = r0 + np.random.normal(0, position_noise, 3)
        v_perturbed = v0 + np.random.normal(0, velocity_noise, 3)
        return r_perturbed, v_perturbed

    def run_simulation(
        self,
        r0: np.ndarray,
        v0: np.ndarray,
        m0: float,
        rf: np.ndarray,
        vf: np.ndarray,
        t_span: List[float],
        position_noise: float = 100.0,
        velocity_noise: float = 5.0,
    ) -> Dict:
        """
        运行单次模拟

        Parameters:
            r0, v0, m0: 标称初始状态
            rf, vf: 终端状态
            t_span: 时间区间
            position_noise: 位置扰动
            velocity_noise: 速度扰动

        Returns:
            result: 模拟结果字典
        """
        # 扰动初始状态
        r_perturbed, v_perturbed = self.perturb_initial_state(
            r0, v0, position_noise, velocity_noise
        )

        try:
            # 执行优化
            if hasattr(self.optimizer, "optimize_with_multiple_guesses"):
                result = self.optimizer.optimize_with_multiple_guesses(
                    r_perturbed, v_perturbed, m0, rf, vf, t_span
                )
            else:
                result = self.optimizer.optimize(
                    r_perturbed, v_perturbed, m0, rf, vf, t_span
                )

            success = result.get("success", False)

            return {
                "success": success,
                "r0_perturbed": r_perturbed,
                "v0_perturbed": v_perturbed,
                "final_mass": result.get("final_mass", m0),
                "fuel_consumption": m0 - result.get("final_mass", m0),
                "error": result.get("final_error", float("inf")),
            }
        except Exception as e:
            return {
                "success": False,
                "r0_perturbed": r_perturbed,
                "v0_perturbed": v_perturbed,
                "error": str(e),
            }

    def run_monte_carlo(
        self,
        r0: np.ndarray,
        v0: np.ndarray,
        m0: float,
        rf: np.ndarray,
        vf: np.ndarray,
        t_span: List[float],
        position_noise: float = 100.0,
        velocity_noise: float = 5.0,
    ) -> Dict:
        """
        运行蒙特卡洛模拟

        Parameters:
            r0, v0, m0: 标称初始状态
            rf, vf: 终端状态
            t_span: 时间区间
            position_noise: 位置扰动标准差
            velocity_noise: 速度扰动标准差

        Returns:
            stats: 统计结果
        """
        print(f"\n=== 蒙特卡洛模拟 ===")
        print(f"模拟次数: {self.n_simulations}")
        print(f"位置扰动: ±{position_noise} m")
        print(f"速度扰动: ±{velocity_noise} m/s")

        self.results = []
        start_time = time.time()

        for i in range(self.n_simulations):
            print(f"\n模拟 {i + 1}/{self.n_simulations}")

            result = self.run_simulation(
                r0, v0, m0, rf, vf, t_span, position_noise, velocity_noise
            )

            self.results.append(result)

            if result["success"]:
                print(f"  成功 - 燃料消耗: {result['fuel_consumption']:.2f} kg")
            else:
                print(f"  失败")

        elapsed_time = time.time() - start_time

        # 统计分析
        stats = self.analyze_results()
        stats["elapsed_time"] = elapsed_time

        print(f"\n=== 模拟完成 ===")
        print(f"总耗时: {elapsed_time:.2f} s")
        print(f"成功率: {stats['success_rate']:.1f}%")

        return stats

    def analyze_results(self) -> Dict:
        """
        分析模拟结果

        Returns:
            stats: 统计结果字典
        """
        successes = [r for r in self.results if r.get("success", False)]
        failures = [r for r in self.results if not r.get("success", False)]

        success_rate = len(successes) / len(self.results) * 100

        stats = {
            "n_total": len(self.results),
            "n_success": len(successes),
            "n_failure": len(failures),
            "success_rate": success_rate,
        }

        if successes:
            fuel_consumptions = [r["fuel_consumption"] for r in successes]
            stats["fuel_mean"] = np.mean(fuel_consumptions)
            stats["fuel_std"] = np.std(fuel_consumptions)
            stats["fuel_min"] = np.min(fuel_consumptions)
            stats["fuel_max"] = np.max(fuel_consumptions)

        return stats

    def plot_results(self, save_path: Optional[str] = None):
        """
        绘制蒙特卡洛结果

        Parameters:
            save_path: 保存路径（可选）
        """
        if not self.results:
            print("没有模拟结果可以绘制")
            return

        fig = plt.figure(figsize=(15, 5))

        # 子图1: 成功率
        ax1 = fig.add_subplot(1, 3, 1)
        successes = [r for r in self.results if r.get("success", False)]
        failures = [r for r in self.results if not r.get("success", False)]

        labels = ["Success", "Failure"]
        sizes = [len(successes), len(failures)]
        colors = ["#2ecc71", "#e74c3c"]

        ax1.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
        ax1.set_title("Success Rate")

        # 子图2: 燃料消耗分布
        ax2 = fig.add_subplot(1, 3, 2)
        if successes:
            fuel_consumptions = [r["fuel_consumption"] for r in successes]
            ax2.hist(fuel_consumptions, bins=20, color="#3498db", edgecolor="black")
            ax2.set_xlabel("Fuel Consumption (kg)")
            ax2.set_ylabel("Frequency")
            ax2.set_title("Fuel Consumption Distribution")
            ax2.axvline(
                np.mean(fuel_consumptions),
                color="r",
                linestyle="--",
                label=f"Mean: {np.mean(fuel_consumptions):.2f} kg",
            )
            ax2.legend()

        # 子图3: 初始位置扰动分布
        ax3 = fig.add_subplot(1, 3, 3, projection="3d")
        r_perturbed = np.array([r["r0_perturbed"] for r in self.results])
        ax3.scatter(
            r_perturbed[:, 0],
            r_perturbed[:, 1],
            r_perturbed[:, 2],
            c=["green" if r["success"] else "red" for r in self.results],
            alpha=0.5,
        )
        ax3.set_xlabel("X (m)")
        ax3.set_ylabel("Y (m)")
        ax3.set_zlabel("Z (m)")
        ax3.set_title("Initial Position Dispersion")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"结果图已保存至: {save_path}")

        plt.show()


def _state_derivative(asteroid, spacecraft, state: np.ndarray, control: np.ndarray) -> np.ndarray:
    r = state[0:3]
    v = state[3:6]
    m = max(state[6], 1.0)

    # Gravity
    if hasattr(asteroid, "compute_gravity"):
        g = asteroid.compute_gravity(r)
    else:
        r_norm = np.linalg.norm(r)
        if r_norm > 1e-10:
            g = -asteroid.mu * r / (r_norm ** 3)
        else:
            g = np.zeros(3)

    omega = getattr(asteroid, "omega", np.zeros(3))
    coriolis = -2 * np.cross(omega, v)
    centrifugal = -np.cross(omega, np.cross(omega, r))

    thrust_mag = np.linalg.norm(control)
    thrust_accel = control / m

    r_dot = v
    v_dot = g + coriolis + centrifugal + thrust_accel
    m_dot = -thrust_mag / (spacecraft.I_sp * spacecraft.g0) if thrust_mag > 0 else 0.0

    return np.concatenate([r_dot, v_dot, [m_dot]])


def simulate_tracking(
    tracker,
    asteroid,
    spacecraft,
    r0: np.ndarray,
    v0: np.ndarray,
    m0: float,
    t_span: List[float],
    dt: float,
) -> Dict:
    """闭环PID轨迹跟踪仿真"""
    t0, tf = t_span
    n_steps = int(np.ceil((tf - t0) / dt))
    t_nodes = np.linspace(t0, tf, n_steps + 1)

    X = np.zeros((n_steps + 1, 7))
    U = np.zeros((n_steps, 3))
    X[0, 0:3] = r0
    X[0, 3:6] = v0
    X[0, 6] = m0

    tracker.reset()

    for k in range(n_steps):
        u, _ = tracker.compute_control(
            t_nodes[k], X[k, 0:3], X[k, 3:6], X[k, 6], dt
        )
        U[k] = u

        k1 = _state_derivative(asteroid, spacecraft, X[k], u)
        k2 = _state_derivative(asteroid, spacecraft, X[k] + 0.5 * dt * k1, u)
        k3 = _state_derivative(asteroid, spacecraft, X[k] + 0.5 * dt * k2, u)
        k4 = _state_derivative(asteroid, spacecraft, X[k] + dt * k3, u)

        X[k + 1] = X[k] + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        X[k + 1, 6] = max(X[k + 1, 6], 0.2 * m0)

    ref_final = tracker.get_reference_at_time(tf)
    pos_err = float(np.linalg.norm(X[-1, 0:3] - ref_final["position"]))
    vel_err = float(np.linalg.norm(X[-1, 3:6] - ref_final["velocity"]))

    return {
        "t": t_nodes,
        "r": X[:, 0:3],
        "v": X[:, 3:6],
        "m": X[:, 6],
        "U": U,
        "pos_error": pos_err,
        "vel_error": vel_err,
        "success": True,
    }


class TrackingMonteCarloSimulator:
    """基于PID闭环跟踪的蒙特卡罗模拟"""

    def __init__(self, tracker, asteroid, spacecraft, n_simulations: int = 50):
        self.tracker = tracker
        self.ast = asteroid
        self.sc = spacecraft
        self.n_simulations = n_simulations
        self.results = []

    def run_monte_carlo(
        self,
        r0: np.ndarray,
        v0: np.ndarray,
        m0: float,
        t_span: List[float],
        dt: float,
        position_noise: float = 20.0,
        velocity_noise: float = 1.0,
        max_position_error: float = 1.0,
        max_velocity_error: float = 1.0,
    ) -> Dict:
        self.results = []

        for _ in range(self.n_simulations):
            r_perturbed = r0 + np.random.normal(0, position_noise, 3)
            v_perturbed = v0 + np.random.normal(0, velocity_noise, 3)

            sim = simulate_tracking(
                self.tracker,
                self.ast,
                self.sc,
                r_perturbed,
                v_perturbed,
                m0,
                t_span,
                dt,
            )

            success = (sim["pos_error"] <= max_position_error) and (
                sim["vel_error"] <= max_velocity_error
            )

            self.results.append(
                {
                    "success": success,
                    "pos_error": sim["pos_error"],
                    "vel_error": sim["vel_error"],
                }
            )

        successes = [r for r in self.results if r["success"]]
        success_rate = len(successes) / len(self.results) * 100.0
        pos_errs = [r["pos_error"] for r in self.results]
        vel_errs = [r["vel_error"] for r in self.results]

        return {
            "n_total": len(self.results),
            "success_rate": success_rate,
            "pos_error_max": float(np.max(pos_errs)),
            "vel_error_max": float(np.max(vel_errs)),
            "pos_error_mean": float(np.mean(pos_errs)),
            "vel_error_mean": float(np.mean(vel_errs)),
        }

    def save_report(self, filepath: str):
        """
        保存模拟报告

        Parameters:
            filepath: 报告文件路径
        """
        stats = self.analyze_results()

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("# 蒙特卡洛模拟报告\n\n")
            f.write(f"模拟次数: {stats['n_total']}\n")
            f.write(f"成功次数: {stats['n_success']}\n")
            f.write(f"失败次数: {stats['n_failure']}\n")
            f.write(f"成功率: {stats['success_rate']:.1f}%\n\n")

            if "fuel_mean" in stats:
                f.write("## 燃料消耗统计\n")
                f.write(f"平均值: {stats['fuel_mean']:.2f} kg\n")
                f.write(f"标准差: {stats['fuel_std']:.2f} kg\n")
                f.write(f"最小值: {stats['fuel_min']:.2f} kg\n")
                f.write(f"最大值: {stats['fuel_max']:.2f} kg\n")

        print(f"报告已保存至: {filepath}")
